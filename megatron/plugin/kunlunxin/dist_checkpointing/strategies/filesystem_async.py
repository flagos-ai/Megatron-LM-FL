"""KunLunXin plugin implementations for filesystem async checkpointing."""

import importlib.util
import inspect
import logging
import os
import warnings
from time import time
from typing import List, Tuple, Union

import torch
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItemType
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    WriteBucket,
    _process_memory,
    _split_by_separation_hint,
    _split_by_size_and_type,
    get_write_results_queue,
    logger,
)


def _bridge_available() -> bool:
    """Check if megatron.bridge is available."""
    return importlib.util.find_spec("megatron.bridge") is not None


def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
    """Prepare async checkpoint write buckets for KunLunXin bridge path.

    Args:
        plan: Save plan from PyTorch distributed planner.
        planner: Save planner for resolving bytes and tensor data.
    """
    if not _bridge_available():
        return FileSystemWriterAsync.prepare_write_data.__wrapped__(self, plan, planner)

    storage_plan: _StoragePrefix = plan.storage_data
    start = time()
    logger.debug(f"thread_count: {self.thread_count}, time: {start}")
    if self.separation_hint:
        assert self.thread_count > 1, "thread_count must be at least 2 if separation_hint is provided"
    bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
    item_buckets = _split_by_size_and_type(bins, plan.items)
    logger.debug(f"bucket_prep, time: {time() - start}")

    start = time()
    file_count = 0

    def gen_file(prefix=""):
        """Generate a unique checkpoint file name with optional prefix."""
        nonlocal file_count
        file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
        file_count += 1
        return file_name

    def _clone_or_dequantize_if_needed(ten: torch.Tensor):
        """Detach and dequantize GPU tensors if needed."""
        ten = ten.detach()
        if ten.device.type != "cpu" and ten.device.type == "cuda" and "dequantize" in type(ten).__dict__:
            ten = ten.dequantize()
        return ten

    self.write_buckets = []
    for group_name, group_buckets in _split_by_separation_hint(
        item_buckets, self.separation_hint
    ).items():
        for bucket in group_buckets:
            bytes_data = [
                (item, planner.resolve_data(item))
                for item in bucket
                if item.type == WriteItemType.BYTE_IO
            ]
            tensor_data = [
                (item, _clone_or_dequantize_if_needed(planner.resolve_data(item)))
                for item in bucket
                if item.type != WriteItemType.BYTE_IO
            ]
            if len(bytes_data) > 0 or len(tensor_data) > 0:
                file_name = gen_file(prefix=group_name)
                self.write_buckets.append(
                    (
                        os.path.join(self.checkpoint_dir, file_name),
                        file_name,
                        (bytes_data, tensor_data),
                    )
                )

    if len(self.write_buckets) > 0:
        assert len(self.write_buckets) <= self.thread_count, (
            len(self.write_buckets),
            self.thread_count,
        )
        self.results_queue = get_write_results_queue()
    else:
        self.results_queue = None
    end = time()
    logger.debug(f"D2H and push, time: {end - start}")


def preload_tensors_kunlunxin(
    write_buckets: List[WriteBucket], non_blocking=True
) -> List[WriteBucket]:
    """Preload tensors with blocking D2H copies for KunLunXin.

    Args:
        write_buckets: List of write buckets to preload.
        non_blocking: Whether to use non-blocking D2H (forced to False).

    Returns:
        List of write buckets with tensors moved to CPU.
    """
    if _bridge_available():
        return write_buckets

    if non_blocking:
        warnings.warn(
            "D2H might fail when `non_blocking` is True, force it to be False for KunLunXin."
        )
    return FileSystemWriterAsync.preload_tensors.__wrapped__(write_buckets, non_blocking=False)


def write_preloaded_data(
    transform_list,
    local_proc_idx: int,
    write_bucket: WriteBucket,
    results_queue,
    count_queue,
    use_fsync: bool,
    **kwargs,
) -> Union[Tuple[int, Exception], None]:
    """Write preloaded checkpoint data for KunLunXin bridge path.

    Args:
        transform_list: Storage writer transforms.
        local_proc_idx: Index of the worker performing writing.
        write_bucket: Data to write to storage.
        results_queue: Queue to return write results (may be None for main worker).
        count_queue: Queue to signal task completion (may be None).
        use_fsync: If True, call os.fsync at end of saving.

    Returns:
        Tuple of (proc_idx, results) or (proc_idx, exception), or None.
    """
    if not _bridge_available():
        return FileSystemWriterAsync.write_preloaded_data.__wrapped__(
            transform_list,
            local_proc_idx,
            write_bucket,
            results_queue,
            count_queue,
            use_fsync,
            **kwargs,
        )

    logger = logging.getLogger(__name__)
    logger.debug(f"{local_proc_idx} started")
    mem_before = _process_memory()
    use_msc = kwargs.get("use_msc", False)

    local_results = []
    try:
        file_name, storage_key, (bytes_data, tensor_data) = write_bucket
        extra_kwargs = {}
        if "serialization_format" in inspect.signature(_write_item).parameters:
            from torch.distributed.checkpoint.filesystem import SerializationFormat

            extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
        if use_msc:
            import multistorageclient as msc

            open_file = msc.open
        else:
            open_file = open
        with open_file(file_name, "wb") as stream:
            for write_item, data in bytes_data:
                local_results.append(
                    _write_item(
                        *transform_list, stream, data, write_item, storage_key, **extra_kwargs
                    )
                )

            for write_item, tensor in tensor_data:
                assert tensor.is_cpu
                if tensor.dtype != torch.bfloat16:
                    tensor = tensor.bfloat16()
                if tensor.untyped_storage().size() != tensor.numel() * tensor.itemsize:
                    tensor = tensor.clone()
                local_results.append(
                    _write_item(
                        *transform_list, stream, tensor, write_item, storage_key, **extra_kwargs
                    )
                )

            if use_fsync:
                if use_msc:
                    stream.fsync()
                else:
                    os.fsync(stream.fileno())
        local_output = (local_proc_idx, local_results)
    except Exception as e:
        logger.debug(f"{local_proc_idx} failed")
        local_output = (local_proc_idx, e)
    if results_queue is not None:
        results_queue.put(local_output)
    if count_queue is not None:
        count_queue.get()
        count_queue.task_done()

    mem_after = _process_memory()
    logger.debug(
        f"{local_proc_idx} consumed: {mem_after - mem_before},"
        f" before: {mem_before}, after: {mem_after}"
    )
    return local_output

