# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
from datetime import timedelta

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps
from megatron.plugin.platform import get_platform


cur_platform = get_platform()


def get_current_device():
    """Get current accelerator device as torch.device object.

    Returns platform-agnostic device (cuda/xpu/npu) based on the active platform.
    Use this instead of hardcoded device=get_current_device() for cross-platform compatibility.
    """
    return cur_platform.device(cur_platform.current_device())


def get_device_str():
    """Get current accelerator device type as string (e.g., 'cuda', 'xpu', 'npu').

    Returns device type without index, matching tensor.device.type behavior.
    """
    return cur_platform.device_name()


class TestModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


def clear_nvte_env_vars():
    """Clear NVTE env vars set by conftest set_env fixture."""
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)


class Utils:

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    inited = False
    store = None

    @staticmethod
    def initialize_distributed():

        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            device_count = cur_platform.device_count()
            if device_count > 0:
                cur_platform.set_device(Utils.rank % device_count)
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            ######## FlagScale Begin ########
            # Multi-backend CI can spend more than one minute compiling kernels on a rank.
            # Keep the test store alive long enough for slower ranks to reach group creation.
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=5)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=5))
            ######## FlagScale End ########

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend=Utils.get_backend(), world_size=Utils.world_size, rank=Utils.rank, store=store
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def get_backend():
        """Get the appropriate distributed backend for the current platform.

        Returns 'mccl' for MUSA, 'nccl' for all other platforms.
        Can be overridden via DISTRIBUTED_BACKEND environment variable.
        """
        default_backend = 'mccl' if cur_platform.device_name() == 'musa' else 'nccl'
        return os.getenv('DISTRIBUTED_BACKEND', default_backend)

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = cur_platform.device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['LOCAL_RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def _destroy_model_parallel_groups():
        # parallel_state resets device-group references without destroying the
        # groups. Release groups owned by that state between tests while keeping
        # the default group available to tests that create their own subgroups.
        groups = tuple(ps._global_process_group_list or ())
        ps.destroy_model_parallel()
        # In the current Kunlunxin CI image, explicit subgroup shutdown can
        # invalidate device handles used by the next test. Keep these groups
        # registered until session teardown; parallel_state has already performed
        # its normal reference and Gloo cleanup.
        if os.getenv('MEGATRON_TEST_PLATFORM') == 'kunlunxin':
            return
        for group in groups:
            # Gloo groups may already have been destroyed by parallel_state.
            if group is not None and group in torch.distributed.distributed_c10d._world.pg_map:
                torch.distributed.destroy_process_group(group)

    @staticmethod
    def destroy_model_parallel():
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        if not Utils.inited:
            return

        try:
            # Flush pending device work before tearing down process groups.
            cur_platform.synchronize()
            torch.distributed.barrier()
        except Exception:
            Utils.inited = False
            return
        Utils._destroy_model_parallel_groups()
        Utils.inited = False
        cur_platform.empty_cache()  # FlagScale Modify

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        # Need to unset these variables to make sure previous
        # tests setting them doesn't interfere current test.
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        Utils._destroy_model_parallel_groups()
        Utils.initialize_distributed()
        if cur_platform.device_name() == 'musa' and 'create_gloo_process_groups' not in kwargs:
            kwargs['create_gloo_process_groups'] = False
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

    @staticmethod
    def fake_initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
    ):
        """Used for layer-wise UT as a proxy for NeMo-style intialization."""
        ps.set_tensor_model_parallel_world_size(tensor_model_parallel_size)
        ps.set_tensor_model_parallel_rank(0)

        ps.set_expert_model_parallel_world_size(expert_model_parallel_size)
        ps.set_expert_model_parallel_rank(0)
        if virtual_pipeline_model_parallel_size is not None:
            ps.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        ps.set_virtual_pipeline_model_parallel_rank(0)

        ps.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)
