# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.strategies import async_utils
from megatron.core.dist_checkpointing.strategies import torch as torch_strategy


def test_async_request_sync_freeze_preload_and_finalize_paths(monkeypatch):
    calls = []

    def async_fn(*args, **kwargs):
        calls.append(("async", args, kwargs))

    def preload():
        calls.append("preload")
        return ["cpu-bucket"]

    req = async_utils.AsyncRequest(
        async_fn=async_fn,
        async_fn_args=("planner", ["gpu-bucket"], "storage"),
        async_fn_kwargs={"flag": True},
        finalize_fns=[lambda: calls.append("finalize-0")],
        preload_fn=preload,
    )
    req.add_finalize_fn(lambda: calls.append("finalize-1"))
    frozen = req.freeze()
    assert frozen.is_frozen is True
    with pytest.raises(RuntimeError, match="frozen"):
        frozen.add_finalize_fn(lambda: None)

    monkeypatch.setattr(async_utils.torch.distributed, "barrier", lambda: calls.append("barrier"))
    req.execute_sync()
    assert calls == [
        "preload",
        ("async", ("planner", ["cpu-bucket"], "storage"), {"flag": True}),
        "barrier",
        "finalize-0",
        "finalize-1",
    ]

    no_op_calls = []
    no_op = async_utils.AsyncRequest(
        async_fn=None,
        async_fn_args=(),
        finalize_fns=[lambda: no_op_calls.append("done")],
    )
    no_op.execute_sync()
    assert no_op_calls == ["done"]

    bad_preload = async_utils.AsyncRequest(
        async_fn=async_fn,
        async_fn_args=("too", "short"),
        finalize_fns=[],
        preload_fn=preload,
    )
    with pytest.raises(AssertionError, match="Expected 3 args"):
        bad_preload.execute_sync()


def test_async_calls_queue_schedule_finalize_close_and_persistent_paths(monkeypatch):
    calls = []

    class _Caller:
        def __init__(self):
            self.done = False
            self.scheduled = []
            self.closed = []

        def schedule_async_call(self, async_req):
            self.scheduled.append(async_req)
            calls.append(("schedule", async_req.call_idx, async_req.finalize_fns))

        def is_current_async_call_done(self, blocking=False, no_dist=False):
            calls.append(("done?", blocking, no_dist))
            return self.done or blocking

        def close(self, abort=False):
            self.closed.append(abort)
            calls.append(("close", abort))

    caller = _Caller()
    monkeypatch.setattr(async_utils.AsyncCallsQueue, "_persistent_caller", None)
    monkeypatch.setattr(async_utils.AsyncCallsQueue, "_get_async_caller", lambda self: caller)
    monkeypatch.setattr(async_utils.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        async_utils.torch.distributed,
        "all_reduce",
        lambda tensor, op=None: calls.append(("all_reduce", int(tensor.item()), op)),
    )
    monkeypatch.setattr(
        async_utils.torch.distributed,
        "ReduceOp",
        SimpleNamespace(MAX="max"),
        raising=False,
    )

    finalized = []
    queue = async_utils.AsyncCallsQueue(persistent=False)
    call_idx = queue.schedule_async_request(
        async_utils.AsyncRequest(
            async_fn=lambda: None,
            async_fn_args=(),
            finalize_fns=[lambda: finalized.append("first")],
        )
    )
    assert call_idx == 0
    assert queue.get_num_unfinalized_calls() == 1
    assert caller.scheduled[0].is_frozen is True
    assert caller.scheduled[0].finalize_fns == []
    assert queue.maybe_finalize_async_calls(blocking=False, no_dist=True) == []

    caller.done = True
    assert queue.maybe_finalize_async_calls(blocking=False, no_dist=True) == [0]
    assert finalized == ["first"]
    assert queue.get_num_unfinalized_calls() == 0

    queue.schedule_async_request(
        async_utils.AsyncRequest(async_fn=None, async_fn_args=(), finalize_fns=[])
    )
    queue.close(abort=False)
    assert queue.get_num_unfinalized_calls() == 0

    persistent = async_utils.AsyncCallsQueue(persistent=True)
    async_utils.AsyncCallsQueue._persistent_caller = caller
    persistent.close(abort=True)
    assert async_utils.AsyncCallsQueue._persistent_caller is None


def test_async_calls_queue_backward_compatible_request_and_mismatch_detection(monkeypatch):
    class _OldRequest(tuple):
        _fields = ("async_fn", "async_fn_args", "finalize_fns")

        def _asdict(self):
            return {"async_fn": None, "async_fn_args": (), "finalize_fns": []}

    class _Caller:
        def schedule_async_call(self, async_req):
            self.async_req = async_req

        def is_current_async_call_done(self, blocking=False, no_dist=False):
            return True

    caller = _Caller()
    monkeypatch.setattr(async_utils.AsyncCallsQueue, "_get_async_caller", lambda self: caller)
    monkeypatch.setattr(async_utils.cur_platform, "current_device", lambda: "cpu")

    def _bad_all_reduce(tensor, op=None):
        tensor.add_(1)

    monkeypatch.setattr(async_utils.torch.distributed, "all_reduce", _bad_all_reduce)
    queue = async_utils.AsyncCallsQueue()
    assert queue.schedule_async_request(_OldRequest()) == 0
    with pytest.raises(AssertionError, match="Unmatched async calls"):
        queue.maybe_finalize_async_calls(blocking=True)


def test_torch_strategy_flatten_restore_and_key_replacement_paths():
    tensor = torch.arange(4).reshape(2, 2)
    shard = ShardedTensor.from_rank_offsets("weight", tensor, (0, 0, 1), replica_id=0)
    replica = ShardedTensor.from_rank_offsets("weight", tensor + 1, (0, 0, 1), replica_id=1)
    obj = ShardedObject("metadata", {"x": 1}, global_shape=(1,), global_offset=(0,), replica_id=0)
    state_dict = {"layer": {"w": shard, "replica": replica}, "objects": [obj]}

    flat, mapping = torch_strategy.flatten_state_dict(state_dict)
    assert set(flat) == {"layer.w", "layer.replica", "objects.0"}
    assert mapping["objects.0"] == ("objects", 0)

    with pytest.raises(ValueError, match="duplicated flatten key"):
        torch_strategy.flatten_state_dict({"a.b": shard, "a": {"b": obj}})

    new_flat, flat_mapping, rename_mapping = torch_strategy._replace_state_dict_keys_with_sharded_keys(
        state_dict, keep_only_main_replica=True
    )
    assert set(new_flat) == {"weight", obj.unique_key}
    assert rename_mapping["weight"] == ["layer.w"]
    assert "layer.replica" not in sum(rename_mapping.values(), [])

    restored = torch_strategy._replace_sharded_keys_with_state_dict_keys(
        {"weight": [torch.ones(2, 2)], obj.unique_key: [b"payload"]},
        flat_mapping,
        rename_mapping,
    )
    assert torch.equal(restored["layer"]["w"], torch.ones(2, 2))
    assert restored["objects"][0] == b"payload"

    assert torch_strategy._unwrap_pyt_sharded_tensor("plain") == "plain"

    template = {1: [{"x": 2}], "keep": {"nested": 3}}
    restored_types = {"1": [{"x": 2}], "keep": {"nested": 3}}
    torch_strategy._restore_dict_types(restored_types, template)
    assert 1 in restored_types and "1" not in restored_types
    with pytest.raises(AssertionError):
        torch_strategy._restore_dict_types([], {"not": "dict"})
    with pytest.raises(AssertionError):
        torch_strategy._restore_dict_types({}, ["not", "list"])
