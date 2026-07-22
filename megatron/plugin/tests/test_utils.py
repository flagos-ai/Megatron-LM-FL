from types import SimpleNamespace

import torch

from megatron.core.transformer.moe import moe_utils
from megatron.plugin import utils


def test_get_device_type_for_comm_handles_single_and_group_lists(monkeypatch):
    cpu_group = object()
    device_group = object()
    backends = {cpu_group: "cpu:gloo", device_group: "nccl"}
    monkeypatch.setattr(
        utils.torch.distributed,
        "get_backend",
        lambda group: backends[group],
    )
    monkeypatch.setattr(utils, "cur_platform", SimpleNamespace(device_name=lambda: "cuda"))

    assert utils.get_device_type_for_comm(cpu_group) == "cpu"
    assert utils.get_device_type_for_comm([cpu_group]) == "cpu"
    assert utils.get_device_type_for_comm(device_group) == "cuda"


def test_is_built_on_zero_rank_respects_shared_filesystem_policy(monkeypatch):
    import megatron.training

    args = SimpleNamespace(no_shared_fs=False)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert utils.is_built_on_zero_rank() is True

    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 2)
    assert utils.is_built_on_zero_rank() is False

    args.no_shared_fs = True
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert utils.is_built_on_zero_rank() is True


def test_is_built_on_zero_rank_falls_back_when_training_args_are_unavailable(monkeypatch):
    import megatron.training

    monkeypatch.setattr(
        megatron.training,
        "get_args",
        lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
    )
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert utils.is_built_on_zero_rank() is True


def test_reduce_aux_losses_tracker_uses_configured_groups(monkeypatch):
    reduce_group = object()
    average_group = object()
    pipeline_groups = [object(), object()]
    values = torch.tensor([2.0])
    tracker = {
        "load_balancing": {
            "values": values,
            "reduce_group": reduce_group,
            "avg_group": average_group,
        }
    }
    reductions = []

    monkeypatch.setattr(
        moe_utils,
        "get_moe_layer_wise_logging_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        utils.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: pipeline_groups,
    )
    monkeypatch.setattr(utils.torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(
        utils.torch.distributed,
        "all_reduce",
        lambda tensor, group, op=None: reductions.append((group, op, tensor.clone())),
    )

    utils.reduce_aux_losses_tracker_across_ranks_hetero()

    assert [item[0] for item in reductions] == [
        reduce_group,
        average_group,
        pipeline_groups[0],
        pipeline_groups[1],
    ]
    assert reductions[1][1] == torch.distributed.ReduceOp.AVG
