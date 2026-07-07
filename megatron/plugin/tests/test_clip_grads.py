from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.optimizer import clip_grads


@pytest.fixture(autouse=True)
def cpu_clip_grad_runtime(monkeypatch):
    monkeypatch.setattr(
        clip_grads,
        "cur_platform",
        SimpleNamespace(device_name=lambda: "cpu"),
    )
    monkeypatch.setattr(clip_grads, "get_device_type_for_comm", lambda group: "cpu")
    monkeypatch.setattr(
        clip_grads,
        "get_data_parallel_group_if_dtensor",
        lambda grad, group: group,
    )
    monkeypatch.setattr(clip_grads, "to_local_if_dtensor", lambda grad: grad)


def test_get_grad_norm_fp32_infinity_and_general_norms(monkeypatch):
    reductions = []
    monkeypatch.setattr(
        clip_grads.torch.distributed,
        "all_reduce",
        lambda tensor, op, group: reductions.append((tensor.clone(), op, group)),
    )

    grads = [torch.tensor([3.0, -4.0]), torch.tensor([1.0])]

    assert clip_grads.get_grad_norm_fp32(grads, float("inf"), "model") == 4.0
    assert clip_grads.get_grad_norm_fp32(grads, 1, "model") == pytest.approx(8.0)
    assert len(reductions) == 2
    assert reductions[0][1] == torch.distributed.ReduceOp.MAX
    assert reductions[1][1] == torch.distributed.ReduceOp.SUM


def test_get_grad_norm_fp32_l2_uses_multi_tensor_boundary(monkeypatch):
    calls = []

    def fake_multi_tensor_applier(impl, overflow_buf, tensor_lists, per_parameter):
        calls.append((impl, tensor_lists, per_parameter))
        return torch.tensor(5.0), None

    monkeypatch.setattr(clip_grads, "multi_tensor_applier", fake_multi_tensor_applier)
    monkeypatch.setattr(clip_grads, "l2_norm_impl", object())
    monkeypatch.setattr(clip_grads, "multi_tensor_scale_tensor_impl", None)
    monkeypatch.setattr(clip_grads.torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    grad = torch.tensor([3.0, 4.0])
    assert clip_grads.get_grad_norm_fp32(grad, 2, "model") == pytest.approx(5.0)
    assert calls[0][1][0][0] is grad
    assert calls[0][2] is False

    assert clip_grads.get_grad_norm_fp32([], 2, "model") == pytest.approx(0.0)


def test_get_grad_norm_fp32_reduces_data_and_model_groups(monkeypatch):
    reductions = []
    data_group = object()
    model_groups = [object(), object()]

    monkeypatch.setattr(
        clip_grads,
        "get_data_parallel_group_if_dtensor",
        lambda grad, group: data_group,
    )
    monkeypatch.setattr(
        clip_grads.torch.distributed,
        "all_reduce",
        lambda tensor, op, group: reductions.append((op, group)),
    )

    result = clip_grads.get_grad_norm_fp32(torch.tensor([2.0]), 1, model_groups)

    assert result == pytest.approx(2.0)
    assert reductions == [
        (torch.distributed.ReduceOp.SUM, data_group),
        (torch.distributed.ReduceOp.SUM, model_groups[0]),
        (torch.distributed.ReduceOp.SUM, model_groups[1]),
    ]
