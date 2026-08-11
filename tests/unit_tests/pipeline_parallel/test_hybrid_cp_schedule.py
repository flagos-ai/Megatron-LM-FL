# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib

import torch

import megatron.core.pipeline_parallel.schedules as schedules
from megatron.core.pipeline_parallel import hybrid_cp_schedule


def test_hybrid_cp_schedule_uses_dynamic_cp_size_on_both_tp_sides(monkeypatch):
    real_tensor = torch.tensor
    broadcasts = []
    forward_calls = []
    backward_calls = []
    forward_output = object()
    config = object()
    input_tensor = object()
    output_tensor_grad = object()

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    def forward_step(*args, **kwargs):
        cp_group_size = kwargs.get("cp_group_size", args[7] if len(args) > 7 else None)
        collect_non_loss_data = kwargs.get(
            "collect_non_loss_data", args[8] if len(args) > 8 else False
        )
        sample = None if args[1] is None else next(args[1])
        forward_calls.append(
            (
                cp_group_size,
                collect_non_loss_data,
                None if sample is None else int(sample["local_cp_size"].item()),
                kwargs["current_microbatch"],
            )
        )
        return forward_output, real_tensor(1, dtype=torch.int)

    def run(data_iterator):
        return hybrid_cp_schedule.hybrid_context_parallel_forward_backward(
            forward_step_func=object(),
            data_iterator=data_iterator,
            model=object(),
            num_microbatches=3,
            input_tensor=input_tensor,
            output_tensor_grad=output_tensor_grad,
            forward_data_store=[],
            config=config,
            collect_non_loss_data=True,
            first_val_step=None,
            forward_only=False,
            no_sync_func=contextlib.nullcontext,
            total_num_tokens=real_tensor(0, dtype=torch.int),
            check_first_val_step=lambda *args: args[-1],
            model_type=object(),
        )

    monkeypatch.setattr(hybrid_cp_schedule.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(hybrid_cp_schedule.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(hybrid_cp_schedule.torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hybrid_cp_schedule.torch.distributed,
        "broadcast",
        lambda item, *args, **kwargs: broadcasts.append(item.detach().cpu().tolist()),
    )
    for name, value in {
        "get_data_parallel_rank": lambda **kwargs: 0,
        "get_data_parallel_group": lambda **kwargs: object(),
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_src_rank": lambda: 0,
        "get_tensor_model_parallel_group": lambda: object(),
    }.items():
        monkeypatch.setattr(hybrid_cp_schedule.parallel_state, name, value)
    monkeypatch.setattr(schedules, "forward_step", forward_step)
    monkeypatch.setattr(
        schedules,
        "backward_step",
        lambda *args, **kwargs: backward_calls.append((args, kwargs)),
    )

    samples = [{"tokens": real_tensor([index])} for index in range(3)]
    groups = [[[0], [0], []], [[1, 2], [1, 2], [1]]]
    _, total_num_tokens = run(iter([(samples, groups)]))

    assert forward_calls == [(2, True, 2, 0), (3, True, 3, 1), (2, True, 2, 2)]
    assert backward_calls == [
        ((input_tensor, forward_output, output_tensor_grad, config), {})
    ] * 3
    assert broadcasts[-3:] == [[2], [3], [2]]
    assert int(total_num_tokens.item()) == 3

    forward_calls.clear()
    backward_calls.clear()
    received = iter(([1], [1], [7]))

    def receive_broadcast(item, *args, **kwargs):
        item.copy_(real_tensor(next(received), dtype=item.dtype))

    monkeypatch.setattr(hybrid_cp_schedule.parallel_state, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(hybrid_cp_schedule.torch.distributed, "broadcast", receive_broadcast)
    _, total_num_tokens = run(None)

    assert forward_calls == [(7, True, None, 0)]
    assert backward_calls == [
        ((input_tensor, forward_output, output_tensor_grad, config), {})
    ]
    assert int(total_num_tokens.item()) == 1
