# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Implicit pipeline phase metadata and scopes; scheduling remains in Core."""

import torch

from megatron.core.observability import open_trace_scope, prepare_trace_scope


def _set_pipeline_operation_identity(scope, current_microbatch, vp_stage):
    """Fill operation identity only when the phase scope is active."""
    if scope.get("timing_phase") != "framework_phase":
        return
    if current_microbatch is None:
        operation_id = None
    else:
        vp_identity = "none" if vp_stage is None else vp_stage
        operation_id = f"pp:microbatch={current_microbatch}:vp={vp_identity}"
    scope.set("operation_id", operation_id)


def _pipeline_workload_from_output(
    output_tensor, config, *, cp_group_size, tp_group_size, fallback_num_tokens
):
    """Derive the source MixedPara workload from the pre-loss output shape."""
    current_num_tokens = 0
    actual_tensor = output_tensor[0] if isinstance(output_tensor, list) else output_tensor
    if isinstance(actual_tensor, torch.Tensor) and actual_tensor.ndim >= 2:
        current_num_tokens = int(actual_tensor.shape[0]) * int(actual_tensor.shape[1])
        current_num_tokens *= cp_group_size
        if getattr(config, "sequence_parallel", False):
            current_num_tokens *= tp_group_size

    if current_num_tokens == 0:
        current_num_tokens = fallback_num_tokens

    return int(current_num_tokens), float(current_num_tokens**2)


def _set_pipeline_workload(scope, workload):
    """Fill workload slots only when the phase scope is active."""
    if workload is None or scope.get("timing_phase") != "framework_phase":
        return
    num_tokens, sum_sq_seq_len = workload
    scope.set("num_tokens", num_tokens)
    scope.set("sum_sq_seq_len", sum_sq_seq_len)


def _optional_bool(value):
    """Preserve unknown schedule metadata instead of asserting a false value."""
    return None if value is None else bool(value)


def _pipeline_phase_context(*, current_microbatch, vp_stage, is_first_microbatch, is_last_stage):
    """Build phase metadata after the facade confirms that tracing is active."""
    ctx = {
        "current_microbatch": current_microbatch,
        "vp_stage": vp_stage,
        "is_first_microbatch": _optional_bool(is_first_microbatch),
        "is_last_stage": _optional_bool(is_last_stage),
        "timing_phase": "framework_phase",
    }
    return ctx


def _pipeline_phase_scope(
    name, *, current_microbatch, vp_stage, is_first_microbatch, is_last_stage, slots
):
    gate = prepare_trace_scope(name)
    ctx = None
    if gate is not None:
        ctx = _pipeline_phase_context(
            current_microbatch=current_microbatch,
            vp_stage=vp_stage,
            is_first_microbatch=is_first_microbatch,
            is_last_stage=is_last_stage,
        )
    return open_trace_scope(gate, name, ctx=ctx, slots=slots)


def _pipeline_grad_sync_context(schedule):
    """Build schedule metadata only for an enabled grad-sync probe."""
    return {"schedule": schedule, "timing_phase": "framework_phase"}


def _pipeline_grad_sync_scope(schedule):
    gate = prepare_trace_scope("grad-sync")
    ctx = _pipeline_grad_sync_context(schedule) if gate is not None else None
    return open_trace_scope(gate, "grad-sync", ctx=ctx)


def _combined_operation_id(microbatch, vp_stage):
    if microbatch is None:
        return None
    vp_identity = "none" if vp_stage is None else vp_stage
    return f"pp:microbatch={microbatch}:vp={vp_identity}"


def _build_combined_step_context(
    *, f_model, b_model, current_microbatch, backward_microbatch, f_vp_stage, b_vp_stage
):
    forward_active = f_model is not None
    backward_active = b_model is not None
    forward_operation_id = (
        _combined_operation_id(current_microbatch, f_vp_stage) if forward_active else None
    )
    backward_operation_id = (
        _combined_operation_id(backward_microbatch, b_vp_stage) if backward_active else None
    )
    if forward_active and backward_active:
        execution_mode = "combined"
    elif forward_active:
        execution_mode = "forward"
    else:
        execution_mode = "backward"
    return {
        "operation_id": (
            f"pp-combined:forward={forward_operation_id}:backward={backward_operation_id}"
        ),
        "forward_operation_id": forward_operation_id,
        "backward_operation_id": backward_operation_id,
        "forward_microbatch": current_microbatch if forward_active else None,
        "backward_microbatch": backward_microbatch if backward_active else None,
        "forward_vp_stage": f_vp_stage if forward_active else None,
        "backward_vp_stage": b_vp_stage if backward_active else None,
        "execution_mode": execution_mode,
        "overlap_active": execution_mode == "combined",
        "schedule": "combined-1f1b",
        "timing_phase": "framework_phase",
    }


def _combined_step_context(
    forward_step_func,
    data_iterator,
    f_model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    b_model,
    b_input_tensor,
    b_output_tensor,
    b_output_tensor_grad,
    config,
    f_model_chunk_id=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    fsdp_wrapper=None,
    backward_microbatch=None,
    b_model_chunk_id=None,
):
    del (
        forward_step_func,
        data_iterator,
        num_microbatches,
        input_tensor,
        forward_data_store,
        b_input_tensor,
        b_output_tensor,
        b_output_tensor_grad,
        config,
        pre_forward,
        pre_backward,
        post_forward,
        post_backward,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
        is_first_microbatch,
        encoder_decoder_xattn,
    )
    return _build_combined_step_context(
        f_model=f_model,
        b_model=b_model,
        current_microbatch=current_microbatch,
        backward_microbatch=backward_microbatch,
        f_vp_stage=f_model_chunk_id,
        b_vp_stage=b_model_chunk_id,
    )
