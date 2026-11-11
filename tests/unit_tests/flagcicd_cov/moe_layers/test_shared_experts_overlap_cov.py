# Copyright (c) 2026, BAAI. All rights reserved.
"""SharedExpertMLP overlap staging: pre-forward communication guards, grad-node sequence
numbering and the staged forward sequence's state machine with cleared caches."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.shared_experts import (
    SharedExpertMLP,
    SharedExpertState,
    set_tensor_grad_fn_sequence_sr,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _shared_expert_config(**overrides):
    """Config with one small shared expert, following the developer test conventions."""
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        moe_shared_expert_intermediate_size=16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        moe_token_dispatcher_type='alltoall',
        moe_router_load_balancing_type='aux_loss',
        moe_aux_loss_coeff=0.0,
        activation_func=F.silu,
        gated_linear_unit=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _local_submodules():
    """MLP submodules with local (non-TE) linear modules."""
    return MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)


def _shared_expert(config, gate):
    """SharedExpertMLP with deterministic silu-glu weights and gate."""
    expert = SharedExpertMLP(
        config=config,
        submodules=_local_submodules(),
        gate=gate,
        pg_collection=get_default_pg_collection(),
    )
    with torch.no_grad():
        fc1_weight = expert.linear_fc1.weight
        fc1_weight.copy_(
            torch.arange(fc1_weight.numel(), dtype=torch.float32).reshape(fc1_weight.shape) / 64.0
        )
        fc2_weight = expert.linear_fc2.weight
        fc2_weight.copy_(
            torch.arange(fc2_weight.numel(), dtype=torch.float32).reshape(fc2_weight.shape) / 128.0
        )
        if gate:
            expert.gate_weight.copy_(
                torch.arange(expert.gate_weight.numel(), dtype=torch.float32).reshape(
                    expert.gate_weight.shape
                )
                / 8.0
            )
    return expert


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestOverlapStateGuards:

    def test_pre_forward_comm_requires_overlap_enabled(self, model_parallel):
        model_parallel()
        expert = _shared_expert(_shared_expert_config(moe_shared_expert_overlap=False), gate=True)

        with pytest.raises(
            AssertionError, match='pre_forward_comm requires --moe-shared-expert-overlap to be set'
        ):
            expert.pre_forward_comm(torch.zeros(2, 2, 8))

    def test_set_tensor_grad_fn_sequence_sr_reorders_the_grad_node(self, model_parallel):
        model_parallel()
        base = torch.ones(3, requires_grad=True)
        early = base * 2.0
        late = base * 3.0

        set_tensor_grad_fn_sequence_sr(late, torch.iinfo(torch.int).max)
        set_tensor_grad_fn_sequence_sr(early, 0)

        assert late.grad_fn._sequence_nr() == torch.iinfo(torch.int).max
        assert early.grad_fn._sequence_nr() == 0


class TestOverlappedForward:
    """Overlapped forward staging on the A100 CI node; the sequence runs on a side stream."""

    def _run_overlap_sequence(self, expert, hidden, overlapped_comm_output=None):
        expert.pre_forward_comm(hidden)
        expert.linear_fc1_forward_and_act(overlapped_comm_output)
        expert.linear_fc2_forward(overlapped_comm_output)
        expert.post_forward_comm()
        return expert.get_output()

    @pytest.mark.parametrize(
        'gate', [pytest.param(True, id='gated'), pytest.param(False, id='ungated')]
    )
    def test_overlap_sequence_matches_plain_forward_and_clears_caches(self, model_parallel, gate):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        overlap = _shared_expert(
            _shared_expert_config(moe_shared_expert_overlap=True), gate=gate
        ).cuda()
        plain = _shared_expert(_shared_expert_config(), gate=gate).cuda()
        overlap.load_state_dict(plain.state_dict())
        assert overlap.linear_fc1.explicit_expert_comm is True
        assert overlap.linear_fc2.explicit_expert_comm is True
        assert overlap._overlap_state is SharedExpertState.IDLE
        hidden = (
            torch.arange(32, dtype=torch.float32, device=device).reshape(2, 2, 8) / 8.0
        ).requires_grad_(True)

        output = self._run_overlap_sequence(overlap, hidden)

        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference = plain(reference_hidden)
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
        assert overlap._overlap_state is SharedExpertState.IDLE
        assert overlap.cached_fc1_input is None
        assert overlap.cached_fc2_input is None
        assert overlap.cached_fc2_output is None
        assert overlap.cached_output is None
        assert overlap.gate_score is None

        output.sum().backward()
        reference.sum().backward()
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            overlap.linear_fc1.weight.grad, plain.linear_fc1.weight.grad, rtol=1e-5, atol=1e-6
        )
        if gate:
            torch.testing.assert_close(
                overlap.gate_weight.grad, plain.gate_weight.grad, rtol=1e-5, atol=1e-6
            )

    def test_overlap_sequence_with_comm_output_keeps_output_and_pins_grad_order(
        self, model_parallel
    ):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        overlap = _shared_expert(
            _shared_expert_config(moe_shared_expert_overlap=True), gate=False
        ).cuda()
        plain = _shared_expert(_shared_expert_config(), gate=False).cuda()
        overlap.load_state_dict(plain.state_dict())
        hidden = (
            torch.arange(32, dtype=torch.float32, device=device).reshape(2, 2, 8) / 8.0
        ).requires_grad_(True)
        comm_base = torch.ones(2, 2, 8, device=device, requires_grad=True)
        overlapped_comm_output = comm_base * 2.0

        output = self._run_overlap_sequence(overlap, hidden, overlapped_comm_output)

        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference = plain(reference_hidden)
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
        assert overlapped_comm_output.grad_fn._sequence_nr() == torch.iinfo(torch.int).max

        output.sum().backward()
        reference.sum().backward()

        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            overlap.linear_fc1.weight.grad, plain.linear_fc1.weight.grad, rtol=1e-5, atol=1e-6
        )

    def test_overlap_sequence_gathers_input_for_sequence_parallel(self, model_parallel):
        model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        device = torch.cuda.current_device()
        config = dict(
            moe_shared_expert_overlap=True, sequence_parallel=True, tensor_model_parallel_size=2
        )
        overlap = _shared_expert(_shared_expert_config(**config), gate=False).cuda()
        plain = _shared_expert(
            _shared_expert_config(sequence_parallel=True, tensor_model_parallel_size=2), gate=False
        ).cuda()
        overlap.load_state_dict(plain.state_dict())
        hidden = (
            torch.arange(16, dtype=torch.float32, device=device).reshape(1, 2, 8) / 8.0
        ).requires_grad_(True)

        output = self._run_overlap_sequence(overlap, hidden)

        reference_hidden = hidden.detach().clone().requires_grad_(True)
        torch.testing.assert_close(output, plain(reference_hidden), rtol=1e-4, atol=1e-5)

    def test_overlap_rejects_wrong_call_order(self, model_parallel):
        model_parallel()
        model_parallel_cuda_manual_seed(123)
        overlap = _shared_expert(_shared_expert_config(moe_shared_expert_overlap=True), gate=False)

        with pytest.raises(
            AssertionError,
            match='linear_fc1_forward_and_act must be called from PRE_FORWARD_COMM_DONE '
            'state, but current state is IDLE',
        ):
            overlap.linear_fc1_forward_and_act()
