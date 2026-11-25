# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(transformer_config, submodules).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerFP16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2), (4, 2)])
    def test_moe_layer_fp16_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, tp_size, ep_size
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=True,
            params_dtype=torch.float16,
        )

        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
            requires_grad=True,
        )

        # Forward pass
        output, _ = moe_layer(hidden_states)

        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.float16
        ), f"Expected fp16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRecompute:
    """Test MoE layer with recompute enabled (activation checkpointing).

    Tests both code paths:
    - fp8=False: uses tensor_parallel.checkpoint
    - fp8=True: uses te_checkpoint (requires TE >= 1.7.0)
    """

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("with_padding_mask", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (4, 2)])
    @pytest.mark.parametrize("fp8", [False, True])
    def test_moe_layer_recompute_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, with_padding_mask, tp_size, ep_size, fp8
    ):
        """Test MoE layer forward and backward pass with recompute enabled.

        When fp8=False, uses tensor_parallel.checkpoint.
        When fp8=True, uses te_checkpoint (requires TE >= 1.7.0).
        """
        # Skip fp8 tests if TE version is not sufficient
        if fp8 and not is_te_min_version("1.7.0.dev0"):
            pytest.skip("FP8 MoE recompute requires TE 1.7.0 and later.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            # Enable recompute for MoE layer
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp8=fp8,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Use TE spec for fp8, local spec otherwise
        if fp8:
            submodules = get_submodules(
                get_gpt_layer_with_transformer_engine_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        else:
            submodules = get_submodules(
                get_gpt_layer_local_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create padding mask if needed: shape [batch_size, sequence_length]
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.ones(
                micro_batch_size,
                sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.bool,
            )
            # Mark last 4 tokens as padding for each batch
            padding_mask[:, -4:] = False

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)

        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass - this is where recompute/checkpoint is actually used
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.bfloat16
        ), f"Expected bf16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "tp_size, etp_size, ep_size, deny_fusion, routing_case, permute_fusion",
    [
        (1, 1, 8, False, "random", True),
        (4, 1, 8, False, "random", True),
        (4, 2, 8, False, "random", True),
        (4, 1, 8, True, "random", True),
        (4, 1, 8, False, "local_disabled", True),
        (1, 1, 2, False, "random", True),
        (4, 1, 8, False, "skewed", True),
        (1, 1, 8, False, "small_payload", True),
        (1, 1, 8, False, "empty_destination", True),
        (1, 1, 8, False, "overflow", True),
        (4, 1, 8, False, "random", False),
    ],
)
@pytest.mark.parametrize("recompute", [False, True])
def test_npu_mc2_layer_matches_unfused(
    monkeypatch, tp_size, etp_size, ep_size, deny_fusion, routing_case, recompute, permute_fusion
):
    """Compare complete expert/input/router gradients and the disabled path on NPU."""
    import copy
    import os
    from dataclasses import replace
    from types import MethodType

    from torch.utils.checkpoint import checkpoint

    from megatron.plugin.platform import get_platform

    if get_platform().device_name() != "npu" or Utils.world_size < ep_size * etp_size:
        pytest.skip(f"MC2 layer comparison requires at least {ep_size * etp_size} Ascend ranks")
    import flagcx  # noqa: F401; register the optional test backend before group initialization
    from transformer_engine.plugin.core.backends.vendor.npu import npu
    from transformer_engine.plugin.core.policy import policy_context, policy_from_env

    if not npu.is_moe_fusion_available():
        pytest.skip("Native MC2 interfaces are unavailable under the current policy")
    monkeypatch.setenv("NPU_MOE_MC2", "1")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    try:
        model_parallel_cuda_manual_seed(321)
        # Top-k over all experts keeps ETP fallback splits equal for FlagCX.
        num_experts = 256 if etp_size == 1 and ep_size == 8 else 8
        overflow = routing_case == "overflow"
        hidden_size = 128 if overflow else 2048
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=16,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=64 if overflow else 512,
            moe_router_topk=2 if overflow else 8,
            moe_grouped_gemm=True,
            moe_permute_fusion=permute_fusion,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.001,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            bias_activation_fusion=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            hidden_dropout=0,
            attention_dropout=0,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_experts, moe_grouped_gemm=True
            ).mlp
        )
        device = get_platform().device(get_platform().current_device())
        candidate = MoELayer(config, submodules=submodules, layer_number=1).to(device)
        reference = MoELayer(copy.deepcopy(config), submodules=submodules, layer_number=1).to(
            device
        )
        if routing_case in ("skewed", "overflow", "empty_destination"):
            with torch.no_grad():
                weight = candidate.router.weight
                weight.zero_()
                weight[:, 0] = torch.linspace(-0.2, 0.2, num_experts, device=device)
                hot = torch.arange(config.moe_router_topk, device=device) * 32 + 5
                # Exercise skew with one active expert per destination for either input sign.
                if routing_case == "empty_destination":
                    hot = torch.arange(config.moe_router_topk) + 8
                cold = hot + (8 if routing_case == "empty_destination" else 1)
                weight[cold, 0] = -1
                weight[hot, 0] = (
                    0.8 + torch.arange(config.moe_router_topk, device=device) * 0.02
                ).to(weight.dtype)
        reference.load_state_dict(candidate.state_dict())
        reference.compute_experts = MethodType(MoELayer.compute_experts.__wrapped__, reference)
        reference.preprocess = MethodType(MoELayer.preprocess.__wrapped__, reference)
        torch.manual_seed(37 + int(os.environ["RANK"]))
        tokens = 65536 if overflow and candidate.token_dispatcher.ep_rank == 7 else 256
        if routing_case == "small_payload":
            tokens = 32
        x = torch.randn(tokens, 1, hidden_size, device=device, dtype=torch.bfloat16) / 4
        if routing_case in ("skewed", "overflow", "empty_destination"):
            x[:, :, 0] = 1

            def check_skewed_routing(module, inputs, result):
                counts = result[1].sum(0)
                assert (counts == 0).any()
                assert counts.max().item() == inputs[0].shape[0]
                assert torch.count_nonzero(counts).item() == config.moe_router_topk

            candidate.router.register_forward_hook(check_skewed_routing)
            reference.router.register_forward_hook(check_skewed_routing)
        grad = torch.randn_like(x) / 32
        metadata_calls = []
        gather = candidate.token_dispatcher._gather_tokens_per_expert

        def count_metadata(*args, **kwargs):
            metadata_calls.append(1)
            return gather(*args, **kwargs)

        monkeypatch.setattr(candidate.token_dispatcher, "_gather_tokens_per_expert", count_metadata)
        policy = policy_from_env()
        if deny_fusion and candidate.token_dispatcher.ep_rank == 0:
            policy = replace(
                policy,
                per_op_order=(
                    ("moe_alltoallv_grouped_matmul", ("reference",)),
                    ("moe_grouped_matmul_alltoallv", ("reference",)),
                ),
            )
        with policy_context(policy):
            assert npu.is_moe_fusion_available() == (
                not deny_fusion or candidate.token_dispatcher.ep_rank != 0
            )
        if routing_case == "local_disabled" and candidate.token_dispatcher.ep_rank == 0:
            monkeypatch.setenv("NPU_MOE_MC2", "0")
        expect_fused = (
            etp_size == 1
            and ep_size == 8
            and not deny_fusion
            and routing_case not in ("overflow", "local_disabled", "small_payload", "empty_destination")
        )

        def run(module):
            module.zero_grad(set_to_none=True)
            inp = x.detach().clone().requires_grad_()
            with policy_context(policy):
                output = (
                    checkpoint(module, inp, use_reentrant=True)[0] if recompute else module(inp)[0]
                )
                other = (
                    checkpoint(module, -inp, use_reentrant=True)[0]
                    if recompute
                    else module(-inp)[0]
                )
                torch.autograd.backward((output, other), (grad, grad / 2))
            grads = {
                name: p.grad.detach().clone()
                for name, p in module.named_parameters()
                if p.requires_grad
            }
            return output.detach(), inp.grad.detach(), grads

        actual, expected = run(candidate), run(reference)
        forwards = 4 if recompute else 2
        assert len(metadata_calls) == forwards
        if expect_fused and permute_fusion:
            mapping = candidate.token_dispatcher.reversed_local_input_permutation_mapping
            assert mapping.shape == (tokens, 2 * config.num_moe_experts + 1)
            assert mapping.dtype == torch.int32
            order = candidate.token_dispatcher._npu_moe_expert_order[0]
            assert order != list(range(len(order)))
        if overflow:
            assert candidate.token_dispatcher._npu_moe_expert_order is None
        if routing_case in ("skewed", "overflow", "empty_destination"):
            assert actual[2]["router.weight"].norm().item() > 0
        assert bool(getattr(candidate, "_npu_moe_mc2_logged", False)) == expect_fused
        if ep_size == 2:
            assert candidate.token_dispatcher._npu_moe_counts is None
        def compare(result, baseline, fused):
            assert result[2].keys() == baseline[2].keys()
            for value, ref in [*zip(result[:2], baseline[:2]), *((result[2][k], baseline[2][k]) for k in result[2])]:
                assert torch.isfinite(value).all() and torch.isfinite(ref).all()
                if fused:
                    error = (value.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
                    assert error.item() < 0.03
                else:
                    torch.testing.assert_close(value, ref, rtol=0, atol=0)

        compare(actual, expected, expect_fused)
        with monkeypatch.context() as disabled_env:
            disabled_env.setenv("NPU_MOE_MC2", "0")
            disabled, plain = run(candidate), run(reference)
        compare(disabled, plain, False)
        with monkeypatch.context() as denied_env:
            denied_env.setenv("TE_FL_DENY_VENDORS", "NPU")
            with policy_context(policy_from_env()):
                assert not npu.is_moe_fusion_available()
                assert not npu.is_moe_fusion_available(candidate.token_dispatcher.ep_group, device)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "case, supported",
    [("threshold", True), ("small_send", False), ("small_receive", False),
     ("empty_rank", False), ("empty_expert", True), ("count_limit", True), ("overflow", False)],
)
def test_npu_mc2_global_count_limit(case, supported):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import supports_moe_counts

    # 512 BF16 rows of width 2048 are exactly 2 MiB per source/destination.
    counts = torch.full((8, 16), 32, dtype=torch.int64)
    if case == "small_send":
        counts[-1, 0] -= 1
        counts[0, 0] += 1
    elif case == "small_receive":
        counts[0, -1] -= 1
        counts[0, 0] += 1
    elif case == "empty_rank":
        counts[:, :2] += counts[:, -2:]
        counts[:, -2:] = 0
    elif case == "empty_expert":
        counts[:, 0] += counts[:, 1]
        counts[:, 1] = 0
    elif case in ("count_limit", "overflow"):
        counts[-1, -1] = 65535 if case == "count_limit" else 65536
    assert supports_moe_counts(counts, 2048, 2) is supported


def test_npu_mc2_ddp_weight_readiness(monkeypatch):
    """Direct weight reads respect native DDP hook disable/re-enable and real param gathers."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.plugin.Ascend.transformer.moe.fused_experts import _weights
    from megatron.plugin.platform import get_platform

    if get_platform().device_name() != "npu" or Utils.world_size < 16:
        pytest.skip("MC2 readiness requires Ascend EP8 with at least two expert DP replicas")
    import flagcx  # noqa: F401

    monkeypatch.setenv("NPU_MOE_MC2", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=8, expert_tensor_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(321)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_ffn_hidden_size=64,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(config, submodules=submodules, layer_number=1).to(
            get_platform().device(get_platform().current_device())
        )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True, overlap_param_gather=True, overlap_grad_reduce=True
            ),
            layer,
        )
        linears = (layer.experts.linear_fc1, layer.experts.linear_fc2)
        with torch.no_grad():
            params = (p for linear in linears for p in linear.parameters(recurse=False))
            for i, param in enumerate(params):
                param.fill_(i + 1)
        expected = [
            tuple(p.detach().clone() for p in linear.parameters(recurse=False)) for linear in linears
        ]
        # Exercise normal DDP scheduling, an external gather with hooks removed, and re-enable.
        for enabled in (True, False, True):
            if not enabled:
                ddp.disable_forward_pre_hook(param_sync=False)
            for group in ddp.expert_parallel_bucket_groups:
                for bucket in group.buckets:
                    shards = bucket.param_data.chunk(group.intra_distributed_optimizer_instance_size)
                    for rank, shard in enumerate(shards):
                        if rank != group.intra_distributed_optimizer_instance_rank:
                            shard.zero_()
                group.start_param_sync()
                assert group.param_gather_handle is not None
            for linear, baseline in zip(linears, expected):
                weights = _weights(linear)
                if isinstance(weights, torch.Tensor):
                    weights = (weights,)
                assert len(weights) == len(baseline)
                for actual, original in zip(weights, baseline):
                    torch.testing.assert_close(actual, original, rtol=0, atol=0)
                for param in linear.parameters(recurse=False):
                    assert ddp.param_to_bucket_group[param].param_gather_handle is None
            if not enabled:
                ddp.enable_forward_pre_hook()
    finally:
        Utils.destroy_model_parallel()
