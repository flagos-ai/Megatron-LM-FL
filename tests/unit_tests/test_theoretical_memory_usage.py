# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.training import theoretical_memory_usage as tmu


def make_args(**overrides):
    defaults = {
        "kv_channels": 4,
        "num_attention_heads": 4,
        "hidden_size": 16,
        "group_query_attention": False,
        "num_query_groups": 2,
        "num_experts": None,
        "swiglu": False,
        "moe_shared_expert_intermediate_size": None,
        "moe_layer_freq": 1,
        "num_layers": 4,
        "moe_ffn_hidden_size": 32,
        "mtp_num_layers": None,
        "normalization": "LayerNorm",
        "multi_latent_attention": False,
        "q_lora_rank": None,
        "qk_head_dim": 2,
        "qk_pos_emb_head_dim": 1,
        "kv_lora_rank": 4,
        "v_head_dim": 2,
        "ffn_hidden_size": 64,
        "moe_router_topk": 2,
        "padded_vocab_size": 128,
        "untie_embeddings_and_output_weights": False,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 2,
        "use_distributed_optimizer": False,
        "data_parallel_size": 4,
        "seq_length": 8,
        "micro_batch_size": 2,
        "virtual_pipeline_model_parallel_size": None,
        "sequence_parallel": False,
        "recompute_granularity": "full",
        "hybrid_layer_pattern": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def expected_activation_memory(args, num_microbatches):
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    activation_memory *= args.num_layers
    activation_memory += 8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    if args.virtual_pipeline_model_parallel_size is not None:
        activation_memory *= 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (
                args.pipeline_model_parallel_size
                * args.virtual_pipeline_model_parallel_size
            )
        )

    if (
        args.virtual_pipeline_model_parallel_size is None
        and args.pipeline_model_parallel_size > 1
        and num_microbatches is not None
    ):
        activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)

    if args.pipeline_model_parallel_size == 1:
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    return activation_memory / args.tensor_model_parallel_size


def expected_activation_memory_without_sp(args, num_microbatches):
    total_activation_memory = (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * (10 + (24 / args.tensor_model_parallel_size))
    )
    total_activation_memory *= args.num_layers
    total_activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    total_activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    if args.virtual_pipeline_model_parallel_size is not None:
        total_activation_memory *= 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (
                args.pipeline_model_parallel_size
                * args.virtual_pipeline_model_parallel_size
            )
        )

    if (
        args.virtual_pipeline_model_parallel_size is None
        and args.pipeline_model_parallel_size > 1
        and num_microbatches is not None
    ):
        total_activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)

    if args.pipeline_model_parallel_size == 1:
        logits_size = args.seq_length * args.micro_batch_size * args.padded_vocab_size
        logits_size /= args.tensor_model_parallel_size
        final_ln_output = args.seq_length * args.micro_batch_size * args.hidden_size
        total_activation_memory += (logits_size + final_ln_output) * 2

    return total_activation_memory * 1.05


def test_compute_weight_and_optimizer_memory_dense_variants(capsys):
    baseline_args = make_args()
    baseline_memory = tmu.compute_weight_and_optimizer_memory(baseline_args)

    assert baseline_memory > 0
    assert baseline_args.num_query_groups == baseline_args.num_attention_heads

    untied_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(untie_embeddings_and_output_weights=True)
    )
    assert untied_memory > baseline_memory

    pipeline_tied_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(pipeline_model_parallel_size=2), verbose=True
    )
    pipeline_untied_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(pipeline_model_parallel_size=2, untie_embeddings_and_output_weights=True),
        verbose=True,
    )
    assert pipeline_untied_memory == pytest.approx(pipeline_tied_memory)

    distributed_optimizer_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(use_distributed_optimizer=True)
    )
    assert distributed_optimizer_memory < baseline_memory

    captured = capsys.readouterr().out
    assert "most loaded shard" in captured
    assert "other shards" in captured


def test_compute_weight_and_optimizer_memory_moe_and_mtp_variants(capsys):
    moe_args = make_args(
        num_experts=4,
        moe_layer_freq=2,
        moe_ffn_hidden_size=24,
        moe_shared_expert_intermediate_size=8,
        swiglu=True,
    )
    moe_memory = tmu.compute_weight_and_optimizer_memory(moe_args, verbose=True)
    moe_with_mtp_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(
            num_experts=4,
            moe_layer_freq=2,
            moe_ffn_hidden_size=24,
            moe_shared_expert_intermediate_size=8,
            swiglu=True,
            mtp_num_layers=2,
        )
    )

    assert moe_memory > 0
    assert moe_with_mtp_memory > moe_memory
    assert "active parameters" in capsys.readouterr().out


def test_compute_weight_and_optimizer_memory_moe_list_pattern_and_invalid_length():
    list_pattern_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(
            num_experts=2,
            moe_layer_freq=[1, 0, 1, 0],
            moe_ffn_hidden_size=24,
        )
    )

    assert list_pattern_memory > 0

    with pytest.raises(AssertionError, match="Invalid length of moe_layer_pattern"):
        tmu.compute_weight_and_optimizer_memory(
            make_args(
                num_experts=2,
                moe_layer_freq=[1, 0],
                moe_ffn_hidden_size=24,
            )
        )


def test_compute_weight_and_optimizer_memory_multi_latent_attention_variants(capsys):
    mla_args = make_args(
        multi_latent_attention=True,
        group_query_attention=False,
        normalization="RMSNorm",
        q_lora_rank=None,
        qk_head_dim=3,
        qk_pos_emb_head_dim=2,
        kv_lora_rank=5,
        v_head_dim=2,
    )
    mla_memory = tmu.compute_weight_and_optimizer_memory(mla_args)
    mla_lora_memory = tmu.compute_weight_and_optimizer_memory(
        make_args(
            multi_latent_attention=True,
            group_query_attention=False,
            normalization="RMSNorm",
            q_lora_rank=4,
            qk_head_dim=3,
            qk_pos_emb_head_dim=2,
            kv_lora_rank=5,
            v_head_dim=2,
        ),
        verbose=True,
    )

    assert mla_memory > 0
    assert mla_lora_memory > 0
    assert mla_memory != mla_lora_memory
    assert "Total number of parameters" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args,num_microbatches,expected_message",
    [
        (make_args(), 2, "Activation memory footprint per transformer layer"),
        (
            make_args(pipeline_model_parallel_size=4, tensor_model_parallel_size=1),
            2,
            "Number of in-flight microbatches",
        ),
        (
            make_args(
                pipeline_model_parallel_size=4,
                tensor_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=2,
            ),
            2,
            "Memory penalty from interleaved schedule",
        ),
        (
            make_args(pipeline_model_parallel_size=3, tensor_model_parallel_size=1),
            None,
            "Number of in-flight microbatches",
        ),
    ],
)
def test_compute_activation_memory_matches_formula(
    args, num_microbatches, expected_message, capsys
):
    result = tmu.compute_activation_memory(args, num_microbatches, verbose=True)

    assert result == pytest.approx(expected_activation_memory(args, num_microbatches))
    assert expected_message in capsys.readouterr().out


@pytest.mark.parametrize(
    "args,num_microbatches,expected_message",
    [
        (
            make_args(),
            2,
            "Activation memory footprint per transformer layer (precise, without SP)",
        ),
        (
            make_args(pipeline_model_parallel_size=4, tensor_model_parallel_size=1),
            2,
            "Number of in-flight microbatches",
        ),
        (
            make_args(
                pipeline_model_parallel_size=4,
                tensor_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=2,
            ),
            2,
            "Memory penalty from interleaved schedule",
        ),
        (
            make_args(pipeline_model_parallel_size=3, tensor_model_parallel_size=1),
            None,
            "Number of in-flight microbatches",
        ),
    ],
)
def test_compute_activation_memory_without_sp_matches_formula(
    args, num_microbatches, expected_message, capsys
):
    result = tmu.compute_activation_memory_without_sp(args, num_microbatches, verbose=True)

    assert result == pytest.approx(
        expected_activation_memory_without_sp(args, num_microbatches)
    )
    assert expected_message in capsys.readouterr().out


def test_report_theoretical_memory_uses_sp_path():
    args = make_args(sequence_parallel=True, recompute_granularity="selective")

    with (
        patch.object(
            tmu,
            "compute_weight_and_optimizer_memory",
            return_value=2 * tmu.NUM_BYTES_IN_MEGABYTE,
        ) as mock_weight,
        patch.object(
            tmu,
            "compute_activation_memory",
            return_value=3 * tmu.NUM_BYTES_IN_MEGABYTE,
        ) as mock_activation,
        patch.object(tmu, "compute_activation_memory_without_sp") as mock_activation_without_sp,
        patch.object(tmu, "print_rank_0") as mock_print_rank_0,
    ):
        result = tmu.report_theoretical_memory(args, num_microbatches=7, verbose=True)

    assert result == pytest.approx((2.0, 3.0, 5.0))
    mock_weight.assert_called_once_with(args, verbose=True)
    mock_activation.assert_called_once_with(args, num_microbatches=7, verbose=True)
    mock_activation_without_sp.assert_not_called()
    mock_print_rank_0.assert_called_once_with("compute_activation_memory with SP")


def test_report_theoretical_memory_uses_non_sp_path():
    args = make_args(sequence_parallel=False, recompute_granularity="selective")

    with (
        patch.object(
            tmu,
            "compute_weight_and_optimizer_memory",
            return_value=4 * tmu.NUM_BYTES_IN_MEGABYTE,
        ) as mock_weight,
        patch.object(tmu, "compute_activation_memory") as mock_activation,
        patch.object(
            tmu,
            "compute_activation_memory_without_sp",
            return_value=5 * tmu.NUM_BYTES_IN_MEGABYTE,
        ) as mock_activation_without_sp,
        patch.object(tmu, "print_rank_0") as mock_print_rank_0,
    ):
        result = tmu.report_theoretical_memory(args, num_microbatches=3, verbose=False)

    assert result == pytest.approx((4.0, 5.0, 9.0))
    mock_weight.assert_called_once_with(args, verbose=False)
    mock_activation.assert_not_called()
    mock_activation_without_sp.assert_called_once_with(
        args, num_microbatches=3, verbose=False
    )
    mock_print_rank_0.assert_called_once_with("compute_activation_memory_without_sp")


def test_report_theoretical_memory_returns_early_for_hybrid_model(capsys):
    args = make_args(hybrid_layer_pattern=[1, 0, 1, 0])

    assert tmu.report_theoretical_memory(args) is None
    assert "not yet supported for hybrid Mamba-Transformer models" in capsys.readouterr().out