# Copyright (c) 2026, BAAI. All rights reserved.
"""TransformerBlock layer planning, per-microbatch recompute tables and forward outputs."""

import json

import pytest
import torch

from megatron.core import mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.parallel_state import get_pipeline_model_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    defaults = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        pipeline_dtype=torch.bfloat16,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _block(model_parallel, **overrides):
    model_parallel(tensor_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(123)
    block = TransformerBlock(_config(**overrides), get_gpt_layer_local_spec()).cuda().eval()
    hidden_states = torch.randn(8, 2, 32, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")
    return block, hidden_states, attention_mask


@pytest.mark.parametrize(("pp_rank", "expected"), [(0, 1), (1, 1)], ids=["first", "last"])
def test_get_num_layers_to_build_without_middle_stages(pp_rank, expected):
    config = _config(
        num_layers=2,
        pipeline_model_parallel_size=2,
        num_layers_in_first_pipeline_stage=1,
        num_layers_in_last_pipeline_stage=1,
    )

    assert get_num_layers_to_build(config, None, pp_rank) == expected


@pytest.mark.parametrize(("pp_rank", "expected"), [(0, 1), (1, 1)], ids=["first", "second"])
def test_get_num_layers_to_build_dualpipev_halves_stage_layers(pp_rank, expected):
    config = _config(
        num_layers=4,
        pipeline_model_parallel_size=2,
        use_dualpipev=True,
        dualpipev_pipeline_model_parallel_size=2,
    )

    assert get_num_layers_to_build(config, None, pp_rank) == expected


@pytest.mark.parametrize(
    ("vp_stage", "pp_rank", "expected"),
    [(0, 0, 0), (1, 0, 1), (0, 1, 1)],
    ids=["embedding-deducted", "second-chunk", "second-stage"],
)
def test_get_num_layers_to_build_deducts_embedding_after_first_vp_stage(
    vp_stage, pp_rank, expected
):
    config = _config(
        num_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        account_for_embedding_in_pipeline_split=True,
    )

    assert get_num_layers_to_build(config, vp_stage, pp_rank) == expected


def test_get_num_layers_to_build_deducts_loss_on_second_dualpipev_stage():
    config = _config(
        num_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        account_for_loss_in_pipeline_split=True,
    )

    assert get_num_layers_to_build(config, 0, 0, 1) == 0


@pytest.mark.parametrize(
    ("vp_stage", "pp_rank", "expected"),
    [(1, 1, 0), (0, 1, 1)],
    ids=["last-vp-stage-deducts-loss", "first-vp-stage-keeps-layer"],
)
def test_get_num_layers_to_build_deducts_loss_on_last_vp_stage(vp_stage, pp_rank, expected):
    config = _config(
        num_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        account_for_loss_in_pipeline_split=True,
    )

    assert get_num_layers_to_build(config, vp_stage, pp_rank) == expected


def test_get_num_layers_to_build_counts_layout_decoders_per_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    layout = PipelineParallelLayerLayout(
        [["embedding", "decoder", "decoder"], ["decoder", "mtp", "loss"]],
        pipeline_model_parallel_size=2,
    )
    config = _config(
        num_layers=3,
        pipeline_model_parallel_size=2,
        mtp_num_layers=1,
        pipeline_model_parallel_layout=layout,
    )

    built = get_num_layers_to_build(config, None)

    assert built == (2 if get_pipeline_model_parallel_rank() == 0 else 1)


def test_block_forward_returns_intermediate_hidden_states(model_parallel):
    block, hidden_states, attention_mask = _block(model_parallel, num_layers=2)

    with torch.no_grad():
        extracted, intermediates = block(
            hidden_states=hidden_states, attention_mask=attention_mask, extract_layer_indices={0}
        )
        plain = block(hidden_states=hidden_states, attention_mask=attention_mask)
        layer_zero_output, _ = block.layers[0](
            hidden_states=hidden_states, attention_mask=attention_mask
        )

    assert len(intermediates) == 1
    torch.testing.assert_close(extracted, plain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(intermediates[0], layer_zero_output, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("per_microbatch_table", "expected_method"),
    [([[0]], "uniform"), ([[1]], "block")],
    ids=["uniform-entry", "block-entry"],
)
def test_block_forward_sets_recompute_method_from_per_microbatch_table(
    model_parallel, per_microbatch_table, expected_method
):
    block, hidden_states, attention_mask = _block(model_parallel)
    with torch.no_grad():
        reference = block(hidden_states=hidden_states, attention_mask=attention_mask)

    table_block, table_hidden, table_mask = _block(
        model_parallel, recompute_method_per_stage_micro_batch=per_microbatch_table
    )
    with torch.no_grad():
        output = table_block(hidden_states=table_hidden, attention_mask=table_mask)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
    assert table_block.config.recompute_method == expected_method


def test_block_forward_rejects_invalid_recompute_method_entry(model_parallel):
    block, hidden_states, attention_mask = _block(
        model_parallel, recompute_method_per_stage_micro_batch=[[2]]
    )

    with pytest.raises(ValueError, match="must be '0' or '1'"):
        block(hidden_states=hidden_states, attention_mask=attention_mask)


@pytest.mark.parametrize(
    ("per_microbatch_table", "expected_num_layers", "expected_method"),
    [([[0]], 0, None), ([[1]], 1, None)],
    ids=["zero-disables-recompute", "one-sets-num-layers"],
)
def test_block_forward_recompute_num_layers_table_sets_num_layers(
    model_parallel, per_microbatch_table, expected_num_layers, expected_method
):
    block, hidden_states, attention_mask = _block(model_parallel)
    with torch.no_grad():
        reference = block(hidden_states=hidden_states, attention_mask=attention_mask)

    table_block, table_hidden, table_mask = _block(
        model_parallel, recompute_num_layers_per_stage_micro_batch=per_microbatch_table
    )
    with torch.no_grad():
        output = table_block(hidden_states=table_hidden, attention_mask=table_mask)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
    assert table_block.config.recompute_num_layers == expected_num_layers
    assert table_block.config.recompute_method == expected_method


@pytest.mark.parametrize(
    ("granularity_table", "expected_method"),
    [([[0]], None), ([[1]], "uniform")],
    ids=["zero-entry-clears-method", "nonzero-entry-keeps-method"],
)
def test_block_forward_zero_recompute_granularity_entry_clears_method(
    model_parallel, granularity_table, expected_method
):
    block, hidden_states, attention_mask = _block(model_parallel)
    with torch.no_grad():
        reference = block(hidden_states=hidden_states, attention_mask=attention_mask)

    table_block, table_hidden, table_mask = _block(
        model_parallel,
        recompute_method="uniform",
        recompute_granularity_per_stage_micro_batch=granularity_table,
    )
    with torch.no_grad():
        output = table_block(hidden_states=table_hidden, attention_mask=table_mask)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
    assert table_block.config.recompute_method == expected_method


def test_block_forward_empty_stage_clones_input_tensor(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    layout = PipelineParallelLayerLayout(
        [["embedding", "decoder", "decoder"], ["loss"]], pipeline_model_parallel_size=2
    )
    config = _config(
        num_layers=2, pipeline_model_parallel_size=2, pipeline_model_parallel_layout=layout
    )
    block = TransformerBlock(
        config, get_gpt_layer_local_spec(), pre_process=False, post_process=False
    )

    if get_pipeline_model_parallel_rank() == 1:
        assert len(block.layers) == 0
        block.set_input_tensor(torch.randn(8, 2, 32, device="cuda"))
        hidden_states = block.input_tensor

        with torch.no_grad():
            output = block(hidden_states=hidden_states, attention_mask=None)

        assert output is not hidden_states
        torch.testing.assert_close(output, hidden_states, rtol=0.0, atol=0.0)
    else:
        assert len(block.layers) == 2


def test_block_call_runs_plain_forward_outside_inference_mode(model_parallel):
    block, hidden_states, attention_mask = _block(model_parallel)

    with torch.no_grad():
        called = block(hidden_states=hidden_states, attention_mask=attention_mask)
        direct = block.forward(hidden_states=hidden_states, attention_mask=attention_mask)

    torch.testing.assert_close(called, direct, rtol=0.0, atol=0.0)


def _hetero_config(tmp_path):
    block_configs = {
        "block_configs": [
            {"attention": {"n_heads_in_group": 2}, "mlp": {"ffn_mult": 2.625}},
            {"attention": {"n_heads_in_group": 4}, "mlp": {"ffn_hidden_size": 512}},
        ]
    }
    path = tmp_path / "hetero.json"
    path.write_text(json.dumps(block_configs))
    return HeterogeneousTransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        heterogeneous_layers_config_path=str(path),
        hetereogenous_dist_checkpoint=True,
    )


def test_hetero_block_builds_per_layer_qkv_shapes(model_parallel, tmp_path):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _hetero_config(tmp_path)

    block = TransformerBlock(config, get_gpt_heterogeneous_layer_spec(config, use_te=False))

    assert len(block.layers) == 2
    assert block.layers[0].self_attention.linear_qkv.weight.shape == (64, 32)
    assert block.layers[1].self_attention.linear_qkv.weight.shape == (48, 32)
    assert block.layers[0].mlp.linear_fc1.weight.shape == (256, 32)
    assert block.layers[1].mlp.linear_fc1.weight.shape == (512, 32)


def test_hyper_connection_block_sharded_state_dict_keeps_head_params(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(enable_hyper_connections=True, num_residual_streams=2)
    block = TransformerBlock(
        config, get_gpt_layer_local_spec(enable_hyper_connection=True), post_process=True
    )

    sharded = block.sharded_state_dict(
        metadata={
            "dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True),
            "non_homogeneous_layers": False,
        }
    )

    assert sharded['hc_head_fn'].global_shape == (2, 64)
    assert sharded['hc_head_base'].global_shape == (2,)
    assert sharded['hc_head_scale'].global_shape == (1,)
    assert sharded['hc_head_fn'].data is block.hc_head_fn
    assert sharded['hc_head_base'].data is block.hc_head_base


def test_hetero_block_sharded_state_dict_indexes_layers_globally(model_parallel, tmp_path):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _hetero_config(tmp_path)
    block = TransformerBlock(config, get_gpt_heterogeneous_layer_spec(config, use_te=False))

    sharded = block.sharded_state_dict(
        metadata={
            "dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True),
            "non_homogeneous_layers": True,
        }
    )

    assert "layers.0.self_attention.linear_qkv.weight" in sharded
    assert "layers.1.self_attention.linear_qkv.weight" in sharded
    assert sharded["layers.0.self_attention.linear_qkv.weight"].global_shape == (64, 32)
    assert sharded["layers.1.self_attention.linear_qkv.weight"].global_shape == (48, 32)
    assert sharded["layers.1.mlp.linear_fc1.weight"].global_shape == (512, 32)
