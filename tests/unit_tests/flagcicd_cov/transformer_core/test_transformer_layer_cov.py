# Copyright (c) 2026, BAAI. All rights reserved.
"""TransformerLayer numbering offsets, selective recompute, graph scoping and static inputs."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.parallel_state import get_pipeline_model_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
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


def _layer(config, **kwargs):
    return build_module(get_gpt_layer_local_spec(), config=config, layer_number=1, **kwargs)


@pytest.mark.parametrize(
    ("pp_rank", "dualpipev_stage", "expected_offset"),
    [(0, 0, 0), (0, 1, 2), (1, 0, 0), (1, 1, 1)],
    ids=["first-stage0", "first-stage1", "second-stage0", "second-stage1"],
)
def test_get_transformer_layer_offset_dualpipev_uneven_first_stage(
    pp_rank, dualpipev_stage, expected_offset
):
    config = _config(
        num_layers=4,
        pipeline_model_parallel_size=2,
        num_layers_in_first_pipeline_stage=1,
        use_dualpipev=True,
        dualpipev_pipeline_model_parallel_size=2,
    )

    offset = get_transformer_layer_offset(config, None, pp_rank, dualpipev_stage)

    assert offset == expected_offset


def test_get_transformer_layer_offset_dualpipev_uneven_requires_stage():
    config = _config(
        num_layers=4,
        pipeline_model_parallel_size=2,
        num_layers_in_first_pipeline_stage=1,
        use_dualpipev=True,
        dualpipev_pipeline_model_parallel_size=2,
    )

    with pytest.raises(AssertionError, match="dualpipev_stage must be provided if use_dualpipev"):
        get_transformer_layer_offset(config, None, 0)


@pytest.mark.parametrize("dualpipev_stage", [0, 1])
def test_get_transformer_layer_offset_dualpipev_two_stage_pipeline(model_parallel, dualpipev_stage):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    config = _config(
        num_layers=4,
        pipeline_model_parallel_size=2,
        use_dualpipev=True,
        dualpipev_pipeline_model_parallel_size=2,
    )
    pp_rank = get_pipeline_model_parallel_rank()

    offset = get_transformer_layer_offset(config, None, pp_rank, dualpipev_stage)

    assert offset == (pp_rank if dualpipev_stage == 0 else 3 - pp_rank)


@pytest.mark.parametrize(("pp_rank", "expected_offset"), [(0, 0), (1, 1)])
def test_get_transformer_layer_offset_without_middle_stages(pp_rank, expected_offset):
    config = _config(
        num_layers=2,
        pipeline_model_parallel_size=2,
        num_layers_in_first_pipeline_stage=1,
        num_layers_in_last_pipeline_stage=1,
    )

    assert get_transformer_layer_offset(config, None, pp_rank) == expected_offset


@pytest.mark.parametrize(
    ("vp_stage", "pp_rank", "expected_offset"),
    [(0, 0, 0), (0, 1, 0), (1, 0, 1)],
    ids=["first-chunk-first-stage", "first-chunk-second-stage", "second-chunk-first-stage"],
)
def test_get_transformer_layer_offset_deducts_embedding_after_first_stage(
    vp_stage, pp_rank, expected_offset
):
    config = _config(
        num_layers=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        account_for_embedding_in_pipeline_split=True,
    )

    assert get_transformer_layer_offset(config, vp_stage, pp_rank) == expected_offset


def test_layer_get_layer_offset_delegates_with_deprecation_warning(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config()
    layer = _layer(config)

    with pytest.warns(UserWarning, match="Please use get_transformer_layer_offset instead"):
        offset = layer._get_layer_offset(config)

    assert offset == get_transformer_layer_offset(config)


def test_layer_number_adds_decoder_offset_on_second_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    config = _config(
        num_layers=4, pipeline_model_parallel_size=2, num_layers_in_first_pipeline_stage=1
    )

    layer = _layer(config)

    expected_number = 2 if get_pipeline_model_parallel_rank() == 1 else 1
    assert layer.layer_number == expected_number


def test_is_mtp_layer_number_skips_decoder_offset(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    config = _config(
        num_layers=4, pipeline_model_parallel_size=2, num_layers_in_first_pipeline_stage=1
    )

    layer = _layer(config, is_mtp_layer=True)

    assert layer.is_mtp_layer is True
    assert layer.layer_number == 1


def test_layer_passes_list_cp_comm_type_for_first_layer(model_parallel):
    model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    config = _config(num_layers=2, context_parallel_size=2, cp_comm_type=['p2p', 'p2p'])

    layer = TransformerLayer(
        config,
        get_gpt_layer_with_transformer_engine_submodules(),
        layer_number=1,
        is_mtp_layer=True,
    )

    assert layer.self_attention.core_attention.cp_comm_type == 'p2p'


def test_selective_recompute_layernorm_and_mlp_preserves_values_and_grads(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    hidden_states = torch.randn(8, 2, 32)
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool)

    outputs = {}
    grads = {}
    for recompute in (False, True):
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(123)
        overrides = (
            dict(recompute_granularity='selective', recompute_modules=['layernorm', 'mlp'])
            if recompute
            else dict()
        )
        layer = _layer(_config(**overrides)).train()
        output, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        output.sum().backward()
        outputs[recompute] = output.detach()
        grads[recompute] = {
            name: param.grad.detach().clone() for name, param in layer.named_parameters()
        }
        if recompute:
            assert layer.recompute_input_layernorm is True
            assert layer.recompute_pre_mlp_layernorm is True
            assert layer.recompute_mlp is True

    torch.testing.assert_close(outputs[True], outputs[False], rtol=0.0, atol=0.0)
    assert grads[True].keys() == grads[False].keys()
    for name in grads[True]:
        torch.testing.assert_close(grads[True][name], grads[False][name], rtol=0.0, atol=0.0)


def test_fp32_residual_connection_preserves_fp32_layer_values(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    hidden_states = torch.randn(8, 2, 32)
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool)

    outputs = []
    for fp32_residual in (False, True):
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(123)
        layer = _layer(_config(fp32_residual_connection=fp32_residual)).eval()
        with torch.no_grad():
            output, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        outputs.append(output)

    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.0, atol=0.0)


def test_te_cuda_graph_capture_mlp_scope_returns_mlp_output_only(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config(cuda_graph_modules=[CudaGraphModule.mlp])).eval()
    hidden_states = torch.randn(8, 2, 32)
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool)

    with torch.no_grad():
        captured = layer._te_cuda_graph_capture(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        expected = layer._forward_mlp(hidden_states)

    assert isinstance(captured, tuple)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, rtol=0.0, atol=0.0)


def test_get_submodules_under_cudagraphs_returns_whole_layer_without_scope(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config())

    assert layer._get_submodules_under_cudagraphs() == [layer]


@pytest.mark.parametrize(
    ("scope", "expected_names"),
    [
        (
            [CudaGraphModule.attn],
            ['input_layernorm', 'self_attention', 'pre_cross_attn_layernorm', 'cross_attention'],
        ),
        ([CudaGraphModule.mlp], ['pre_mlp_layernorm', 'mlp']),
    ],
    ids=["attn-scope", "mlp-scope"],
)
def test_get_submodules_under_cudagraphs_lists_scoped_modules(
    model_parallel, scope, expected_names
):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config(cuda_graph_modules=scope))

    submodules = layer._get_submodules_under_cudagraphs()

    assert submodules == [getattr(layer, name) for name in expected_names]


def test_get_layer_static_inputs_contains_causal_attention_mask(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config())

    static_inputs = layer.get_layer_static_inputs(seq_length=8, micro_batch_size=2)

    assert set(static_inputs) == {'hidden_states', 'attention_mask'}
    hidden_states = static_inputs['hidden_states']
    assert hidden_states.shape == (8, 2, 32)
    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.requires_grad is True
    expected_mask = (~torch.tril(torch.ones((8, 8))).bool()).reshape(1, 1, 8, 8).tile(2, 1, 1, 1)
    assert torch.equal(static_inputs['attention_mask'].cpu(), expected_mask)
