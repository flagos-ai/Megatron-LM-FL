# Copyright (c) 2026, BAAI. All rights reserved.
"""GPTModel: position-embedding dispatch (rope/yarn/mrope), flash-decode preprocessing, deferred wgrad buffers, and hash-routing forward."""

import json

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def nvte_auto_env(monkeypatch):
    monkeypatch.setenv('NVTE_FLASH_ATTN', '1')
    monkeypatch.setenv('NVTE_FUSED_ATTN', '1')
    monkeypatch.setenv('NVTE_UNFUSED_ATTN', '1')


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _build_gpt(config, **kwargs):
    values = dict(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=8,
    )
    values.update(kwargs)
    return GPTModel(**values)


def _yarn_config(**overrides):
    config = _config(**overrides)
    config.yarn_rotary_scaling_factor = 2.0
    config.yarn_original_max_position_embeddings = 8
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 1.0
    config.yarn_correction_range_round_to_int = True
    return config


def test_gpt_constructor_prefers_position_embedding_type_from_config(model_parallel, nvte_auto_env):
    model_parallel()
    config = _config()
    config.position_embedding_type = 'rope'

    model = _build_gpt(config, position_embedding_type='learned_absolute')

    assert model.position_embedding_type == 'rope'
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    torch.testing.assert_close(
        model.rotary_pos_emb.inv_freq, inv_freq.to(model.rotary_pos_emb.inv_freq.device)
    )
    assert model.rotary_pos_emb.inv_freq.shape == (8,)


def test_gpt_constructor_builds_yarn_rotary_and_preprocess_computes_embedding(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model = _build_gpt(_yarn_config(), position_embedding_type='yarn')

    assert isinstance(model.rotary_pos_emb, YarnRotaryEmbedding)
    input_ids = torch.randint(0, 128, (2, 4))
    position_ids = torch.arange(4).unsqueeze(0).expand(2, 4)
    output = model._preprocess(input_ids, position_ids)
    expected_embedding = model.embedding.word_embeddings(input_ids).transpose(0, 1).contiguous()
    assert output[1].shape == (4, 1, 1, 16)
    torch.testing.assert_close(output[0], expected_embedding)


def test_gpt_constructor_builds_mrope_and_preprocess_computes_embedding(
    model_parallel, nvte_auto_env
):
    model_parallel()
    config = _config()
    config.mrope_section = [4, 2, 2]

    model = _build_gpt(config, position_embedding_type='mrope')

    assert isinstance(model.rotary_pos_emb, MultimodalRotaryEmbedding)
    assert model.mrope_section == [4, 2, 2]
    input_ids = torch.randint(0, 128, (2, 4))
    position_ids = torch.arange(4.0).unsqueeze(0).unsqueeze(0).expand(3, 2, 4).contiguous()
    expected_embedding = model.embedding.word_embeddings(input_ids).transpose(0, 1).contiguous()
    output = model._preprocess(input_ids, position_ids)
    assert output[1].shape == (4, 2, 1, 16)
    torch.testing.assert_close(output[0], expected_embedding)


@pytest.mark.parametrize(
    ('position_embedding_type', 'match'),
    [
        pytest.param('yarn', 'not implemented in YarnRotaryEmbedding', id='yarn'),
        pytest.param('mrope', 'not implemented in MultimodalRotaryEmbedding', id='mrope'),
    ],
)
def test_gpt_precompute_raises_when_flash_decode_with_rope_variant(
    model_parallel, nvte_auto_env, position_embedding_type, match
):
    model_parallel()
    if position_embedding_type == 'yarn':
        config = _yarn_config(flash_decode=True)
    else:
        config = _config(flash_decode=True)
        config.mrope_section = [4, 2, 2]
    model = _build_gpt(config, position_embedding_type=position_embedding_type)

    with InferenceMode.active():
        with pytest.raises(NotImplementedError, match=match):
            model._preprocess(
                torch.randint(0, 128, (2, 4)), torch.arange(4).unsqueeze(0).expand(2, 4)
            )


def test_gpt_preprocess_precomputes_cos_sin_and_offset_for_flash_decode(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model = _build_gpt(_config(flash_decode=True), position_embedding_type='rope')
    context = StaticInferenceContext(max_batch_size=2, max_sequence_length=8)

    with InferenceMode.active():
        output = model._preprocess(
            torch.randint(0, 128, (2, 4)),
            torch.arange(4).unsqueeze(0).expand(2, 4),
            inference_context=context,
        )

    assert len(output) == 6
    assert output[2].shape == (8, 8)
    assert output[3].shape == (8, 8)
    assert output[4].dtype == torch.int32
    assert torch.equal(output[4], torch.zeros(2, dtype=torch.int32, device=output[4].device))
    assert 8 in model.rotary_pos_emb_cache
    assert output[2] is model.rotary_pos_emb_cache[8][0]
    assert output[3] is model.rotary_pos_emb_cache[8][1]
    expected_freqs = torch.outer(torch.arange(8.0), model.rotary_pos_emb.inv_freq)
    torch.testing.assert_close(output[2], torch.cos(expected_freqs))
    torch.testing.assert_close(output[3], torch.sin(expected_freqs))


def test_gpt_constructor_allocates_wgrad_deferral_buffers_on_last_stage_when_deferred(
    model_parallel, nvte_auto_env
):
    model_parallel(pipeline_model_parallel_size=2)
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = _build_gpt(
        _config(
            num_layers=2,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            defer_embedding_wgrad_compute=True,
            gradient_accumulation_fusion=True,
        ),
        pre_process=pre_process,
        post_process=post_process,
    )

    if post_process:
        assert model.embedding_activation_buffer == []
        assert model.grad_output_buffer == []
    else:
        assert not hasattr(model, 'embedding_activation_buffer')
        assert not hasattr(model, 'grad_output_buffer')


def test_gpt_constructor_writes_config_logger_payloads(model_parallel, nvte_auto_env, tmp_path):
    model_parallel()

    model = _build_gpt(_config(config_logger_dir=str(tmp_path)))

    # Each process writes exactly one payload per prefix, carrying its own tp/dp/cp/pp/ep ranks
    # (config_logger.py names them f'{prefix}.rank_{rank_str}.iter{count}'; the state-dict
    # payload is an OrderedDict and is dumped as .pth).
    rank_str = parallel_state.get_all_ranks()
    names = [path.name for path in tmp_path.iterdir()]
    assert [name for name in names if name.startswith('GPTModel.rank_')] == [
        f'GPTModel.rank_{rank_str}.iter0.json'
    ]
    assert [name for name in names if name.startswith('GPTModel_init_ckpt.rank_')] == [
        f'GPTModel_init_ckpt.rank_{rank_str}.iter0.pth'
    ]
    payload = json.loads((tmp_path / f'GPTModel.rank_{rank_str}.iter0.json').read_text())
    assert payload['vocab_size'] == 128
    assert payload['max_sequence_length'] == 8


def test_gpt_forward_threads_input_ids_to_decoder_for_hash_routing(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config(moe_n_hash_layers=1, actual_vocab_size=128)
    model = _build_gpt(config).eval().to(get_current_device())
    device = get_current_device()
    captured = {}
    model.decoder.register_forward_hook(
        lambda module, args, kwargs, output: captured.__setitem__('input_ids', kwargs['input_ids']),
        with_kwargs=True,
    )
    input_ids = torch.randint(0, 128, (2, 4), device=device)

    logits = model(
        input_ids,
        torch.arange(4, device=device).unsqueeze(0).expand(2, 4),
        torch.ones(2, 1, 4, 4, dtype=torch.bool, device=device),
    )

    assert torch.equal(captured['input_ids'], input_ids)
    assert logits.shape == (2, 4, 128)
