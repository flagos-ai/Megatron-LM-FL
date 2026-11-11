# Copyright (c) 2026, BAAI. All rights reserved.
"""BertModel: extended attention mask, cross-entropy loss reference, rope and pipeline-stage forwards."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.bert.bert_layer_specs import bert_layer_local_spec
from megatron.core.models.bert.bert_model import BertModel
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


def _bert_config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _build_bert(config, **kwargs):
    values = dict(
        config=config,
        num_tokentypes=2,
        transformer_layer_spec=bert_layer_local_spec,
        vocab_size=128,
        max_sequence_length=8,
    )
    values.update(kwargs)
    return BertModel(**values)


def test_bert_extended_attention_mask_broadcasts_padding_to_b1ss(model_parallel, nvte_auto_env):
    model_parallel()
    model = _build_bert(_bert_config())

    attention_mask = torch.ones(2, 4)
    attention_mask[1, 2] = 0.0
    extended = model.bert_extended_attention_mask(attention_mask)

    assert model.attn_mask_dimensions == 'b1ss'
    assert extended.shape == (2, 1, 4, 4)
    expected = torch.zeros(2, 4, 4, dtype=torch.bool)
    expected[1, :, 2] = True
    expected[1, 2, :] = True
    assert torch.equal(extended.squeeze(1), expected)


def test_bert_forward_returns_loss_matching_cross_entropy_reference(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_bert(_bert_config(), parallel_output=False).eval().to(get_current_device())
    device = get_current_device()
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)
    labels = torch.randint(0, 128, (2, 4), device=device)

    logits, binary_logits = model(input_ids, attention_mask, torch.zeros_like(input_ids))
    loss, loss_binary_logits = model(
        input_ids, attention_mask, torch.zeros_like(input_ids), lm_labels=labels
    )

    expected_loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)
    assert loss.shape == (2, 4)
    assert binary_logits.shape == (2, 2)
    torch.testing.assert_close(loss_binary_logits, binary_logits)


def test_bert_forward_uses_shared_embedding_weight_when_sharing_with_output(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = (
        _build_bert(_bert_config(), parallel_output=False, share_embeddings_and_output_weights=True)
        .eval()
        .to(get_current_device())
    )
    device = get_current_device()
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, output: captured.__setitem__('encoder_hidden', output)
    )
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)

    logits, _ = model(input_ids, attention_mask, torch.zeros_like(input_ids))

    assert model.shared_embedding_or_output_weight() is model.embedding.word_embeddings.weight
    hidden_after_lm_head = model.lm_head(hidden_states=captured['encoder_hidden'])
    expected_logits, _ = model.output_layer(
        hidden_after_lm_head, weight=model.shared_embedding_or_output_weight()
    )
    torch.testing.assert_close(
        logits, expected_logits.transpose(0, 1).contiguous(), rtol=1e-5, atol=1e-5
    )
    assert logits.shape == (2, 4, 128)


def test_bert_forward_without_binary_head_matches_logits_with_binary_head(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(1234)
    with_head = (
        _build_bert(_bert_config(), add_binary_head=True, parallel_output=False)
        .eval()
        .to(get_current_device())
    )
    torch.manual_seed(1234)
    without_head = (
        _build_bert(_bert_config(), add_binary_head=False, parallel_output=False)
        .eval()
        .to(get_current_device())
    )
    device = get_current_device()
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)

    logits_with, binary_logits_with = with_head(
        input_ids, attention_mask, torch.zeros_like(input_ids)
    )
    logits_without, binary_logits_without = without_head(
        input_ids, attention_mask, torch.zeros_like(input_ids)
    )

    assert binary_logits_without is None
    assert binary_logits_with.shape == (2, 2)
    torch.testing.assert_close(logits_without, logits_with, rtol=1e-5, atol=1e-6)
    assert logits_without.shape == (2, 4, 128)


def test_bert_with_rope_position_embedding_passes_reference_rotary_to_encoder(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(1234)
    model = (
        _build_bert(_bert_config(), position_embedding_type='rope', parallel_output=False)
        .eval()
        .to(get_current_device())
    )
    device = get_current_device()
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, kwargs, output: captured.__setitem__(
            'rotary_pos_emb', kwargs['rotary_pos_emb']
        ),
        with_kwargs=True,
    )
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)

    logits, _ = model(input_ids, attention_mask, torch.zeros_like(input_ids))

    torch.manual_seed(1234)
    twin = (
        _build_bert(_bert_config(), position_embedding_type='rope', parallel_output=False)
        .eval()
        .to(device)
    )
    twin_logits, _ = twin(input_ids, attention_mask, torch.zeros_like(input_ids))

    assert getattr(model.embedding, 'position_embeddings', None) is None
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 16, 2, dtype=torch.float32, device=device) / 16))
    torch.testing.assert_close(model.rotary_pos_emb.inv_freq, inv_freq)
    freqs = torch.outer(torch.arange(4, dtype=torch.float32, device=device), inv_freq)
    expected_rotary = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    torch.testing.assert_close(captured['rotary_pos_emb'], expected_rotary)
    assert torch.equal(twin_logits, logits)
    assert logits.shape == (2, 4, 128)


def test_bert_constructor_writes_config_logger_payload(model_parallel, nvte_auto_env, tmp_path):
    model_parallel()

    model = _build_bert(_bert_config(config_logger_dir=str(tmp_path)))

    assert isinstance(model, BertModel)
    logged = [path.name for path in tmp_path.iterdir() if path.name.startswith('BertModel.rank_')]
    assert len(logged) == Utils.world_size
    assert all(name.endswith('.json') for name in logged)


def test_bert_forward_without_post_process_returns_last_layer_hidden_states(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    device = get_current_device()
    full_model = _build_bert(_bert_config(), parallel_output=False).eval().to(device)
    no_post_model = _build_bert(_bert_config(), post_process=False).eval().to(device)
    no_post_sd = {
        key: value
        for key, value in full_model.state_dict().items()
        if key.startswith('embedding.') or key.startswith('encoder.layers.')
    }
    missing, unexpected = no_post_model.load_state_dict(no_post_sd, strict=False)
    assert missing == []
    assert unexpected == []
    layer_outputs = []
    full_model.encoder.layers[1].register_forward_hook(
        lambda module, args, output: layer_outputs.append(output[0])
    )
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)
    full_model(input_ids, attention_mask, torch.zeros_like(input_ids))

    hidden_states = no_post_model(input_ids, attention_mask, torch.zeros_like(input_ids))

    assert hidden_states.shape == (4, 2, full_model.config.hidden_size)
    torch.testing.assert_close(hidden_states, layer_outputs[0], rtol=1e-5, atol=1e-5)


def test_bert_second_pipeline_stage_fed_first_stage_output_matches_full_model(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(1234)
    device = get_current_device()
    full_model = _build_bert(_bert_config(), parallel_output=False).eval().to(device)
    input_ids = torch.randint(0, 128, (2, 4), device=device)
    attention_mask = torch.ones(2, 4, device=device)
    first_stage_outputs = []
    full_model.encoder.layers[0].register_forward_hook(
        lambda module, args, output: first_stage_outputs.append(output[0])
    )
    full_logits, full_binary_logits = full_model(
        input_ids, attention_mask, torch.zeros_like(input_ids)
    )

    model_parallel(pipeline_model_parallel_size=2)
    stage_two_model = (
        _build_bert(
            _bert_config(pipeline_model_parallel_size=2, pipeline_dtype=torch.float32),
            pre_process=False,
            post_process=True,
            parallel_output=False,
        )
        .eval()
        .to(device)
    )
    stage_two_sd = {}
    for key, value in full_model.state_dict().items():
        if key.startswith('embedding.'):
            continue
        if key.startswith('encoder.layers.'):
            if not key.startswith('encoder.layers.1.'):
                continue
            key = 'encoder.layers.0.' + key.split('.', 3)[3]
        stage_two_sd[key] = value
    missing, unexpected = stage_two_model.load_state_dict(stage_two_sd, strict=False)
    assert missing == []
    assert unexpected == []
    stage_two_model.set_input_tensor(first_stage_outputs[0])

    logits, binary_logits = stage_two_model(input_ids, attention_mask, torch.zeros_like(input_ids))

    torch.testing.assert_close(logits, full_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(binary_logits, full_binary_logits, rtol=1e-5, atol=1e-5)
    assert logits.shape == (2, 4, 128)
    assert binary_logits.shape == (2, 2)
