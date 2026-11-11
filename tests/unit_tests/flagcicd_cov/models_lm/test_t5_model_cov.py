# Copyright (c) 2026, BAAI. All rights reserved.
"""T5Model: position ids, attention mask, loss reference, rope/relative embeddings, stage routing."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.T5.t5_model import T5Model, t5_extended_attention_mask, t5_position_ids
from megatron.core.models.T5.t5_spec import (
    get_t5_decoder_with_local_block_spec,
    get_t5_decoder_with_transformer_engine_block_spec,
    get_t5_encoder_with_local_block_spec,
    get_t5_encoder_with_transformer_engine_block_spec,
)
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


def _t5_config(**overrides):
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


def _build_t5(config, **kwargs):
    values = dict(
        config=config,
        encoder_config=config,
        transformer_encoder_layer_spec=get_t5_encoder_with_local_block_spec(config.num_layers),
        transformer_decoder_layer_spec=get_t5_decoder_with_local_block_spec(config.num_layers),
        vocab_size=128,
        max_sequence_length=8,
        parallel_output=False,
    )
    values.update(kwargs)
    return T5Model(**values)


def _inputs(device):
    encoder_ids = torch.randint(0, 128, (2, 8), device=device)
    decoder_ids = torch.randint(0, 128, (2, 6), device=device)
    encoder_mask = torch.ones(2, 1, 8, 8, dtype=torch.bool, device=device)
    decoder_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool, device=device)).expand(2, 1, 6, 6)
    cross_mask = torch.ones(2, 1, 6, 8, dtype=torch.bool, device=device)
    return encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask


def test_t5_position_ids_expand_arange_over_token_shape():
    token_ids = torch.randint(0, 128, (2, 3))

    position_ids = t5_position_ids(token_ids)

    assert torch.equal(position_ids, torch.arange(3).unsqueeze(0).expand(2, 3))
    assert position_ids.dtype == torch.long


def test_t5_extended_attention_mask_unsqueezes_each_entry_and_keeps_none():
    masks = [torch.ones(2, 4, 4), None, torch.zeros(2, 3, 3)]

    extended = t5_extended_attention_mask(masks)

    assert torch.equal(extended[0], masks[0].unsqueeze(1))
    assert extended[1] is None
    assert torch.equal(extended[2], masks[2].unsqueeze(1))


def test_t5_forward_returns_logits_and_loss_matching_reference(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config()).eval().to(get_current_device())
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    labels = torch.randint(0, 128, (2, 6), device=device)

    logits = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)
    loss = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask, lm_labels=labels)

    expected_loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
    assert logits.shape == (2, 6, 128)
    assert loss.shape == (2, 6)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)


def test_t5_with_rope_position_embedding_passes_reference_rotary_to_encoder(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config(), position_embedding_type='rope').eval().to(get_current_device())
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, kwargs, output: captured.__setitem__(
            'rotary_pos_emb', kwargs['rotary_pos_emb']
        ),
        with_kwargs=True,
    )

    logits = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)
    encoder_hidden = model(
        encoder_ids,
        decoder_ids,
        encoder_mask,
        decoder_mask,
        cross_mask,
        output_encoder_hidden_only=True,
    )
    round_trip_logits = model(
        encoder_ids,
        decoder_ids,
        encoder_mask,
        decoder_mask,
        cross_mask,
        encoder_hidden_states=encoder_hidden,
    )

    assert model.position_embeddings is None
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 16, 2, dtype=torch.float32, device=device) / 16))
    torch.testing.assert_close(model.rotary_pos_emb.inv_freq, inv_freq)
    freqs = torch.outer(torch.arange(8, dtype=torch.float32, device=device), inv_freq)
    expected_rotary = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    torch.testing.assert_close(captured['rotary_pos_emb'], expected_rotary)
    torch.testing.assert_close(round_trip_logits, logits, rtol=1e-5, atol=1e-5)
    assert logits.shape == (2, 6, 128)


def test_t5_with_relative_position_embedding_scatters_attention_bias(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _t5_config()
    model = (
        T5Model(
            config=config,
            encoder_config=config,
            transformer_encoder_layer_spec=get_t5_encoder_with_transformer_engine_block_spec(
                config.num_layers
            ),
            transformer_decoder_layer_spec=get_t5_decoder_with_transformer_engine_block_spec(
                config.num_layers
            ),
            vocab_size=128,
            max_sequence_length=8,
            parallel_output=False,
            position_embedding_type='relative',
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        .eval()
        .to(get_current_device())
    )
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, kwargs, output: captured.__setitem__(
            'encoder_bias', kwargs['attention_bias']
        ),
        with_kwargs=True,
    )
    model.decoder.register_forward_hook(
        lambda module, args, kwargs, output: captured.__setitem__(
            'decoder_bias', kwargs['attention_bias']
        ),
        with_kwargs=True,
    )

    logits = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)

    assert model.encoder_relative_pos_emb.bidirectional is True
    assert model.decoder_relative_pos_emb.bidirectional is False
    assert captured['encoder_bias'].shape == (1, 4, 8, 8)
    assert captured['decoder_bias'].shape == (1, 4, 6, 6)
    for key, embedding, length, bidirectional in (
        ('encoder_bias', model.encoder_relative_pos_emb, 8, True),
        ('decoder_bias', model.decoder_relative_pos_emb, 6, False),
    ):
        table_device = embedding.relative_attention_bias.weight.device
        relative_position = (
            torch.arange(length, device=table_device)[None, :]
            - torch.arange(length, device=table_device)[:, None]
        )
        buckets = embedding._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=embedding.relative_attention_num_buckets,
            max_distance=embedding.relative_attention_max_distance,
        )
        expected_bias = embedding.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0)
        torch.testing.assert_close(captured[key], expected_bias)
    assert logits.shape == (2, 6, 128)


def test_t5_forward_returns_encoder_hidden_states_when_output_encoder_hidden_only(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config()).eval().to(get_current_device())
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, output: captured.__setitem__('encoder_hidden', output)
    )
    model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)

    hidden_states = model(
        encoder_ids,
        decoder_ids,
        encoder_mask,
        decoder_mask,
        cross_mask,
        output_encoder_hidden_only=True,
    )

    assert hidden_states.shape == (8, 2, 64)
    torch.testing.assert_close(hidden_states, captured['encoder_hidden'], rtol=1e-5, atol=1e-5)


def test_t5_forward_consumes_precomputed_encoder_hidden_states(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config()).eval().to(get_current_device())
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    full_logits = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)
    encoder_hidden = model(
        encoder_ids,
        decoder_ids,
        encoder_mask,
        decoder_mask,
        cross_mask,
        output_encoder_hidden_only=True,
    )

    logits = model(
        encoder_ids,
        decoder_ids,
        encoder_mask,
        decoder_mask,
        cross_mask,
        encoder_hidden_states=encoder_hidden,
    )

    torch.testing.assert_close(logits, full_logits, rtol=1e-5, atol=1e-5)
    assert logits.shape == (2, 6, 128)


def test_t5_set_input_tensor_routes_to_encoder_on_encoder_decoder_stage(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config())
    tensor = torch.randn(8, 2, 64)

    model.set_input_tensor(tensor)

    assert torch.equal(model.encoder.input_tensor, tensor)


def test_t5_set_input_tensor_routes_to_encoder_on_encoder_only_stage(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config(), add_encoder=True, add_decoder=False)
    tensor = torch.randn(8, 2, 64)

    model.set_input_tensor(tensor)

    assert torch.equal(model.encoder.input_tensor, tensor)


def test_t5_set_input_tensor_stores_encoder_state_on_decoder_only_stage(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _t5_config()
    model = _build_t5(config, add_encoder=False, add_decoder=True)
    decoder_tensor = torch.randn(6, 2, 64)
    encoder_tensor = torch.randn(8, 2, 64)

    model.set_input_tensor([decoder_tensor, encoder_tensor])

    assert torch.equal(model.decoder.input_tensor, decoder_tensor)
    assert torch.equal(model.encoder_hidden_state, encoder_tensor)

    model.set_input_tensor([encoder_tensor])

    assert model.decoder.input_tensor is None
    assert torch.equal(model.encoder_hidden_state, encoder_tensor)

    with pytest.raises(Exception, match='input_tensor must have either length 1 or 2'):
        model.set_input_tensor([])


def test_t5_set_input_tensor_raises_when_stage_has_no_encoder_or_decoder(
    model_parallel, nvte_auto_env
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config(), add_encoder=False, add_decoder=False)

    with pytest.raises(Exception, match='Stage must have at least either encoder or decoder'):
        model.set_input_tensor(torch.randn(6, 2, 64))


def test_t5_shared_embedding_or_output_weight_per_stage(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _t5_config()
    full_model = _build_t5(config)
    assert full_model.shared_embedding_or_output_weight() is (
        full_model.embedding.word_embeddings.weight
    )

    post_only_model = _build_t5(config, pre_process=False, post_process=True)
    assert post_only_model.shared_embedding_or_output_weight() is (
        post_only_model.lm_head.output_layer.weight
    )

    empty_stage_model = _build_t5(
        config, pre_process=False, post_process=False, add_encoder=False, add_decoder=False
    )
    assert empty_stage_model.shared_embedding_or_output_weight() is None


def test_t5_decoder_only_stage_reproduces_full_model_logits(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = _build_t5(_t5_config()).eval().to(get_current_device())
    device = get_current_device()
    encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask = _inputs(device)
    captured = {}
    model.encoder.register_forward_hook(
        lambda module, args, output: captured.__setitem__('encoder_hidden', output)
    )
    full_logits = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)

    decoder_only = _build_t5(_t5_config(), add_encoder=False, pre_process=False)
    decoder_only_sd = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith('decoder.')
        or key.startswith('lm_head.')
        or key.startswith('output_layer.')
    }
    missing, unexpected = decoder_only.load_state_dict(decoder_only_sd, strict=False)
    assert missing == []
    assert unexpected == []
    decoder_only.set_input_tensor(
        [
            model.embedding(input_ids=decoder_ids, position_ids=t5_position_ids(decoder_ids)),
            captured['encoder_hidden'],
        ]
    )
    decoder_only.eval().to(device)

    logits = decoder_only(encoder_ids, decoder_ids, encoder_mask, decoder_mask, cross_mask)

    torch.testing.assert_close(logits, full_logits, rtol=1e-5, atol=1e-5)
    assert logits.shape == (2, 6, 128)


def test_t5_lm_head_constructor_writes_config_logger_payload(
    model_parallel, nvte_auto_env, tmp_path
):
    model_parallel()

    model = _build_t5(_t5_config(config_logger_dir=str(tmp_path)))

    logged = [path.name for path in tmp_path.iterdir() if path.name.startswith('T5LMHead.rank_')]
    assert len(logged) == Utils.world_size
    assert all(name.endswith('.json') for name in logged)
