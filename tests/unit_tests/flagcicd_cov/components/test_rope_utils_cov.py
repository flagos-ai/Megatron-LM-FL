# Copyright (c) 2026, BAAI. All rights reserved.
"""Rotary position embeddings on CPU: half and interleaved rotation, BSHD and THD application, and unfused fallbacks, checked against a hand-written float64 reference."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.models.common.embeddings.rope_utils as rope_utils


class _CpGroup:
    """Stand-in for a context-parallel process group of the given size and rank."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _rotary_reference(
    t,
    freqs,
    rotary_interleaved=False,
    mla_rotary_interleaved=False,
    mscale=1.0,
    inverse=False,
    mla_output_remove_interleaving=False,
):
    cos_ = torch.cos(freqs) * mscale
    sin_ = torch.sin(freqs) * mscale
    if inverse:
        sin_ = -sin_
    x = t
    if mla_rotary_interleaved:
        x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        stacked = torch.stack((-x2, x1), dim=-1)
        rotated = stacked.view(*stacked.shape[:-2], -1)
    y = x * cos_ + rotated * sin_
    if mla_rotary_interleaved and mla_output_remove_interleaving:
        x1, x2 = torch.chunk(y, 2, dim=-1)
        y = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)
    return y


def test_get_pos_emb_on_this_cp_rank_requires_group():
    with pytest.raises(ValueError, match='cp_group must be provided'):
        rope_utils.get_pos_emb_on_this_cp_rank(torch.zeros(4, 2, 3), 0, None)


def test_rotate_half_interleaved_negates_odd_positions():
    x = torch.arange(16.0).reshape(1, 1, 2, 8)
    rotated = rope_utils._rotate_half(x, True)
    expected = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)
    assert torch.equal(rotated, expected)


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='plain'),
        pytest.param({'rotary_interleaved': True}, id='interleaved'),
        pytest.param({'mscale': 2.0}, id='mscale'),
        pytest.param(
            {
                'mla_rotary_interleaved': True,
                'mscale': 2.0,
                'inverse': True,
                'mla_output_remove_interleaving': True,
            },
            id='mla-inverse',
        ),
    ],
)
def test_apply_rotary_pos_emb_bshd_matches_hand_written_reference(kwargs):
    torch.manual_seed(0)
    t = torch.randn(4, 2, 3, 8, dtype=torch.float64)
    freqs = torch.randn(4, 2, 1, 8, dtype=torch.float64)
    torch.testing.assert_close(
        rope_utils._apply_rotary_pos_emb_bshd(t, freqs, **kwargs),
        _rotary_reference(t, freqs, **kwargs),
    )


def test_apply_rotary_pos_emb_bshd_passes_through_unrotated_tail():
    torch.manual_seed(1)
    t = torch.randn(4, 2, 3, 12, dtype=torch.float64)
    freqs = torch.randn(4, 2, 1, 8, dtype=torch.float64)
    result = rope_utils._apply_rotary_pos_emb_bshd(t, freqs)
    expected = torch.cat((_rotary_reference(t[..., :8], freqs), t[..., 8:]), dim=-1)
    torch.testing.assert_close(result, expected)


def test_apply_rotary_pos_emb_bshd_squeezes_extra_singleton_axis():
    torch.manual_seed(2)
    t = torch.randn(4, 2, 3, 8, dtype=torch.float64)
    freqs = torch.randn(4, 2, 1, 1, 8, dtype=torch.float64)
    torch.testing.assert_close(
        rope_utils._apply_rotary_pos_emb_bshd(t, freqs),
        rope_utils._apply_rotary_pos_emb_bshd(t, freqs.squeeze(-2)),
    )


def test_apply_rotary_pos_emb_bshd_warns_on_deprecated_multi_latent_attention():
    torch.manual_seed(3)
    t = torch.randn(4, 2, 3, 8, dtype=torch.float64)
    freqs = torch.randn(4, 2, 1, 8, dtype=torch.float64)
    with pytest.warns(DeprecationWarning, match='multi_latent_attention is deprecated'):
        result = rope_utils._apply_rotary_pos_emb_bshd(t, freqs, multi_latent_attention=True)
    torch.testing.assert_close(result, _rotary_reference(t, freqs, mla_rotary_interleaved=True))


def test_apply_rotary_pos_emb_thd_requires_cp_group():
    with pytest.raises(ValueError, match='cp_group must be provided for THD format RoPE'):
        rope_utils._apply_rotary_pos_emb_thd(
            torch.zeros(4, 2, 8), torch.tensor([0, 4], dtype=torch.int32), torch.zeros(4, 1, 1, 8)
        )


def test_apply_rotary_pos_emb_thd_warns_on_deprecated_multi_latent_attention():
    torch.manual_seed(6)
    t = torch.randn(4, 2, 8, dtype=torch.float64)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    freqs = torch.randn(4, 1, 1, 8, dtype=torch.float64)
    with pytest.warns(DeprecationWarning, match='multi_latent_attention is deprecated'):
        rope_utils._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=_CpGroup(1, 0), multi_latent_attention=True
        )


def test_apply_rotary_pos_emb_thd_maps_exact_positions_with_offsets():
    torch.manual_seed(4)
    t = torch.randn(4, 2, 8, dtype=torch.float64)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    freqs = torch.randn(8, 1, 1, 8, dtype=torch.float64)
    result = rope_utils._apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_group=_CpGroup(2, 0))
    seg = 1
    full_seqlen = 4
    first = torch.cat((freqs[0:seg], freqs[full_seqlen - seg : full_seqlen]))
    second = torch.cat((freqs[4 : 4 + seg], freqs[4 + full_seqlen - seg : 4 + full_seqlen]))
    expected_freqs = torch.cat((first, second))
    expected = _rotary_reference(t.unsqueeze(1), expected_freqs).squeeze(1)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ('config_kwargs', 'call_kwargs', 'message'),
    [
        pytest.param({'mrope_section': [1, 1, 2]}, {}, 'does not support mRoPE', id='mrope'),
        pytest.param({}, {'mscale': 2.0}, 'is not supported by TE', id='mscale'),
        pytest.param({}, {'mla_rotary_interleaved': True}, 'does not support MLA-style', id='mla'),
        pytest.param({}, {'inverse': True}, 'inverse RoPE is not supported', id='inverse'),
    ],
)
def test_apply_rotary_pos_emb_falls_back_to_unfused_when_fusion_unsupported(
    config_kwargs, call_kwargs, message
):
    torch.manual_seed(5)
    t = torch.randn(4, 2, 3, 8, dtype=torch.float64)
    freqs = torch.randn(4, 2, 1, 8, dtype=torch.float64)
    config_values = dict(apply_rope_fusion=True, mrope_section=None, rotary_interleaved=False)
    config_values.update(config_kwargs)
    config = SimpleNamespace(**config_values)
    with pytest.warns(UserWarning, match=message):
        result = rope_utils.apply_rotary_pos_emb(
            t, freqs, config, cp_group=_CpGroup(1, 0), **call_kwargs
        )
    torch.testing.assert_close(result, _rotary_reference(t, freqs, **call_kwargs))
