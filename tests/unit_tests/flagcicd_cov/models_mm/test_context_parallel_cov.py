# Copyright (c) 2026, BAAI. All rights reserved.
"""Multimodal context parallelism: padding arithmetic, sample splits across CP ranks, gather/reduce-scatter gradients, tubelet-aware split points, and dynamic-resolution vision splits."""
import pytest
import torch

from megatron.core.models.multimodal.context_parallel import (
    GatherFromContextParallelRanks,
    _compute_tubelet_aware_split_points,
    gather_from_context_parallel_ranks,
    get_padding,
    split_to_context_parallel_ranks,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size
from tests.unit_tests.test_utilities import Utils

PATCH_DIM = 16
PATCH_HIDDEN = 3 * PATCH_DIM * PATCH_DIM


@pytest.fixture
def cp_two_ranks():
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def cp_four_ranks():
    Utils.initialize_model_parallel(1, 1, context_parallel_size=4)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "seq_len,cp_size,tp_size,has_sp,expected",
    [(8, 1, 1, False, 0), (10, 1, 4, True, 2), (9, 2, 1, False, 3), (9, 2, 2, True, 7)],
    ids=["no_parallelism", "sp_only", "cp_only", "cp_and_sp"],
)
def test_padding_rounds_sequence_up_to_parallelism_multiple(
    seq_len, cp_size, tp_size, has_sp, expected
):
    assert get_padding(seq_len, cp_size, tp_size, has_sp) == expected


def test_fp8_padding_rounds_sequence_up_to_sixteen():
    assert get_padding(17, 1, 1, False, fp8_enabled=True, fp8_recipe=None) == 15


def test_mxfp8_padding_rounds_sequence_up_to_thirty_two():
    assert get_padding(33, 1, 1, False, fp8_enabled=True, fp8_recipe="mxfp8") == 31


def test_tp_comm_overlap_padding_is_decoder_minus_vision_length():
    assert get_padding(8, 1, 1, True, decoder_tp_comm_overlap=True, decoder_seq_len=12) == 4


def test_tp_comm_overlap_padding_requires_decoder_seq_len():
    with pytest.raises(AssertionError, match="Please provide decoder seq length"):
        get_padding(8, 1, 1, True, decoder_tp_comm_overlap=True, decoder_seq_len=None)


def test_split_pads_last_rank_to_equal_samples(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    global_t = torch.arange(5, dtype=torch.float32).unsqueeze(1).repeat(1, 3).cuda()

    local_t, global_pad = split_to_context_parallel_ranks(global_t)

    assert get_context_parallel_world_size() == 2
    assert global_pad == 1
    assert local_t.shape == torch.Size([3, 3])
    if cp_rank == 0:
        torch.testing.assert_close(local_t, global_t[:3])
    else:
        expected = torch.cat([global_t[3:], torch.zeros(1, 3, device=global_t.device)])
        torch.testing.assert_close(local_t, expected)


def test_gather_and_reduce_scatter_round_trip_with_gradients(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    local_t = torch.full((2, 4, 5), float(cp_rank + 1), device='cuda', requires_grad=True)

    global_t = gather_from_context_parallel_ranks(local_t, 0)

    assert global_t.shape == torch.Size([2, 8, 5])
    torch.testing.assert_close(global_t[:, :4], torch.full_like(global_t[:, :4], 1.0))
    torch.testing.assert_close(global_t[:, 4:], torch.full_like(global_t[:, 4:], 2.0))

    global_t.backward(torch.ones_like(global_t))
    torch.testing.assert_close(local_t.grad, torch.full_like(local_t, 2.0))


def test_symbolic_gather_matches_functional_gather(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    local_t = torch.full((2, 4, 5), float(cp_rank + 1), device='cuda')

    symbolic_global = GatherFromContextParallelRanks.symbolic(None, local_t)
    forward_global = GatherFromContextParallelRanks.apply(local_t)

    torch.testing.assert_close(symbolic_global, forward_global)
    assert symbolic_global.shape == torch.Size([2, 8, 5])


def test_gather_strips_trailing_padding(cp_two_ranks):
    local_t = torch.full((2, 4, 5), float(get_context_parallel_rank() + 1), device='cuda')

    gathered = gather_from_context_parallel_ranks(local_t, 4)

    assert gathered.shape == torch.Size([2, 4, 5])
    torch.testing.assert_close(gathered, torch.full_like(gathered, 1.0))


@pytest.mark.parametrize(
    "num_frames,temporal_patch_size,cp_size,total_frames,expected",
    [
        ([1, 1], 1, 2, 2, [0, 1, 2]),
        ([4], 2, 2, 4, [0, 2, 4]),
        ([1, 4], 2, 2, 5, [0, 3, 5]),
        ([3], 3, 2, 3, [0, 0, 3]),
        ([1, 1], 1, 2, 10, [0, 2, 10]),
    ],
    ids=[
        "media_boundary_split",
        "tubelet_boundary_inside_media",
        "split_inside_second_media",
        "single_tubelet_closer_to_start",
        "target_beyond_all_media",
    ],
)
def test_tubelet_aware_split_points(
    num_frames, temporal_patch_size, cp_size, total_frames, expected
):
    split_points = _compute_tubelet_aware_split_points(
        num_frames, temporal_patch_size, cp_size, total_frames
    )

    assert split_points == expected
    assert split_points[0] == 0
    assert split_points[-1] == total_frames


def test_dynamic_res_split_assigns_whole_tubelets_per_rank(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    num_frames = [4]
    global_t = torch.zeros(1, 16, PATCH_HIDDEN).cuda()
    global_t[0, :, :] = torch.arange(16, device='cuda').unsqueeze(1).float()
    imgs_sizes = torch.tensor([[32, 32]] * 4, device='cuda')
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device='cuda')
    )

    local_t, local_imgs_sizes, local_params, has_padding, num_padded_ranks, local_num_frames = (
        split_to_context_parallel_ranks_dynamic_res(
            global_t,
            imgs_sizes,
            packed_seq_params,
            patch_dim=PATCH_DIM,
            num_frames=torch.tensor(num_frames, device='cuda'),
            temporal_patch_size=2,
        )
    )

    assert has_padding is False
    assert num_padded_ranks == 0
    if cp_rank == 0:
        assert local_t.shape == torch.Size([1, 8, PATCH_HIDDEN])
        torch.testing.assert_close(local_t[0, :, 0], torch.arange(0, 8, device='cuda').float())
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [32, 32]], device='cuda')
        )
        # The split is per frame: each rank keeps one cu_seqlens entry per frame it owns.
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32, device='cuda')
        )
        assert local_num_frames.tolist() == [2]
    else:
        assert local_t.shape == torch.Size([1, 8, PATCH_HIDDEN])
        torch.testing.assert_close(local_t[0, :, 0], torch.arange(8, 16, device='cuda').float())
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [32, 32]], device='cuda')
        )
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32, device='cuda')
        )
        assert local_num_frames.tolist() == [2]


def test_dynamic_res_split_last_rank_absorbs_remainder_media(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    global_t = torch.zeros(1, 9, PATCH_HIDDEN).cuda()
    global_t[0, :, :] = torch.arange(9, device='cuda').unsqueeze(1).float()
    imgs_sizes = torch.tensor([[32, 32], [32, 32], [16, 16]], device='cuda')
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 8, 9], dtype=torch.int32, device='cuda')
    )

    local_t, local_imgs_sizes, local_params, has_padding, num_padded_ranks, local_num_frames = (
        split_to_context_parallel_ranks_dynamic_res(
            global_t, imgs_sizes, packed_seq_params, patch_dim=PATCH_DIM
        )
    )

    assert has_padding is False
    assert num_padded_ranks == 0
    assert local_num_frames is None
    if cp_rank == 0:
        torch.testing.assert_close(local_t[0, :, 0], torch.arange(4, device='cuda').float())
        torch.testing.assert_close(local_imgs_sizes, torch.tensor([[32, 32]], device='cuda'))
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4], dtype=torch.int32, device='cuda')
        )
    else:
        torch.testing.assert_close(local_t[0, :, 0], torch.arange(4, 9, device='cuda').float())
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [16, 16]], device='cuda')
        )
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4, 5], dtype=torch.int32, device='cuda')
        )


def test_dynamic_res_split_pads_local_sequence_for_fp8(cp_two_ranks):
    cp_rank = get_context_parallel_rank()
    global_t = torch.zeros(1, 9, PATCH_HIDDEN).cuda()
    global_t[0, :, :] = torch.arange(9, device='cuda').unsqueeze(1).float()
    imgs_sizes = torch.tensor([[32, 32], [32, 32], [16, 16]], device='cuda')
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 8, 9], dtype=torch.int32, device='cuda')
    )

    local_t, local_imgs_sizes, local_params, has_padding, num_padded_ranks, local_num_frames = (
        split_to_context_parallel_ranks_dynamic_res(
            global_t, imgs_sizes, packed_seq_params, patch_dim=PATCH_DIM, fp8_enabled=True
        )
    )

    assert has_padding is True
    if cp_rank == 0:
        assert local_t.shape == torch.Size([1, 16, PATCH_HIDDEN])
        torch.testing.assert_close(local_t[0, :4, 0], torch.arange(4, device='cuda').float())
        assert torch.count_nonzero(local_t[0, 4:, :]) == 0
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4, 16], dtype=torch.int32, device='cuda')
        )
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [16, 192]], device='cuda')
        )
    else:
        assert local_t.shape == torch.Size([1, 16, PATCH_HIDDEN])
        torch.testing.assert_close(local_t[0, :5, 0], torch.arange(4, 9, device='cuda').float())
        assert torch.count_nonzero(local_t[0, 5:, :]) == 0
        torch.testing.assert_close(
            local_params.cu_seqlens_q, torch.tensor([0, 4, 5, 16], dtype=torch.int32, device='cuda')
        )
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [16, 16], [16, 176]], device='cuda')
        )


def test_dynamic_res_split_retries_until_every_rank_gets_a_media(cp_four_ranks):
    cp_rank = get_context_parallel_rank()
    global_t = torch.zeros(1, 8, PATCH_HIDDEN).cuda()
    global_t[0, :, :] = torch.arange(8, device='cuda').unsqueeze(1).float()
    imgs_sizes = torch.tensor([[32, 32], [32, 32]], device='cuda')
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32, device='cuda')
    )

    local_t, local_imgs_sizes, local_params, has_padding, num_padded_ranks, local_num_frames = (
        split_to_context_parallel_ranks_dynamic_res(
            global_t,
            imgs_sizes,
            packed_seq_params,
            patch_dim=PATCH_DIM,
            num_frames=torch.tensor([2], device='cuda'),
            temporal_patch_size=2,
        )
    )

    assert has_padding is False
    assert num_padded_ranks == 3
    if cp_rank == 0:
        assert local_t.shape == torch.Size([1, 8, PATCH_HIDDEN])
        torch.testing.assert_close(local_t[0, :, 0], torch.arange(8, device='cuda').float())
        torch.testing.assert_close(
            local_imgs_sizes, torch.tensor([[32, 32], [32, 32]], device='cuda')
        )
        assert local_num_frames.tolist() == [2]
    else:
        assert cp_rank in (1, 2, 3)
        expected_rows = 1 if cp_rank in (1, 2) else 2
        assert local_t.shape == torch.Size([1, expected_rows, PATCH_HIDDEN])
        assert torch.count_nonzero(local_t) == 0
        assert local_num_frames.tolist() == [1] * expected_rows
        if cp_rank == 1:
            torch.testing.assert_close(local_imgs_sizes, torch.tensor([[16, 16]], device='cuda'))
        elif cp_rank == 2:
            torch.testing.assert_close(local_imgs_sizes, torch.tensor([[16, 16]], device='cuda'))
        else:
            torch.testing.assert_close(
                local_imgs_sizes, torch.tensor([[16, 16], [16, 16]], device='cuda')
            )
