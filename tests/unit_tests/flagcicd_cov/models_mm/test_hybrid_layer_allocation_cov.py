# Copyright (c) 2026, BAAI. All rights reserved.
"""Hybrid layer patterns: ratio-to-pattern generation, unified pattern parsing, layer counting, and pipeline segment selection per rank."""
import pytest

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_layer_counts,
    get_layer_maps_from_layer_type_list,
    parse_hybrid_pattern,
    pattern_from_ratios,
    select_pipeline_segment,
    validate_segment_layers,
)
from megatron.core.parallel_state import get_pipeline_model_parallel_group
from tests.unit_tests.test_utilities import Utils


def test_pattern_from_ratios_places_attention_evenly_for_third_ratio():
    pattern = pattern_from_ratios(6, attention_ratio=1 / 3)
    assert pattern == "M*MM*M"
    assert pattern.count(Symbols.ATTENTION) == 2
    assert pattern.count(Symbols.MAMBA) == 4


def test_pattern_from_ratios_replaces_mamba_with_mlp_not_attention():
    pattern = pattern_from_ratios(8, attention_ratio=0.25, mlp_ratio=0.25)
    assert pattern == "MM*-M*M-"
    assert pattern.count(Symbols.ATTENTION) == 2
    assert pattern.count(Symbols.MLP) == 2
    assert pattern.count(Symbols.MAMBA) == 4


def test_pattern_from_ratios_interleaves_attention_across_full_sequence():
    assert pattern_from_ratios(7, attention_ratio=3 / 7) == "M*M*M*M"
    assert pattern_from_ratios(9, attention_ratio=1.0) == "*********"


@pytest.mark.parametrize(
    "num_layers,attention_ratio,mlp_ratio",
    [(0, 0.0, 0.0), (4, -0.1, 0.0), (4, 0.5, -0.1), (4, 0.6, 0.6)],
    ids=["zero_layers", "negative_attention", "negative_mlp", "ratios_exceed_one"],
)
def test_pattern_from_ratios_rejects_invalid_ratios(num_layers, attention_ratio, mlp_ratio):
    with pytest.raises(AssertionError):
        pattern_from_ratios(num_layers, attention_ratio, mlp_ratio)


def test_get_hybrid_layer_counts_multiplies_mtp_layers_by_depth():
    counts = get_hybrid_layer_counts("MM/-M/-M")
    assert counts == {'*': 0, 'G': 0, 'D': 0, 'M': 4, '-': 2, 'E': 0}


def test_parse_hybrid_pattern_returns_empty_components_for_none():
    parsed = parse_hybrid_pattern(None)
    assert parsed.main_pattern is None
    assert parsed.mtp_pattern is None
    assert parsed.mtp_num_depths == 0


def test_parse_hybrid_pattern_ignores_trailing_separator():
    parsed = parse_hybrid_pattern("M*/")
    assert parsed.main_pattern == "M*"
    assert parsed.mtp_pattern is None
    assert parsed.mtp_num_depths == 0


def test_parse_hybrid_pattern_rejects_mismatched_mtp_patterns():
    with pytest.raises(ValueError, match="All MTP patterns must be identical"):
        parse_hybrid_pattern("M*/MM/M")


def test_parse_hybrid_pattern_rejects_unknown_symbols():
    with pytest.raises(ValueError, match="'\\?' is not a valid layer symbol"):
        parse_hybrid_pattern("M?")


def test_parse_hybrid_pattern_rejects_attention_and_dsa_mix():
    with pytest.raises(ValueError, match="Not supported to have both Attention and MLA/DSA"):
        parse_hybrid_pattern("*D")


def test_validate_segment_layers_rejects_pipe_symbol():
    with pytest.raises(ValueError, match="one of"):
        validate_segment_layers("M|M")


def test_validate_segment_layers_rejects_attention_and_dsa_mix():
    with pytest.raises(ValueError, match="Not supported to have both Attention and MLA/DSA"):
        validate_segment_layers("*D*")


def test_select_pipeline_segment_rejects_first_stage_layers_with_pipes():
    with pytest.raises(ValueError, match="Cannot specify"):
        select_pipeline_segment("M|M", pp_group=None, vp_stage=None, first_stage_layers=1)


def test_select_pipeline_segment_rejects_out_of_range_vp_stage():
    with pytest.raises(ValueError, match="out of range"):
        select_pipeline_segment("M*|M*", pp_group=None, vp_stage=5)


def test_get_layer_maps_from_layer_type_list_indexes_layers_per_type():
    maps = get_layer_maps_from_layer_type_list(["M", "*", "M", "-", "*"])
    assert set(maps.keys()) == Symbols.VALID_LAYERS
    assert maps[Symbols.MAMBA] == {0: 0, 2: 1}
    assert maps[Symbols.ATTENTION] == {1: 0, 4: 1}
    assert maps[Symbols.MLP] == {3: 0}
    assert maps[Symbols.DS_ATTENTION] == {}
    assert maps[Symbols.GDN] == {}
    assert maps[Symbols.MOE] == {}


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestSelectPipelineSegmentLegacySplit:
    """Legacy no-pipe fallback: pattern without '|' is sliced at runtime for pp>1."""

    def test_even_split_slices_layers_per_rank(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()
        pp_rank = pp_group.rank()

        layers, offset = select_pipeline_segment("M-M*M*M", pp_group=pp_group, vp_stage=None)

        assert layers == (["M", "-", "M"] if pp_rank == 0 else ["*", "M", "*"])
        assert offset == pp_rank * 3

    def test_uneven_first_stage_layers_shift_middle_ranks(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()
        pp_rank = pp_group.rank()

        layers, offset = select_pipeline_segment(
            "M*M*M*", pp_group=pp_group, vp_stage=None, first_stage_layers=3
        )

        if pp_rank == 0:
            assert layers == ["M", "*", "M"]
            assert offset == 0
        else:
            assert layers == ["*"]
            assert offset == 3

    def test_uneven_last_stage_layers_gives_tail_to_last_rank(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()
        pp_rank = pp_group.rank()

        layers, offset = select_pipeline_segment(
            "M*M*M*", pp_group=pp_group, vp_stage=None, last_stage_layers=2
        )

        if pp_rank == 0:
            assert layers == ["M", "*"]
            assert offset == 0
        else:
            assert layers == ["*", "M"]
            assert offset == 2

    def test_first_and_last_stage_layers_leave_empty_middle(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()
        pp_rank = pp_group.rank()

        layers, offset = select_pipeline_segment(
            "M*M*M*", pp_group=pp_group, vp_stage=None, first_stage_layers=2, last_stage_layers=2
        )

        assert layers == (["M", "*"] if pp_rank == 0 else ["*", "M"])
        assert offset == pp_rank * 2

    def test_vpp_without_pipes_raises(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()

        with pytest.raises(ValueError, match="Virtual pipeline parallelism"):
            select_pipeline_segment("M*M*", pp_group=pp_group, vp_stage=0)

    def test_layers_not_divisible_by_pp_size_raises(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()

        with pytest.raises(ValueError, match="evenly divisible"):
            select_pipeline_segment("M*M*M", pp_group=pp_group, vp_stage=None)

    def test_middle_layers_not_divisible_by_middle_stages_raises(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)
        pp_group = get_pipeline_model_parallel_group()

        with pytest.raises(ValueError, match="Middle layers"):
            select_pipeline_segment(
                "M" * 10,
                pp_group=pp_group,
                vp_stage=None,
                first_stage_layers=2,
                last_stage_layers=1,
            )

    def test_segment_count_not_divisible_by_pp_size_raises(self, model_parallel):
        model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        pp_group = get_pipeline_model_parallel_group()

        with pytest.raises(ValueError, match="evenly divisible"):
            select_pipeline_segment("M|M|M", pp_group=pp_group, vp_stage=None)
