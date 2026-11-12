# Copyright (c) 2026, BAAI. All rights reserved.
"""Flexible pipeline layer layout stage counting, validation, and encoder rejection."""

import pytest

from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout


@pytest.mark.parametrize(
    ('layout_str', 'num_stages'),
    [('Et|tt|m|L', 4), ('(tt|)*2', 3), ('EttttL', 1)],
    ids=['pipelined', 'multiplied_group', 'single_stage'],
)
def test_get_num_stages_from_str_counts_every_stage_segment(layout_str, num_stages):
    assert PipelineParallelLayerLayout.get_num_stages_from_str(layout_str) == num_stages


@pytest.mark.parametrize(
    ('stage_layouts', 'num_layers', 'mtp_num_layers', 'mtp_standalone'),
    [
        ([[LayerType.embedding, LayerType.decoder, LayerType.loss]], 1, None, False),
        ([[LayerType.embedding, LayerType.decoder], [LayerType.mtp, LayerType.loss]], 1, 1, False),
        ([[LayerType.embedding, LayerType.decoder, LayerType.mtp], [LayerType.loss]], 1, 1, True),
    ],
    ids=['without_mtp', 'mtp_in_final_stage', 'mtp_before_final_stage'],
)
def test_validate_layer_layout_reports_whether_mtp_runs_standalone(
    stage_layouts, num_layers, mtp_num_layers, mtp_standalone
):
    layout = PipelineParallelLayerLayout(stage_layouts, len(stage_layouts))

    assert layout.validate_layer_layout(num_layers, mtp_num_layers) is mtp_standalone


def test_validate_layer_layout_rejects_encoder_layers():
    layout = PipelineParallelLayerLayout(
        [[LayerType.embedding, LayerType.encoder, LayerType.loss]], 1
    )

    with pytest.raises(
        NotImplementedError, match="Encoder layer is not supported for flexible pipeline layout"
    ):
        layout.validate_layer_layout(num_layers=0, mtp_num_layers=None)
