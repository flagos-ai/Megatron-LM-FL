# Copyright (c) 2026, BAAI. All rights reserved.
"""Heterogeneous auxiliary-loss reduction: moving tracked values to CPU for cpu:gloo pipeline groups and reducing across the pipeline group list."""

import pytest
import torch
import torch.distributed as dist

import megatron.core.parallel_state
import megatron.plugin.utils as plugin_utils
from megatron.core.transformer.moe import moe_logging, moe_utils


def test_reduce_aux_losses_tracker_across_ranks_hetero_moves_values_to_cpu_for_cpu_gloo(
    monkeypatch,
):
    monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)
    reductions = []

    # monkeypatch torch.distributed so the single-process result is the identity reduction
    def _identity_all_reduce(tensor, group=None, op=None):
        reductions.append((tensor.clone(), group))
        return tensor

    monkeypatch.setattr(dist, 'all_reduce', _identity_all_reduce)
    monkeypatch.setattr(dist, 'get_backend', lambda group: 'cpu:gloo')
    cpu_pp_group = object()
    monkeypatch.setattr(
        megatron.core.parallel_state, 'get_pipeline_model_parallel_group', lambda: [cpu_pp_group]
    )
    moe_utils.save_to_aux_losses_tracker(
        'load_balancing_loss', torch.tensor(2.0), layer_number=1, num_layers=2
    )

    plugin_utils.reduce_aux_losses_tracker_across_ranks_hetero(track_names=['load_balancing_loss'])

    ((reduced, group),) = reductions
    assert group is cpu_pp_group
    assert reduced.tolist() == [2.0, 0.0]
    assert reduced.device.type == 'cpu'
