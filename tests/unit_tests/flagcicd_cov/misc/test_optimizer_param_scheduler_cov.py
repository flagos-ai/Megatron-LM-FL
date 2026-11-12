# Copyright (c) 2026, BAAI. All rights reserved.
"""Param-group override combination and rejection of unsupported weight-decay increment styles."""

import pytest
import torch

from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    combine_param_group_overrides,
)


def test_combine_param_group_overrides_skips_none_entries_and_merges_disjoint_keys():
    overrides = [None, {'lr': 0.1}, None, {'weight_decay': 0.01}, None]

    combined = combine_param_group_overrides(overrides)

    assert combined == {'lr': 0.1, 'weight_decay': 0.01}


def test_combine_param_group_overrides_rejects_conflicting_values_for_one_key():
    with pytest.raises(ValueError, match=r"Conflicting overrides for lr: 0.1 and 0.2"):
        combine_param_group_overrides([{'lr': 0.1}, {'lr': 0.2}])


def test_unknown_weight_decay_increment_style_is_rejected_at_construction():
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([param], lr=0.0)

    with pytest.raises(Exception, match="quadratic weight decay increment style is not supported"):
        OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=0.0,
            max_lr=1.0,
            min_lr=0.0,
            lr_warmup_steps=5,
            lr_decay_steps=50,
            lr_decay_style='linear',
            start_wd=0.0,
            end_wd=0.1,
            wd_incr_steps=50,
            wd_incr_style='quadratic',
        )
