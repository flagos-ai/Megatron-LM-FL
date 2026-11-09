# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest

from megatron.training import training


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({'indexer_type_rule': 'repeat(full, share)'}, (6, 3)),
        ({'csa_compress_ratios': [4, 128, 4, 0, 4, 128]}, (6, 3)),
        ({'csa_compress_ratios': [4, 128, 4, 0, 4, 128], 'csa_dense_mode': True}, (6, 0)),
    ],
)
def test_indexer_logging_layer_counts(overrides, expected):
    values = dict(
        num_layers=4,
        mtp_num_layers=2,
        experimental_attention_variant='dsa',
        indexer_types=None,
        indexer_type_rule=None,
        dsa_indexer_topk_freq=2,
        dsa_indexer_skip_topk_offset=0,
        mtp_use_repeated_layer=False,
        csa_dense_mode=False,
        hybrid_layer_pattern=None,
    )
    values.update(overrides)
    assert training._get_indexer_logging_layer_counts(SimpleNamespace(**values)) == expected


@pytest.mark.parametrize('legacy', [False, True])
def test_training_log_passes_indexer_groups_and_layer_counts(legacy):
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=2,
        seq_length=16,
        num_experts=None,
        mtp_num_layers=2,
        dsa_indexer_loss_coeff=1.0,
        num_layers=4,
        experimental_attention_variant='dsa',
        dsa_indexer_topk_freq=2,
        dsa_indexer_skip_topk_offset=0,
        indexer_types=None,
        indexer_type_rule=None,
        mtp_use_repeated_layer=False,
        cuda_graph_impl='none',
    )
    groups = object()

    class ReachedLogger(Exception):
        pass

    logger = create_autospec(
        training.DSAIndexerLossLoggingHelper.track_indexer_metrics, side_effect=ReachedLogger
    )
    with (
        patch.object(training, 'get_args', return_value=args),
        patch.object(training, 'get_timers'),
        patch.object(training, 'get_tensorboard_writer', return_value=None),
        patch.object(training, 'get_wandb_writer', return_value=None),
        patch.object(training, 'get_one_logger', return_value=None),
        patch.object(training, 'get_energy_monitor', return_value=None),
        patch.object(training, 'get_num_microbatches', return_value=1),
        patch.object(training.one_logger_utils, 'track_app_tag'),
        patch.object(training, 'reduce_max_stat_across_model_parallel_group', return_value=0.1),
        patch.object(training.MTPLossLoggingHelper, 'track_mtp_metrics'),
        patch.object(
            training.ProcessGroupCollection, 'use_mpu_process_groups', return_value=groups
        ),
        patch.object(training.DSAIndexerLossLoggingHelper, 'track_indexer_metrics', logger),
        pytest.raises(ReachedLogger),
    ):
        training.training_log(
            {},
            {},
            0.1,
            1,
            1,
            False,
            0,
            None,
            None,
            None,
            None,
            pg_collection=None if legacy else groups,
        )
    assert logger.call_args.kwargs['pg_collection'] is groups
    assert logger.call_args.kwargs['num_layers'] == 6
    assert logger.call_args.kwargs['num_indexer_layers'] == 3
    assert logger.call_args.kwargs['preserve_groups'] is False
