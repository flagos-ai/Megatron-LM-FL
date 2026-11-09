# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.training import training
from tests.unit_tests.test_utilities import Utils


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
@pytest.mark.parametrize('per_token', [False, True])
def test_training_log_passes_indexer_groups_and_layer_counts(legacy, per_token):
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
        calculate_per_token_loss=per_token,
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
    assert logger.call_args.kwargs['calculate_per_token_loss'] is per_token


@pytest.mark.parametrize('per_token', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_indexer_logging_pp2_cp2_empty_stage(per_token, legacy):
    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip('requires a multiple of four ranks')
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2, context_parallel_size=2)
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['pp', 'cp', 'dp'])
    logger = DSAIndexerLossLoggingHelper
    logger.tracker = {}
    if legacy:
        # Legacy callers must provide saved group metadata on every PP stage.
        logger.tracker.update(
            reduce_group=groups.cp if per_token else None,
            avg_group=None if per_token else groups.cp,
        )
    try:
        # A second iteration checks the cached tracker shape and retained groups.
        for _ in range(2):
            if groups.pp.rank() == 0:
                local_loss = torch.tensor(float(1 + 2 * groups.cp.rank()), device='cuda')
                logger.save_loss_to_tracker(
                    local_loss,
                    layer_number=1,
                    num_layers=1,
                    reduce_group=groups.cp if per_token else None,
                    avg_group=None if per_token else groups.cp,
                )
            logged = {}
            logger.track_indexer_metrics(
                loss_scale=1.0,
                iteration=1,
                writer=None,
                pg_collection=groups,
                total_loss_dict=logged,
                num_layers=1,
                num_indexer_layers=1,
                preserve_groups=True,
                calculate_per_token_loss=None if legacy else per_token,
            )
            torch.testing.assert_close(
                logged['indexer loss'], torch.tensor(4.0 if per_token else 2.0, device='cuda')
            )
    finally:
        logger.tracker = {}
        Utils.destroy_model_parallel()
