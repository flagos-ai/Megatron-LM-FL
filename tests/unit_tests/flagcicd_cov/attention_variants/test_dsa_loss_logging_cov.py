# Copyright (c) 2026, BAAI. All rights reserved.
"""DSAIndexerLossLoggingHelper: tracker save/clean lifecycle, cross-group loss reduction, and averaged indexer-loss reporting to TensorBoard and wandb."""

import pytest
import torch

import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


class _RecordingWandb:
    """Stand-in for the wandb network client; records what would be sent."""

    def __init__(self):
        self.records = []

    def log(self, data, step=None):
        self.records.append((dict(data), step))


class TestDSAIndexerLossLoggingHelper:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def isolate_tracker(self, monkeypatch):
        monkeypatch.setattr(dsa_module.DSAIndexerLossLoggingHelper, 'tracker', {})

    def test_save_loss_to_tracker_skips_when_layer_number_is_none(self):
        loss = torch.tensor(3.0, device=torch.cuda.current_device())
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss, layer_number=None, num_layers=4
        )
        assert dsa_module.DSAIndexerLossLoggingHelper.tracker == {}

    def test_clean_loss_in_tracker_zeroes_values_and_clears_groups(self):
        tracker = dsa_module.DSAIndexerLossLoggingHelper.tracker
        tracker['values'] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tracker['reduce_group'] = self.pg.tp
        tracker['avg_group'] = self.pg.tp
        dsa_module.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        assert torch.equal(tracker['values'], torch.zeros(4))
        assert tracker['reduce_group'] is None
        assert tracker['avg_group'] is None

    def test_clean_loss_in_tracker_without_values_only_clears_groups(self):
        tracker = dsa_module.DSAIndexerLossLoggingHelper.tracker
        tracker['reduce_group'] = self.pg.tp
        tracker['avg_group'] = self.pg.tp
        dsa_module.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        assert tracker == {'reduce_group': None, 'avg_group': None}

    def test_reduce_loss_in_tracker_returns_when_empty_and_num_layers_is_none(self):
        dsa_module.DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        assert dsa_module.DSAIndexerLossLoggingHelper.tracker == {}

    def test_reduce_loss_in_tracker_reduces_across_pipeline_and_optional_groups(self):
        device = torch.cuda.current_device()
        tracker = dsa_module.DSAIndexerLossLoggingHelper.tracker
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(3.0, device=device),
            layer_number=1,
            num_layers=4,
            reduce_group=self.pg.tp,
            avg_group=self.pg.tp,
        )
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(5.0, device=device),
            layer_number=3,
            num_layers=4,
            reduce_group=self.pg.tp,
            avg_group=self.pg.tp,
        )
        dsa_module.DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(num_layers=4)
        expected = torch.tensor([3.0, 0.0, 5.0, 0.0], device=tracker['values'].device)
        assert torch.equal(tracker['values'], expected)

    def test_track_indexer_metrics_returns_when_tracker_stays_empty(self):
        dsa_module.DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=2.0, iteration=1, writer=None, num_layers=None
        )
        assert dsa_module.DSAIndexerLossLoggingHelper.tracker == {}

    @pytest.mark.parametrize(
        'csa_compress_ratios,expected_avg',
        [
            pytest.param(None, 4.0, id='legacy_all_layers'),
            pytest.param([0, 4, 128, 4], 8.0, id='indexer_layers_only'),
        ],
    )
    def test_track_indexer_metrics_logs_averaged_indexer_loss(
        self, tmp_path, csa_compress_ratios, expected_avg
    ):
        from torch.utils.tensorboard import SummaryWriter

        device = torch.cuda.current_device()
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(3.0, device=device), layer_number=1, num_layers=4
        )
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(5.0, device=device), layer_number=3, num_layers=4
        )
        writer = SummaryWriter(log_dir=str(tmp_path))
        wandb = _RecordingWandb()
        total_loss_dict = {}
        dsa_module.DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=2.0,
            iteration=7,
            writer=writer,
            wandb_writer=wandb,
            total_loss_dict=total_loss_dict,
            num_layers=4,
            csa_compress_ratios=csa_compress_ratios,
        )
        writer.flush()

        tracker = dsa_module.DSAIndexerLossLoggingHelper.tracker
        assert torch.equal(tracker['values'], torch.zeros(4, device=tracker['values'].device))
        assert tracker['reduce_group'] is None
        assert tracker['avg_group'] is None
        torch.testing.assert_close(
            total_loss_dict['indexer loss'].cpu(), torch.tensor(expected_avg), rtol=1e-6, atol=1e-6
        )
        assert wandb.records[0][1] == 7
        torch.testing.assert_close(
            wandb.records[0][0]['indexer loss'].cpu(),
            torch.tensor(expected_avg),
            rtol=1e-6,
            atol=1e-6,
        )

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(tmp_path))
        accumulator.Reload()
        scalars = accumulator.Scalars('indexer loss')
        assert scalars[0].step == 7
        assert scalars[0].value == pytest.approx(expected_avg, abs=1e-6)

    def test_track_indexer_metrics_accumulates_into_existing_loss_dict(self):
        device = torch.cuda.current_device()
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(3.0, device=device), layer_number=1, num_layers=4
        )
        dsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(5.0, device=device), layer_number=3, num_layers=4
        )
        total_loss_dict = {'indexer loss': torch.tensor(1.0, device=device)}
        dsa_module.DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=2.0,
            iteration=1,
            writer=None,
            wandb_writer=None,
            total_loss_dict=total_loss_dict,
            num_layers=4,
        )
        torch.testing.assert_close(
            total_loss_dict['indexer loss'].cpu(), torch.tensor(5.0), rtol=1e-6, atol=1e-6
        )
