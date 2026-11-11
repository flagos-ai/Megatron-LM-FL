# Copyright (c) 2026, BAAI. All rights reserved.
"""MoE metric tracking: per-layer accumulation, PP/DP reduction of recorded values, percentile
aggregation, console formatting and writer output."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_logging
from megatron.core.transformer.moe.moe_logging import (
    MoEMetricsTracker,
    destroy_moe_metrics_tracker,
    get_moe_metrics_tracker,
    set_moe_metrics_tracker,
)
from tests.unit_tests.test_utilities import Utils


class _RecordingWriter:
    """TensorBoard / W&B boundary double that records add_scalar and log calls."""

    def __init__(self):
        self.scalars = []
        self.wandb_logs = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def log(self, payload, step):
        self.wandb_logs.append((payload, step))


@pytest.fixture
def tracker():
    """A fresh tracker whose metrics do not leak into other tests."""
    return MoEMetricsTracker()


@pytest.fixture
def model_parallel(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv('DISTRIBUTED_BACKEND', 'gloo')

    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


class TestGlobalTracker:

    def test_set_and_destroy_moe_metrics_tracker_swaps_the_global_instance(self, monkeypatch):
        monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)
        replacement = MoEMetricsTracker()

        set_moe_metrics_tracker(replacement)
        assert get_moe_metrics_tracker() is replacement

        destroy_moe_metrics_tracker()
        recreated = get_moe_metrics_tracker()
        assert recreated is not replacement
        assert isinstance(recreated, MoEMetricsTracker)
        assert recreated.metrics == {}

    def test_get_moe_metrics_tracker_creates_tracker_lazily(self, monkeypatch):
        monkeypatch.setattr(moe_logging, '_MOE_METRICS_TRACKER', None)

        assert get_moe_metrics_tracker() is get_moe_metrics_tracker()


class TestRecord:

    def test_record_accumulates_values_per_layer_and_stores_reduction_groups(self, tracker):
        tracker.record('load_balancing_loss', torch.tensor(1.5), layer_number=1, num_layers=3)
        pp_group = object()
        dp_group = object()
        tracker.record(
            'load_balancing_loss',
            torch.tensor(2.5),
            layer_number=3,
            num_layers=3,
            reduce_group=pp_group,
            avg_group=dp_group,
            needs_dp_avg=False,
        )

        entry = tracker.metrics['load_balancing_loss']
        assert torch.equal(entry.values, torch.tensor([1.5, 0.0, 2.5]))
        assert entry.values.dtype == torch.float32
        assert entry.reduce_group is pp_group
        assert entry.avg_group is dp_group
        assert entry.needs_dp_avg is False


class TestResolveNames:

    @pytest.mark.parametrize(
        ('track_names', 'expected'),
        [(None, ['alpha', 'beta']), ('alpha', ['alpha']), (['beta', 'alpha'], ['beta', 'alpha'])],
        ids=['none-uses-all', 'string-wraps', 'list-passes-through'],
    )
    def test_resolve_names_normalizes_track_names_argument(self, tracker, track_names, expected):
        tracker.record('alpha', torch.tensor(1.0), layer_number=1, num_layers=2)
        tracker.record('beta', torch.tensor(2.0), layer_number=1, num_layers=2)

        assert tracker._resolve_names(track_names) == expected


class TestCountMoeLayers:

    @pytest.mark.parametrize(
        ('num_layers', 'moe_layer_freq', 'mtp_num_layers', 'expected'),
        [
            (12, None, None, 12),
            (6, 2, None, 3),
            ([0, 1, 1, 1], [0, 1, 1, 1], None, 3),
            (12, None, 1, 13),
        ],
        ids=['no-freq', 'int-freq-counts-divisible-layers', 'list-freq-sums', 'mtp-adds-layers'],
    )
    def test_count_moe_layers_computes_effective_layer_count(
        self, num_layers, moe_layer_freq, mtp_num_layers, expected
    ):
        assert (
            MoEMetricsTracker._count_moe_layers(num_layers, moe_layer_freq, mtp_num_layers)
            == expected
        )

    def test_count_moe_layers_rejects_invalid_frequency_type(self):
        with pytest.raises(ValueError, match='Invalid moe_layer_freq: dense'):
            MoEMetricsTracker._count_moe_layers(4, 'dense', None)


class TestAggregate:

    def test_aggregate_averages_layers_and_reports_sums(self, tracker):
        tracker.record('load_balancing', torch.tensor(1.0), layer_number=1, num_layers=2)
        tracker.record('load_balancing', torch.tensor(3.0), layer_number=2, num_layers=2)

        scalars = tracker._aggregate(0.5, 2, ['load_balancing'])

        assert torch.equal(scalars['load_balancing'], torch.tensor(1.0))

    def test_aggregate_computes_percentiles_over_nonzero_layers(self, tracker):
        values = torch.tensor([1.0, 0.0, 3.0, 0.0])
        tracker.metrics['load_balancing'] = moe_logging.MetricEntry(values=values.clone())

        scalars = tracker._aggregate(
            1.0, 4, ['load_balancing'], percentiles={'load_balancing': [0.5, 1.0]}
        )

        assert scalars['load_balancing_p50'] == 2.0
        assert scalars['load_balancing_p100'] == 3.0
        assert torch.equal(scalars['load_balancing'], torch.tensor(1.0))

    def test_aggregate_skips_percentiles_when_all_layers_are_zero(self, tracker):
        tracker.metrics['load_balancing'] = moe_logging.MetricEntry(values=torch.zeros(4))

        scalars = tracker._aggregate(1.0, 4, ['load_balancing'], {'load_balancing': [0.5]})

        assert list(scalars) == ['load_balancing']
        assert scalars['load_balancing'] == 0.0

    def test_aggregate_skips_metric_names_that_were_never_recorded(self, tracker):
        assert tracker._aggregate(1.0, 4, ['missing']) == {}


class TestLoggers:

    def test_log_scalars_writes_each_metric_to_both_writers(self, tracker):
        writer = _RecordingWriter()

        tracker._log_scalars({'load_balancing': 1.5, 'z_loss': 2.25}, 7, writer, writer)

        assert writer.scalars == [('load_balancing', 1.5, 7), ('z_loss', 2.25, 7)]
        assert writer.wandb_logs == [({'load_balancing': 1.5}, 7), ({'z_loss': 2.25}, 7)]

    def test_log_per_layer_writes_every_layer_value(self, tracker):
        writer = _RecordingWriter()
        tracker.metrics['load_balancing'] = moe_logging.MetricEntry(values=torch.tensor([2.0, 4.0]))

        tracker._log_per_layer(0.5, ['load_balancing'], 3, writer, writer, None)

        assert writer.scalars == [
            ('moe/load_balancing_layer_0', 1.0, 3),
            ('moe/load_balancing_layer_1', 2.0, 3),
        ]
        assert writer.wandb_logs == [
            ({'moe/load_balancing_layer_0': 1.0}, 3),
            ({'moe/load_balancing_layer_1': 2.0}, 3),
        ]

    def test_log_per_layer_skips_zero_layers_of_sparse_metrics(self, tracker):
        writer = _RecordingWriter()
        tracker.metrics['load_balancing'] = moe_logging.MetricEntry(values=torch.tensor([2.0, 0.0]))

        tracker._log_per_layer(
            1.0, ['load_balancing'], 5, writer, writer, {'load_balancing': [0.5]}
        )

        assert writer.scalars == [('moe/load_balancing_layer_0', 2.0, 5)]
        assert writer.wandb_logs == [({'moe/load_balancing_layer_0': 2.0}, 5)]

    def test_log_per_layer_skips_unknown_metric_names(self, tracker):
        writer = _RecordingWriter()

        tracker._log_per_layer(1.0, ['missing'], 5, writer, writer, None)

        assert writer.scalars == []
        assert writer.wandb_logs == []


class TestFormatAndInit:

    def test_format_renders_scalars_with_two_decimals(self):
        formatted = MoEMetricsTracker._format({'load_balancing': 1.0, 'z_loss': 2.25})

        assert formatted == ' load_balancing: 1.00 | z_loss: 2.25 |'

    def test_ensure_initialized_creates_zero_entry_for_new_metric(self, tracker):
        tracker.ensure_initialized('load_balancing', 4)

        entry = tracker.metrics['load_balancing']
        assert torch.equal(entry.values, torch.zeros(4))
        assert entry.values.dtype == torch.float32

    def test_ensure_initialized_keeps_existing_entry_values(self, tracker):
        tracker.record('load_balancing', torch.tensor(3.0), layer_number=2, num_layers=4)

        tracker.ensure_initialized('load_balancing', 8)

        assert torch.equal(
            tracker.metrics['load_balancing'].values, torch.tensor([0.0, 3.0, 0.0, 0.0])
        )


def test_report_requires_num_layers_when_force_initialize_is_set(tracker):
    with pytest.raises(ValueError, match='num_layers must be provided when force_initialize=True.'):
        tracker.report(loss_scale=1.0, iteration=1, force_initialize=True)


class TestReportDistributed:
    """report() end-to-end; requires the 8-rank CI launcher to run collectives."""

    def test_report_averages_groups_and_routes_loss_metrics_into_total_loss_dict(
        self, model_parallel, tracker
    ):
        model_parallel()
        rank = torch.distributed.get_rank()
        pg = ProcessGroupCollection()
        pg.pp = parallel_state.get_pipeline_model_parallel_group()
        dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
        pg.dp = dp_group
        device = torch.cuda.current_device()
        tracker.record(
            'load_balancing_loss',
            torch.tensor(float(rank + 1), device=device),
            layer_number=1,
            num_layers=2,
            avg_group=dp_group,
            needs_dp_avg=False,
        )
        tracker.record(
            'load_balancing_loss',
            torch.tensor(2.0, device=device),
            layer_number=2,
            num_layers=2,
            avg_group=dp_group,
            needs_dp_avg=False,
        )
        tracker.record(
            'load_balancing', torch.tensor(4.0, device=device), layer_number=1, num_layers=2
        )
        total_loss_dict = {'load_balancing_loss': torch.tensor(2.0, device=device)}

        log_string = tracker.report(
            loss_scale=0.5,
            iteration=3,
            num_layers=2,
            track_names=['load_balancing_loss', 'load_balancing'],
            total_loss_dict=total_loss_dict,
            percentiles={'load_balancing': [0.5]},
            pg_collection=pg,
        )

        assert log_string == ' load_balancing: 1.00 | load_balancing_p50: 2.00 |'
        torch.testing.assert_close(
            total_loss_dict['load_balancing_loss'],
            torch.tensor(3.625, device=device),
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.equal(tracker.metrics['load_balancing'].values, torch.zeros(2, device=device))

    def test_report_sums_reduce_group_and_assigns_new_loss_totals(self, model_parallel, tracker):
        model_parallel()
        rank = torch.distributed.get_rank()
        pg = ProcessGroupCollection()
        pg.pp = parallel_state.get_pipeline_model_parallel_group()
        pg.dp = parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
        device = torch.cuda.current_device()
        tracker.record(
            'load_balancing_loss',
            torch.tensor(float(rank + 1), device=device),
            layer_number=1,
            num_layers=2,
            reduce_group=pg.dp,
            needs_dp_avg=False,
        )
        tracker.record(
            'load_balancing_loss',
            torch.tensor(2.0, device=device),
            layer_number=2,
            num_layers=2,
            reduce_group=pg.dp,
            needs_dp_avg=False,
        )
        total_loss_dict = {}

        log_string = tracker.report(
            loss_scale=0.5,
            iteration=4,
            num_layers=2,
            total_loss_dict=total_loss_dict,
            pg_collection=pg,
        )

        assert log_string == ''
        torch.testing.assert_close(
            total_loss_dict['load_balancing_loss'],
            torch.tensor(13.0, device=device),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_report_uses_global_parallel_groups_when_pg_collection_is_none(
        self, model_parallel, tracker
    ):
        model_parallel()
        device = torch.cuda.current_device()
        tracker.record(
            'load_balancing', torch.tensor(4.0, device=device), layer_number=1, num_layers=2
        )

        log_string = tracker.report(loss_scale=0.5, iteration=5, num_layers=2)

        assert log_string == ' load_balancing: 1.00 |'

    def test_report_force_initialize_precreates_entries_with_mtp_layers(
        self, model_parallel, tracker
    ):
        model_parallel()
        device = torch.cuda.current_device()

        log_string = tracker.report(
            loss_scale=1.0,
            iteration=6,
            force_initialize=True,
            track_names='load_balancing_loss',
            num_layers=4,
            mtp_num_layers=1,
        )

        assert log_string == ' load_balancing_loss: 0.00 |'
        assert torch.equal(
            tracker.metrics['load_balancing_loss'].values, torch.zeros(5, device=device)
        )

    def test_report_writes_per_layer_scalars_when_per_layer_logging_is_enabled(
        self, model_parallel, tracker
    ):
        model_parallel()
        device = torch.cuda.current_device()
        writer = _RecordingWriter()
        tracker.record(
            'load_balancing_loss', torch.tensor(4.0, device=device), layer_number=1, num_layers=2
        )
        tracker.record(
            'load_balancing_loss', torch.tensor(2.0, device=device), layer_number=2, num_layers=2
        )

        log_string = tracker.report(
            loss_scale=0.5,
            iteration=7,
            per_layer_logging=True,
            writer=writer,
            wandb_writer=writer,
            num_layers=2,
        )

        assert log_string == ' load_balancing_loss: 1.50 |'
        assert writer.scalars == [
            ('moe/load_balancing_loss_layer_0', 2.0, 7),
            ('moe/load_balancing_loss_layer_1', 1.0, 7),
        ]
        assert writer.wandb_logs == [
            ({'moe/load_balancing_loss_layer_0': 2.0}, 7),
            ({'moe/load_balancing_loss_layer_1': 1.0}, 7),
        ]

    def test_report_skips_track_names_that_were_never_recorded(self, model_parallel, tracker):
        model_parallel()

        log_string = tracker.report(
            loss_scale=1.0, iteration=8, num_layers=2, track_names=['missing']
        )

        assert log_string == ''
        assert tracker.metrics == {}
