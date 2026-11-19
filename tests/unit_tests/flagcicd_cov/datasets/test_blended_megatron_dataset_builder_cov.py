# Copyright (c) 2026, BAAI. All rights reserved.
"""BlendedMegatronDatasetBuilder: split construction from blends, per-rank gating."""

import numpy
import pytest
import torch

from megatron.core.datasets import blended_megatron_dataset_builder as builder_module
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split


class _NumpyDataset(MegatronDataset):
    def __init__(self, dataset, dataset_path, indices, num_samples, index_split, config):
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        if self.num_samples is None:
            self.num_samples = len(self.indices)
        self.sample_index = numpy.arange(self.num_samples, dtype=numpy.int64) % len(self.indices)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset):
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        return numpy.load(dataset_path)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        return {"text": self.dataset[self.sample_index[idx]]}


class _UnwritableDataset(_NumpyDataset):
    def __init__(self, dataset_path, *args):
        open(dataset_path, "w")


def _write_npy(tmp_path, name, rows=50, offset=0):
    path = tmp_path / f"{name}.npy"
    numpy.save(path, numpy.arange(rows, dtype=numpy.int64) + offset)
    return str(path)


def _config(**overrides):
    kwargs = dict(random_seed=11, sequence_length=2)
    kwargs.update(overrides)
    return BlendedMegatronDatasetConfig(**kwargs)


def _simulate_distributed(monkeypatch, rank=0):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)


def test_builder_wraps_mock_generator_failures_with_the_dataset_class_name(tmp_path):
    config = _config(path_to_cache=str(tmp_path))
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, 10, 10], lambda: True, config)
    assert config.mock

    with pytest.raises(Exception, match="_NumpyDataset failed to build as a mock") as caught:
        builder.build()

    assert isinstance(caught.value.__cause__, TypeError)


def test_builder_builds_each_split_from_a_single_prefix_blend(tmp_path):
    prefix = _write_npy(tmp_path, "single")
    config = _config(blend=([prefix], None), split="500,500,0")
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, 10, None], lambda: True, config)

    datasets = builder.build()

    assert len(datasets) == 3
    assert datasets[0].dataset_path == prefix
    assert datasets[0].num_samples == 10
    assert datasets[1].num_samples == 10
    assert datasets[2] is None
    assert len(datasets[0]) == 10 and len(datasets[1]) == 10
    numpy.testing.assert_array_equal(datasets[0].indices, numpy.arange(0, 25, dtype=numpy.int32))
    numpy.testing.assert_array_equal(datasets[1].indices, numpy.arange(25, 50, dtype=numpy.int32))
    numpy.testing.assert_array_equal(datasets[0].sample_index, numpy.arange(10, dtype=numpy.int64))
    assert datasets[0][4]["text"].item() == 4


@pytest.mark.parametrize(
    ("full_validation", "sizes", "expected_lengths"),
    [
        pytest.param(True, [None, None, None], [50, 50], id="full-validation"),
        pytest.param(False, [None, 10, None], [10, 10], id="sized-validation"),
    ],
)
def test_builder_builds_one_dataset_per_validation_prefix(
    tmp_path, full_validation, sizes, expected_lengths
):
    first = _write_npy(tmp_path, "valid_a")
    second = _write_npy(tmp_path, "valid_b", offset=100)
    config = _config(
        blend_per_split=[None, ([first, second], None), None],
        multiple_validation_sets=True,
        full_validation=full_validation,
    )
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, sizes, lambda: True, config)

    datasets = builder.build()

    assert datasets[0] is None and datasets[2] is None
    assert [len(dataset) for dataset in datasets[1]] == expected_lengths
    assert [dataset.dataset_path for dataset in datasets[1]] == [first, second]
    numpy.testing.assert_array_equal(datasets[1][0][3]["text"], numpy.array(3))
    numpy.testing.assert_array_equal(datasets[1][1][3]["text"], numpy.array(103))


def test_builder_returns_none_splits_for_ranks_that_do_not_build(tmp_path, monkeypatch):
    prefix = _write_npy(tmp_path, "unused")
    config = _config(blend_per_split=[([prefix], None), None, None])
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, 0, 0], lambda: False, config)
    _simulate_distributed(monkeypatch)

    datasets = builder.build()

    assert datasets == [None, None, None]


def test_builder_treats_unbuilt_mid_level_datasets_as_zero_weights_in_blend_per_split(
    tmp_path, monkeypatch
):
    first = _write_npy(tmp_path, "first")
    second = _write_npy(tmp_path, "second", offset=100)
    config = _config(blend_per_split=[([first, second], None), None, None])
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, 0, 0], lambda: False, config)
    _simulate_distributed(monkeypatch)

    datasets = builder.build()

    assert datasets == [None, None, None]


def test_builder_scales_worker_threads_and_blends_on_rank_zero(tmp_path, monkeypatch):
    first = _write_npy(tmp_path, "first")
    second = _write_npy(tmp_path, "second", offset=100)
    config = _config(
        blend=([first, second], None),
        split="500,500,0",
        path_to_cache=str(tmp_path),
        defer_npy_index_mmap=True,
        num_dataset_builder_threads=2,
    )
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, 10, None], lambda: True, config)
    _simulate_distributed(monkeypatch)
    monkeypatch.setattr(
        builder_module,
        "cur_platform",
        type("_Platform", (), {"device_count": staticmethod(lambda: 3)})(),
    )

    datasets = builder.build()

    assert isinstance(datasets[0], BlendedDataset) and len(datasets[0]) == 10
    assert datasets[0].weights == [0.5, 0.5]
    assert isinstance(datasets[1], BlendedDataset) and len(datasets[1]) == 10
    assert datasets[2] is None


def test_builder_propagates_dataset_worker_failures(tmp_path):
    present = _write_npy(tmp_path, "present")
    missing = str(tmp_path / "missing.npy")
    config = _config(blend=([present, missing], None), split="1,0,0")
    builder = BlendedMegatronDatasetBuilder(_NumpyDataset, [10, None, None], lambda: True, config)

    with pytest.raises(FileNotFoundError, match="missing.npy"):
        builder.build()


def test_build_generic_dataset_wraps_oserror_from_rank_zero_construction(tmp_path, monkeypatch):
    _simulate_distributed(monkeypatch)

    with pytest.raises(Exception, match="Failed to write dataset materials") as caught:
        BlendedMegatronDatasetBuilder.build_generic_dataset(
            _UnwritableDataset, lambda: True, True, str(tmp_path / "no_such_dir" / "cache.txt")
        )

    assert isinstance(caught.value.__cause__, OSError)


def test_build_generic_dataset_constructs_the_dataset_without_distribution(tmp_path):
    prefix = _write_npy(tmp_path, "direct")
    dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
        _NumpyDataset,
        lambda: True,
        True,
        numpy.load(prefix),
        prefix,
        numpy.arange(50, dtype=numpy.int32),
        7,
        Split.train,
        _config(),
    )

    assert dataset.dataset_path == prefix
    assert dataset.num_samples == 7
    assert len(dataset) == 7
    assert dataset.sample_index.tolist() == list(range(7))
    assert dataset[2]["text"].item() == 2
