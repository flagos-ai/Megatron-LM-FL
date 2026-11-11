# Copyright (c) 2026, BAAI. All rights reserved.
"""BlendedDataset: weight rejection and normalisation, single-dataset warning, deferred indices."""

import logging

import numpy
import pytest

from megatron.core.datasets.blended_dataset import BlendedDataset
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


def _dataset(tmp_path, index, offset=0):
    path = tmp_path / f"data_{index}.npy"
    numpy.save(path, numpy.arange(20, dtype=numpy.int64).reshape(10, 2) + offset)
    return _NumpyDataset(
        numpy.load(path),
        str(path),
        numpy.arange(10, dtype=numpy.int32),
        10,
        Split.train,
        BlendedMegatronDatasetConfig(random_seed=7, sequence_length=2),
    )


def _config(tmp_path):
    return BlendedMegatronDatasetConfig(
        random_seed=7, sequence_length=2, path_to_cache=str(tmp_path), defer_npy_index_mmap=True
    )


def test_blended_dataset_rejects_fractional_float_weights_when_the_size_is_none(tmp_path):
    datasets = [_dataset(tmp_path, 0), _dataset(tmp_path, 1)]

    with pytest.raises(AssertionError):
        BlendedDataset(datasets, [1.5, 2.5], None, _config(tmp_path))


def test_blended_dataset_normalizes_integer_weights_when_a_size_is_given(tmp_path):
    datasets = [_dataset(tmp_path, 0), _dataset(tmp_path, 1)]

    blended = BlendedDataset(datasets, [1, 3], 4, _config(tmp_path))

    assert blended.weights == [0.25, 0.75]
    assert len(blended) == 4


def test_blended_dataset_warns_when_blending_a_single_dataset(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("RANK", "0")
    datasets = [_dataset(tmp_path, 0)]

    with caplog.at_level(logging.WARNING):
        blended = BlendedDataset(datasets, [1], 2, _config(tmp_path))

    assert blended.weights == [1.0]
    assert len(blended) == 2
    assert "Building a BlendedDataset for a single MegatronDataset" in caplog.text


def test_blended_dataset_loads_deferred_indices_on_first_access(tmp_path):
    config = _config(tmp_path)
    initial = BlendedDataset(
        [_dataset(tmp_path, 0, 0), _dataset(tmp_path, 1, 100)], [1, 1], 2, config
    )
    numpy.save(initial.path_to_dataset_index, numpy.array([1, 0], dtype=numpy.int16))
    numpy.save(initial.path_to_dataset_sample_index, numpy.array([2, 3], dtype=numpy.int64))

    loaded = BlendedDataset(
        [_dataset(tmp_path, 0, 0), _dataset(tmp_path, 1, 100)], [1, 1], 2, config
    )

    assert loaded.dataset_index is None
    first = loaded[0]
    second = loaded[1]
    assert first["dataset_id"] == numpy.int16(1)
    numpy.testing.assert_array_equal(first["text"], [104, 105])
    assert second["dataset_id"] == numpy.int16(0)
    numpy.testing.assert_array_equal(second["text"], [6, 7])
    numpy.testing.assert_array_equal(loaded.dataset_index, numpy.array([1, 0], dtype=numpy.int16))
