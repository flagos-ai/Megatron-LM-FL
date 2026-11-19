# Copyright (c) 2026, BAAI. All rights reserved.
"""MaskedWordPiece dataset: config validation, cached sample indices, masked-LM prediction spans."""

import hashlib
import json
import logging
import os
from collections import OrderedDict

import numpy
import pytest
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.masked_dataset import (
    MaskedWordPieceDataset,
    MaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.utils import Split


class _TinyTokenizer:
    cls = 1
    sep = 2
    mask = 3
    pad = 0
    inv_vocab = {
        0: "[PAD]",
        1: "[CLS]",
        2: "[SEP]",
        3: "<mask>",
        10: "apple",
        11: "##s",
        12: "banana",
        13: "##ing",
        14: "pear",
    }
    special_tokens_dict = {"pad_token": 0, "cls_token": 1, "sep_token": 2}
    unique_identifiers = {"class": "TinyTokenizer"}


class _CountingDataset(MaskedWordPieceDataset):
    def __init__(
        self, indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
    ):
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.sample_index = self._build_sample_index(config.sequence_length - 1, 1)

    def __getitem__(self, idx):
        return {"idx": idx}

    def _get_token_mask(self, numpy_random_state):
        return super()._get_token_mask(numpy_random_state)


def _config(**overrides):
    kwargs = dict(
        random_seed=42,
        sequence_length=16,
        tokenizer=_TinyTokenizer(),
        masking_probability=0.2,
        short_sequence_probability=0.1,
        masking_max_ngram=3,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=False,
    )
    kwargs.update(overrides)
    return MaskedWordPieceDatasetConfig(**kwargs)


def _predictions(**config_overrides):
    dataset = _CountingDataset.__new__(_CountingDataset)
    dataset.config = _config(**config_overrides)
    return dataset


def _write_indexed_dataset(tmp_path, name, token_sequences):
    prefix = str(tmp_path / name)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    for tokens in token_sequences:
        builder.add_item(torch.tensor(tokens, dtype=torch.int64))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    return prefix


def _prime_sample_index_cache(cache_dir, config, dataset_path, num_samples, sample_index):
    unique_identifiers = OrderedDict()
    unique_identifiers["class"] = "_CountingDataset"
    unique_identifiers["dataset_path"] = dataset_path
    unique_identifiers["num_samples"] = num_samples
    unique_identifiers["index_split"] = Split.train.name
    for attr in _CountingDataset._key_config_attributes():
        unique_identifiers[attr] = getattr(config, attr)
    description = json.dumps(
        unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
    )
    description_hash = hashlib.md5(description.encode("utf-8"), usedforsecurity=False).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{description_hash}-_CountingDataset-description.txt"), "w"
    ) as writer:
        writer.write(description)
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-_CountingDataset-sample_index.npy"),
        sample_index,
    )


@pytest.mark.parametrize(
    "config_overrides",
    [
        pytest.param({"tokenizer": None}, id="missing-tokenizer"),
        pytest.param({"masking_probability": 0.0}, id="zero-probability"),
        pytest.param({"masking_probability": 1.0}, id="unit-probability"),
        pytest.param({"short_sequence_probability": -0.1}, id="negative-short-probability"),
        pytest.param({"short_sequence_probability": 1.5}, id="short-probability-above-one"),
        pytest.param({"masking_max_ngram": 0}, id="zero-ngram"),
        pytest.param(
            {"masking_use_geometric_distribution": True, "masking_do_permutation": True},
            id="geometric-with-permutation",
        ),
    ],
)
def test_masked_dataset_config_rejects_invalid_parameter_combinations(config_overrides):
    with pytest.raises(AssertionError):
        _config(**config_overrides)


def test_masked_dataset_config_warns_when_geometric_distribution_overrides_the_default(
    monkeypatch, caplog
):
    monkeypatch.setenv("RANK", "0")

    with caplog.at_level(logging.WARNING):
        config = _config(masking_use_geometric_distribution=True, masking_use_longer_ngrams=True)

    assert config.masking_use_geometric_distribution is True
    assert "The use of a geometric distribution overrides the default distribution" in caplog.text


def test_masked_dataset_numel_counts_the_low_level_documents():
    low_level_dataset = type("_LowLevel", (), {"document_indices": numpy.array([0, 5, 12])})()

    assert MaskedWordPieceDataset.numel_low_level_dataset(low_level_dataset) == 2


def test_masked_dataset_build_low_level_dataset_opens_the_indexed_dataset(tmp_path):
    prefix = _write_indexed_dataset(tmp_path, "masked_data", [[1, 2, 3], [4, 5]])

    low_level_dataset = MaskedWordPieceDataset.build_low_level_dataset(
        prefix, _config(path_to_cache=str(tmp_path))
    )

    assert low_level_dataset.path_prefix == prefix
    assert len(low_level_dataset) == 2
    assert low_level_dataset[1].tolist() == [4, 5]
    assert low_level_dataset.document_indices.tolist() == [0, 1, 2]
    numpy.testing.assert_array_equal(low_level_dataset[1], numpy.array([4, 5], dtype=numpy.int32))


def test_masked_dataset_key_config_attributes_list_the_masking_knobs():
    assert MaskedWordPieceDataset._key_config_attributes() == [
        "random_seed",
        "sequence_length",
        "split",
        "split_matrix",
        "tokenizer",
        "masking_probability",
        "short_sequence_probability",
        "masking_max_ngram",
        "masking_do_full_word",
        "masking_do_permutation",
        "masking_use_longer_ngrams",
        "masking_use_geometric_distribution",
    ]


def test_masked_dataset_get_token_mask_has_no_default_replacement():
    dataset = _predictions()

    assert dataset._get_token_mask(numpy.random.RandomState(0)) is None


def test_masked_dataset_length_counts_the_cached_sample_index(tmp_path):
    config = _config(path_to_cache=str(tmp_path))
    prefix = _write_indexed_dataset(tmp_path, "cached_data", [[1, 2, 3], [4, 5]])
    _prime_sample_index_cache(str(tmp_path), config, "cached_path", None, numpy.array([[0, 1, 8]]))

    dataset = _CountingDataset(
        IndexedDataset(prefix, mmap=False),
        "cached_path",
        numpy.arange(2, dtype=numpy.int64),
        None,
        Split.train,
        config,
    )

    assert len(dataset) == 1
    assert dataset.sample_index.tolist() == [[0, 1, 8]]


def test_masked_dataset_falls_back_to_the_dataset_prefix_for_the_cache_directory(tmp_path):
    config = _config()
    prefix = _write_indexed_dataset(tmp_path, "default_cache_data", [[1, 2, 3], [4, 5]])
    cache_dir = os.path.join(prefix, "cache", "_CountingDataset_indices")
    _prime_sample_index_cache(
        cache_dir,
        config,
        prefix,
        5,
        numpy.array([[0, 1, 8], [1, 2, 8], [0, 1, 8]], dtype=numpy.int32),
    )

    dataset = _CountingDataset(
        IndexedDataset(prefix, mmap=False),
        prefix,
        numpy.arange(2, dtype=numpy.int64),
        5,
        Split.train,
        config,
    )

    assert len(dataset) == 3


def test_masked_predictions_mask_every_candidate_token():
    dataset = _predictions(masking_probability=0.8, masking_max_ngram=1, masking_do_full_word=False)

    masked_token_ids, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 14, 2], 7, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 11, 12, 13, 14, 2]
    assert positions == [1, 2, 3, 4, 5]
    assert labels == [10, 11, 12, 13, 14]
    assert boundaries == [1, 1, 0, 1, 0, 1, 1]
    assert spans == [([1], [10]), ([2], [11]), ([3], [12]), ([4], [13]), ([5], [14])]


def test_masked_predictions_group_subword_tokens_into_full_word_spans():
    dataset = _predictions(masking_probability=0.8, masking_max_ngram=1, masking_do_full_word=True)

    masked_token_ids, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 14, 2], 7, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 11, 12, 13, 14, 2]
    assert positions == [1, 2, 3, 4, 5]
    assert labels == [10, 11, 12, 13, 14]
    assert boundaries == [1, 1, 0, 1, 0, 1, 1]
    assert spans == [([1, 2], [10, 11]), ([3, 4], [12, 13]), ([5], [14])]


def test_masked_predictions_cover_all_candidates_with_a_single_long_ngram():
    dataset = _predictions(masking_probability=0.8, masking_max_ngram=3, masking_do_full_word=True)

    masked_token_ids, positions, labels, _, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 14, 2], 7, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 11, 12, 13, 14, 2]
    assert positions == [1, 2, 3, 4, 5]
    assert labels == [10, 11, 12, 13, 14]
    assert spans == [([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])]


def test_masked_predictions_sample_ngram_lengths_from_the_geometric_distribution():
    dataset = _predictions(
        masking_probability=0.8, masking_use_geometric_distribution=True, masking_do_full_word=False
    )

    masked_token_ids, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [1, 10, 12, 13, 14, 2], 6, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 12, 13, 14, 2]
    assert positions == [1, 2, 3, 4]
    assert labels == [10, 12, 13, 14]
    assert boundaries == [1, 1, 1, 0, 1, 1]
    assert sorted(index for indices, _ in spans for index in indices) == [1, 2, 3, 4]
    span_labels = sorted(label for _, labels in spans for label in labels)
    assert span_labels == [10, 12, 13, 14]


def test_masked_predictions_prefer_longer_ngrams_when_requested():
    dataset = _predictions(
        masking_probability=0.8,
        masking_max_ngram=3,
        masking_use_longer_ngrams=True,
        masking_do_full_word=True,
    )

    masked_token_ids, positions, labels, _, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 14, 2], 7, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 11, 12, 13, 14, 2]
    assert positions == [1, 2, 3, 4, 5]
    assert labels == [10, 11, 12, 13, 14]
    assert spans == [([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])]


def test_masked_predictions_leave_special_tokens_unmasked_when_no_candidates_exist():
    dataset = _predictions(masking_probability=0.8)

    masked_token_ids, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [1, 2], 2, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 2]
    assert positions == []
    assert labels == []
    assert boundaries == [1, 1]
    assert spans == []


def test_masked_predictions_skip_candidates_whose_ngram_exceeds_the_budget():
    dataset = _predictions(masking_probability=0.25, masking_max_ngram=1, masking_do_full_word=True)

    masked_token_ids, positions, labels, _, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 2], 5, numpy.random.RandomState(3)
    )

    assert masked_token_ids == [1, 10, 11, 12, 2]
    assert positions == [3]
    assert labels == [12]
    assert spans == [([3], [12])]


def test_masked_predictions_permute_tokens_that_the_masking_phase_left_untouched():
    dataset = _predictions(
        masking_probability=0.4,
        masking_max_ngram=1,
        masking_do_full_word=False,
        masking_do_permutation=True,
    )

    masked_token_ids, positions, labels, _, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 14, 2], 7, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 12, 11, 10, 13, 14, 2]
    assert positions == [1, 2, 3, 5]
    assert labels == [10, 11, 12, 14]
    assert spans == [([2], [11]), ([5], [14])]


def test_masked_predictions_stop_permutating_once_the_swapping_budget_is_spent():
    dataset = _predictions(
        masking_probability=0.2,
        masking_max_ngram=1,
        masking_do_full_word=False,
        masking_do_permutation=True,
    )

    masked_token_ids, positions, labels, boundaries, spans = dataset._create_masked_lm_predictions(
        [1, 10, 11, 12, 13, 2], 5, numpy.random.RandomState(42)
    )

    assert masked_token_ids == [1, 10, 11, 12, 13, 2]
    assert positions == [2, 4]
    assert labels == [11, 13]
    assert boundaries == [1, 1, 0, 1, 0, 1]
    assert spans == [([2], [11])]
