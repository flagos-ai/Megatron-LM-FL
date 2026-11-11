# Copyright (c) 2026, BAAI. All rights reserved.
"""GPT dataset: document/shuffle index building, cached index loading, deferred-mmap lengths."""

import hashlib
import json
import os
import shutil
from collections import OrderedDict

import numpy
import pytest
import torch

from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _build_document_index,
    _build_shuffle_index,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split


class _TinyTokenizer:
    vocab_size = 24
    eod = 9
    pad = 0
    eos = 2
    special_tokens_dict = {"pad_token": 0}
    unique_identifiers = {"class": "TinyGPTTokenizer"}


class _FakeS3Client:
    def __init__(self, root):
        self._root = root

    def download_file(self, bucket, key, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copyfile(f"{self._root}/{bucket}/{key}", filename)

    def get_object(self, Bucket, Key, Range):
        start, end = map(int, Range.split("=")[1].split("-"))
        with open(f"{self._root}/{Bucket}/{Key}", "rb") as stream:
            stream.seek(start)
            payload = stream.read(end - start + 1)
        return {"Body": type("_Body", (), {"read": lambda _, p=payload: p})()}

    def close(self):
        pass


class _FakeBoto3:
    def __init__(self, root):
        self._root = root

    def client(self, service):
        return _FakeS3Client(self._root)


def _config(**overrides):
    kwargs = dict(
        random_seed=5,
        sequence_length=4,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_TinyTokenizer(),
    )
    kwargs.update(overrides)
    return GPTDatasetConfig(**kwargs)


def _write_indexed_dataset(tmp_path, name, token_sequences):
    prefix = str(tmp_path / name)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    for tokens in token_sequences:
        builder.add_item(torch.tensor(tokens, dtype=torch.int64))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    return prefix


def test_build_document_index_separates_the_final_epoch_from_the_global_shuffle():
    rng = numpy.random.RandomState(42)
    documents = numpy.array([10, 20, 30])

    document_index = _build_document_index(
        documents, num_epochs=3, numpy_random_state=rng, separate_final_epoch=True
    )

    assert document_index.dtype == numpy.int32
    assert len(document_index) == 9
    assert sorted(document_index[:6].tolist()) == [10, 10, 20, 20, 30, 30]
    assert sorted(document_index[6:].tolist()) == [10, 20, 30]


def test_build_shuffle_index_concatenates_two_shuffled_ranges():
    rng = numpy.random.RandomState(42)

    shuffle_index = _build_shuffle_index(num_samples=3, total_size=7, numpy_random_state=rng)

    assert shuffle_index.dtype == numpy.uint32
    assert sorted(shuffle_index[:3].tolist()) == [0, 1, 2]
    assert sorted(shuffle_index[3:].tolist()) == [3, 4, 5, 6]


@pytest.mark.parametrize(
    ("index_split", "drop_last_partial_validation_sequence", "expected_length"),
    [
        pytest.param(Split.train, True, 4, id="train-drops-partial"),
        pytest.param(Split.valid, False, 5, id="valid-keeps-partial"),
    ],
)
def test_gpt_dataset_length_handles_partial_sequences_when_mmap_is_deferred(
    tmp_path, index_split, drop_last_partial_validation_sequence, expected_length
):
    prefix = _write_indexed_dataset(tmp_path, "lengths", [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]])
    cache = tmp_path / "cache"
    config = _config(
        blend=([prefix], None),
        split="500,500,0",
        path_to_cache=str(cache),
        defer_npy_index_mmap=True,
        drop_last_partial_validation_sequence=drop_last_partial_validation_sequence,
    )

    dataset = GPTDataset(
        IndexedDataset(prefix, mmap=False),
        prefix,
        numpy.arange(3, dtype=numpy.int32),
        3,
        index_split,
        config,
    )

    assert dataset.dataset.sequence_lengths.tolist() == [3, 3, 4]
    assert len(dataset) == expected_length
    assert dataset.path_to_sample_index == os.path.join(
        str(cache),
        f"{dataset.unique_description_hash}-GPTDataset-{index_split.name}-sample_index.npy",
    )


def test_gpt_dataset_loads_indices_from_the_default_cache_directory(tmp_path):
    prefix = _write_indexed_dataset(tmp_path, "cached", [[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    config = _config(blend=([prefix], None), split="500,500,0")
    documents = numpy.array([2, 0, 1], dtype=numpy.int32)
    samples = numpy.array([[0, 3], [3, 6], [6, 9]], dtype=numpy.int64)
    shuffle = numpy.array([1, 0, 2], dtype=numpy.uint32)

    unique_identifiers = OrderedDict()
    unique_identifiers["class"] = "GPTDataset"
    unique_identifiers["dataset_path"] = prefix
    unique_identifiers["num_samples"] = None
    unique_identifiers["index_split"] = Split.train.name
    for attr in GPTDataset._key_config_attributes():
        unique_identifiers[attr] = getattr(config, attr)
    description = json.dumps(
        unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
    )
    description_hash = hashlib.md5(description.encode("utf-8"), usedforsecurity=False).hexdigest()
    cache_dir = os.path.join(prefix, "cache", "GPTDataset_indices")
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{description_hash}-GPTDataset-train-description.txt"), "w"
    ) as writer:
        writer.write(description)
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-GPTDataset-train-document_index.npy"),
        documents,
    )
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-GPTDataset-train-sample_index.npy"), samples
    )
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-GPTDataset-train-shuffle_index.npy"), shuffle
    )

    dataset = GPTDataset(
        IndexedDataset(prefix, mmap=False),
        prefix,
        numpy.arange(3, dtype=numpy.int32),
        None,
        Split.train,
        config,
    )

    assert dataset.unique_description_hash == description_hash
    numpy.testing.assert_array_equal(dataset.document_index, documents)
    numpy.testing.assert_array_equal(dataset.sample_index, samples)
    numpy.testing.assert_array_equal(dataset.shuffle_index, shuffle)


def test_gpt_dataset_build_low_level_dataset_requires_a_cache_path_for_object_storage():
    with pytest.raises(AssertionError):
        GPTDataset.build_low_level_dataset("s3://bucket/gpt_data", _config())


def test_gpt_dataset_build_low_level_dataset_streams_the_index_from_object_storage(
    tmp_path, monkeypatch
):
    prefix = _write_indexed_dataset(
        tmp_path, os.path.join("bucket", "gpt_data"), [[1, 2, 3], [4, 5]]
    )
    fake_boto3 = _FakeBoto3(str(tmp_path))
    monkeypatch.setattr("megatron.core.datasets.indexed_dataset.boto3", fake_boto3, raising=False)
    monkeypatch.setattr(
        "megatron.core.datasets.object_storage_utils.boto3", fake_boto3, raising=False
    )
    config = _config(object_storage_cache_path=str(tmp_path / "idx_cache"), mmap_bin_files=False)

    dataset = GPTDataset.build_low_level_dataset("s3://bucket/gpt_data", config)

    assert isinstance(dataset, IndexedDataset)
    assert dataset.path_prefix == "s3://bucket/gpt_data"
    assert len(dataset) == 2
    numpy.testing.assert_array_equal(dataset[1], numpy.array([4, 5], dtype=numpy.int32))
    assert (tmp_path / "idx_cache" / "bucket" / "gpt_data.idx").is_file()
