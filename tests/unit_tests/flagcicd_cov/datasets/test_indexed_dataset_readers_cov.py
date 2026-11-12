# Copyright (c) 2026, BAAI. All rights reserved.
"""IndexedDataset bin readers: retries, chunked S3/MSC reads, object-storage-backed opening."""

import os
from types import SimpleNamespace

import numpy
import pytest
import torch

from megatron.core.datasets import indexed_dataset as indexed_module
from megatron.core.datasets import object_storage_utils
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    ObjectStorageConfig,
    _FileBinReader,
    _MMapBinReader,
    _MultiStorageClientBinReader,
    _S3BinReader,
)


def _build_dataset(tmp_path, name, dtype=numpy.uint16, multimodal=False):
    prefix = str(tmp_path / name)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype, multimodal=multimodal)
    if multimodal:
        builder.add_item(torch.tensor([1, 2], dtype=torch.int64), mode=0)
        builder.end_document()
        builder.add_document(torch.tensor([3, 4, 5], dtype=torch.int64), [1, 2], modes=[0, 0])
    else:
        builder.add_item(torch.tensor([1, 2], dtype=torch.int64))
        builder.end_document()
        builder.add_document(torch.tensor([3, 4, 5], dtype=torch.int64), [1, 2])
    builder.finalize(prefix + ".idx")
    return prefix


class _Range:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size


class _FakeMscStorageClient:
    def __init__(self):
        self.reads = []

    def read(self, path, byte_range):
        self.reads.append((path, byte_range.offset, byte_range.size))
        with open(path, "rb") as stream:
            stream.seek(byte_range.offset)
            return stream.read(byte_range.size)


class _FakeMsc:
    def __init__(self, tmp_path):
        self.root = str(tmp_path)
        self.types = SimpleNamespace(Range=_Range)
        self.client = _FakeMscStorageClient()

    def open(self, path, mode="r", **kwargs):
        return open(path, mode, **kwargs)

    def resolve_storage_client(self, path):
        return self.client, os.path.join(self.root, path.removeprefix("msc://profile/"))

    def exists(self, path):
        return os.path.exists(os.path.join(self.root, path.removeprefix("msc://profile/")))

    def download_file(self, remote_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        source = os.path.join(self.root, remote_path.removeprefix("msc://profile/"))
        with open(source, "rb") as stream, open(local_path, "wb") as target:
            target.write(stream.read())


class _FakeMscFeature:
    def __init__(self, package):
        self._package = package

    def is_enabled(self):
        return True

    def import_package(self):
        return self._package


class _FlakyMscFeature:
    def __init__(self, package, failures):
        self._inner = _FakeMscFeature(package)
        self._failures = failures
        self.attempts = []

    def is_enabled(self):
        return True

    def import_package(self):
        if self._failures > 0:
            self._failures -= 1
            self.attempts.append(False)
            raise ConnectionError("storage unavailable")
        self.attempts.append(True)
        return self._inner.import_package()


def _install_fake_msc(monkeypatch, tmp_path):
    package = _FakeMsc(tmp_path)
    feature = _FakeMscFeature(package)
    monkeypatch.setattr(indexed_module, "MultiStorageClientFeature", feature)
    monkeypatch.setattr(object_storage_utils, "MultiStorageClientFeature", feature)
    return package


class _FakeS3Client:
    def __init__(self, root):
        self.root = root
        self.ranges = []
        self.closed = False

    def _local(self, bucket, key):
        return os.path.join(self.root, bucket, key)

    def get_object(self, Bucket, Key, Range):
        self.ranges.append(Range)
        start, end = map(int, Range.split("=")[1].split("-"))
        with open(self._local(Bucket, Key), "rb") as stream:
            stream.seek(start)
            data = stream.read(end - start + 1)
        return {"Body": SimpleNamespace(read=lambda: data)}

    def download_file(self, bucket, key, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(self._local(bucket, key), "rb") as source, open(filename, "wb") as target:
            target.write(source.read())

    def head_object(self, bucket, key):
        assert os.path.exists(self._local(bucket, key))
        return {"ContentLength": 8}

    def close(self):
        self.closed = True


def _install_fake_s3(monkeypatch, client):
    fake_boto3 = SimpleNamespace(client=lambda _name: client)
    monkeypatch.setattr(indexed_module, "boto3", fake_boto3, raising=False)
    monkeypatch.setattr(object_storage_utils, "boto3", fake_boto3, raising=False)


def test_file_bin_reader_reraises_persistent_read_errors_without_retrying(tmp_path):
    reader = _FileBinReader(str(tmp_path / "missing.bin"), num_max_retries=0)

    with pytest.raises(FileNotFoundError, match="missing.bin"):
        reader.read(numpy.uint16, 2, 0)


def test_file_bin_reader_retries_a_failed_read_and_succeeds(tmp_path, monkeypatch):
    prefix = _build_dataset(tmp_path, "flaky")
    flaky = _FlakyMscFeature(_FakeMsc(tmp_path), failures=1)
    monkeypatch.setattr(indexed_module, "MultiStorageClientFeature", flaky)
    reader = _FileBinReader(prefix + ".bin", num_max_retries=2, sleep_duration_start=0)

    sequence = reader.read(numpy.uint16, 2, 0)

    assert sequence.tolist() == [1, 2]
    assert flaky.attempts == [False, True]
    numpy.testing.assert_array_equal(sequence, numpy.array([1, 2]))


def test_s3_bin_reader_serves_cached_byte_ranges_without_refetching(tmp_path, monkeypatch):
    _build_dataset(tmp_path, os.path.join("bucket", "s3_sample"))
    client = _FakeS3Client(str(tmp_path))
    _install_fake_s3(monkeypatch, client)

    reader = _S3BinReader(
        "s3://bucket/s3_sample.bin", ObjectStorageConfig(str(tmp_path), bin_chunk_nbytes=8)
    )

    numpy.testing.assert_array_equal(reader.read(numpy.uint16, 2, 0), numpy.array([1, 2]))
    numpy.testing.assert_array_equal(reader.read(numpy.uint16, 1, 2), numpy.array([2]))
    assert client.ranges == ["bytes=0-7"]

    numpy.testing.assert_array_equal(reader.read(numpy.uint16, 2, 6), numpy.array([4, 5]))
    assert client.ranges == ["bytes=0-7", "bytes=0-9"]


def test_s3_bin_reader_requires_a_positive_chunk_size(tmp_path, monkeypatch):
    _install_fake_s3(monkeypatch, _FakeS3Client(str(tmp_path)))

    with pytest.raises(AssertionError):
        _S3BinReader("s3://bucket/data.bin", ObjectStorageConfig(str(tmp_path), bin_chunk_nbytes=0))


def test_s3_bin_reader_closes_its_client_on_deletion(tmp_path, monkeypatch):
    client = _FakeS3Client(str(tmp_path))
    _install_fake_s3(monkeypatch, client)

    reader = _S3BinReader("s3://bucket/data.bin", ObjectStorageConfig(str(tmp_path)))
    assert client.closed is False
    del reader

    assert client.closed is True


def test_msc_bin_reader_reads_the_resolved_byte_range(tmp_path, monkeypatch):
    package = _install_fake_msc(monkeypatch, tmp_path)
    prefix = _build_dataset(tmp_path, "msc_sample")

    reader = _MultiStorageClientBinReader(
        "msc://profile/msc_sample.bin", ObjectStorageConfig(str(tmp_path))
    )

    numpy.testing.assert_array_equal(reader.read(numpy.uint16, 2, 0), numpy.array([1, 2]))
    assert package.client.reads == [(prefix + ".bin", 0, 4)]


def test_msc_backed_builders_and_readers_exchange_real_files(tmp_path, monkeypatch):
    _install_fake_msc(monkeypatch, tmp_path)
    prefix = _build_dataset(tmp_path, "msc_dataset", multimodal=True)

    dataset = IndexedDataset(
        "msc://profile/msc_dataset",
        multimodal=True,
        mmap=False,
        object_storage_config=ObjectStorageConfig(str(tmp_path)),
    )
    assert isinstance(dataset.bin_reader, _MultiStorageClientBinReader)
    sequence, mode = dataset[0]
    numpy.testing.assert_array_equal(sequence, numpy.array([1, 2], dtype=numpy.uint16))
    assert int(mode) == 0

    mmap_dataset = IndexedDataset(prefix, mmap=True)
    assert isinstance(mmap_dataset.bin_reader, _MMapBinReader)
    numpy.testing.assert_array_equal(mmap_dataset[0], numpy.array([1, 2], dtype=numpy.uint16))

    file_reader = _FileBinReader(prefix + ".bin")
    numpy.testing.assert_array_equal(file_reader.read(numpy.uint16, 2, 0), numpy.array([1, 2]))
    assert IndexedDataset.exists("msc://profile/msc_dataset")


def test_s3_backed_indexed_dataset_caches_the_index_and_streams_the_bin(tmp_path, monkeypatch):
    prefix = _build_dataset(tmp_path, os.path.join("bucket", "s3_dataset"))
    client = _FakeS3Client(str(tmp_path))
    _install_fake_s3(monkeypatch, client)

    dataset = IndexedDataset(
        "s3://bucket/s3_dataset",
        mmap=False,
        object_storage_config=ObjectStorageConfig(str(tmp_path / "cache")),
    )

    assert isinstance(dataset.bin_reader, _S3BinReader)
    assert len(dataset) == 3
    numpy.testing.assert_array_equal(dataset[1], numpy.array([3], dtype=numpy.uint16))
    assert (tmp_path / "cache" / "bucket" / "s3_dataset.idx").is_file()
    assert IndexedDataset.exists("s3://bucket/s3_dataset")
    assert client.ranges == [f"bytes=0-{256 * 1024 * 1024 - 1}"]
