# Copyright (c) 2026, BAAI. All rights reserved.
"""Object-storage helpers: scheme parsing, cache paths, existence checks, rank-gated caching."""

import os
from types import SimpleNamespace

import pytest

from megatron.core.datasets import object_storage_utils
from megatron.core.datasets.object_storage_utils import (
    ObjectStorageConfig,
    _remove_msc_prefix,
    _remove_s3_prefix,
    _s3_download_file,
    _s3_object_exists,
    cache_index_file,
    dataset_exists,
    get_index_cache_path,
    get_object_storage_access,
    is_object_storage_path,
    parse_s3_path,
)


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeS3Client:
    def __init__(self, root, downloads=None):
        self.root = root
        self.downloads = downloads if downloads is not None else []
        self.head_calls = []

    def _local(self, bucket, key):
        return os.path.join(self.root, bucket, key)

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(self._local(bucket, key), "rb") as source, open(filename, "wb") as target:
            target.write(source.read())

    def head_object(self, bucket, key):
        self.head_calls.append((bucket, key))
        if not os.path.exists(self._local(bucket, key)):
            raise _ClientError("404")
        return {"ContentLength": 16}


class _ForbiddenS3Client:
    def head_object(self, bucket, key):
        raise _ClientError("403")


def _patch_s3(monkeypatch, client):
    monkeypatch.setattr(
        object_storage_utils, "boto3", SimpleNamespace(client=lambda _name: client), raising=False
    )
    monkeypatch.setattr(
        object_storage_utils, "exceptions", SimpleNamespace(ClientError=_ClientError), raising=False
    )


def _patch_msc(monkeypatch, package):
    monkeypatch.setattr(
        object_storage_utils,
        "MultiStorageClientFeature",
        SimpleNamespace(import_package=lambda: package),
    )


def test_prefix_removal_strips_the_s3_scheme():
    assert _remove_s3_prefix("s3://bucket/key/data.bin") == "bucket/key/data.bin"


def test_prefix_removal_strips_the_msc_scheme():
    assert _remove_msc_prefix("msc://profile/key/data.bin") == "profile/key/data.bin"


def test_prefix_removal_leaves_local_paths_untouched():
    assert _remove_s3_prefix("/local/data.bin") == "/local/data.bin"
    assert _remove_msc_prefix("/local/data.bin") == "/local/data.bin"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("s3://bucket/data", True, id="s3"),
        pytest.param("msc://profile/data", True, id="msc"),
        pytest.param("/local/path/data", False, id="local"),
    ],
)
def test_is_object_storage_path_detects_the_supported_schemes(path, expected):
    assert is_object_storage_path(path) is expected


@pytest.mark.parametrize(
    ("path", "bucket", "key"),
    [
        pytest.param("s3://my-bucket/a/b.idx", "my-bucket", "a/b.idx", id="with-key"),
        pytest.param("s3://bare-bucket", "bare-bucket", "", id="bucket-only"),
    ],
)
def test_parse_s3_path_splits_bucket_from_key(path, bucket, key):
    assert parse_s3_path(path) == (bucket, key)


def test_parse_s3_path_rejects_paths_without_the_s3_scheme():
    with pytest.raises(AssertionError):
        parse_s3_path("msc://profile/a/b.idx")


def test_get_object_storage_access_maps_schemes_to_reader_kinds():
    assert get_object_storage_access("s3://bucket/data") == "s3"
    assert get_object_storage_access("msc://profile/data") == "msc"


@pytest.mark.parametrize(
    ("idx_path", "suffix"),
    [
        pytest.param("s3://bucket/nested/data.idx", "bucket/nested/data.idx", id="s3"),
        pytest.param("msc://profile/nested/data.idx", "profile/nested/data.idx", id="msc"),
    ],
)
def test_get_index_cache_path_appends_the_scheme_free_path_to_the_cache_dir(idx_path, suffix):
    config = ObjectStorageConfig("cache_dir")

    assert get_index_cache_path(idx_path, config) == os.path.join("cache_dir", suffix)


def test_get_index_cache_path_rejects_local_paths():
    with pytest.raises(ValueError, match="Invalid path: /local/data.idx"):
        get_index_cache_path("/local/data.idx", ObjectStorageConfig("cache_dir"))


def test_s3_download_file_parses_the_path_and_writes_the_object(tmp_path):
    source = tmp_path / "bucket" / "prefix" / "file.idx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"index-bytes")
    downloads = []
    destination = tmp_path / "cache" / "bucket" / "prefix" / "file.idx"

    _s3_download_file(
        _FakeS3Client(str(tmp_path), downloads), "s3://bucket/prefix/file.idx", str(destination)
    )

    assert downloads == [("bucket", "prefix/file.idx", str(destination))]
    assert destination.read_bytes() == b"index-bytes"


def test_s3_object_exists_returns_true_when_head_object_succeeds(tmp_path):
    (tmp_path / "bucket").mkdir()
    (tmp_path / "bucket" / "present.idx").write_bytes(b"idx")
    client = _FakeS3Client(str(tmp_path))

    assert _s3_object_exists(client, "s3://bucket/present.idx") is True
    assert client.head_calls == [("bucket", "present.idx")]


def test_s3_object_exists_reraises_client_errors_other_than_404(monkeypatch):
    _patch_s3(monkeypatch, _ForbiddenS3Client())

    with pytest.raises(_ClientError) as caught:
        _s3_object_exists(_ForbiddenS3Client(), "s3://bucket/forbidden.idx")

    assert caught.value.response["Error"]["Code"] == "403"


def test_dataset_exists_heads_both_the_idx_and_the_bin_on_s3(tmp_path, monkeypatch):
    (tmp_path / "bucket").mkdir()
    (tmp_path / "bucket" / "data.idx").write_bytes(b"idx")
    (tmp_path / "bucket" / "data.bin").write_bytes(b"bin")
    client = _FakeS3Client(str(tmp_path))
    _patch_s3(monkeypatch, client)

    assert (
        dataset_exists("s3://bucket/data", "s3://bucket/data.idx", "s3://bucket/data.bin") is True
    )
    assert client.head_calls == [("bucket", "data.idx"), ("bucket", "data.bin")]


def test_dataset_exists_reports_missing_msc_objects(monkeypatch, tmp_path):
    present = tmp_path / "present.idx"
    present.write_bytes(b"idx")
    seen = []

    class _FakeMsc:
        def exists(self, path):
            seen.append(path)
            return os.path.exists(path)

    _patch_msc(monkeypatch, _FakeMsc())

    assert dataset_exists("msc://p/data", str(present), str(tmp_path / "missing.bin")) is False
    assert seen == [str(present), str(tmp_path / "missing.bin")]


def test_dataset_exists_rejects_local_prefixes():
    with pytest.raises(ValueError, match="Invalid path: /local/data"):
        dataset_exists("/local/data", "/local/data.idx", "/local/data.bin")


def test_cache_index_file_caches_the_s3_index_bytes_when_distributed_training_is_disabled(
    monkeypatch, tmp_path
):
    remote = tmp_path / "bucket" / "file.idx"
    remote.parent.mkdir(parents=True)
    remote.write_bytes(b"downloaded-index")
    downloads = []
    _patch_s3(monkeypatch, _FakeS3Client(str(tmp_path), downloads))
    destination = tmp_path / "cache" / "bucket" / "file.idx"

    cache_index_file("s3://bucket/file.idx", str(destination))

    assert destination.read_bytes() == b"downloaded-index"
    assert downloads == [("bucket", "file.idx", str(destination))]


def test_cache_index_file_skips_the_download_for_nonzero_ranks(monkeypatch, tmp_path):
    destination = tmp_path / "cache" / "file.idx"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"already-cached")
    downloads = []
    _patch_s3(monkeypatch, _FakeS3Client(str(tmp_path), downloads))
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 3)

    cache_index_file("s3://bucket/file.idx", str(destination))

    assert downloads == []
    assert destination.read_bytes() == b"already-cached"


def test_cache_index_file_caches_the_msc_index_bytes_when_distributed_training_is_disabled(
    monkeypatch, tmp_path
):
    class _FakeMsc:
        def __init__(self, objects):
            self.objects = objects
            self.calls = []

        def download_file(self, remote_path, local_path):
            self.calls.append((remote_path, local_path))
            with open(local_path, "wb") as target:
                target.write(self.objects[remote_path])

    fake_msc = _FakeMsc({"msc://profile/file.idx": b"msc-index"})
    _patch_msc(monkeypatch, fake_msc)
    destination = tmp_path / "cache" / "file.idx"
    destination.parent.mkdir(parents=True)

    cache_index_file("msc://profile/file.idx", str(destination))

    assert destination.read_bytes() == b"msc-index"
    assert fake_msc.calls == [("msc://profile/file.idx", str(destination))]


def test_cache_index_file_rejects_unknown_schemes(tmp_path):
    with pytest.raises(ValueError, match="Invalid path: ftp://host/file.idx"):
        cache_index_file("ftp://host/file.idx", str(tmp_path / "file.idx"))
