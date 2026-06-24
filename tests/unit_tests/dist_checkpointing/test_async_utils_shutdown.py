# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for shutdown-safety of async checkpoint callers.

Verifies that ``PersistentAsyncCaller.__del__`` / ``close()`` does not raise
when the distributed process group has already been destroyed (the scenario
described in issue #3775).
"""

from unittest import mock
import importlib

from megatron.core.dist_checkpointing.strategies.async_utils import (
    PersistentAsyncCaller,
    TemporalAsyncCaller,
)
from megatron.plugin.decorators import get_override_method
from megatron.plugin.kunlunxin.dist_checkpointing.strategies.filesystem_async import (
    preload_tensors_kunlunxin,
)


class TestPersistentAsyncCallerShutdown:
    """Verify ``PersistentAsyncCaller`` does not crash during GC shutdown."""

    def test_close_without_process_group(self):
        """Calling close() after process group destruction must not raise."""
        caller = PersistentAsyncCaller()
        # Simulate the state where no process was ever spawned (process is None)
        # but close() still logs with the rank.
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            # Must not raise
            caller.close()

    def test_del_without_process_group(self):
        """``__del__`` must not raise when dist is uninitialised."""
        caller = PersistentAsyncCaller()
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            # Must not raise
            caller.__del__()


class TestFileSystemAsyncKunlunxin:
    """Verify KunLunXin filesystem async checkpoint overrides."""

    def test_filesystem_async_routes_to_kunlunxin_vendor(self, monkeypatch):
        """KunLunXin vendor must own all migrated filesystem async patch points."""
        monkeypatch.setenv("MG_FL_PREFER", "kunlunxin")
        registry = importlib.import_module("megatron.plugin.override_registry")
        importlib.reload(registry)
        expected_module = "megatron.plugin.kunlunxin.dist_checkpointing.strategies.filesystem_async"
        for key in (
            "FileSystemWriterAsync.prepare_write_data",
            "filesystem_async.preload_tensors",
            "filesystem_async.write_preloaded_data",
        ):
            impl = get_override_method(key)
            assert impl.__module__ == expected_module

    def test_preload_tensors_skips_d2h_when_bridge_exists(self):
        """Bridge mode must keep XME behavior and skip D2H staging."""
        tensor = mock.Mock()
        buckets = [("file", "key", ([], [("item", tensor)]))]

        with mock.patch("importlib.util.find_spec", return_value=object()):
            result = preload_tensors_kunlunxin(buckets, non_blocking=True)

        assert result is buckets
        tensor.to.assert_not_called()


class TestTemporalAsyncCallerShutdown:
    """Verify ``TemporalAsyncCaller`` does not crash during GC shutdown."""

    def test_close_without_process_group(self):
        """Calling close() after process group destruction must not raise."""
        caller = TemporalAsyncCaller()
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            caller.close()
