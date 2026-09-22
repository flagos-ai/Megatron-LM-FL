import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "te_fl_local_cache.py"
SPEC = importlib.util.spec_from_file_location("te_fl_local_cache", SCRIPT)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class LocalWheelCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "cache"
        self.source = self.directory / "built"
        self.source.mkdir()
        self.filename = "transformer_engine-2.17.0-cp312-cp312-linux_x86_64.whl"
        (self.source / self.filename).write_bytes(b"built-wheel")
        self.destination = self.directory / "restored"
        self.key = "te-fl-backup-" + "a" * 64
        self.output = self.directory / "github-output"
        self.env = {
            **os.environ,
            "TE_FL_CACHE_DIRECTORY": str(self.root),
            "TE_FL_CACHE_KEY": self.key,
            "TE_FL_WHEEL_DIR": str(self.destination),
            "GITHUB_OUTPUT": str(self.output),
        }

    def run_cache(self, operation, *args, **env):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), operation, *args],
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=15,
        )
        return result

    def publish(self):
        result = self.run_cache("publish", TE_FL_WHEEL_DIR=str(self.source))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preparation_miss_requests_build(self):
        result = self.run_cache("restore")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "cache-hit=false\n")
        self.assertFalse(self.root.exists())

    def test_consumer_miss_fails_without_network_fallback(self):
        result = self.run_cache("restore", "--required")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("same runner host", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_publish_and_restore_do_not_modify_cache(self):
        self.publish()
        entry = self.root / self.key
        manifest = (entry / "manifest.json").read_bytes()
        result = self.run_cache("restore", "--required")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "cache-hit=true\n")
        self.assertEqual((self.destination / self.filename).read_bytes(), b"built-wheel")
        (self.destination / self.filename).write_bytes(b"consumer-change")
        self.assertEqual((entry / self.filename).read_bytes(), b"built-wheel")
        self.assertEqual((entry / "manifest.json").read_bytes(), manifest)

    def test_existing_key_is_immutable(self):
        self.publish()
        (self.source / self.filename).write_bytes(b"different-build")
        self.publish()
        self.assertEqual((self.root / self.key / self.filename).read_bytes(), b"built-wheel")

    def test_other_key_does_not_restore_old_wheel(self):
        self.publish()
        result = self.run_cache("restore", "--required", TE_FL_CACHE_KEY="different-version")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.destination.exists())

    def test_corrupt_wheel_fails_before_installation(self):
        self.publish()
        (self.root / self.key / self.filename).write_bytes(b"corrupt")
        for arguments in ((), ("--required",)):
            with self.subTest(arguments=arguments):
                result = self.run_cache("restore", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse(self.destination.exists())
        result = self.run_cache("publish", TE_FL_WHEEL_DIR=str(self.source))
        self.assertNotEqual(result.returncode, 0)

    def test_manifest_must_match_key_and_schema(self):
        self.publish()
        manifest = self.root / self.key / "manifest.json"
        for content in ({"schema": 1, "key": "wrong"}, {"schema": 2, "key": self.key}, []):
            with self.subTest(content=content):
                manifest.write_text(json.dumps(content))
                result = self.run_cache("restore", "--required")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("identity mismatch", result.stderr)

    def test_incomplete_entry_is_not_a_hit(self):
        entry = self.root / self.key
        entry.mkdir(parents=True)
        (entry / self.filename).write_bytes(b"partial")
        result = self.run_cache("restore", "--required")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_symlinked_cache_entries_are_rejected(self):
        self.publish()
        (self.root / "alias").symlink_to(self.root / self.key, target_is_directory=True)
        result = self.run_cache("restore", "--required", TE_FL_CACHE_KEY="alias")
        self.assertNotEqual(result.returncode, 0)

    def test_symlinked_wheel_and_manifest_are_rejected(self):
        self.publish()
        entry = self.root / self.key
        wheel = entry / self.filename
        wheel.unlink()
        wheel.symlink_to(self.source / self.filename)
        result = self.run_cache("restore", "--required")
        self.assertNotEqual(result.returncode, 0)
        wheel.unlink()
        wheel.write_bytes(b"built-wheel")
        manifest = entry / "manifest.json"
        external_manifest = self.directory / "external.json"
        manifest.rename(external_manifest)
        manifest.symlink_to(external_manifest)
        result = self.run_cache("restore", "--required")
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_key_and_cache_root_are_rejected(self):
        for key in ("", "../escape", "/absolute", "with\nnewline"):
            with self.subTest(key=key):
                result = self.run_cache("restore", TE_FL_CACHE_KEY=key)
                self.assertNotEqual(result.returncode, 0)
        for root in ("relative", "/"):
            with self.subTest(root=root):
                result = self.run_cache("restore", TE_FL_CACHE_DIRECTORY=root)
                self.assertNotEqual(result.returncode, 0)

    def test_multiple_wheels_are_not_published(self):
        (self.source / "transformer_engine-other.whl").write_bytes(b"extra")
        result = self.run_cache("publish", TE_FL_WHEEL_DIR=str(self.source))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / self.key).exists())

    def test_nonempty_restore_destination_is_not_overwritten(self):
        self.publish()
        self.destination.mkdir()
        existing = self.destination / "transformer_engine-old.whl"
        existing.write_bytes(b"old")
        result = self.run_cache("restore", "--required")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(existing.read_bytes(), b"old")

    def test_failed_publish_leaves_no_partial_entry(self):
        with mock.patch.object(CACHE.shutil, "copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                CACHE.publish(self.root, self.key, self.source)
        self.assertFalse((self.root / self.key).exists())
        self.assertEqual(list(self.root.glob(".publish-*")), [])

    def test_corruption_during_copy_is_detected(self):
        self.publish()
        with mock.patch.object(
            CACHE.shutil, "copyfile", side_effect=lambda source, target: target.write_bytes(b"bad")
        ):
            with self.assertRaisesRegex(ValueError, "changed during restore"):
                CACHE.restore(self.root, self.key, self.destination, required=True)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_concurrent_publishers_leave_one_complete_entry(self):
        processes = [
            subprocess.Popen(
                [sys.executable, str(SCRIPT), "publish"],
                env={**self.env, "TE_FL_WHEEL_DIR": str(self.source)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(4)
        ]
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stdout + stderr)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        self.assertEqual(list(self.root.glob(".publish-*")), [])
        result = self.run_cache("restore", "--required")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_downstream_rerun_restores_producer_identity(self):
        result = self.run_cache("publish", TE_FL_WHEEL_DIR=str(self.source), GITHUB_RUN_ATTEMPT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_cache("restore", "--required", GITHUB_RUN_ATTEMPT="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.destination / self.filename).read_bytes(), b"built-wheel")

    def expire(self, entry):
        timestamp = CACHE.time.time() - CACHE.RETENTION_SECONDS - 1
        os.utime(entry, (timestamp, timestamp))

    def test_publish_prunes_expired_backups_but_preserves_recent_ones(self):
        self.publish()
        old = self.root / self.key
        self.expire(old)
        recent_key = "te-fl-backup-" + "b" * 64
        new_key = "te-fl-backup-" + "c" * 64
        self.assertTrue(CACHE.publish(self.root, recent_key, self.source))
        self.assertFalse(old.exists())
        self.assertTrue(CACHE.publish(self.root, new_key, self.source))
        self.assertTrue((self.root / recent_key).exists())
        self.assertTrue((self.root / new_key).exists())

    def test_capacity_limit_skips_backup_without_evicting_recent_entry(self):
        self.publish()
        original_size = CACHE.cache_size(self.root)
        new_key = "te-fl-backup-" + "b" * 64
        self.assertFalse(
            CACHE.publish(self.root, new_key, self.source, max_bytes=original_size + 4096)
        )
        self.assertTrue((self.root / self.key).exists())
        self.assertFalse((self.root / new_key).exists())
        self.assertEqual(CACHE.cache_size(self.root), original_size)
        self.assertEqual(list(self.root.glob(".publish-*")), [])

    def test_oversized_wheel_is_not_copied(self):
        with mock.patch.object(CACHE.shutil, "copyfile") as copy:
            self.assertFalse(CACHE.publish(self.root, self.key, self.source, max_bytes=1))
        copy.assert_not_called()
        self.assertFalse((self.root / self.key).exists())

    def test_active_reader_prevents_expiration_and_publication(self):
        self.publish()
        old = self.root / self.key
        self.expire(old)
        with (self.root / ".publish.lock").open("r") as lock:
            CACHE.fcntl.flock(lock, CACHE.fcntl.LOCK_SH)
            self.assertFalse(CACHE.publish(self.root, "te-fl-backup-" + "b" * 64, self.source))
            self.assertTrue(old.exists())

    def test_restore_holds_read_lock_during_copy(self):
        self.publish()
        original_copy = CACHE.shutil.copyfile

        def copy_under_lock(source, target):
            with (self.root / ".publish.lock").open("r") as lock:
                with self.assertRaises(BlockingIOError):
                    CACHE.fcntl.flock(lock, CACHE.fcntl.LOCK_EX | CACHE.fcntl.LOCK_NB)
            return original_copy(source, target)

        with mock.patch.object(CACHE.shutil, "copyfile", side_effect=copy_under_lock):
            self.assertTrue(CACHE.restore(self.root, self.key, self.destination, required=True))

    def test_pruning_ignores_unmanaged_directories_and_symlinks(self):
        self.publish()
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        (unrelated / "keep").write_bytes(b"keep")
        self.expire(unrelated)
        alias = self.root / ("te-fl-backup-" + "b" * 64)
        alias.symlink_to(self.source, target_is_directory=True)
        self.expire(self.source)
        self.assertTrue(CACHE.publish(self.root, "te-fl-backup-" + "c" * 64, self.source))
        self.assertTrue((unrelated / "keep").exists())
        self.assertTrue(alias.is_symlink())
        self.assertTrue((self.source / self.filename).exists())

    def test_pruning_removes_only_expired_abandoned_staging(self):
        self.publish()
        old = self.root / ".publish-abandoned"
        old.mkdir()
        (old / "partial").write_bytes(b"partial")
        self.expire(old)
        recent = self.root / ".publish-recent"
        recent.mkdir()
        self.assertTrue(CACHE.publish(self.root, "te-fl-backup-" + "b" * 64, self.source))
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())


if __name__ == "__main__":
    unittest.main()
