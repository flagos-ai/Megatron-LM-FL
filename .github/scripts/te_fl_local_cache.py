#!/usr/bin/env python3

"""Publish and restore immutable TE-FL wheels in a runner-mounted directory."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_CACHE_BYTES = 20 * 1024**3


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_wheel(directory):
    wheels = list(directory.glob("transformer_engine*.whl"))
    if len(wheels) != 1 or wheels[0].is_symlink() or not wheels[0].is_file():
        raise ValueError(f"Expected exactly one regular TE-FL wheel in {directory}")
    return wheels[0]


def validate(entry, key):
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError(f"Invalid local TE-FL cache entry: {entry}")
    manifest = entry / "manifest.json"
    if manifest.is_symlink():
        raise ValueError(f"Invalid local TE-FL manifest: {manifest}")
    record = json.loads(manifest.read_text())
    if not isinstance(record, dict) or record.get("schema") != 1 or record.get("key") != key:
        raise ValueError(f"Local TE-FL cache identity mismatch: {entry}")
    wheel = single_wheel(entry)
    if record.get("wheel") != wheel.name or record.get("sha256") != checksum(wheel):
        raise ValueError(f"Local TE-FL wheel checksum mismatch: {entry}")
    return wheel, record["sha256"]


def restore(root, key, destination, required):
    if not root.exists():
        return restore_locked(root, key, destination, required)
    # CONTRACT: readers hold a shared lock until their private copy is complete.
    with (root / ".publish.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return restore_locked(root, key, destination, required)


def restore_locked(root, key, destination, required):
    entry = root / key
    if not entry.exists() and not entry.is_symlink():
        if required:
            raise ValueError(
                f"Prepared TE-FL wheel is missing: {entry}. "
                "Preparation and tests must use the same runner host and persistent mount."
            )
        print(f"Local TE-FL cache miss: {key}")
        return False
    wheel, expected_checksum = validate(entry, key)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"TE-FL restore destination must be empty: {destination}")
    # Copy before installation so a consumer never modifies the persistent entry.
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination) as temporary:
        copied = Path(temporary) / wheel.name
        shutil.copyfile(wheel, copied)
        if checksum(copied) != expected_checksum:
            raise ValueError(f"TE-FL wheel changed during restore: {entry}")
        copied.replace(destination / wheel.name)
    print(f"Local TE-FL cache hit: {key}")
    return True


def prune_expired(root, now):
    for entry in root.iterdir():
        managed = re.fullmatch(r"te-fl-backup-[0-9a-f]{64}", entry.name)
        abandoned = entry.name.startswith(".publish-")
        if not (managed or abandoned) or entry.is_symlink() or not entry.is_dir():
            continue
        if now - entry.stat().st_mtime > RETENTION_SECONDS:
            shutil.rmtree(entry)


def cache_size(root):
    return sum(
        path.stat().st_size for path in root.rglob("*") if not path.is_symlink() and path.is_file()
    )


def publish(root, key, source, max_bytes=MAX_CACHE_BYTES):
    wheel = single_wheel(source)
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    # CONTRACT: never evict recent backups to admit a new one, or interrupt readers.
    with (root / ".publish.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("::warning::Local TE-FL cache is busy; skipping optional backup")
            return False
        prune_expired(root, time.time())
        if entry.exists() or entry.is_symlink():
            validate(entry, key)
            print(f"Local TE-FL cache already published: {key}")
            return True
        # Reserve metadata space before copying; verify the exact size before publication.
        if cache_size(root) + wheel.stat().st_size + 4096 > max_bytes:
            print("::warning::Local TE-FL cache capacity reached; skipping optional backup")
            return False
        with tempfile.TemporaryDirectory(prefix=".publish-", dir=root) as temporary:
            staged = Path(temporary)
            copied = staged / wheel.name
            shutil.copyfile(wheel, copied)
            record = {"schema": 1, "key": key, "wheel": wheel.name, "sha256": checksum(copied)}
            (staged / "manifest.json").write_text(json.dumps(record, sort_keys=True) + "\n")
            validate(staged, key)
            if cache_size(root) > max_bytes:
                print("::warning::Local TE-FL cache capacity reached; discarding optional backup")
                return False
            staged.chmod(0o755)
            staged.rename(entry)
    print(f"Published local TE-FL wheel: {entry / wheel.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("restore", "publish"))
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args()
    try:
        root = Path(os.environ["TE_FL_CACHE_DIRECTORY"])
        key = os.environ["TE_FL_CACHE_KEY"]
        wheel_dir = Path(os.environ["TE_FL_WHEEL_DIR"])
        if not root.is_absolute() or root == Path("/") or root.is_symlink():
            raise ValueError("TE_FL_CACHE_DIRECTORY must be a dedicated absolute directory")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key):
            raise ValueError("Invalid TE_FL_CACHE_KEY")
        if args.operation == "publish":
            publish(root, key, wheel_dir)
        else:
            hit = restore(root, key, wheel_dir, args.required)
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                    output.write(f"cache-hit={str(hit).lower()}\n")
    except (KeyError, OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
