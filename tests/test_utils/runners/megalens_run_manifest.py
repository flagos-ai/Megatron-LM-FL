# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compact evidence manifest for FlagScale MegaLens training runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "manifest.json"

_GLOBAL_RANK_PATTERN = re.compile(r"benchmark-global-(?P<rank>\d+)-")


@dataclass(frozen=True)
class EventRequirement:
    """One event and the fields that must be present on its selected phase."""

    name: str
    fields: tuple[str, ...] = ()
    phase: str | None = None


@dataclass(frozen=True)
class TraceProfile:
    """Observable trace requirements for one runnable FlagScale profile."""

    name: str
    rank_count: int
    events: tuple[EventRequirement, ...] = ()
    contract: Callable[[Path], Sequence[Failure]] | None = None
    run_contract: Callable[[Path, bool], Sequence[Failure]] | None = None


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    evidence: str | None = None


@dataclass(frozen=True)
class TraceShard:
    path: str
    rank: int | None
    record_count: int


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    failures: tuple[Failure, ...]
    shards: tuple[TraceShard, ...]
    ranks: tuple[int, ...]
    event_counts: Mapping[str, int]
    total_records: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [asdict(failure) for failure in self.failures],
            "trace": {
                "shards": [asdict(shard) for shard in self.shards],
                "ranks": list(self.ranks),
                "event_counts": dict(sorted(self.event_counts.items())),
                "total_records": self.total_records,
            },
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("events", payload.get("records"))
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("trace shard must contain a JSON event array")
    return list(rows)


def _rank_from_records(records: Sequence[Mapping[str, Any]]) -> int | None:
    ranks = {
        row["g_rk"]
        for row in records
        if isinstance(row.get("g_rk"), int) and not isinstance(row.get("g_rk"), bool)
    }
    return next(iter(ranks)) if len(ranks) == 1 else None


def _rank_from_path(path: Path) -> int | None:
    match = _GLOBAL_RANK_PATTERN.search(path.name)
    return int(match.group("rank")) if match is not None else None


def discover_trace_shards(trace_root: Path) -> tuple[Path, ...]:
    """Return rank-local JSON shards produced under the configured trace root."""

    if not trace_root.is_dir():
        return ()
    return tuple(sorted(trace_root.rglob("benchmark-*.json")))


def validate_trace(
    trace_root: Path,
    profile: TraceProfile,
    *,
    trace_enabled: bool,
) -> ValidationReport:
    """Load trace shards and check the profile's event, field, and rank contract."""

    paths = discover_trace_shards(trace_root)
    failures: list[Failure] = []
    shards: list[TraceShard] = []
    all_records: list[Mapping[str, Any]] = []
    observed_ranks: set[int] = set()

    if not trace_enabled:
        if paths:
            failures.append(
                Failure(
                    "trace.unexpected",
                    "trace-off run produced trace shards",
                    str(trace_root),
                )
            )
        return ValidationReport(
            passed=not failures,
            failures=tuple(failures),
            shards=(),
            ranks=(),
            event_counts={},
            total_records=0,
        )

    if not paths:
        failures.append(
            Failure(
                "trace.missing",
                "trace-on run produced no trace shards",
                str(trace_root),
            )
        )

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = _trace_records(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            failures.append(Failure("trace.invalid", str(error), str(path)))
            continue

        path_rank = _rank_from_path(path)
        record_rank = _rank_from_records(records)
        rank = path_rank if path_rank is not None else record_rank
        if rank is None:
            failures.append(
                Failure(
                    "trace.rank_unknown",
                    "cannot determine one global rank for trace shard",
                    str(path),
                )
            )
        else:
            observed_ranks.add(rank)
        shards.append(
            TraceShard(
                path=path.as_posix(),
                rank=rank,
                record_count=len(records),
            )
        )
        all_records.extend(records)

    ranks = tuple(sorted(observed_ranks))
    if len(ranks) != profile.rank_count:
        failures.append(
            Failure(
                "trace.rank_count",
                f"profile {profile.name!r} expects {profile.rank_count} ranks; "
                f"observed {len(ranks)}",
                ",".join(str(rank) for rank in ranks),
            )
        )

    event_counts = Counter(
        str(row["name"]) for row in all_records if isinstance(row.get("name"), str)
    )
    for requirement in profile.events:
        matching = [
            row
            for row in all_records
            if row.get("name") == requirement.name
            and (requirement.phase is None or row.get("ph") == requirement.phase)
        ]
        if not matching:
            phase = (
                f" phase {requirement.phase}" if requirement.phase is not None else ""
            )
            failures.append(
                Failure(
                    "trace.event_missing",
                    f"required event {requirement.name!r}{phase} is absent",
                    profile.name,
                )
            )
            continue
        for field in requirement.fields:
            missing_count = sum(field not in row for row in matching)
            if missing_count:
                failures.append(
                    Failure(
                        "trace.field_missing",
                        f"event {requirement.name!r} is missing field {field!r} "
                        f"on {missing_count} matching records",
                        profile.name,
                    )
                )

    if profile.contract is not None and not failures:
        try:
            failures.extend(profile.contract(trace_root))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            failures.append(
                Failure(
                    "trace.contract_invalid",
                    f"profile {profile.name!r} contract could not read the trace: {error}",
                    profile.name,
                )
            )

    return ValidationReport(
        passed=not failures,
        failures=tuple(failures),
        shards=tuple(shards),
        ranks=ranks,
        event_counts=dict(event_counts),
        total_records=len(all_records),
    )


def validate_run_artifacts(
    run_root: Path,
    profile: TraceProfile,
    report: ValidationReport,
    *,
    trace_enabled: bool,
) -> ValidationReport:
    """Add profile-specific run artifacts to a trace validation report."""

    if profile.run_contract is None:
        return report
    failures = list(report.failures)
    try:
        failures.extend(profile.run_contract(run_root, trace_enabled))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        failures.append(
            Failure(
                "run.contract_invalid",
                f"profile {profile.name!r} run contract could not read artifacts: {error}",
                profile.name,
            )
        )
    return ValidationReport(
        passed=not failures,
        failures=tuple(failures),
        shards=report.shards,
        ranks=report.ranks,
        event_counts=report.event_counts,
        total_records=report.total_records,
    )


def build_manifest(
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    profile: TraceProfile,
    mode: str,
    command: Sequence[str],
    config_path: Path,
    source_root: Path,
    source_head: str,
    image: str,
    returncode: int,
    timed_out: bool,
    log_path: Path,
    report: ValidationReport,
) -> dict[str, Any]:
    """Build the small, reviewable evidence record written after a run."""

    completed = returncode == 0 and report.passed
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed" if completed else "failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "profile": profile.name,
        "mode": mode,
        "command": list(command),
        "config": {
            "path": config_path.as_posix(),
            "sha256": sha256(config_path),
        },
        "source": {
            "path": source_root.as_posix(),
            "head": source_head,
        },
        "image": image,
        "execution": {
            "returncode": returncode,
            "timed_out": timed_out,
            "log": log_path.as_posix(),
        },
        "validation": report.as_manifest(),
    }


def write_manifest(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write one JSON result without lock directories or publication state."""

    path = run_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
