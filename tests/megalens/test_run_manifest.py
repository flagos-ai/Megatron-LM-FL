# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from tests.test_utils.runners import megalens_run_manifest as manifest

_FIELDS = ("iteration", "g_rk", "dp_rk", "pp_rk", "tp_rk", "value")
_PROFILE = manifest.TraceProfile(
    "two-rank-probe",
    2,
    (manifest.EventRequirement("target-event", _FIELDS, "E"),),
)


def _write_shard(
    root: Path,
    rank: int,
    *,
    include_value: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attributes = {
        "iteration": 2,
        "g_rk": rank,
        "dp_rk": rank,
        "pp_rk": 0,
        "tp_rk": 0,
    }
    end = {"name": "target-event", "ph": "E", **attributes}
    if include_value:
        end["value"] = rank
    path = root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
    path.write_text(
        json.dumps(
            [
                {"name": "target-event", "ph": "B", **attributes},
                end,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_trace_report_records_shards_ranks_events_and_fields(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0)
    _write_shard(trace_root, 1).rename(
        trace_root / "benchmark-data-1-pipeline-0-tensor-0.json"
    )

    report = manifest.validate_trace(trace_root, _PROFILE, trace_enabled=True)

    assert report.passed
    assert report.ranks == (0, 1)
    assert report.total_records == 4
    assert report.event_counts["target-event"] == 4
    assert sorted((shard.rank, shard.record_count) for shard in report.shards) == [
        (0, 2),
        (1, 2),
    ]


def test_trace_report_rejects_mismatched_and_noncanonical_ranks(tmp_path: Path) -> None:
    mismatch_root = tmp_path / "mismatch"
    _write_shard(mismatch_root, 0).rename(
        mismatch_root / "benchmark-global-1-data-0-pipeline-0-tensor-0.json"
    )

    mismatch = manifest.validate_trace(
        mismatch_root,
        manifest.TraceProfile("one-rank", 1),
        trace_enabled=True,
    )

    assert mismatch.ranks == (0,)
    assert [failure.code for failure in mismatch.failures] == ["trace.rank_mismatch"]

    mixed_root = tmp_path / "mixed"
    mixed_path = _write_shard(mixed_root, 0)
    mixed_rows = json.loads(mixed_path.read_text(encoding="utf-8"))
    mixed_rows[-1]["g_rk"] = 1
    mixed_path.write_text(json.dumps(mixed_rows), encoding="utf-8")

    mixed = manifest.validate_trace(
        mixed_root,
        manifest.TraceProfile("one-rank", 1),
        trace_enabled=True,
    )

    assert mixed.shards[0].rank is None
    assert {failure.code for failure in mixed.failures} == {
        "trace.rank_count",
        "trace.rank_unknown",
    }

    invalid_root = tmp_path / "invalid"
    invalid_path = _write_shard(invalid_root, 0)
    invalid_rows = json.loads(invalid_path.read_text(encoding="utf-8"))
    for row in invalid_rows:
        row["g_rk"] = "0"
    invalid_path.write_text(json.dumps(invalid_rows), encoding="utf-8")

    invalid = manifest.validate_trace(
        invalid_root,
        manifest.TraceProfile("one-rank", 1),
        trace_enabled=True,
    )

    assert invalid.shards[0].rank is None
    assert {failure.code for failure in invalid.failures} == {
        "trace.rank_count",
        "trace.rank_unknown",
    }

    shifted_root = tmp_path / "shifted"
    _write_shard(shifted_root, 4)
    _write_shard(shifted_root, 5)

    shifted = manifest.validate_trace(shifted_root, _PROFILE, trace_enabled=True)

    assert shifted.ranks == (4, 5)
    assert [failure.code for failure in shifted.failures] == ["trace.rank_count"]


def test_trace_report_explains_missing_field_and_rank(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0, include_value=False)

    report = manifest.validate_trace(trace_root, _PROFILE, trace_enabled=True)

    assert not report.passed
    assert {failure.code for failure in report.failures} == {
        "trace.field_missing",
        "trace.rank_count",
    }


def test_trace_off_reports_any_discovered_shard(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0)

    report = manifest.validate_trace(trace_root, _PROFILE, trace_enabled=False)

    assert not report.passed
    assert [failure.code for failure in report.failures] == ["trace.unexpected"]


def test_profile_contract_failures_join_the_trace_report(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0)
    _write_shard(trace_root, 1)
    observed_roots: list[Path] = []

    def check_contract(root: Path) -> tuple[manifest.Failure, ...]:
        observed_roots.append(root)
        return (
            manifest.Failure(
                "trace.contract",
                "the profile-specific event order is invalid",
                "two-rank-probe",
            ),
        )

    profile = manifest.TraceProfile(
        _PROFILE.name,
        _PROFILE.rank_count,
        _PROFILE.events,
        check_contract,
    )

    report = manifest.validate_trace(trace_root, profile, trace_enabled=True)

    assert observed_roots == [trace_root]
    assert not report.passed
    assert [failure.code for failure in report.failures] == ["trace.contract"]


def test_trace_off_does_not_run_the_profile_contract(tmp_path: Path) -> None:
    profile = manifest.TraceProfile(
        "trace-off-contract",
        1,
        contract=lambda root: (_ for _ in ()).throw(
            AssertionError(f"trace-off evaluated {root}")
        ),
    )

    report = manifest.validate_trace(tmp_path / "traces", profile, trace_enabled=False)

    assert report.passed


def test_invalid_trace_json_is_reported_without_hiding_other_shards(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0)
    bad = trace_root / "benchmark-global-1-data-1-pipeline-0-tensor-0.json"
    bad.write_text("{", encoding="utf-8")

    report = manifest.validate_trace(trace_root, _PROFILE, trace_enabled=True)

    assert not report.passed
    assert "trace.invalid" in {failure.code for failure in report.failures}
    assert report.ranks == (0,)


def test_manifest_records_minimum_reproducible_run_evidence(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    _write_shard(trace_root, 0)
    _write_shard(trace_root, 1)
    report = manifest.validate_trace(trace_root, _PROFILE, trace_enabled=True)
    config = tmp_path / "inputs" / "smoke.yaml"
    config.parent.mkdir()
    config.write_text("action: test\n", encoding="utf-8")
    log = tmp_path / "launcher.log"
    log.write_text("training output\n", encoding="utf-8")

    payload = manifest.build_manifest(
        run_id="run-001",
        started_at="2026-07-30T00:00:00+00:00",
        finished_at="2026-07-30T00:01:00+00:00",
        profile=_PROFILE,
        mode="trace-on",
        command=("flagscale", "run"),
        config_path=config,
        source_root=tmp_path,
        source_head="a" * 40,
        image="example/image:dev",
        returncode=0,
        timed_out=False,
        log_path=log,
        report=report,
    )
    path = manifest.write_manifest(tmp_path, payload)
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert reloaded["status"] == "completed"
    assert reloaded["command"] == ["flagscale", "run"]
    assert reloaded["config"]["path"] == config.as_posix()
    assert len(reloaded["config"]["sha256"]) == 64
    assert reloaded["source"]["head"] == "a" * 40
    assert reloaded["image"] == "example/image:dev"
    assert reloaded["execution"] == {
        "log": log.as_posix(),
        "returncode": 0,
        "timed_out": False,
    }
    assert reloaded["validation"]["passed"] is True


def test_schema_fixture_matches_the_compact_manifest_shape() -> None:
    schema_path = Path(__file__).parent / "fixtures" / "c5_2_run_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert set(schema["required"]) == {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "finished_at",
        "profile",
        "mode",
        "command",
        "config",
        "source",
        "image",
        "execution",
        "validation",
    }
    assert set(schema["properties"]["validation"]["required"]) == {
        "passed",
        "failures",
        "trace",
    }
