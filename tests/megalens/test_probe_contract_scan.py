# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import probe_contract_scan as gate

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures/probe_scan_gate.json"


def _write_module(
    repo: Path,
    *,
    producers: tuple[str, ...],
    exact_consumers: tuple[str, ...] = (),
    substring_consumers: tuple[str, ...] = (),
    base: tuple[str, ...] = (),
    full: tuple[str, ...] = (),
    dead_producers: tuple[str, ...] = (),
) -> None:
    package = repo / "megatron"
    package.mkdir(parents=True, exist_ok=True)
    lines = [
        f"BASE_TRACING_EVENTS = {list(base)!r}",
        f"FULL_TRACING_EVENTS = {list(full)!r}",
    ]
    lines.extend(f"tracer.scope({name!r})" for name in producers)
    if dead_producers:
        lines.append("if False:")
        lines.extend(f"    tracer.scope({name!r})" for name in dead_producers)
    lines.extend(f"loader.get_events_by_name({name!r})" for name in exact_consumers)
    lines.extend(
        f"loader.get_events_matching({name!r})" for name in substring_consumers
    )
    (package / "sample.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _commit_source(repo: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.name", "MegaLens Test"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "megalens-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "megatron"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source fixture"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _build_fixture(
    path: Path, source_repo: Path, target_repo: Path
) -> dict[str, object]:
    source_commit = _commit_source(source_repo)
    fixture = gate.build_fixture(source_repo, target_repo, source_commit)
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return fixture


def test_scanner_extracts_producers_consumers_and_event_sets(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        producers=("event-a",),
        dead_producers=("event-dead",),
        exact_consumers=("event-a",),
        substring_consumers=("event-",),
        base=("event-a",),
        full=("event-a", "event-dead"),
    )

    scan = gate.scan_repo(tmp_path)

    assert scan["producers"] == {"live": ["event-a"], "dead": ["event-dead"]}
    assert scan["consumers"] == {"exact": ["event-a"], "substring": ["event-"]}
    assert scan["event_sets"] == {
        "BASE_TRACING_EVENTS": ["event-a"],
        "FULL_TRACING_EVENTS": ["event-a", "event-dead"],
    }
    assert scan["parse_errors"] == []


def test_current_target_matches_reviewed_fixture() -> None:
    result = gate.run_target_gate(REPOSITORY_ROOT, FIXTURE)

    assert result.errors == ()


def test_fixture_contains_only_reviewable_probe_facts(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(
        source_repo,
        producers=("source-event",),
        exact_consumers=("source-event",),
        base=("source-event",),
        full=("source-event",),
    )
    _write_module(
        target_repo,
        producers=("source-event", "target-event"),
        exact_consumers=("source-event",),
        base=("source-event", "target-event"),
        full=("source-event", "target-event"),
    )
    fixture_path = tmp_path / "fixture.json"

    fixture = _build_fixture(fixture_path, source_repo, target_repo)

    assert set(fixture) == {
        "mixedpara_baseline_commit",
        "source",
        "target",
        "event_set_differences",
        "declared_target_only_events",
    }
    assert set(fixture["source"]) == {"producers", "consumers", "event_sets"}
    assert set(fixture["target"]) == {"producers", "consumers"}
    assert fixture["event_set_differences"] == {
        "BASE_TRACING_EVENTS": {"source_only": [], "target_only": ["target-event"]},
        "FULL_TRACING_EVENTS": {"source_only": [], "target_only": ["target-event"]},
    }
    assert fixture["declared_target_only_events"] == ["target-event"]


def test_target_gate_detects_producer_name_drift_with_the_same_count(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",))
    _write_module(target_repo, producers=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    _build_fixture(fixture_path, source_repo, target_repo)
    _write_module(target_repo, producers=("event-b",))

    result = gate.run_target_gate(target_repo, fixture_path)

    assert any("target producers.live changed" in error for error in result.errors)


def test_target_gate_detects_consumer_name_drift(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",), exact_consumers=("event-a",))
    _write_module(target_repo, producers=("event-a",), exact_consumers=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    _build_fixture(fixture_path, source_repo, target_repo)
    _write_module(target_repo, producers=("event-a",), exact_consumers=("event-b",))

    result = gate.run_target_gate(target_repo, fixture_path)

    assert any("target consumers.exact changed" in error for error in result.errors)


def test_target_gate_detects_event_set_drift(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",), base=("event-a",))
    _write_module(target_repo, producers=("event-a",), base=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    _build_fixture(fixture_path, source_repo, target_repo)
    _write_module(target_repo, producers=("event-a",), base=("event-b",))

    result = gate.run_target_gate(target_repo, fixture_path)

    assert any(
        "target event_sets.BASE_TRACING_EVENTS changed" in error
        for error in result.errors
    )


def test_declared_target_only_events_are_derived_from_source_and_target(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",), base=("event-a",))
    _write_module(
        target_repo,
        producers=("event-a", "target-event"),
        base=("event-a", "target-event"),
    )
    fixture_path = tmp_path / "fixture.json"
    fixture = _build_fixture(fixture_path, source_repo, target_repo)
    fixture["declared_target_only_events"] = []
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    result = gate.run_target_gate(target_repo, fixture_path)

    assert any(
        "declared_target_only_events changed" in error for error in result.errors
    )


def test_full_gate_reads_the_locked_source_commit(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",))
    _write_module(target_repo, producers=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    fixture = _build_fixture(fixture_path, source_repo, target_repo)
    _write_module(source_repo, producers=("dirty-working-tree-event",))

    result = gate.run_full_gate(source_repo, target_repo, fixture_path)

    assert result.errors == ()
    assert result.source is not None
    assert result.source["producers"]["live"] == ["event-a"]
    assert fixture["mixedpara_baseline_commit"]


def test_source_ref_override_must_match_the_locked_commit(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",))
    _write_module(target_repo, producers=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    fixture = _build_fixture(fixture_path, source_repo, target_repo)

    result = gate.run_full_gate(
        source_repo, target_repo, fixture_path, source_ref="HEAD"
    )

    assert (
        "source ref differs from fixture: "
        f"expected={fixture['mixedpara_baseline_commit']!r} actual='HEAD'"
    ) in result.errors


@pytest.mark.parametrize(
    "argv",
    (
        (
            "target",
            "--target-repo",
            ".",
            "--fixture",
            "fixture.json",
            "--contract",
            "missing.json",
        ),
        (
            "gate",
            "--source-repo",
            ".",
            "--target-repo",
            ".",
            "--fixture",
            "fixture.json",
            "--contract",
            "missing.json",
        ),
    ),
)
def test_scanner_rejects_the_removed_contract_option(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit, match="2"):
        gate._parser().parse_args(argv)


def test_target_parse_errors_are_reported(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",))
    _write_module(target_repo, producers=("event-a",))
    fixture_path = tmp_path / "fixture.json"
    _build_fixture(fixture_path, source_repo, target_repo)
    (target_repo / "megatron/broken.py").write_text("if (:\n", encoding="utf-8")

    result = gate.run_target_gate(target_repo, fixture_path)

    assert any(error.startswith("target parse errors:") for error in result.errors)


def test_rebuild_command_writes_a_fixture_that_passes(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    target_repo = tmp_path / "target"
    _write_module(source_repo, producers=("event-a",))
    _write_module(target_repo, producers=("event-a", "target-event"))
    source_commit = _commit_source(source_repo)
    fixture_path = tmp_path / "fixture.json"

    returncode = gate.main(
        [
            "rebuild",
            "--source-repo",
            str(source_repo),
            "--source-ref",
            source_commit,
            "--target-repo",
            str(target_repo),
            "--fixture",
            str(fixture_path),
        ]
    )

    assert returncode == 0
    assert gate.run_full_gate(source_repo, target_repo, fixture_path).errors == ()
