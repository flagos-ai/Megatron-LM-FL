# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare literal Probe names in MixedPara and Megatron-LM-FL.

The fixture records the locked MixedPara commit, source/target producer and
consumer names, event-set differences, and the reviewed target-only event
vocabulary.  The scan is static and makes no runtime timing or completion
claims.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROBE_METHODS = {
    "scope",
    "scoped",
    "tick",
    "trace_scope",
    "open_trace_scope",
    "_pipeline_phase_scope",
}
PROBE_NAME_POSITIONS = {"open_trace_scope": 1}
PROBE_DECORATORS = {"scoped_forward"}
CONSUMER_METHODS = {"get_events_by_name", "get_events_matching"}
EVENT_SET_NAMES = ("BASE_TRACING_EVENTS", "FULL_TRACING_EVENTS")
RUNTIME_EXCLUDED_PARTS = {"migration_checks"}
FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class GateResult:
    source: dict[str, Any] | None
    target: dict[str, Any]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_strings(node: ast.AST | None) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    values = [_literal_string(item) for item in node.elts]
    return (
        [value for value in values if value is not None] if None not in values else []
    )


def _call_method_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _call_arg(node: ast.Call, position: int, keyword: str) -> ast.AST | None:
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    return node.args[position] if len(node.args) > position else None


class _FileScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.live_producers: set[str] = set()
        self.dead_producers: set[str] = set()
        self.exact_consumers: set[str] = set()
        self.substring_consumers: set[str] = set()
        self.event_sets: dict[str, list[str]] = {}
        self._dead_depth = 0

    def _record_producer(self, event_name: str | None) -> None:
        if event_name is None:
            return
        producers = self.dead_producers if self._dead_depth else self.live_producers
        producers.add(event_name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and _call_method_name(decorator) in PROBE_DECORATORS
            ):
                self._record_producer(_literal_string(_call_arg(decorator, 0, "name")))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_If(self, node: ast.If) -> Any:
        is_false = isinstance(node.test, ast.Constant) and node.test.value is False
        if is_false:
            self._dead_depth += 1
        for item in node.body:
            self.visit(item)
        if is_false:
            self._dead_depth -= 1
        for item in node.orelse:
            self.visit(item)

    def _record_assignment(self, name: str, value: ast.AST | None) -> None:
        values = _literal_strings(value)
        if name in EVENT_SET_NAMES and values:
            self.event_sets[name] = sorted(set(values))

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record_assignment(target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name):
            self._record_assignment(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        method = _call_method_name(node)
        if method in PROBE_METHODS:
            self._record_producer(
                _literal_string(
                    _call_arg(node, PROBE_NAME_POSITIONS.get(method, 0), "name")
                )
            )
        elif method in CONSUMER_METHODS:
            query = _literal_string(_call_arg(node, 0, "name"))
            if query is not None:
                consumers = (
                    self.exact_consumers
                    if method == "get_events_by_name"
                    else self.substring_consumers
                )
                consumers.add(query)
        self.generic_visit(node)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _python_sources(repo: Path, git_ref: str | None) -> Iterable[tuple[str, str]]:
    if git_ref is None:
        for path in sorted((repo / "megatron").rglob("*.py")):
            if not RUNTIME_EXCLUDED_PARTS.intersection(path.parts):
                yield str(path.relative_to(repo)), path.read_text(encoding="utf-8-sig")
        return

    listing = _git(
        repo, "ls-tree", "-r", "--name-only", git_ref, "--", "megatron"
    ).decode("utf-8")
    for relative_path in sorted(
        item
        for item in listing.splitlines()
        if item.endswith(".py")
        and not RUNTIME_EXCLUDED_PARTS.intersection(Path(item).parts)
    ):
        yield (
            relative_path,
            _git(repo, "show", f"{git_ref}:{relative_path}").decode("utf-8-sig"),
        )


def scan_repo(repo: Path, git_ref: str | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    if not (repo / "megatron").is_dir():
        raise ValueError(f"{repo} does not contain megatron/")

    aggregate = _FileScanner()
    parse_errors: list[dict[str, str]] = []
    try:
        source_items = list(_python_sources(repo, git_ref))
    except (subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read git ref {git_ref!r}: {exc}") from exc

    for relative_path, source in source_items:
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError as exc:
            parse_errors.append({"file": relative_path, "error": str(exc)})
            continue
        scanner = _FileScanner()
        scanner.visit(tree)
        aggregate.live_producers.update(scanner.live_producers)
        aggregate.dead_producers.update(scanner.dead_producers)
        aggregate.exact_consumers.update(scanner.exact_consumers)
        aggregate.substring_consumers.update(scanner.substring_consumers)
        aggregate.event_sets.update(scanner.event_sets)

    return {
        "producers": {
            "live": sorted(aggregate.live_producers),
            "dead": sorted(aggregate.dead_producers),
        },
        "consumers": {
            "exact": sorted(aggregate.exact_consumers),
            "substring": sorted(aggregate.substring_consumers),
        },
        "event_sets": {
            name: aggregate.event_sets.get(name, []) for name in EVENT_SET_NAMES
        },
        "parse_errors": parse_errors,
    }


def _snapshot(scan: dict[str, Any], *, include_event_sets: bool) -> dict[str, Any]:
    value = {
        "producers": scan["producers"],
        "consumers": scan["consumers"],
    }
    if include_event_sets:
        value["event_sets"] = scan["event_sets"]
    return value


def _event_set_differences(
    source_event_sets: dict[str, list[str]],
    target_event_sets: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    differences = {}
    for name in EVENT_SET_NAMES:
        source = set(source_event_sets[name])
        target = set(target_event_sets[name])
        differences[name] = {
            "source_only": sorted(source - target),
            "target_only": sorted(target - source),
        }
    return differences


def _event_vocabulary(snapshot: dict[str, Any]) -> set[str]:
    names = set(snapshot["producers"]["live"]) | set(snapshot["producers"]["dead"])
    for event_names in snapshot["event_sets"].values():
        names.update(event_names)
    return names


def build_fixture(
    source_repo: Path,
    target_repo: Path,
    mixedpara_baseline_commit: str,
) -> dict[str, Any]:
    """Build the complete review fixture from one source commit and one target tree."""

    if FULL_COMMIT_RE.fullmatch(mixedpara_baseline_commit) is None:
        raise ValueError(
            "MixedPara baseline must be a full lowercase 40-character commit SHA"
        )
    source_scan = scan_repo(source_repo, mixedpara_baseline_commit)
    target_scan = scan_repo(target_repo)
    if source_scan["parse_errors"] or target_scan["parse_errors"]:
        raise ValueError(
            "cannot build fixture with parse errors: "
            f"source={source_scan['parse_errors']} target={target_scan['parse_errors']}"
        )
    source = _snapshot(source_scan, include_event_sets=True)
    target = _snapshot(target_scan, include_event_sets=False)
    differences = _event_set_differences(
        source_scan["event_sets"], target_scan["event_sets"]
    )
    target_with_event_sets = {**target, "event_sets": target_scan["event_sets"]}
    return {
        "mixedpara_baseline_commit": mixedpara_baseline_commit,
        "source": source,
        "target": target,
        "event_set_differences": differences,
        "declared_target_only_events": sorted(
            _event_vocabulary(target_with_event_sets) - _event_vocabulary(source)
        ),
    }


def write_fixture(
    source_repo: Path,
    target_repo: Path,
    fixture_path: Path,
    mixedpara_baseline_commit: str,
) -> dict[str, Any]:
    fixture = build_fixture(source_repo, target_repo, mixedpara_baseline_commit)
    fixture_path.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return fixture


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _list_field(
    value: Any,
    *path: str,
    errors: list[str],
) -> list[str]:
    current = value
    for part in path:
        if not isinstance(current, dict):
            current = None
            break
        current = current.get(part)
    label = ".".join(path)
    if not isinstance(current, list) or not all(
        isinstance(item, str) for item in current
    ):
        errors.append(f"fixture.{label} must be an array of strings")
        return []
    if current != sorted(set(current)):
        errors.append(f"fixture.{label} must be sorted and unique")
    return current


def _fixture_snapshot(
    fixture: dict[str, Any],
    label: str,
    *,
    include_event_sets: bool,
    errors: list[str],
) -> dict[str, Any]:
    snapshot = {
        "producers": {
            kind: _list_field(fixture, label, "producers", kind, errors=errors)
            for kind in ("live", "dead")
        },
        "consumers": {
            kind: _list_field(fixture, label, "consumers", kind, errors=errors)
            for kind in ("exact", "substring")
        },
    }
    if include_event_sets:
        snapshot["event_sets"] = {
            name: _list_field(fixture, label, "event_sets", name, errors=errors)
            for name in EVENT_SET_NAMES
        }
    return snapshot


def _fixture_differences(
    fixture: dict[str, Any], errors: list[str]
) -> dict[str, dict[str, list[str]]]:
    return {
        name: {
            direction: _list_field(
                fixture,
                "event_set_differences",
                name,
                direction,
                errors=errors,
            )
            for direction in ("source_only", "target_only")
        }
        for name in EVENT_SET_NAMES
    }


def _target_event_sets(
    source_event_sets: dict[str, list[str]],
    differences: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    return {
        name: sorted(
            (set(source_event_sets[name]) - set(differences[name]["source_only"]))
            | set(differences[name]["target_only"])
        )
        for name in EVENT_SET_NAMES
    }


def _compare_snapshot(
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    errors: list[str],
) -> None:
    groups = ("producers", "consumers", "event_sets")
    for group in groups:
        if group not in expected:
            continue
        for name, expected_names in expected[group].items():
            actual_names = actual[group][name]
            if actual_names != expected_names:
                errors.append(
                    f"{label} {group}.{name} changed: "
                    f"missing={sorted(set(expected_names) - set(actual_names))} "
                    f"unexpected={sorted(set(actual_names) - set(expected_names))}"
                )


def _read_fixture(
    fixture_path: Path,
) -> tuple[dict[str, Any], str | None, dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    fixture = _load_json(fixture_path)
    baseline = fixture.get("mixedpara_baseline_commit")
    if not isinstance(baseline, str) or FULL_COMMIT_RE.fullmatch(baseline) is None:
        errors.append(
            "fixture.mixedpara_baseline_commit must be a full lowercase "
            "40-character commit SHA"
        )
        baseline = None

    source = _fixture_snapshot(
        fixture, "source", include_event_sets=True, errors=errors
    )
    target = _fixture_snapshot(
        fixture, "target", include_event_sets=False, errors=errors
    )
    differences = _fixture_differences(fixture, errors)
    target["event_sets"] = _target_event_sets(source["event_sets"], differences)

    declared = _list_field(fixture, "declared_target_only_events", errors=errors)
    derived_differences = _event_set_differences(
        source["event_sets"], target["event_sets"]
    )
    if differences != derived_differences:
        errors.append("fixture.event_set_differences is internally inconsistent")
    derived_target_only = sorted(_event_vocabulary(target) - _event_vocabulary(source))
    if declared != derived_target_only:
        errors.append(
            "fixture.declared_target_only_events changed: "
            f"missing={sorted(set(derived_target_only) - set(declared))} "
            f"unexpected={sorted(set(declared) - set(derived_target_only))}"
        )
    return fixture, baseline, source, target, errors


def run_target_gate(
    target_repo: Path,
    fixture_path: Path,
) -> GateResult:
    """Validate the target tree against the reviewed fixture."""

    _, _, _, expected_target, errors = _read_fixture(fixture_path)
    target_scan = scan_repo(target_repo)
    if target_scan["parse_errors"]:
        errors.append(f"target parse errors: {target_scan['parse_errors']}")
    _compare_snapshot("target", target_scan, expected_target, errors)
    return GateResult(source=None, target=target_scan, errors=tuple(errors))


def run_full_gate(
    source_repo: Path,
    target_repo: Path,
    fixture_path: Path,
    *,
    source_ref: str | None = None,
) -> GateResult:
    """Validate the locked MixedPara commit and the current target tree."""

    target_result = run_target_gate(target_repo, fixture_path)
    _, baseline, expected_source, _, _ = _read_fixture(fixture_path)
    errors = list(target_result.errors)
    if source_ref is not None and source_ref != baseline:
        errors.append(
            f"source ref differs from fixture: expected={baseline!r} actual={source_ref!r}"
        )
    if baseline is None:
        return GateResult(
            source=None, target=target_result.target, errors=tuple(errors)
        )

    source_scan = scan_repo(source_repo, baseline)
    if source_scan["parse_errors"]:
        errors.append(f"source parse errors: {source_scan['parse_errors']}")
    _compare_snapshot("source", source_scan, expected_source, errors)
    return GateResult(
        source=source_scan, target=target_result.target, errors=tuple(errors)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser("target", help="validate the current target tree")
    target.add_argument("--target-repo", type=Path, required=True)
    target.add_argument("--fixture", type=Path, required=True)

    gate = subparsers.add_parser(
        "gate", help="validate the locked source and current target"
    )
    gate.add_argument("--source-repo", type=Path, required=True)
    gate.add_argument("--source-ref")
    gate.add_argument("--target-repo", type=Path, required=True)
    gate.add_argument("--fixture", type=Path, required=True)

    rebuild = subparsers.add_parser("rebuild", help="rebuild the review fixture")
    rebuild.add_argument("--source-repo", type=Path, required=True)
    rebuild.add_argument("--source-ref", required=True)
    rebuild.add_argument("--target-repo", type=Path, required=True)
    rebuild.add_argument("--fixture", type=Path, required=True)
    return parser


def _print_result(result: GateResult) -> None:
    output = {
        "status": "PASS" if result.ok else "FAIL",
        "source": result.source,
        "target": result.target,
        "errors": list(result.errors),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "target":
            result = run_target_gate(args.target_repo, args.fixture)
        elif args.command == "gate":
            result = run_full_gate(
                args.source_repo,
                args.target_repo,
                args.fixture,
                source_ref=args.source_ref,
            )
        else:
            fixture = write_fixture(
                args.source_repo,
                args.target_repo,
                args.fixture,
                args.source_ref,
            )
            print(json.dumps(fixture, indent=2, ensure_ascii=False))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"probe contract gate failed to run: {exc}", file=sys.stderr)
        return 2
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
