#!/usr/bin/env python3
"""Logical unit-test group routing shared by the planner and its validator.

The catalog decides which logical groups a change affects. Platform configs
decide which groups a platform can actually execute. This module keeps those
two layers separate: a logical group is materialized only when the platform
declares an execution group with the same name.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import pathlib
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable

import flagcicd_contract
import yaml

GROUP_NAME = re.compile(r"[A-Za-z0-9_-]+")
_STATUSES = ("active", "planned", "unsupported")


class CatalogError(ValueError):
    """Raised when the catalog or a platform config cannot be loaded."""


@dataclass(frozen=True)
class Group:
    """A logical unit-test group and its routing metadata."""

    name: str
    description: str
    status: str
    reason: str
    execution_group: str | None
    paths: tuple[str, ...]
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class Level:
    """Required groups for one routing level."""

    name: str
    description: str
    required_groups: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """Parsed unit-group catalog."""

    path: pathlib.Path
    data: dict[str, Any]
    groups: dict[str, Group]
    levels: dict[str, Level]
    documentation_paths: tuple[str, ...]
    documentation_suffixes: tuple[str, ...]
    shared_paths: tuple[str, ...]
    coverage_sources: tuple[str, ...]
    digest: str

    def matches_all(self, path: str) -> list[str]:
        """Return every logical group whose path patterns match the repository path."""
        return sorted(
            {
                group.name
                for group in self.groups.values()
                for pattern in group.paths
                if _matches(path, pattern)
            }
        )

    def matches(self, path: str) -> str | None:
        """Return the most specific logical group for a repository path."""
        matches = [
            (len(pattern), group.name)
            for group in self.groups.values()
            for pattern in group.paths
            if _matches(path, pattern)
        ]
        if not matches:
            return None
        # Longest pattern wins, so a specific directory beats a broader one.
        return max(matches)[1]

    def is_documentation(self, path: str) -> bool:
        if path.startswith(self.documentation_paths):
            return True
        return pathlib.PurePosixPath(path).suffix in self.documentation_suffixes

    def is_shared(self, path: str) -> bool:
        return any(_matches(path, pattern) for pattern in self.shared_paths)

    def closure(self, names: Iterable[str]) -> set[str]:
        """Return the dependency closure of the given logical groups."""
        result: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in result:
                continue
            result.add(name)
            stack.extend(self.groups[name].depends_on)
        return result

    def required_groups(self, level: str) -> tuple[str, ...]:
        if level not in self.levels:
            raise CatalogError(f"Unknown routing level: {level}")
        return self.levels[level].required_groups

    def validate(self, root: pathlib.Path | None = None) -> list[str]:
        """Return catalog errors; an empty list means the catalog is usable."""
        errors: list[str] = []
        if not self.groups:
            errors.append("Catalog declares no groups")
        if not self.coverage_sources:
            errors.append("Catalog declares no coverage sources")
        if root is not None:
            for source in self.coverage_sources:
                if not path_exists(root, source):
                    errors.append(f"Coverage source does not exist: {source}")
        for name, group in self.groups.items():
            if not GROUP_NAME.fullmatch(name):
                errors.append(f"Invalid group name: {name!r}")
            if group.status not in _STATUSES:
                errors.append(f"Group {name} has an invalid status: {group.status!r}")
            if group.status != "active" and not group.reason:
                errors.append(f"Group {name} needs a reason for status {group.status}")
            if group.execution_group is not None:
                if not GROUP_NAME.fullmatch(group.execution_group):
                    errors.append(f"Group {name} has an invalid execution_group")
                else:
                    target = self.groups.get(group.execution_group)
                    if target is None:
                        errors.append(
                            f"Group {name} execution_group targets unknown group: "
                            f"{group.execution_group}"
                        )
                    elif target.status != "active":
                        errors.append(
                            f"Group {name} execution_group targets {target.status} group: "
                            f"{group.execution_group}"
                        )
            if not group.paths:
                errors.append(f"Group {name} declares no paths")
            for pattern in group.paths:
                if not pattern:
                    errors.append(f"Group {name} declares an empty path")
                elif root is not None and not path_exists(root, pattern):
                    errors.append(f"Group {name} path does not exist: {pattern}")
            for dependency in group.depends_on:
                if dependency not in self.groups:
                    errors.append(f"Group {name} depends on unknown group: {dependency}")
                if dependency == name:
                    errors.append(f"Group {name} depends on itself")
        errors.extend(self._cycle_errors())
        for level in self.levels.values():
            for name in level.required_groups:
                if name not in self.groups:
                    errors.append(f"Level {level.name} requires unknown group: {name}")
                elif self.groups[name].status != "active":
                    errors.append(
                        f"Level {level.name} requires {self.groups[name].status} group: {name}"
                    )
        if root is not None:
            unit_tests = root / "tests/unit_tests"
            if unit_tests.is_dir():
                for directory in sorted(unit_tests.iterdir()):
                    if not directory.is_dir() or directory.name == "__pycache__":
                        continue
                    probe = f"tests/unit_tests/{directory.name}/"
                    if self.matches(probe) is None:
                        errors.append(f"Unit-test directory has no logical group: {probe}")
        return errors

    def _cycle_errors(self) -> list[str]:
        errors: list[str] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str, stack: list[str]) -> None:
            if name in done:
                return
            if name in visiting:
                cycle = " -> ".join(stack + [name])
                errors.append(f"Dependency cycle: {cycle}")
                return
            visiting.add(name)
            for dependency in self.groups[name].depends_on:
                if dependency in self.groups:
                    visit(dependency, stack + [name])
            visiting.remove(name)
            done.add(name)

        for name in sorted(self.groups):
            visit(name, [])
        return errors


@dataclass(frozen=True)
class PlatformUnit:
    """Unit-test execution capabilities of one platform config."""

    name: str
    path: pathlib.Path
    data: dict[str, Any]
    device_types: tuple[str, ...]
    nproc_per_node: int
    groups: dict[str, dict[str, Any]]
    unsupported_groups: dict[str, str] = field(default_factory=dict)
    digest: str = ""


def _matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    if any(character in pattern for character in "*?["):
        # Keep globs inside one directory: tests/unit_tests/test_*.py must not
        # match tests/unit_tests/some_directory/test_file.py.
        return fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")
    return path == pattern


def execution_covers(executed: str, test_path: str) -> bool:
    """Return whether an execution group's paths collect a catalog test path.

    A directory covers everything beneath it; a file or glob only covers the
    path it actually matches.  In particular, a single file must not be treated
    as covering the directory that contains it.
    """
    return any(_matches(test_path, token) for token in executed.split())


@functools.lru_cache(maxsize=8)
def tracked_paths(root: pathlib.Path) -> tuple[str, ...] | None:
    """Return repository-relative paths tracked by git, or None outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return tuple(path for path in result.stdout.split("\0") if path)


def path_exists(root: pathlib.Path, pattern: str) -> bool:
    # CI validates a clean checkout. Checking the git index here keeps local
    # validation honest when ignored or generated directories are present.
    tracked = tracked_paths(root)
    if tracked is not None:
        if pattern.endswith("/"):
            return any(path.startswith(pattern) for path in tracked)
        if any(character in pattern for character in "*?["):
            return any(_matches(path, pattern) for path in tracked)
        return pattern in tracked
    if any(character in pattern for character in "*?["):
        return any(root.glob(pattern))
    target = root / pattern
    return target.is_dir() if pattern.endswith("/") else target.exists()


def load_catalog(path: pathlib.Path) -> Catalog:
    """Load the project contract and build its validated routing catalog in memory."""
    try:
        project, digest = flagcicd_contract.load_project(path)
    except ValueError as error:
        raise CatalogError(f"Invalid FlagCICD project contract {path}: {error}") from error
    data = flagcicd_contract.project_to_catalog(project)
    global_config = data.get("global") or {}
    if not isinstance(global_config, dict):
        raise CatalogError("global must be a mapping")
    documentation_paths = _string_tuple(global_config, "documentation_paths")
    documentation_suffixes = _string_tuple(global_config, "documentation_suffixes")
    shared_paths = _string_tuple(global_config, "shared_paths")
    _string_tuple(global_config, "impact_paths")
    coverage = data.get("coverage") or {}
    if not isinstance(coverage, dict):
        raise CatalogError("coverage must be a mapping")
    coverage_sources = _string_tuple(coverage, "sources", required=True)
    groups: dict[str, Group] = {}
    for name, entry in _mapping(data.get("groups"), "groups").items():
        if not isinstance(name, str) or not GROUP_NAME.fullmatch(name):
            raise CatalogError(f"Invalid group name: {name!r}")
        if not isinstance(entry, dict):
            raise CatalogError(f"Group {name} must be a mapping")
        paths = _string_tuple(entry, "paths", required=True)
        depends_on = _string_tuple(entry, "depends_on")
        status = entry.get("status", "active")
        if status not in _STATUSES:
            raise CatalogError(f"Group {name} has an invalid status: {status!r}")
        reason = entry.get("reason", "")
        if not isinstance(reason, str):
            raise CatalogError(f"Group {name} has an invalid reason")
        if status != "active" and not reason.strip():
            raise CatalogError(f"Group {name} needs a reason for status {status}")
        execution_group = entry.get("execution_group")
        if execution_group is not None and (
            not isinstance(execution_group, str) or not GROUP_NAME.fullmatch(execution_group)
        ):
            raise CatalogError(f"Group {name} has an invalid execution_group")
        groups[name] = Group(
            name=name,
            description=str(entry.get("description", "")),
            status=status,
            reason=reason,
            execution_group=execution_group,
            paths=paths,
            depends_on=depends_on,
        )

    levels: dict[str, Level] = {}
    for name, entry in _mapping(data.get("levels"), "levels").items():
        if not isinstance(name, str) or not name:
            raise CatalogError(f"Invalid level name: {name!r}")
        if not isinstance(entry, dict):
            raise CatalogError(f"Level {name} must be a mapping")
        levels[name] = Level(
            name=name,
            description=str(entry.get("description", "")),
            required_groups=_string_tuple(entry, "required_groups"),
        )

    catalog = Catalog(
        path=path,
        data=data,
        groups=groups,
        levels=levels,
        documentation_paths=documentation_paths,
        documentation_suffixes=documentation_suffixes,
        shared_paths=shared_paths,
        coverage_sources=coverage_sources,
        digest=digest,
    )
    errors = catalog.validate()
    if errors:
        raise CatalogError("; ".join(errors))
    return catalog


def load_platform(config_path: pathlib.Path) -> PlatformUnit:
    """Load the unit-test section of a platform configuration."""
    try:
        raw = config_path.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise CatalogError(f"Cannot read platform config {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise CatalogError(f"Platform config must be a mapping: {config_path}")

    unit = (data.get("test_matrix") or {}).get("unit") or {}
    if not isinstance(unit, dict):
        raise CatalogError(f"test_matrix.unit must be a mapping: {config_path}")
    nproc = unit.get("nproc_per_node")
    if not isinstance(nproc, int) or isinstance(nproc, bool) or nproc <= 0:
        raise CatalogError(
            f"test_matrix.unit.nproc_per_node must be a positive integer: {config_path}"
        )

    groups: dict[str, dict[str, Any]] = {}
    entries = unit.get("groups") or []
    if not isinstance(entries, list):
        raise CatalogError(f"test_matrix.unit.groups must be a list: {config_path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError(f"Each unit group must be a mapping: {config_path}")
        name = entry.get("name")
        if not isinstance(name, str) or not GROUP_NAME.fullmatch(name):
            raise CatalogError(f"Invalid unit group name {name!r}: {config_path}")
        if name in groups:
            raise CatalogError(f"Duplicate unit group {name}: {config_path}")
        if not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise CatalogError(f"Unit group {name} has no test path: {config_path}")
        groups[name] = entry

    unsupported_raw = unit.get("unsupported_groups") or {}
    if not isinstance(unsupported_raw, dict):
        raise CatalogError(f"unsupported_groups must be a mapping: {config_path}")
    unsupported_groups: dict[str, str] = {}
    for name, reason in unsupported_raw.items():
        if not isinstance(name, str) or not GROUP_NAME.fullmatch(name):
            raise CatalogError(f"Invalid unsupported group name {name!r}: {config_path}")
        if not isinstance(reason, str) or not reason.strip():
            raise CatalogError(f"Unsupported group {name} needs a reason: {config_path}")
        unsupported_groups[name] = reason

    device_types = data.get("device_types") or []
    if (
        not isinstance(device_types, list)
        or not device_types
        or any(not isinstance(item, str) or not item.strip() for item in device_types)
        or len(device_types) != len(set(device_types))
    ):
        raise CatalogError(
            f"device_types must be a non-empty list of unique strings: {config_path}"
        )

    return PlatformUnit(
        name=config_path.stem,
        path=config_path,
        data=data,
        device_types=tuple(device_types),
        nproc_per_node=nproc,
        groups=groups,
        unsupported_groups=unsupported_groups,
        digest="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def select_groups(
    catalog: Catalog, changed: list[str], *, selection: str, level: str, unknown_diff: bool = False
) -> dict[str, Any]:
    """Select logical groups and explain every selection.

    Returns selected names, per-group reasons, fallback metadata, direct matches,
    and the paths/conflicts that forced a fallback. The caller materializes the
    result per platform.
    """
    all_groups = set(catalog.groups)
    required = set(catalog.required_groups(level))

    def result(
        selected: Iterable[str],
        reasons: dict[str, list[dict[str, str]]],
        *,
        fallback: bool,
        fallback_reason: str | None,
        unmapped: list[str] | None = None,
        documentation_only: bool = False,
        direct: Iterable[str] = (),
        conflicts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "selected": set(selected),
            "reasons": reasons,
            "required": required,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "unmapped_paths": unmapped or [],
            "documentation_only": documentation_only,
            "direct_groups": sorted(set(direct)),
            "route_conflicts": conflicts or [],
        }

    def everything(
        reason: str,
        *,
        fallback: bool,
        unmapped: list[str] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return result(
            all_groups,
            {name: [{"type": "fallback", "reason": reason}] for name in all_groups},
            fallback=fallback,
            fallback_reason=reason if fallback else None,
            unmapped=unmapped,
            conflicts=conflicts,
        )

    if selection == "all":
        return result(
            all_groups,
            {name: [{"type": "all", "reason": selection}] for name in all_groups},
            fallback=False,
            fallback_reason=None,
        )
    if selection in {"configured", "requested"}:
        configured = {name for name, group in catalog.groups.items() if group.status == "active"}
        selected = configured | required
        reasons: dict[str, list[dict[str, str]]] = {}
        for name in selected:
            if name in required:
                reasons[name] = [{"type": "required", "level": level}]
            else:
                reasons[name] = [
                    {
                        "type": selection,
                        "reason": (
                            "all active groups"
                            if selection == "configured"
                            else "requested platform groups"
                        ),
                    }
                ]
        return result(selected, reasons, fallback=False, fallback_reason=None)
    if selection != "affected":
        raise CatalogError(f"Unknown selection: {selection}")
    if unknown_diff:
        return everything("Cannot read changed files", fallback=True)
    if not changed:
        return everything("No changed-file information", fallback=True)

    code_paths = [path for path in changed if not catalog.is_documentation(path)]
    if not code_paths:
        return result(set(), {}, fallback=False, fallback_reason=None, documentation_only=True)

    direct: dict[str, list[str]] = {}
    shared, unmapped, conflicts, impact = [], [], [], []
    for path in sorted(set(code_paths)):
        matches = catalog.matches_all(path)
        if len(matches) > 1:
            conflicts.append({"path": path, "groups": matches})
        elif matches:
            direct.setdefault(matches[0], []).append(path)
        if catalog.is_shared(path):
            shared.append(path)
        elif not matches:
            unmapped.append(path)
        if any(
            _matches(path, pattern)
            for pattern in catalog.data.get("global", {}).get("impact_paths", [])
        ):
            impact.append(path)

    # Expansion must never erase evidence about directly affected gaps.
    if shared or unmapped or conflicts:
        reasons = []
        reasons.extend(f"Shared path: {path}" for path in shared)
        reasons.extend(f"Unmapped path: {path}" for path in unmapped)
        reasons.extend(
            f"Route conflict: {item['path']} matches {', '.join(item['groups'])}"
            for item in conflicts
        )
        reason = "; ".join(reasons)
        return result(
            all_groups,
            {name: [{"type": "fallback", "reason": reason}] for name in all_groups},
            fallback=True,
            fallback_reason=reason,
            direct=direct,
            unmapped=unmapped,
            conflicts=conflicts,
        )

    selected = catalog.closure(direct) | required
    if impact:
        selected |= {name for name, group in catalog.groups.items() if group.status == "active"}
    reasons: dict[str, list[dict[str, str]]] = {}
    for name, paths in direct.items():
        reasons.setdefault(name, []).extend(
            {"type": "direct", "path": path} for path in sorted(paths)
        )
    for name in selected:
        if name in direct:
            continue
        sources = sorted(source for source in direct if name in catalog.closure([source]))
        if sources:
            reasons[name] = [{"type": "dependency", "source_group": source} for source in sources]
        elif name in required:
            reasons[name] = [{"type": "required", "level": level}]
        else:
            reasons[name] = [{"type": "shared-runtime", "path": path} for path in impact]

    for name in required:
        if not any(reason["type"] == "required" for reason in reasons.setdefault(name, [])):
            reasons[name].append({"type": "required", "level": level})

    return result(
        selected,
        {name: _dedupe(reasons[name]) for name in selected},
        fallback=False,
        fallback_reason=None,
        direct=direct,
    )


def materialize_groups(
    catalog: Catalog,
    platform: PlatformUnit,
    selected: set[str],
    reasons: dict[str, list[dict[str, str]]],
    required: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Split logical groups into executable, planned, unsupported, unresolved, and blocked.

    A selected planned group means its tests are not wired into any platform yet.
    In that case the platform conservatively executes every active group it can
    run, so the plan never narrows coverage around a known gap.
    """
    executable: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    fallback_excluded: list[dict[str, Any]] = []

    def entry_for(
        name: str, group: Group, execution_group: str | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        return {
            "name": name,
            "execution_group": execution_group,
            "status": group.status,
            "reason": group.reason if reason is None else reason,
            "required": name in required,
            "reasons": reasons.get(name, []),
        }

    def exclusion(name: str, status: str, reason: str) -> dict[str, Any]:
        return {"name": name, "platform": platform.name, "status": status, "reason": reason}

    for name in sorted(selected):
        group = catalog.groups[name]
        effective = group.execution_group or name
        if group.status == "planned":
            if name in required:
                blocked.append(entry_for(name, group, reason="required group is planned"))
            else:
                planned.append(entry_for(name, group))
        elif group.status == "unsupported":
            if name in required:
                blocked.append(entry_for(name, group, reason="required group is unsupported"))
            else:
                unsupported.append(entry_for(name, group))
        elif name in platform.unsupported_groups:
            if name in required:
                blocked.append(entry_for(name, group))
            else:
                unsupported.append(entry_for(name, group, reason=platform.unsupported_groups[name]))
        elif effective in platform.groups:
            executable.append(entry_for(name, group, effective))
        elif name in required:
            blocked.append(entry_for(name, group))
        else:
            unresolved.append(entry_for(name, group))

    if planned:
        reason = "planned group selected: " + ", ".join(sorted(entry["name"] for entry in planned))
        executable = []
        for name, group in sorted(catalog.groups.items()):
            if group.status != "active":
                continue
            effective = group.execution_group or name
            if name in platform.unsupported_groups:
                detail = platform.unsupported_groups[name]
                if name in required:
                    fallback_excluded.append(
                        exclusion(
                            name, "blocked", f"required group is declared unsupported: {detail}"
                        )
                    )
                else:
                    fallback_excluded.append(exclusion(name, "unsupported", detail))
                continue
            if effective not in platform.groups:
                if name in required:
                    fallback_excluded.append(
                        exclusion(
                            name, "blocked", f"required group has no execution group: {effective}"
                        )
                    )
                else:
                    fallback_excluded.append(
                        exclusion(name, "unresolved", f"no execution group mapping: {effective}")
                    )
                continue
            executable.append(entry_for(name, group, effective))
        for entry in executable:
            entry["reasons"] = [{"type": "fallback", "reason": reason}]

    return {
        "selected_groups": executable,
        "planned": planned,
        "unsupported": unsupported,
        "unresolved": unresolved,
        "blocked": blocked,
        "fallback_excluded": fallback_excluded,
    }


def _dedupe(reasons: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for reason in reasons:
        key = tuple(sorted(reason.items()))
        if key not in seen:
            seen.add(key)
            result.append(reason)
    return result


def _mapping(value: Any, label: str) -> dict[Any, Any]:
    if not isinstance(value, dict) or not value:
        raise CatalogError(f"{label} must be a non-empty mapping")
    return value


def _string_tuple(entry: dict[str, Any], key: str, *, required: bool = False) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CatalogError(f"{key} must be a list of non-empty strings")
    if required and not value:
        raise CatalogError(f"{key} must not be empty")
    return tuple(value)
