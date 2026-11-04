#!/usr/bin/env python3
"""Validate the logical unit-group catalog against the repository and platforms."""

from __future__ import annotations

import argparse
import pathlib
import sys

import generate_matrix
import unit_group_routing


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path


def _path_conflict_errors(root: pathlib.Path, catalog: unit_group_routing.Catalog) -> list[str]:
    """Report tracked paths that belong to more than one logical group."""
    tracked = unit_group_routing.tracked_paths(root)
    if tracked is None:
        return []
    conflicts: dict[tuple[str, ...], list[str]] = {}
    for path in tracked:
        groups = tuple(catalog.matches_all(path))
        if len(groups) > 1:
            conflicts.setdefault(groups, []).append(path)
    errors = []
    for groups, paths in sorted(conflicts.items()):
        sample = ", ".join(paths[:3])
        if len(paths) > 3:
            sample += f" (+{len(paths) - 3} more)"
        errors.append(
            "Tracked path matches multiple logical groups " f"({', '.join(groups)}): {sample}"
        )
    return errors


def collect_errors(
    root: pathlib.Path,
    configs_dir: pathlib.Path,
    catalog: unit_group_routing.Catalog,
    platforms: dict[str, unit_group_routing.PlatformUnit],
    level: str,
) -> list[str]:
    errors = catalog.validate(root)
    errors.extend(_path_conflict_errors(root, catalog))
    required = set(catalog.required_groups(level))
    owners: dict[str, set[str]] = {}
    for name, platform in platforms.items():
        for group_name, group in catalog.groups.items():
            effective = group.execution_group or group_name
            if group_name in platform.unsupported_groups:
                continue
            if effective in platform.groups:
                owners.setdefault(group_name, set()).add(name)
                if group.status == "active":
                    executed = platform.groups[effective]["path"]
                    for test_path in group.paths:
                        if _is_test_path(test_path) and not unit_group_routing.execution_covers(
                            executed, test_path
                        ):
                            errors.append(
                                f"Active group {group_name!r} test path {test_path!r} "
                                f"is not executed by platform {name}"
                            )
        for group_name in platform.groups:
            if group_name not in catalog.groups:
                errors.append(
                    f"Platform {name} executes group {group_name!r} that is absent from the catalog"
                )
        for group_name, entry in platform.groups.items():
            for token in entry["path"].split():
                if not unit_group_routing.path_exists(root, token):
                    errors.append(
                        f"Platform {name} group {group_name} path does not exist: {token}"
                    )
        for group_name in platform.unsupported_groups:
            group = catalog.groups.get(group_name)
            if group is None:
                errors.append(
                    f"Platform {name} declares unsupported group {group_name!r} "
                    "that is absent from the catalog"
                )
            elif group.status != "active":
                errors.append(
                    f"Platform {name} declares {group.status} group {group_name!r} as unsupported"
                )
            elif group_name in required:
                errors.append(
                    f"Platform {name} declares required group {group_name!r} as unsupported"
                )
            elif (group.execution_group or group_name) in platform.groups:
                executed = platform.groups[group.execution_group or group_name]["path"]
                test_paths = [path for path in group.paths if _is_test_path(path)]
                if test_paths and all(
                    unit_group_routing.execution_covers(executed, path) for path in test_paths
                ):
                    errors.append(
                        f"Platform {name} declares unsupported group {group_name!r} "
                        "but its execution group already covers every test path"
                    )

    for group_name, group in catalog.groups.items():
        group_owners = owners.get(group_name, set())
        if group.status == "active" and not group_owners:
            errors.append(f"Active group {group_name!r} has no platform execution mapping")
        if group.status != "active" and group_owners:
            errors.append(
                f"{group.status.title()} group {group_name!r} has platform execution mappings: "
                + ", ".join(sorted(group_owners))
            )
        if group.status != "active":
            continue
        for name, platform in platforms.items():
            if name in group_owners or group_name in platform.unsupported_groups:
                continue
            errors.append(
                f"Active group {group_name!r} has no execution mapping or unsupported "
                f"declaration on platform {name}"
            )
    for group_name in required:
        group = catalog.groups[group_name]
        effective = group.execution_group or group_name
        missing = sorted(
            name for name, platform in platforms.items() if effective not in platform.groups
        )
        if missing:
            errors.append(
                f"Required group {group_name!r} (execution group {effective!r}) "
                f"is not executable on: {', '.join(missing)}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument("--level", default="L1")
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    configs_dir = root / args.configs_dir
    try:
        catalog = unit_group_routing.load_catalog(root / args.project)
        rules = generate_matrix._load_watch_rules(configs_dir)
        platforms = {
            name: unit_group_routing.load_platform(configs_dir / f"{name}.yml")
            for name in sorted(rules)
        }
        errors = collect_errors(root, configs_dir, catalog, platforms, args.level)
    except (unit_group_routing.CatalogError, OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    executable: dict[str, list[str]] = {}
    for name, platform in platforms.items():
        for group_name, group in catalog.groups.items():
            if group_name in platform.unsupported_groups:
                continue
            if (group.execution_group or group_name) in platform.groups:
                executable.setdefault(group_name, []).append(name)

    print(
        f"Unit group catalog: {len(catalog.groups)} logical groups, "
        f"{len(platforms)} platforms, required level {args.level}"
    )
    print(f"Coverage sources: {', '.join(catalog.coverage_sources)}")
    for group_name in sorted(catalog.groups):
        group = catalog.groups[group_name]
        owners = executable.get(group_name, [])
        if group.status != "active":
            print(f"  {group_name}: {group.status} ({group.reason})")
        elif owners:
            print(f"  {group_name}: {', '.join(owners)}")
        else:
            print(f"  {group_name}: active without a platform execution mapping")

    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    print("Unit group catalog validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
