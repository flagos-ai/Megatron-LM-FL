#!/usr/bin/env python3
"""Resolve one platform's execution groups into the test paths they run.

The scope of a unit group lives in ``.flagcicd/project.yaml`` only, split into
``paths`` (what triggers the group) and ``test_targets`` (what it runs). A
platform config lists no groups: every active group runs unless the platform
declares it in ``unsupported_groups``. This script turns the contract into the
pytest paths handed to ``run_unit_groups.py``, including the plan's ephemeral
``planned-<group>`` execution groups and their explicitly selected files.

Group order follows the catalog, so the execution order has one owner.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

import unit_group_routing

TEMPORARY_GROUP = re.compile(r"planned-[A-Za-z0-9_-]+")


class ResolutionError(ValueError):
    """Raised when the platform declaration and the routing plan disagree."""


def _temporary_groups(root: pathlib.Path, identity: dict) -> dict[str, str]:
    """Return the plan's ephemeral groups and their selected test files."""
    temporary: dict[str, str] = {}
    planned = identity.get("planned_execution", [])
    if not isinstance(planned, list):
        raise ResolutionError("planned_execution must be a list")
    for entry in planned:
        if not isinstance(entry, dict):
            raise ResolutionError("Each planned execution entry must be a mapping")
        name, paths = entry.get("name"), entry.get("paths")
        if not isinstance(name, str) or not TEMPORARY_GROUP.fullmatch(name) or name in temporary:
            raise ResolutionError(f"Invalid or duplicate temporary group: {name!r}")
        if not isinstance(paths, list) or not paths:
            raise ResolutionError(f"Temporary group {name} requires changed test files")
        for path in paths:
            if (
                not isinstance(path, str)
                or not re.fullmatch(r"tests/unit_tests/[A-Za-z0-9_./-]+", path)
                or ".." in pathlib.PurePosixPath(path).parts
                or not pathlib.PurePosixPath(path).match("test_*.py")
                or not (root / path).is_file()
                or not (root / path).resolve().is_relative_to(root.resolve())
            ):
                raise ResolutionError(f"Invalid temporary test path for {name}: {path!r}")
        temporary[name] = " ".join(paths)
    return temporary


def resolve(
    catalog: unit_group_routing.Catalog,
    platform: unit_group_routing.PlatformUnit,
    expected: list[str] | None = None,
    identity: dict | None = None,
) -> list[dict[str, str]]:
    """Return the ``{"name", "path"}`` entries this platform must execute."""
    root = catalog.path.parent.parent
    groups: dict[str, str] = {}
    for name, paths in unit_group_routing.execution_groups(catalog, platform).items():
        if not paths:
            raise ResolutionError(
                f"Platform {platform.name} group {name!r} resolves to no test path"
            )
        groups[name] = " ".join(paths)
    for name, paths in _temporary_groups(root, identity or {}).items():
        if name in groups:
            raise ResolutionError(f"Reserved temporary execution group: {name}")
        groups[name] = paths

    if expected:
        if len(set(expected)) != len(expected):
            raise ResolutionError("Expected unit groups must not contain duplicates")
        missing = sorted(set(expected) - set(groups))
        if missing:
            raise ResolutionError(
                "Expected unit groups are absent from the catalog execution groups: "
                + ", ".join(missing)
            )
        groups = {name: path for name, path in groups.items() if name in set(expected)}
        if not groups:
            raise ResolutionError("No configured unit group matches the routing plan")
    return [{"name": name, "path": path} for name, path in groups.items()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-config", required=True)
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument(
        "--expected",
        default=os.environ.get("EXPECTED_UNIT_GROUPS", ""),
        help="JSON array of groups selected by the routing plan; empty means every group",
    )
    parser.add_argument(
        "--identity",
        default=os.environ.get("CONFIG_IDENTITY", ""),
        help="JSON object with the plan's config identity",
    )
    args = parser.parse_args()

    try:
        catalog = unit_group_routing.load_catalog(pathlib.Path(args.project))
        platform = unit_group_routing.load_platform(pathlib.Path(args.platform_config))
        expected = json.loads(args.expected) if args.expected else None
        if expected is not None and (
            not isinstance(expected, list)
            or any(not isinstance(name, str) or not name for name in expected)
        ):
            raise ResolutionError("expected unit groups must be a JSON array of non-empty strings")
        identity = json.loads(args.identity) if args.identity else {}
        if not isinstance(identity, dict):
            raise ResolutionError("config identity must be a JSON object")
        groups = resolve(catalog, platform, expected, identity)
    except (unit_group_routing.CatalogError, ResolutionError, OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    print(json.dumps(groups, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
