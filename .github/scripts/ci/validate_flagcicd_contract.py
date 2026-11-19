#!/usr/bin/env python3
"""Validate the FlagCICD project contract and logical group definitions."""

from __future__ import annotations

import argparse
import pathlib
import sys

import flagcicd_contract
import unit_group_routing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    args = parser.parse_args()
    root = pathlib.Path.cwd()
    try:
        project, digest = flagcicd_contract.load_project(root / args.project)
        catalog = unit_group_routing.load_catalog(root / args.project)
        errors = catalog.validate(root)
    except (OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"FlagCICD project contract: {project['repository']}")
    print(f"Project hash: sha256:{digest}")
    print(f"Stages: {', '.join(project['stages'])}")
    print(f"Levels: {', '.join(project['levels'])}")
    print(f"Logical groups: {len(project['test_groups'])}")
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    print("FlagCICD project contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
