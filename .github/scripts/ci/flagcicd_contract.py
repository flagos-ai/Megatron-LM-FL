#!/usr/bin/env python3
"""Load and validate the repository-facing FlagCICD contract.

The project contract owns stage, level, and logical-group semantics.
Platform YAML remains the adapter for images, runners, setup, and concrete
execution commands. The project file is the only source of routing policy;
logical groups are indexed in memory for routing and validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from typing import Any

import yaml

PROJECT_SCHEMA = "flagcicd.project/v1"
PLAN_SCHEMA = 1


class ContractError(ValueError):
    """Raised when the project contract is malformed or inconsistent."""


def _read(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"Cannot read FlagCICD project contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"FlagCICD project contract must be a mapping: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _check_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(
            f"{context} has unsupported fields: {', '.join(sorted(map(str, unknown)))}"
        )


def load_project(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    """Load the project contract and perform schema-level validation."""
    project, digest = _read(path)
    _check_keys(
        project,
        {
            "schema_version",
            "repository",
            "plan_schema",
            "stages",
            "levels",
            "global",
            "coverage",
            "test_groups",
            "gate_policy",
        },
        "project",
    )
    if project.get("schema_version") != PROJECT_SCHEMA:
        raise ContractError(
            f"Unsupported project contract schema: {project.get('schema_version')!r}"
        )
    if not isinstance(project.get("repository"), str) or not project["repository"].strip():
        raise ContractError("repository must be a non-empty string")
    if project.get("plan_schema") != PLAN_SCHEMA:
        raise ContractError(f"plan_schema must be {PLAN_SCHEMA!r}")
    for key in ("stages", "levels", "test_groups", "coverage", "gate_policy"):
        if key not in project:
            raise ContractError(f"Project contract is missing {key}")
    if not isinstance(project["stages"], dict) or not project["stages"]:
        raise ContractError("stages must be a non-empty mapping")
    if not isinstance(project["levels"], dict) or not project["levels"]:
        raise ContractError("levels must be a non-empty mapping")
    for key in ("global", "coverage"):
        if not isinstance(project.get(key), dict):
            raise ContractError(f"{key} must be a mapping")
    _check_keys(
        project["global"],
        {"documentation_paths", "documentation_suffixes", "shared_paths", "impact_paths"},
        "global",
    )
    _check_keys(project["stages"], {"pr", "branch", "nightly", "manual"}, "stages")
    _check_keys(project["levels"], {"L1", "L2"}, "levels")
    groups = project["test_groups"]
    if not isinstance(groups, list) or not groups:
        raise ContractError("test_groups must be a non-empty list")
    names: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str):
            raise ContractError("every test group must have a string name")
        _check_keys(
            group,
            {"name", "description", "status", "reason", "paths", "depends_on", "execution_group"},
            "test group",
        )
        name = group["name"]
        if name in names:
            raise ContractError(f"duplicate test group: {name}")
        names.add(name)
        status = group.get("status", "active")
        if status not in {"active", "planned", "unsupported"}:
            raise ContractError(f"invalid status for test group {name}: {status!r}")
        if status != "active" and not str(group.get("reason", "")).strip():
            raise ContractError(f"test group {name} needs a reason for status {status}")
        if not isinstance(group.get("paths"), list) or not group["paths"]:
            raise ContractError(f"test group {name} must declare paths")
        if not isinstance(group.get("depends_on", []), list):
            raise ContractError(f"test group {name} depends_on must be a list")
        for dependency in group.get("depends_on", []):
            if dependency not in names and dependency not in {item.get("name") for item in groups}:
                raise ContractError(f"test group {name} depends on unknown group {dependency}")
    for level, config in project["levels"].items():
        if not isinstance(config, dict):
            raise ContractError(f"level {level} must be a mapping")
        _check_keys(config, {"purpose", "required_groups"}, f"level {level}")
        required = config.get("required_groups", [])
        if not isinstance(required, list) or any(item not in names for item in required):
            raise ContractError(f"level {level} has an unknown required group")
    for stage, config in project["stages"].items():
        if not isinstance(config, dict):
            raise ContractError(f"stage {stage} must be a mapping")
        _check_keys(config, {"selection", "default_level", "fixed_level"}, f"stage {stage}")
        selection = config.get("selection")
        if selection not in {"affected", "configured", "requested", "all"}:
            raise ContractError(f"stage {stage} has invalid selection {selection!r}")
        level = config.get("default_level")
        if level not in project["levels"]:
            raise ContractError(f"stage {stage} references unknown level {level!r}")
        if "fixed_level" in config and not isinstance(config["fixed_level"], bool):
            raise ContractError(f"stage {stage} fixed_level must be a boolean")
    if not isinstance(project["gate_policy"], dict):
        raise ContractError("gate_policy must be a mapping")
    for stage in project["stages"]:
        policy = project["gate_policy"].get(stage)
        if not isinstance(policy, dict):
            raise ContractError(f"gate_policy is missing stage {stage}")
        statuses = policy.get("required_plan_status")
        if (
            not isinstance(statuses, list)
            or not statuses
            or any(status not in {"ready", "partial"} for status in statuses)
        ):
            raise ContractError(
                f"gate_policy for {stage} has invalid required_plan_status; "
                "blocked plans cannot be accepted"
            )
        if not isinstance(policy.get("allow_catalog_partial"), bool):
            raise ContractError(f"gate_policy for {stage} allow_catalog_partial must be a boolean")
    # Collection and aggregation currently support this fixed source boundary.
    if project["coverage"].get("sources") != ["megatron/plugin/", "megatron/core/"]:
        raise ContractError("coverage.sources must be megatron/plugin/ and megatron/core/")
    return project, digest


def project_to_catalog(project: dict[str, Any]) -> dict[str, Any]:
    """Build the in-memory routing catalog indexed by logical group name."""
    groups: dict[str, dict[str, Any]] = {}
    for entry in project["test_groups"]:
        group = copy.deepcopy(entry)
        group.pop("name", None)
        groups[entry["name"]] = group
    return {
        "schema_version": 1,
        "global": copy.deepcopy(project.get("global", {})),
        "coverage": copy.deepcopy(project.get("coverage", {})),
        "levels": copy.deepcopy(project["levels"]),
        "groups": groups,
    }


def content_hash(value: Any) -> str:
    """Hash JSON data independently of object ordering and presentation."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def plan_hash(plan: dict[str, Any]) -> str:
    """Hash every plan field except its own digest."""
    return content_hash({key: value for key, value in plan.items() if key != "plan_hash"})
