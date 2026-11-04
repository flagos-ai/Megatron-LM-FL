#!/usr/bin/env python3
"""Validate a FlagCICD unit-test plan and its execution evidence.

The planner owns selection.  This module owns the final evidence contract so
that workflow entry points do not each implement a slightly different gate.
It is deliberately independent of GitHub Actions and can be exercised with
local JSON fixtures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Iterable

import flagcicd_contract
import flagcicd_evidence


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON evidence {path}: {error}") from error


def _statuses(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return ["ready"]
    if isinstance(value, str):
        parsed = json.loads(value or '["ready"]')
    else:
        parsed = list(value)
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(status not in {"ready", "partial"} for status in parsed)
    ):
        raise ValueError("required plan statuses must be a non-empty ready/partial list")
    return parsed


def _expected_devices(plan: dict[str, Any]) -> set[tuple[str, str]]:
    devices: set[tuple[str, str]] = set()
    seen_platforms: set[str] = set()
    platforms = plan.get("platforms", [])
    if not isinstance(platforms, list):
        raise ValueError("plan platforms must be a list")
    for platform in platforms:
        if not isinstance(platform, dict):
            raise ValueError("plan contains an invalid platform entry")
        name = platform.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("plan contains an invalid platform name")
        if name in seen_platforms:
            raise ValueError(f"plan contains duplicate platform: {name}")
        seen_platforms.add(name)
        platform_devices = platform.get("devices", [])
        if not isinstance(platform_devices, list) or not platform_devices:
            raise ValueError(f"devices for platform {name} must be a non-empty list")
        for device in platform_devices:
            if not isinstance(device, str) or not device:
                raise ValueError(f"plan contains an invalid device name for {name}")
            if (name, device) in devices:
                raise ValueError(f"devices for platform {name} must be unique")
            devices.add((name, device))
    return devices


def verify_matrix(expected_groups: dict[str, list[str]], matrix: Any) -> list[str]:
    """Check that the materialized workflow matrix matches the plan output."""
    if not isinstance(matrix, list):
        return ["unit matrix must be a JSON array"]
    actual: dict[str, Any] = {}
    errors: list[str] = []
    for entry in matrix:
        if not isinstance(entry, dict):
            errors.append("unit matrix contains a non-object entry")
            continue
        platform = entry.get("platform")
        if not isinstance(platform, str) or not platform:
            errors.append("unit matrix contains an invalid platform name")
            continue
        if platform in actual:
            errors.append(f"duplicate platform in unit matrix: {platform}")
            continue
        actual[platform] = entry.get("groups")
    if actual != expected_groups:
        errors.append("unit matrix does not match expected groups")
    return errors


def verify_plan(
    plan: dict[str, Any],
    *,
    expected_hash: str,
    expected_groups: dict[str, list[str]],
    required_statuses: Iterable[str] | str | None = None,
    allow_catalog_partial: bool | None = None,
    matrix: Any = None,
) -> list[str]:
    """Return evidence errors; an empty list means the plan is executable."""
    errors: list[str] = []
    allowed = _statuses(required_statuses)
    if plan.get("plan_status") == "blocked":
        errors.append("blocked plans cannot be accepted by the execution gate")
    if plan.get("schema_version") != flagcicd_contract.PLAN_SCHEMA:
        errors.append("unsupported plan schema")
    if flagcicd_contract.plan_hash(plan) != plan.get("plan_hash"):
        errors.append("plan content does not match its hash")
    if plan.get("plan_hash") != expected_hash:
        errors.append("plan hash output does not match the uploaded plan")
    if plan.get("plan_status") not in allowed:
        errors.append(f"plan status {plan.get('plan_status')!r} is not allowed by policy {allowed}")
    partial_allowed = (
        allow_catalog_partial
        if allow_catalog_partial is not None
        else bool(plan.get("gate_policy", {}).get("allow_catalog_partial", False))
    )
    if plan.get("catalog_completeness") == "partial" and not partial_allowed:
        errors.append("catalog is partial but the stage policy does not allow it")
    policy = plan.get("gate_policy", {})
    if policy.get("required_plan_status", ["ready"]) != allowed:
        errors.append("uploaded plan gate policy differs from workflow policy")
    if bool(policy.get("allow_catalog_partial", False)) != partial_allowed:
        errors.append("uploaded catalog-partial policy differs from workflow policy")
    actual_groups = plan.get("expected_unit_groups")
    if actual_groups != expected_groups:
        errors.append("expected execution groups differ from the uploaded plan")
    if matrix is not None:
        errors.extend(verify_matrix(expected_groups, matrix))
        identities = {
            item["name"]: item.get("config_identity") for item in plan.get("platforms", [])
        }
        if isinstance(matrix, list):
            for entry in matrix:
                if isinstance(entry, dict) and (
                    not isinstance(entry.get("config_identity"), dict)
                    or entry["config_identity"] != identities.get(entry.get("platform"))
                ):
                    errors.append("matrix configuration identity differs from plan")
    return errors


def verify_attestations(plan: dict[str, Any], attestations: Iterable[dict[str, Any]]) -> list[str]:
    """Return errors for platform/device execution attestations."""
    expected_groups = plan.get("expected_unit_groups", {})
    if not isinstance(expected_groups, dict):
        return ["plan expected_unit_groups must be a JSON object"]
    expected_devices = _expected_devices(plan)
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for data in attestations:
        if not isinstance(data, dict):
            errors.append("unit result attestation must be a JSON object")
            continue
        platform = data.get("platform")
        device = data.get("device")
        if not isinstance(platform, str) or not platform:
            errors.append("unit result attestation has an invalid platform")
            continue
        if not isinstance(device, str) or not device:
            errors.append(f"unit result attestation has an invalid device for {platform}")
            continue
        key = (platform, device)
        if key in seen:
            errors.append(f"duplicate unit result attestation: {key}")
        seen[key] = data
    if set(seen) != expected_devices:
        errors.append(
            "unit result attestations do not match planned platforms/devices: "
            f"missing={sorted(expected_devices - set(seen))}, "
            f"unexpected={sorted(set(seen) - expected_devices)}"
        )
    for key, data in sorted(seen.items()):
        groups = expected_groups.get(key[0])
        if groups is None:
            errors.append(f"attestation references an unplanned platform: {key[0]}")
            continue
        if data.get("plan_hash") != plan.get("plan_hash"):
            errors.append(f"plan hash mismatch for {key}")
        if data.get("expected_groups") != groups:
            errors.append(f"expected group mismatch for {key}")
        if data.get("executed_groups") != groups:
            errors.append(f"executed group mismatch for {key}")
        if data.get("missing_groups") or data.get("unexpected_groups"):
            errors.append(f"group execution mismatch for {key}")
        details = data.get("groups")
        platform_plan = next(item for item in plan["platforms"] if item["name"] == key[0])
        reports = platform_plan.get("junit_reports")
        if (
            not isinstance(reports, list)
            or not reports
            or any(not isinstance(name, str) for name in reports)
        ):
            errors.append(f"missing JUnit report contract for {key}")
            continue
        if (
            not isinstance(details, list)
            or any(not isinstance(item, dict) for item in details)
            or len(details) != len(groups)
            or sorted(item.get("name", "") for item in details) != sorted(groups)
        ):
            errors.append(f"missing, duplicate or unexpected group evidence for {key}")
        else:
            for item in details:
                if type(item.get("exit_code")) is not int or item["exit_code"] != 0:
                    errors.append(f"group did not succeed for {key}: {item.get('name')}")
                errors.extend(
                    f"{key}/{item.get('name')}: {error}"
                    for error in flagcicd_evidence.validate(item.get("evidence"), reports)
                )
        if data.get("status") != "passed":
            errors.append(f"unit execution did not pass for {key}: {data.get('status')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--expected-groups", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--expected-hash", required=True)
    parser.add_argument("--attestation-root", type=pathlib.Path, required=True)
    parser.add_argument("--required-status", default='["ready"]')
    parser.add_argument("--allow-catalog-partial", action="store_true")
    args = parser.parse_args()
    try:
        plan = _read_json(args.plan)
        expected_groups = json.loads(args.expected_groups)
        matrix = json.loads(args.matrix)
        if not isinstance(plan, dict) or not isinstance(expected_groups, dict):
            raise ValueError("plan and expected groups must be JSON objects")
        errors = verify_plan(
            plan,
            expected_hash=args.expected_hash,
            expected_groups=expected_groups,
            required_statuses=args.required_status,
            allow_catalog_partial=args.allow_catalog_partial,
            matrix=matrix,
        )
        attestations = [
            _read_json(path) for path in args.attestation_root.rglob("unit-results.json")
        ]
        errors.extend(verify_attestations(plan, attestations))
    except (OSError, ValueError, TypeError) as error:
        print(f"::error::{error}")
        return 1
    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1
    print(f"Unit plan verified across {len(attestations)} platform/device results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
