#!/usr/bin/env python3
"""Validate a classifier bundle without modifying the repository or bundle."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True)
    if proc.returncode:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.rstrip("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, label: str, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"invalid {label}: top level must be an object")
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--effective-decisions", type=Path)
    parser.add_argument("--fork-ref", required=True)
    parser.add_argument("--target-ref", required=True)
    args = parser.parse_args()

    errors: list[str] = []
    inventory_path = args.bundle / "inventory.json"
    data = load_json(inventory_path, "inventory", errors)
    approval = load_json(args.approval, "approval file", errors)
    if data is None or approval is None:
        print(json.dumps({"valid": False, "errors": errors}, indent=2, sort_keys=True))
        raise SystemExit(2)

    if data.get("schema_version") != "1.0":
        errors.append("unsupported schema_version")
    refs = data.get("resolved_refs", {})
    fork = git(args.repo.resolve(), "rev-parse", "--verify", f"{args.fork_ref}^{{commit}}")
    target = git(args.repo.resolve(), "rev-parse", "--verify", f"{args.target_ref}^{{commit}}")
    if refs.get("fork") != fork:
        errors.append("fork SHA mismatch")
    if refs.get("target") != target:
        errors.append("target SHA mismatch")
    if data.get("category_counts", {}).get("unclassified", 0):
        errors.append("unclassified paths remain")

    decisions_path = args.effective_decisions or (args.bundle / "effective-decisions.json")
    decisions_data = load_json(decisions_path, "effective decisions", errors)
    decision_ids: list[str] = []
    if decisions_data:
        decision_rows = decisions_data.get("decisions", [])
        decision_ids = [row.get("id") for row in decision_rows]
        if decisions_data.get("refs") != refs:
            errors.append("effective decision refs mismatch")
        if len(decision_rows) != data.get("fork_change_count") or len(decision_ids) != len(set(decision_ids)):
            errors.append("effective decision coverage mismatch")
        if decisions_data.get("manual_count") or any(row.get("effective_action") == "MANUAL" for row in decision_rows):
            errors.append("effective decisions are incomplete")

    routing_path = args.bundle / "domain-routing.json"
    routing = load_json(routing_path, "domain routing", errors)
    if routing:
        routes = routing.get("routes", [])
        route_ids = [row.get("delta_id") for row in routes]
        if routing.get("schema_version") != "1.0":
            errors.append("unsupported domain routing schema_version")
        if routing.get("refs") != refs:
            errors.append("domain routing refs mismatch")
        if set(route_ids) != set(decision_ids) or len(route_ids) != len(set(route_ids)):
            errors.append("domain routing coverage mismatch")
        for row in routes:
            secondary = row.get("secondary_domains", [])
            secondary_handlers = row.get("secondary_handlers", {})
            if not row.get("primary_domain") or not row.get("handler_skill"):
                errors.append(f"unrouted delta: {row.get('delta_id')}")
            if not isinstance(secondary, list) or row.get("primary_domain") in secondary or len(secondary) != len(set(secondary)):
                errors.append(f"invalid secondary domains: {row.get('delta_id')}")
            if set(secondary_handlers) != set(secondary) or any(not secondary_handlers.get(domain) for domain in secondary):
                errors.append(f"missing secondary handlers: {row.get('delta_id')}")
        summary = routing.get("summary", {})
        if summary.get("unrouted", 0):
            errors.append("unrouted deltas remain")
        if summary.get("conflicting_routes", 0):
            errors.append("conflicting domain routes remain")

    gaps_path = args.bundle / "skill-gap-ledger.json"
    gap_ledger = load_json(gaps_path, "skill gap ledger", errors)
    if gap_ledger:
        gaps = gap_ledger.get("gaps", [])
        gap_ids = [row.get("gap_id") for row in gaps]
        if gap_ledger.get("schema_version") != "1.0":
            errors.append("unsupported skill gap schema_version")
        if gap_ledger.get("refs") != refs:
            errors.append("skill gap refs mismatch")
        if len(gap_ids) != len(set(gap_ids)) or any(not gap_id for gap_id in gap_ids):
            errors.append("invalid or duplicate skill gap IDs")
        unresolved = [row.get("gap_id") for row in gaps if row.get("blocking") and row.get("status") != "resolved"]
        if unresolved:
            errors.append("unresolved blocking skill gaps: " + ", ".join(str(value) for value in unresolved))

    approved = approval.get("artifacts", {})
    artifacts = {
        "inventory.json": inventory_path,
        "effective-decisions.json": decisions_path,
        "domain-routing.json": routing_path,
        "skill-gap-ledger.json": gaps_path,
    }
    for name, path in artifacts.items():
        if path.is_file() and approved.get(name) != sha256(path):
            errors.append(f"{name} hash is not approved")
    if approval.get("fork") != fork or approval.get("target") != target:
        errors.append("approval refs mismatch")
    if approval.get("decision") != "approved":
        errors.append("bundle is not approved")

    required = [
        "fork-changes.tsv", "both-changed.tsv", "override-manifest.json",
        "flagscale-block-manifest.json", "platform-manifest.json",
        "upstream-provenance.json", "manual-decisions.md",
        "mg-fl-design-baseline.md", "domain-routing.json", "skill-gap-ledger.json",
    ]
    errors.extend(f"missing {name}" for name in required if not (args.bundle / name).is_file())
    result = {"valid": not errors, "fork": fork, "target": target, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
