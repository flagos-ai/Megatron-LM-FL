#!/usr/bin/env python3
"""Validate a reviewed conflict ledger and emit a machine-readable gate."""

import argparse
import json
from pathlib import Path

ALLOWED = {"TARGET_COVERS", "TARGET_PLUS_FL_DELTA", "REPLAY_FL_NARROW", "REDESIGN_APPROVED"}
COMPATIBLE = {
    "UPSTREAM_ONLY": {"TARGET_COVERS"},
    "UPSTREAM_COVERS": {"TARGET_COVERS"},
    "UPSTREAM_PLUS_FL_DELTA": {"TARGET_PLUS_FL_DELTA"},
    "REPLAY_FL": {"REPLAY_FL_NARROW"},
    "REDESIGN": {"REDESIGN_APPROVED"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.ledger.read_text())
    errors = []
    rows = data.get("rows", [])
    ids = [row.get("delta_id") for row in rows]
    if data.get("schema_version") != "1.0": errors.append("unsupported schema_version")
    if len(ids) != len(set(ids)) or any(not value for value in ids): errors.append("missing or duplicate delta IDs")
    if len(rows) != data.get("summary", {}).get("both_changed"): errors.append("both-changed coverage mismatch")
    for row in rows:
        delta_id = row.get("delta_id")
        for field in ["owner", "fork_invariant", "upstream_change", "reviewer"]:
            if not row.get(field): errors.append(f"{delta_id} missing {field}")
        if not row.get("affected_symbols"): errors.append(f"{delta_id} missing affected_symbols")
        if not row.get("acceptance_tests"): errors.append(f"{delta_id} missing acceptance_tests")
        if row.get("resolution_strategy") not in ALLOWED: errors.append(f"{delta_id} missing or invalid strategy")
        elif row.get("resolution_strategy") not in COMPATIBLE.get(row.get("action"), set()): errors.append(f"{delta_id} strategy is incompatible with approved action")
        if row.get("status") != "planned": errors.append(f"{delta_id} is not planned")
        if (row.get("priority") == "P0" or row.get("resolution_strategy") == "REDESIGN_APPROVED") and row.get("approval_status") != "approved":
            errors.append(f"{delta_id} requires explicit approval")
        if any(test.get("status") == "blocked" and not test.get("external_owner") for test in row.get("acceptance_tests", []) if isinstance(test, dict)):
            errors.append(f"{delta_id} blocked test lacks external owner")
    result = {"valid": not errors, "row_count": len(rows), "errors": errors}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if errors: raise SystemExit(2)


if __name__ == "__main__":
    main()
