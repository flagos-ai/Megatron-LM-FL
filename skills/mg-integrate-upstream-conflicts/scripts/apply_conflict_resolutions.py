#!/usr/bin/env python3
"""Compose reviewed conflict resolutions without editing candidate evidence."""

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
REQUIRED = ["owner", "fork_invariant", "upstream_change", "affected_symbols", "resolution_strategy", "acceptance_tests", "reviewer", "approval_status", "reason"]


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"invalid JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = load(args.candidate)
    overrides = load(args.overrides).get("resolutions", [])
    by_id = {}
    for item in overrides:
        delta_id = item.get("delta_id")
        if not delta_id or delta_id in by_id:
            raise SystemExit("missing or duplicate resolution delta_id")
        missing = [key for key in REQUIRED if not item.get(key)]
        if missing:
            raise SystemExit(f"{delta_id} missing resolution fields: {', '.join(missing)}")
        if item["resolution_strategy"] not in ALLOWED:
            raise SystemExit(f"{delta_id} has invalid strategy")
        candidate_row = next((row for row in candidate.get("rows", []) if row.get("delta_id") == delta_id), None)
        if candidate_row and item["resolution_strategy"] not in COMPATIBLE.get(candidate_row.get("action"), set()):
            raise SystemExit(f"{delta_id} strategy is incompatible with approved action {candidate_row.get('action')}")
        by_id[delta_id] = item
    candidate_ids = {row["delta_id"] for row in candidate.get("rows", [])}
    unknown = sorted(set(by_id) - candidate_ids)
    if unknown:
        raise SystemExit("unknown resolution IDs: " + ", ".join(unknown))
    rows = []
    for source in candidate.get("rows", []):
        row = dict(source)
        override = by_id.get(row["delta_id"])
        if override:
            for key in REQUIRED + ["skill_gap_ids", "blocked_test_owner"]:
                if key in override:
                    row[key] = override[key]
            row["status"] = "planned" if override["approval_status"] == "approved" else "awaiting-approval"
        rows.append(row)
    result = dict(candidate)
    result["rows"] = rows
    result["summary"] = dict(candidate.get("summary", {}))
    result["summary"]["unresolved"] = sum(row["status"] == "unresolved" for row in rows)
    result["summary"]["awaiting_approval"] = sum(row["status"] == "awaiting-approval" for row in rows)
    result["summary"]["planned"] = sum(row["status"] == "planned" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
