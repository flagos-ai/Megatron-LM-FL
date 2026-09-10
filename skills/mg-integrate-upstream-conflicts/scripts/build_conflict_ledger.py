#!/usr/bin/env python3
"""Build a read-only candidate ledger for every both-changed fork path."""

import argparse
import json
import re
import subprocess
from pathlib import Path

HEADER = re.compile(r"^  (base|our|their)\s+\d+\s+[0-9a-f]+\s+(.+)$")
SECTION = re.compile(r"^(changed in both|added in both|removed in both|added in remote|added in local)$")


def load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"invalid {label}: top level must be an object")
    return value


def parse_merge_tree(text: str) -> dict[str, dict]:
    result = {}
    current_kind = None
    block = []

    def flush() -> None:
        nonlocal block
        paths = {}
        for line in block:
            match = HEADER.match(line)
            if match:
                paths[match.group(1)] = match.group(2)
        path = paths.get("our") or paths.get("their") or paths.get("base")
        if path:
            result[path] = {
                "merge_tree_section": current_kind,
                "textual_status": "textual-conflict" if any("<<<<<<< .our" in line for line in block) else "auto-merged-semantic-risk",
            }
        block = []

    for line in text.splitlines():
        match = SECTION.match(line)
        if match:
            flush()
            current_kind = match.group(1)
            block = [line]
        elif current_kind:
            block.append(line)
    flush()
    return result


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo), *args], text=True, encoding="utf-8", errors="replace", capture_output=True)
    if check and proc.returncode:
        raise SystemExit(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = load(args.inventory, "inventory")
    decisions = load(args.decisions, "effective decisions")
    routing = load(args.routing, "domain routing")
    refs = inventory.get("resolved_refs", {})
    if decisions.get("refs") != refs or routing.get("refs") != refs:
        raise SystemExit("artifact refs mismatch")
    required_refs = ["sync_tree_base", "fork", "target"]
    if any(not refs.get(name) for name in required_refs):
        raise SystemExit("missing sync_tree_base, fork, or target SHA")
    repo = args.repo.resolve()
    for name in required_refs:
        resolved = run_git(repo, "rev-parse", "--verify", f"{refs[name]}^{{commit}}").stdout.strip()
        if resolved != refs[name]:
            raise SystemExit(f"{name} does not resolve to the recorded SHA")

    changes = [row for row in inventory.get("fork_changes", []) if row.get("both_changed")]
    if len(changes) != inventory.get("both_changed_count"):
        raise SystemExit("inventory both-changed count mismatch")
    decision_by_id = {row.get("id"): row for row in decisions.get("decisions", [])}
    route_by_id = {row.get("delta_id"): row for row in routing.get("routes", [])}
    if len(decision_by_id) != len(decisions.get("decisions", [])) or len(route_by_id) != len(routing.get("routes", [])):
        raise SystemExit("duplicate decision or route IDs")

    merge = run_git(repo, "merge-tree", refs["sync_tree_base"], refs["fork"], refs["target"], check=False)
    if merge.returncode not in (0, 1):
        raise SystemExit(merge.stderr.strip() or "git merge-tree failed")
    parsed = parse_merge_tree(merge.stdout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "merge-tree.txt").write_text(merge.stdout)

    rows = []
    for change in sorted(changes, key=lambda row: row["id"]):
        delta_id = change["id"]
        decision = decision_by_id.get(delta_id)
        route = route_by_id.get(delta_id)
        if not decision or not route:
            raise SystemExit(f"missing decision or route for {delta_id}")
        merge_info = parsed.get(change["path"], {"merge_tree_section": None, "textual_status": "clean-merge-semantic-risk"})
        rows.append({
            "delta_id": delta_id,
            "path": change["path"],
            "priority": change.get("priority"),
            "action": decision.get("effective_action"),
            "primary_domain": route.get("primary_domain"),
            "secondary_domains": route.get("secondary_domains", []),
            "merge_tree_section": merge_info["merge_tree_section"],
            "textual_status": merge_info["textual_status"],
            "blob_evidence": decision.get("evidence", {}),
            "owner": None,
            "fork_invariant": None,
            "upstream_change": None,
            "affected_symbols": [],
            "resolution_strategy": None,
            "acceptance_tests": [],
            "reviewer": None,
            "approval_status": "unapproved",
            "skill_gap_ids": [],
            "status": "unresolved",
        })
    ledger = {
        "schema_version": "1.0",
        "refs": refs,
        "source_artifacts": {
            "inventory": str(args.inventory.resolve()),
            "decisions": str(args.decisions.resolve()),
            "routing": str(args.routing.resolve()),
        },
        "rows": rows,
        "summary": {
            "both_changed": len(rows),
            "textual_conflicts": sum(row["textual_status"] == "textual-conflict" for row in rows),
            "semantic_risks": sum(row["textual_status"] != "textual-conflict" for row in rows),
            "unresolved": len(rows),
        },
    }
    (args.output / "candidate-conflict-ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
