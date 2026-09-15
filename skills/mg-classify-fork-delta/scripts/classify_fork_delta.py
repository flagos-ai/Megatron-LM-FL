#!/usr/bin/env python3
"""Create a deterministic, read-only Megatron-LM-FL fork delta inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = "1.0"


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True)
    if check and proc.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.rstrip("\n")


def resolve(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def changes(repo: Path, left: str, right: str) -> list[dict]:
    rows = []
    raw = git(repo, "diff", "--name-status", "--find-renames", "--find-copies", f"{left}..{right}")
    for line in raw.splitlines():
        fields = line.split("\t")
        status = fields[0]
        row = {"status": status, "path": fields[-1]}
        if status.startswith(("R", "C")) and len(fields) == 3:
            row["old_path"] = fields[1]
        rows.append(row)
    return rows


def category(path: str, vendor_roots: set[str] | None = None, vendor_tokens: set[str] | None = None) -> str:
    vendor_roots = vendor_roots or set()
    vendor_tokens = {x.lower() for x in (vendor_tokens or set())}
    vendor_path = any(token in path.lower() for token in vendor_tokens)
    name = Path(path).name
    rules = [
        ("plugin-platform", path.startswith("megatron/plugin/platform/")),
        ("plugin-vendor", any(path.lower().startswith(f"megatron/plugin/{vendor.lower()}/") for vendor in vendor_roots)),
        ("plugin-feature", path.startswith(("megatron/plugin/dualpipev/", "megatron/plugin/hetero/", "megatron/plugin/dsa_kernel/", "megatron/plugin/optimizer/"))),
        ("plugin-contract", path.startswith("megatron/plugin/")),
        ("invasive-runtime", path.startswith("megatron/core/")),
        ("training", path.startswith("megatron/") or name.startswith(("pretrain_", "train_", "model_provider"))),
        ("cicd-vendor", path.startswith(".github/") and vendor_path),
        ("cicd-common", path.startswith((".github/", ".gitlab/")) or path == ".gitlab-ci.yml"),
        ("tests-vendor", path.startswith("tests/") and (vendor_path or "/plugin/" in path.lower())),
        ("tests-common", path.startswith("tests/")),
        ("build-packaging", name in {"setup.py", "pyproject.toml", "MANIFEST.in", "uv.lock", "Dockerfile"} or path.startswith(("docker/", "build_tools/"))),
        ("examples-tools-docs", path.startswith(("examples/", "tools/", "docs/")) or name in {"README.md", "CHANGELOG.md"}),
        ("repository-metadata", name in {".gitignore", ".pre-commit-config.yaml", "LICENSE", "CONTRIBUTING.md", "codecov.yml", ".coderabbit.yaml", "greptile.json"}),
    ]
    return next((label for label, matched in rules if matched), "unclassified")


def labels(path: str, text: str, both: bool, status: str) -> list[str]:
    found = []
    if both:
        found.append("both_changed")
    if status.startswith("D"):
        found.append("deleted_in_fork")
    patterns = [
        ("flagscale_block", r"FlagScale Begin|FlagScale End"),
        ("overridable", r"@overridable"),
        ("override_registry", r"@override\s*\(|register_override_method"),
        ("device_abstraction", r"cur_platform|get_platform\s*\("),
        ("raw_cuda_assumption", r"torch\.cuda|[\"']cuda[\"']"),
    ]
    found.extend(label for label, pattern in patterns if re.search(pattern, text))
    if path.endswith(('.py', '.cpp', '.cu', '.cuh')) and path.startswith(("megatron/core/", "megatron/plugin/")):
        found.append("runtime_semantics")
    return sorted(set(found))


def blob(repo: Path, ref: str, path: str) -> str:
    return git(repo, "show", f"{ref}:{path}", check=False)


def python_symbols(text: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    decorators.append(type(dec).__name__)
            if any("overrid" in dec for dec in decorators):
                rows.append({"name": node.name, "kind": type(node).__name__, "line": node.lineno, "decorators": decorators})
    return sorted(rows, key=lambda row: (row["line"], row["name"]))


def marker_blocks(path: str, text: str) -> tuple[list[dict], list[str]]:
    stack = []
    blocks = []
    errors = []
    for number, line in enumerate(text.splitlines(), 1):
        if "FlagScale Begin" in line:
            stack.append((number, line.strip()))
        if "FlagScale End" in line:
            if not stack:
                errors.append(f"{path}:{number}: unmatched FlagScale End")
                continue
            start, start_text = stack.pop()
            body = "\n".join(text.splitlines()[start - 1:number])
            blocks.append({"id": hashlib.sha256(f"{path}:{start}:{body}".encode()).hexdigest()[:16], "path": path, "start": start, "end": number, "begin": start_text, "sha256": hashlib.sha256(body.encode()).hexdigest()})
    errors.extend(f"{path}:{number}: unmatched FlagScale Begin" for number, _ in stack)
    return blocks, errors


def surface(tree: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(path for path in tree if path.startswith(prefixes))


def provenance_entries(data: dict) -> list[dict]:
    repo = Path(data["repo"])
    refs = data["resolved_refs"]
    raw = git(repo, "log", "--format=%H%x09%B%x1e", f"{refs['sync_tree_base']}..{refs['fork']}")
    entries = []
    seen = set()
    for record in raw.split("\x1e"):
        if "\t" not in record:
            continue
        commit, body = record.split("\t", 1)
        commit = commit.strip()
        for match in re.finditer(r"https://github\.com/NVIDIA/Megatron-LM/(?:pull/(\d+)|commit/([0-9a-fA-F]+))", body):
            kind = "pull" if match.group(1) else "commit"
            value = match.group(1) or match.group(2)
            key = (commit, kind, value)
            if key in seen:
                continue
            seen.add(key)
            candidates = []
            if kind == "pull":
                candidates = git(repo, "log", refs["target"], "--format=%H%x09%s", f"--grep=#{value}", check=False).splitlines()
            else:
                resolved = git(repo, "rev-parse", "--verify", f"{value}^{{commit}}", check=False)
                if resolved and subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", resolved, refs["target"]]).returncode == 0:
                    candidates = [resolved]
            entries.append({"fork_commit": commit, "kind": kind, "value": value, "url": match.group(0), "target_candidates": candidates, "coverage": "candidate" if candidates else "manual", "decision": "MANUAL"})
    return sorted(entries, key=lambda row: (row["fork_commit"], row["kind"], row["value"]))

def write_outputs(output: Path, data: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    header = "id\tstatus\tpriority\tcategory\tboth_changed\tpath\trisk_labels\n"
    lines = [f"{r['id']}\t{r['status']}\t{r['priority']}\t{r['category']}\t{str(r['both_changed']).lower()}\t{r['path']}\t{','.join(r['risk_labels'])}\n" for r in data["fork_changes"]]
    (output / "fork-changes.tsv").write_text(header + "".join(lines))
    (output / "both-changed.tsv").write_text(header + "".join(line for line, row in zip(lines, data["fork_changes"]) if row["both_changed"]))
    (output / "override-manifest.json").write_text(json.dumps(data["override_manifest"], indent=2, sort_keys=True) + "\n")
    (output / "flagscale-block-manifest.json").write_text(json.dumps(data["flagscale_blocks"], indent=2, sort_keys=True) + "\n")
    (output / "platform-manifest.json").write_text(json.dumps(data["platform_manifest"], indent=2, sort_keys=True) + "\n")
    (output / "upstream-provenance.json").write_text(json.dumps({"status": "manual-review-required", "entries": provenance_entries(data)}, indent=2, sort_keys=True) + "\n")
    (output / "replay-decision-matrix.md").write_text("# Replay Decision Matrix\n\nAll fork delta actions are MANUAL until reviewed and approved.\n\n" + "\n".join(f"- `{r['id']}` `MANUAL` `{r['path']}`" for r in data["fork_changes"]) + "\n")
    (output / "manual-decisions.md").write_text("# Manual Decisions\n\nResolve provenance, owner, invariant, action, and observing test for every fork delta before integration.\n")
    (output / "mg-fl-design-baseline.md").write_text("# Megatron-LM-FL Design Baseline\n\nGenerated from the complete fork inventory. Marker comments are ownership hints, not completeness evidence.\n")
    cats = data["category_counts"]
    md = ["# Megatron-LM-FL Fork Delta Inventory", "", f"- Schema: `{data['schema_version']}`", *[f"- {key}: `{value}`" for key, value in data["resolved_refs"].items()], f"- Fork-owned paths: {data['fork_change_count']}", f"- Both-changed paths: {data['both_changed_count']}", "", "## Categories", "", "| Category | Count |", "|---|---:|", *[f"| {key} | {value} |" for key, value in sorted(cats.items())], "", "## Acceptance blockers", "", f"- Unclassified paths: {cats.get('unclassified', 0)}", f"- Marker anomalies (non-blocking): {len(data['marker_health']['anomalies'])}", f"- Unresolved provenance entries: {data['provenance_summary']['manual_count']}"]
    (output / "inventory.md").write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--history-base")
    parser.add_argument("--sync-tree-base", required=True)
    parser.add_argument("--release-base", required=True)
    parser.add_argument("--fork", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--allow-divergent-upstream", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"not a Git worktree: {repo}")
    refs = {key: resolve(repo, value) for key, value in {"sync_tree_base": args.sync_tree_base, "release_base": args.release_base, "fork": args.fork, "target": args.target}.items()}
    refs["history_base"] = resolve(repo, args.history_base) if args.history_base else git(repo, "merge-base", refs["fork"], refs["target"])
    linear = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", refs["release_base"], refs["target"]]).returncode == 0
    release_merge_base = git(repo, "merge-base", refs["release_base"], refs["target"])
    if not linear and not args.allow_divergent_upstream:
        raise SystemExit(f"release base is not an ancestor of target; merge-base={release_merge_base}; review and rerun with --allow-divergent-upstream")
    fork_rows = changes(repo, refs["sync_tree_base"], refs["fork"])
    target_paths = {row["path"] for row in changes(repo, refs["sync_tree_base"], refs["target"])}
    tree = git(repo, "ls-tree", "-r", "--name-only", refs["fork"]).splitlines()
    platform_ids = {Path(path).stem.removeprefix("platform_").lower() for path in tree if path.startswith("megatron/plugin/platform/platform_") and Path(path).stem not in {"platform_base", "platform_manager", "platform_register"}}
    plugin_roots = {path.split("/")[2] for path in tree if path.startswith("megatron/plugin/") and len(path.split("/")) > 3}
    registry_text = blob(repo, refs["fork"], "megatron/plugin/override_registry.py")
    vendor_impl_roots = set()
    registry_vendor_ids = set()
    for call in re.findall(r"register\s*\((.*?)\)" , registry_text, re.S):
        impl = re.search(r"impl\s*=\s*[\"']megatron\.plugin\.([A-Za-z0-9_]+)\.", call)
        vendor = re.search(r"vendor\s*=\s*[\"']([^\"']+)", call)
        if impl and vendor:
            vendor_impl_roots.add(impl.group(1))
            registry_vendor_ids.add(vendor.group(1).lower())
    vendor_roots = {root for root in plugin_roots if root.lower() in platform_ids or root in vendor_impl_roots}
    vendor_tokens = platform_ids | registry_vendor_ids | {root.lower() for root in vendor_roots}
    overrides = []
    blocks = []
    marker_anomalies = []
    for index, row in enumerate(fork_rows, 1):
        path = row["path"]
        text = "" if row["status"].startswith("D") else blob(repo, refs["fork"], path)
        both = path in target_paths
        row.update(id=f"MGD-{index:04d}", category=category(path, vendor_roots, vendor_tokens), both_changed=both)
        row["risk_labels"] = labels(path, text, both, row["status"])
        row["priority"] = "P0" if both or row["category"].startswith(("plugin", "invasive")) else ("P1" if row["category"] in {"build-packaging", "training", "tests-common", "tests-vendor"} else "P2")
        if path.endswith(".py"):
            for symbol in python_symbols(text):
                overrides.append({"path": path, **symbol})
            found, errors = marker_blocks(path, text)
            blocks.extend(found)
            marker_anomalies.extend(errors)
    cats = Counter(row["category"] for row in fork_rows)
    platform_files = sorted(path for path in tree if path.startswith("megatron/plugin/platform/"))
    vendor_candidates = sorted(vendor_roots)
    raw_cuda = [row["path"] for row in fork_rows if "raw_cuda_assumption" in row["risk_labels"]]
    data = {"schema_version": SCHEMA_VERSION, "repo": str(repo), "resolved_refs": refs, "upstream_history": {"release_base_is_ancestor": linear, "release_merge_base": release_merge_base, "divergence_explicitly_allowed": args.allow_divergent_upstream}, "dirty_worktree": git(repo, "status", "--short").splitlines(), "fork_change_count": len(fork_rows), "both_changed_count": sum(row["both_changed"] for row in fork_rows), "category_counts": dict(cats), "fork_changes": fork_rows, "override_manifest": overrides, "flagscale_blocks": blocks, "marker_health": {"anomalies": marker_anomalies, "policy": "non-blocking ownership hints; Git diff is authoritative"}, "platform_manifest": {"platform_files": platform_files, "vendor_candidates": vendor_candidates, "raw_cuda_assumption_paths": sorted(raw_cuda)}, "surfaces": {"workflows": surface(tree, (".github/workflows/",)), "ci_configs": surface(tree, (".github/configs/",)), "tests": surface(tree, ("tests/",)), "build_packaging": sorted(path for path in tree if category(path) == "build-packaging"), "plugin": surface(tree, ("megatron/plugin/",))}, "provenance_summary": {"status": "manual-review-required", "manual_count": len(fork_rows)}}
    write_outputs(args.output.resolve(), data)
    summary = {"output": str(args.output.resolve()), "fork_changes": len(fork_rows), "both_changed": data["both_changed_count"], "unclassified": cats.get("unclassified", 0), "marker_anomalies": len(marker_anomalies)}
    print(json.dumps(summary, sort_keys=True))
    if not fork_rows or cats.get("unclassified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
