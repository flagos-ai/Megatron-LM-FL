#!/usr/bin/env python3
"""Generate a reproducible unit-test plan for changed files and platforms.

The planner combines three inputs:

* the project contract (.flagcicd/project.yaml),
* the platform router (generate_matrix.py), and
* each platform's configured unit groups.

PRs use affected selection from the diff. Scheduled nightly runs use configured
selection: every active logical group each enabled platform can execute, with
the diff ignored. Planned catalog groups are reported separately and do not
enter the configured matrix. Selected groups are materialized against each
platform: active groups execute, declared unsupported groups are reported, and
planned groups selected by `all` or an affected fallback expand the platform to
every active group it can execute. A blocked required group or route conflict
fails the plan; a non-required unresolved group marks it partial. The plan is
stable for identical source/base SHAs, configs, and changed files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import flagcicd_contract
import flagcicd_evidence
import generate_matrix
import unit_group_routing
import yaml

PLAN_SCHEMA_VERSION = flagcicd_contract.PLAN_SCHEMA


def _sha256(value: Any) -> str:
    return flagcicd_contract.content_hash(value)


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _default_stage(event_name: str) -> str:
    if event_name == "pull_request":
        return "pr"
    if event_name == "schedule":
        return "nightly"
    if event_name == "push":
        return "branch"
    return "manual"


def _parse_platforms(value: str, known: set[str]) -> list[str]:
    try:
        platforms = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--platforms must be a JSON array: {error}") from error
    if not isinstance(platforms, list) or any(
        not isinstance(platform, str) or not platform for platform in platforms
    ):
        raise ValueError("--platforms must be a JSON array of platform names")
    unknown = sorted(set(platforms) - known)
    if unknown:
        raise ValueError(f"Unknown platforms: {', '.join(unknown)}")
    return sorted(set(platforms))


def _execution_groups(platform: dict[str, Any]) -> list[str]:
    return sorted(
        {
            group["execution_group"]
            for group in platform["selected_groups"]
            if group["execution_group"]
        }
    )


def _group_detail(platform: dict[str, Any], name: str) -> dict[str, str]:
    for bucket in ("selected_groups", "planned", "unsupported", "unresolved", "blocked"):
        for group in platform[bucket]:
            if group["name"] != name:
                continue
            return {
                "platform": platform["name"],
                "status": "executable" if bucket == "selected_groups" else bucket,
                "reason": group["reason"] or "no reason recorded",
            }
    return {
        "platform": platform["name"],
        "status": "not-selected",
        "reason": "group is absent from every materialization bucket",
    }


def _direct_gaps(
    catalog: unit_group_routing.Catalog,
    direct_groups: list[str],
    platform_plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return direct groups that no selected platform actually executes."""
    coverage: dict[str, list[str]] = {}
    for platform in platform_plans:
        for group in platform["selected_groups"]:
            coverage.setdefault(group["name"], []).append(platform["name"])

    gaps = []
    for name in direct_groups:
        if coverage.get(name):
            continue
        group = catalog.groups[name]
        details = [_group_detail(platform, name) for platform in platform_plans]
        gaps.append(
            {
                "group": name,
                "status": group.status,
                "reason": group.reason or "no catalog reason",
                "platforms": details,
            }
        )
    return gaps


def _plan_status(
    *,
    group_selection: dict[str, Any],
    platform_plans: list[dict[str, Any]],
    direct_gaps: list[dict[str, Any]],
) -> str:
    blocking = bool(
        direct_gaps or group_selection["route_conflicts"] or group_selection["unmapped_paths"]
    )
    if not group_selection["documentation_only"]:
        if not platform_plans:
            blocking = True
        for platform in platform_plans:
            if not platform["selected_groups"] or platform["blocked"]:
                blocking = True
    if blocking:
        return "blocked"

    partial = bool(
        group_selection["fallback"]
        or any(
            platform["planned"] or platform["unresolved"] or platform["fallback_excluded"]
            for platform in platform_plans
        )
    )
    return "partial" if partial else "ready"


def build_plan(
    *,
    event_name: str,
    stage: str,
    selection: str,
    level: str,
    changed: list[str],
    unknown_diff: bool,
    platforms: list[str],
    catalog: unit_group_routing.Catalog,
    platform_units: dict[str, unit_group_routing.PlatformUnit],
    source_sha: str,
    base_sha: str,
    project_hash: str = "",
    control_sha: str = "",
    gate_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete plan document, including its stable hash."""
    group_selection = unit_group_routing.select_groups(
        catalog, changed, selection=selection, level=level, unknown_diff=unknown_diff
    )
    required = group_selection["required"]

    platform_plans = []
    if not group_selection["documentation_only"]:
        for name in platforms:
            platform = platform_units[name]
            materialized = unit_group_routing.materialize_groups(
                catalog, platform, group_selection["selected"], group_selection["reasons"], required
            )
            platform_plans.append(
                {
                    "name": name,
                    "devices": list(platform.device_types),
                    "config_identity": {
                        "source_sha": source_sha,
                        "project_hash": project_hash,
                        "platform_hash": platform.digest,
                    },
                    "nproc_per_node": platform.nproc_per_node,
                    "junit_reports": flagcicd_evidence.report_names(
                        platform.nproc_per_node,
                        bool(
                            platform.data.get("test_matrix", {})
                            .get("unit", {})
                            .get("experimental_pytest_extra_args", [])
                        ),
                    ),
                    "configured_groups": sorted(platform.groups),
                    **materialized,
                }
            )

    direct_groups = sorted(group_selection["direct_groups"])
    direct_gaps = _direct_gaps(catalog, direct_groups, platform_plans)
    plan_status = _plan_status(
        group_selection=group_selection, platform_plans=platform_plans, direct_gaps=direct_gaps
    )
    selected_groups = sorted(group_selection["selected"])
    unwired_planned_groups = sorted(
        name for name, group in catalog.groups.items() if group.status == "planned"
    )
    catalog_completeness = "partial" if unwired_planned_groups else "full"
    scope = {
        "affected": "affected_matrix",
        "configured": "configured_matrix",
        "requested": "requested_matrix",
        "all": "all_catalog",
    }.get(selection, selection)
    expected_unit_groups = {
        platform["name"]: _execution_groups(platform) for platform in platform_plans
    }

    config_hash = _sha256(
        {
            "catalog": catalog.data,
            "platforms": {name: platform_units[name].data for name in sorted(platforms)},
        }
    )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "event": event_name,
        "stage": stage,
        "selection": selection,
        "scope": scope,
        "level": level,
        "plan_status": plan_status,
        "catalog_completeness": catalog_completeness,
        "unwired_planned_groups": unwired_planned_groups,
        "source_sha": source_sha,
        "control_sha": control_sha,
        "base_sha": base_sha,
        "project_hash": project_hash,
        "gate_policy": gate_policy or {},
        "changed_files": sorted(set(changed)),
        "catalog_hash": f"sha256:{catalog.digest}",
        "config_hash": config_hash,
        "coverage_sources": list(catalog.coverage_sources),
        "direct_groups": direct_groups,
        "direct_gaps": direct_gaps,
        "expected_unit_groups": expected_unit_groups,
        "selected_logical_groups": selected_groups,
        "global_groups": {
            "selected": selected_groups,
            "required": sorted(required),
            "direct_groups": direct_groups,
            "planned": sorted(
                name
                for name in group_selection["selected"]
                if catalog.groups[name].status == "planned"
            ),
            "unsupported": sorted(
                name
                for name in group_selection["selected"]
                if catalog.groups[name].status == "unsupported"
            ),
            "fallback": group_selection["fallback"],
            "fallback_reason": group_selection["fallback_reason"],
            "unmapped_paths": group_selection["unmapped_paths"],
            "route_conflicts": group_selection["route_conflicts"],
            "documentation_only": group_selection["documentation_only"],
        },
        "platforms": platform_plans,
    }
    plan["plan_hash"] = flagcicd_contract.plan_hash(plan)
    return plan


def execution_matrix(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one workflow-matrix entry per platform with executable group names."""
    return [
        {
            "platform": platform["name"],
            "groups": _execution_groups(platform),
            "config_identity": platform["config_identity"],
        }
        for platform in plan["platforms"]
    ]


def _escape(value: str) -> str:
    return value.replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")


def write_summary(path: pathlib.Path, plan: dict[str, Any]) -> None:
    """Append a human-readable plan to the Actions summary."""
    global_groups = plan["global_groups"]
    plan_blocked = plan["plan_status"] == "blocked"
    lines = [
        "### Unit test plan",
        "",
        f"- Stage: `{plan['stage']}`",
        f"- Selection: `{plan['selection']}`",
        f"- Scope: `{plan['scope']}`",
        f"- Level: `{plan['level']}`",
        f"- Plan status: `{plan['plan_status']}`",
        "- Gate policy: statuses `"
        + ", ".join(plan.get("gate_policy", {}).get("required_plan_status", ["ready"]))
        + "`, catalog partial "
        + (
            "allowed"
            if plan.get("gate_policy", {}).get("allow_catalog_partial", False)
            else "rejected"
        ),
        *(
            [
                "- Configured matrix: "
                f"{'complete' if plan['plan_status'] == 'ready' else 'incomplete'}"
            ]
            if plan["scope"] == "configured_matrix"
            else []
        ),
        f"- Logical catalog completeness: {plan['catalog_completeness']}",
        "- Unwired planned groups: " + (", ".join(plan["unwired_planned_groups"]) or "none"),
        f"- Selected logical groups: {len(plan['selected_logical_groups'])}",
        "- Selected execution groups: "
        + (
            ", ".join(
                f"{name} {len(groups)}"
                for name, groups in sorted(plan["expected_unit_groups"].items())
            )
            or "none"
        ),
        f"- Changed files: {len(plan['changed_files'])}",
        f"- Coverage scope: core/plugin coverage only "
        f"(`{', '.join(plan['coverage_sources'])}`); "
        "not full Megatron unit-group coverage",
        f"- Fallback: {'yes' if global_groups['fallback'] else 'no'}"
        + (
            f" ({_escape(global_groups['fallback_reason'])})"
            if global_groups["fallback_reason"]
            else ""
        ),
        f"- Plan hash: `{plan['plan_hash']}`",
        "",
    ]
    if plan_blocked:
        lines.extend(
            ["**Plan blocked: no unit matrix was emitted and no hardware unit jobs will run.**", ""]
        )
    elif plan["plan_status"] == "partial":
        lines.extend(
            [
                "**Plan status `partial`: this plan is not proof of a complete "
                "logical-group regression.**",
                "",
            ]
        )
    if global_groups["documentation_only"]:
        lines.extend(["Documentation-only change: no hardware tests proposed.", ""])
    if plan["selection"] == "all":
        lines.extend(
            [
                "Selection `all` covers the entire logical catalog, including planned "
                "groups. It is a diagnostic scope, not a configured-matrix run.",
                "",
            ]
        )
    if plan["scope"] == "configured_matrix" and plan["catalog_completeness"] == "partial":
        lines.extend(
            [
                "Planned catalog groups are not part of the configured matrix; they are "
                "reported as catalog gaps and do not block this plan.",
                "",
            ]
        )
    if plan["direct_gaps"]:
        lines.extend(["**Direct group coverage gaps:**", ""])
        for gap in plan["direct_gaps"]:
            lines.append(
                f"- `{gap['group']}` (catalog status `{gap['status']}`): "
                f"{_escape(gap['reason'])}"
            )
            if not gap["platforms"]:
                lines.append("  - No selected platform was available to execute it.")
            for detail in gap["platforms"]:
                lines.append(
                    f"  - `{detail['platform']}`: status `{detail['status']}`, "
                    f"{_escape(detail['reason'])}"
                )
        if any(gap["status"] == "planned" for gap in plan["direct_gaps"]):
            lines.extend(["", "The directly affected group is planned and was not executed.", ""])
        lines.append("")
    if global_groups["route_conflicts"]:
        lines.extend(["**Route conflicts (blocked):**", ""])
        for conflict in global_groups["route_conflicts"]:
            lines.append(
                f"- `{_escape(conflict['path'])}`: "
                + ", ".join(f"`{name}`" for name in conflict["groups"])
            )
        lines.append("")
    if global_groups["unmapped_paths"]:
        lines.extend(
            [
                "**Unmapped paths (need catalog onboarding):**",
                "",
                *[f"- `{_escape(path)}`" for path in global_groups["unmapped_paths"]],
                "",
                "Unknown paths produced this diagnostic fallback plan, but the "
                "fallback plan was not executed.",
                "",
            ]
        )
    if plan["platforms"]:
        if plan_blocked:
            lines.extend(
                [
                    "**Diagnostic fallback materialization (not executed):** the table "
                    "below only shows how the fallback would materialize if the blocking "
                    "issue were resolved.",
                    "",
                ]
            )
        executable_header = "Executable (diagnostic)" if plan_blocked else "Executable"
        lines.extend(
            [
                f"| Platform | Configured | {executable_header} | Planned | Unsupported | "
                "Unresolved | Blocked | Potential reduction |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for platform in plan["platforms"]:
            executable = (
                ", ".join(
                    group["name"]
                    + (
                        f" (executes {group['execution_group']})"
                        if group["execution_group"] != group["name"]
                        else ""
                    )
                    + (" (required)" if group["required"] else "")
                    for group in platform["selected_groups"]
                )
                or "none"
            )
            planned = ", ".join(group["name"] for group in platform["planned"]) or "none"
            unsupported = ", ".join(group["name"] for group in platform["unsupported"]) or "none"
            unresolved = ", ".join(group["name"] for group in platform["unresolved"]) or "none"
            blocked_groups = ", ".join(group["name"] for group in platform["blocked"]) or "none"
            has_gap = bool(
                not platform["selected_groups"]
                or platform["planned"]
                or platform["unsupported"]
                or platform["unresolved"]
                or platform["blocked"]
                or platform["fallback_excluded"]
            )
            executed_groups = {
                group["execution_group"]
                for group in platform["selected_groups"]
                if group["execution_group"]
            }
            reduction = (
                "unavailable; planned/unsupported/unresolved/blocked groups exist"
                if has_gap
                else f"{len(platform['configured_groups']) - len(executed_groups)} groups"
            )
            lines.append(
                f"| {platform['name']} | {len(platform['configured_groups'])} | "
                f"{_escape(executable)} | {_escape(planned)} | {_escape(unsupported)} | "
                f"{_escape(unresolved)} | {_escape(blocked_groups)} | {reduction} |"
            )
        reasons_heading = (
            "Diagnostic fallback reasons (not executed):"
            if plan_blocked
            else "Selected executable group reasons:"
        )
        lines.extend(
            [
                "",
                "Planned groups are not executed. Selecting one expands the platform "
                "to every active group it can execute.",
                "",
                reasons_heading,
                "",
            ]
        )
        for platform in plan["platforms"]:
            for group in platform["selected_groups"]:
                reasons = "; ".join(_reason_text(reason) for reason in group["reasons"])
                lines.append(f"- `{platform['name']}/{group['name']}`: {reasons}")
        for bucket, title in (
            ("planned", "Planned groups"),
            ("unsupported", "Unsupported groups"),
            ("unresolved", "Unresolved groups"),
            ("blocked", "Blocked required groups"),
        ):
            entries = [
                (platform["name"], group)
                for platform in plan["platforms"]
                for group in platform[bucket]
            ]
            if not entries:
                continue
            lines.extend(["", f"**{title}:**", ""])
            for platform_name, group in entries:
                reason = _escape(group["reason"]) if group["reason"] else "no reason recorded"
                lines.append(f"- `{platform_name}/{group['name']}`: {reason}")
        excluded = [
            (platform["name"], group)
            for platform in plan["platforms"]
            for group in platform["fallback_excluded"]
        ]
        if excluded:
            lines.extend(["", "**Fallback excluded groups:**", ""])
            for platform_name, group in excluded:
                lines.append(
                    f"- `{platform_name}/{group['name']}`: status `{group['status']}`, "
                    f"{_escape(group['reason'])}"
                )
        lines.append("")
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def _reason_text(reason: dict[str, str]) -> str:
    if reason["type"] == "direct":
        return f"direct match `{_escape(reason['path'])}`"
    if reason["type"] == "dependency":
        source = reason.get("source_group")
        return f"dependency of `{source}`" if source else "dependency"
    if reason["type"] == "required":
        return f"required for `{reason['level']}`"
    if reason["type"] == "fallback":
        return f"fallback: {_escape(reason['reason'])}"
    return reason["type"]


def _changed_files(args: argparse.Namespace) -> tuple[list[str], bool]:
    if args.changed_files is not None:
        return [path for path in args.changed_files.splitlines() if path], False
    if args.event_name != "pull_request":
        return [], False
    try:
        return generate_matrix._changed_files(args.base_ref), False
    except subprocess.CalledProcessError as error:
        print(f"::warning::Cannot read diff; proposing all groups: {error}", file=sys.stderr)
        return [], True


def _source_sha(value: str | None) -> str:
    return value or os.environ.get("GITHUB_SHA", "") or _git("rev-parse", "HEAD")


def _base_sha(value: str | None, base_ref: str) -> str:
    return (
        value
        or os.environ.get("GITHUB_BASE_SHA", "")
        or _git("merge-base", f"origin/{base_ref}", "HEAD")
        or _git("rev-parse", f"origin/{base_ref}")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument("--changed-files", default=None, help="Newline-separated paths for testing")
    parser.add_argument("--platforms", default=None, help="JSON array of selected platforms")
    parser.add_argument(
        "--selection", choices=("affected", "configured", "requested", "all"), default=None
    )
    parser.add_argument("--stage", choices=("pr", "branch", "nightly", "manual"), default=None)
    parser.add_argument("--level", default=None)
    parser.add_argument("--output", default="unit-test-plan.json")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--github-output", default=None)
    parser.add_argument(
        "--config-root",
        default=".",
        help="Candidate checkout containing configuration data, not control code",
    )
    parser.add_argument("--source-sha", default=None)
    parser.add_argument("--base-sha", default=None)
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    config_root = (root / args.config_root).resolve()
    configs_dir = config_root / args.configs_dir
    stage = args.stage or _default_stage(args.event_name)

    try:
        if config_root != root.resolve():
            actual = subprocess.check_output(
                ["git", "-C", str(config_root), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            if not args.source_sha or actual != args.source_sha:
                raise ValueError("Candidate config checkout differs from source SHA")
        project_path = config_root / args.project
        if config_root != root.resolve() and not project_path.is_file():
            raise ValueError("Candidate checkout must contain the FlagCICD project contract")
        project, project_digest = flagcicd_contract.load_project(project_path)
        project_hash = f"sha256:{project_digest}"
        # Candidate config is data. Stage/gate policy remains controlled by
        # the trusted workflow checkout, pinned for the entire requested run.
        policy_project = project
        if config_root != root.resolve():
            policy_project, _ = flagcicd_contract.load_project(root / args.project)
        stage_config = policy_project["stages"].get(stage)
        if stage_config is None:
            raise ValueError(f"Stage {stage!r} is not declared in the project contract")
        selection = args.selection or stage_config["selection"]
        configured_level = stage_config["default_level"]
        if stage_config.get("fixed_level") and args.level is not None:
            if args.level != configured_level:
                raise ValueError(
                    f"Stage {stage!r} fixes level {configured_level}; received {args.level}"
                )
        level = (
            configured_level
            if stage_config.get("fixed_level")
            else (args.level or configured_level)
        )
        if level not in project["levels"]:
            raise ValueError(f"Level {level!r} is not declared in the project contract")
        catalog = unit_group_routing.load_catalog(project_path)
        rules = generate_matrix._load_watch_rules(configs_dir)
        changed, unknown_diff = _changed_files(args)
        if selection in ("all", "configured", "requested"):
            changed = []
            unknown_diff = False
        if args.platforms is not None:
            platforms = _parse_platforms(args.platforms, set(rules))
        else:
            labels = generate_matrix.event_labels() if args.event_name == "pull_request" else []
            platforms, _ = generate_matrix.select_platforms(rules, changed, args.event_name, labels)
        platform_units = {
            name: unit_group_routing.load_platform(configs_dir / f"{name}.yml")
            for name in platforms
        }
        plan = build_plan(
            event_name=args.event_name,
            stage=stage,
            selection=selection,
            level=level,
            changed=changed,
            unknown_diff=unknown_diff,
            platforms=platforms,
            catalog=catalog,
            platform_units=platform_units,
            source_sha=_source_sha(args.source_sha),
            base_sha=_base_sha(args.base_sha, args.base_ref),
            project_hash=project_hash,
            control_sha=_git("rev-parse", "HEAD"),
            gate_policy=policy_project.get("gate_policy", {}).get(stage, {}),
        )
    except (
        unit_group_routing.CatalogError,
        OSError,
        ValueError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    output = root / args.output
    output.write_text(json.dumps(plan, indent=2, sort_keys=False) + "\n")
    if args.summary:
        write_summary(root / args.summary, plan)

    blocked = [
        (platform["name"], group["name"])
        for platform in plan["platforms"]
        for group in platform["blocked"]
    ]
    unresolved = [
        (platform["name"], group["name"])
        for platform in plan["platforms"]
        for group in platform["unresolved"]
    ]
    empty = [platform["name"] for platform in plan["platforms"] if not platform["selected_groups"]]
    unmapped = plan["global_groups"]["unmapped_paths"]
    conflicts = plan["global_groups"]["route_conflicts"]
    direct = []
    for gap in plan["direct_gaps"]:
        if not gap["platforms"]:
            direct.append(
                f"direct group {gap['group']} (status={gap['status']}): "
                "no selected platform was available"
            )
        for detail in gap["platforms"]:
            direct.append(
                f"direct group {gap['group']} on {detail['platform']}: "
                f"status={detail['status']}, reason={detail['reason']}"
            )
    errors = (
        direct
        + [f"route conflict {item['path']}: {', '.join(item['groups'])}" for item in conflicts]
        + [f"{platform}/{group}: blocked" for platform, group in blocked]
        + [f"{platform}/{group}: unresolved" for platform, group in unresolved]
        + [f"{platform}: no executable groups" for platform in empty]
    )
    if unmapped:
        errors.append(
            "unmapped path(s) need catalog onboarding: "
            + ", ".join(unmapped)
            + "; the fallback plan was not executed"
        )
    if plan["plan_status"] == "blocked":
        message = "; ".join(errors) or "no executable unit-group plan"
        print(f"::error::Unit plan blocked: {message}", file=sys.stderr)
        if args.github_output:
            with open(args.github_output, "a") as stream:
                stream.write("plan_status=blocked\n")
        print("plan_status=blocked")
        return 1
    if errors:
        print(f"::warning::Unit plan has gaps: {'; '.join(errors)}", file=sys.stderr)

    if args.github_output:
        matrix = execution_matrix(plan)
        with open(args.github_output, "a") as stream:
            stream.write(f"unit_matrix={json.dumps(matrix, separators=(',', ':'))}\n")
            stream.write(f"plan_hash={plan['plan_hash']}\n")
            stream.write(f"plan_status={plan['plan_status']}\n")
            stream.write(f"selection={plan['selection']}\n")
            stream.write(f"scope={plan['scope']}\n")
            stream.write(f"catalog_completeness={plan['catalog_completeness']}\n")
            stream.write(
                "required_plan_status="
                + json.dumps(
                    plan.get("gate_policy", {}).get("required_plan_status", ["ready"]),
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.write(
                "allow_catalog_partial="
                + str(plan.get("gate_policy", {}).get("allow_catalog_partial", False)).lower()
                + "\n"
            )
            stream.write(
                "expected_unit_groups="
                + json.dumps(plan["expected_unit_groups"], separators=(",", ":"))
                + "\n"
            )

    print(f"plan_hash={plan['plan_hash']}")
    print(f"plan_status={plan['plan_status']}")
    print(f"selection={plan['selection']}")
    print(f"scope={plan['scope']}")
    print(f"catalog_completeness={plan['catalog_completeness']}")
    print(
        "required_plan_status="
        + json.dumps(
            plan.get("gate_policy", {}).get("required_plan_status", ["ready"]),
            separators=(",", ":"),
        )
    )
    print(
        "unwired_planned_groups="
        + json.dumps(plan["unwired_planned_groups"], separators=(",", ":"))
    )
    print(
        "platforms="
        + json.dumps(
            {
                platform["name"]: [group["name"] for group in platform["selected_groups"]]
                for platform in plan["platforms"]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
