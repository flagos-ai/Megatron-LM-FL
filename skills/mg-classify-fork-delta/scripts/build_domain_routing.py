#!/usr/bin/env python3
"""Build deterministic domain routes and a skill coverage-gap ledger."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"invalid {label}: top level must be an object")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_gap_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{key}".encode()).hexdigest()[:10].upper()
    return f"MG-GAP-{digest}"


def choose_primary(domains: set[str], priority: list[str], observers: set[str]) -> str | None:
    implementation = domains - observers
    candidates = implementation or domains
    order = {name: index for index, name in enumerate(priority)}
    return min(candidates, key=lambda name: (order.get(name, len(order)), name)) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = load_object(args.decisions, "effective decisions")
    policy = load_object(args.policy, "domain routing policy")
    coverage = load_object(args.coverage, "skill coverage")
    if policy.get("schema_version") != "1.0" or coverage.get("schema_version") != "1.0":
        raise SystemExit("unsupported policy or coverage schema_version")

    rows = decisions.get("decisions")
    refs = decisions.get("refs")
    if not isinstance(rows, list) or not isinstance(refs, dict):
        raise SystemExit("effective decisions must contain decisions[] and refs{}")
    ids = [row.get("id") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SystemExit("effective decision IDs are missing or duplicated")

    category_domains = policy.get("category_domains", {})
    risk_domains = policy.get("risk_secondary_domains", {})
    priority = policy.get("coordinator_priority", [])
    observers = set(policy.get("observer_domains", []))
    specialists = policy.get("recommended_specialists", {})
    requirements = policy.get("domain_requirements", {})
    coverage_domains = coverage.get("domains", {})

    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_rows[row.get("group_id") or row["id"]].append(row)

    routes = []
    gaps = []
    gap_keys = set()
    domain_counts = Counter()

    for group_id in sorted(group_rows):
        members = group_rows[group_id]
        domains: set[str] = set()
        intrinsic_domains: set[str] = set()
        unknown_categories: set[str] = set()
        for row in members:
            domain = category_domains.get(row.get("category"))
            if domain:
                domains.add(domain)
                intrinsic_domains.add(domain)
            else:
                unknown_categories.add(str(row.get("category")))
            domains.update(risk_domains[label] for label in row.get("risk_labels", []) if label in risk_domains)
        primary = choose_primary(intrinsic_domains or domains, priority, observers)
        for category in sorted(unknown_categories):
            key = f"unknown-category:{category}"
            if key not in gap_keys:
                gap_keys.add(key)
                affected = sorted(row["id"] for row in rows if str(row.get("category")) == category)
                gaps.append({
                    "gap_id": stable_gap_id("unknown-category", category),
                    "gap_class": "reusable-pattern",
                    "blocking": True,
                    "status": "skill-update-required",
                    "detecting_phase": "domain-routing",
                    "current_domain": "unknown",
                    "reason": f"category has no domain route: {category}",
                    "affected_delta_ids": affected,
                    "proposed_target_skill": "mg-classify-fork-delta",
                    "proposed_change": "extend the generic category-to-domain policy and add a regression fixture",
                    "rerun_phases": ["domain-routing", "classifier-intake"],
                })
        for row in sorted(members, key=lambda item: item["id"]):
            row_domains = set(domains)
            row_primary = primary
            handler = coverage_domains.get(row_primary, {}).get("handler_skill") if row_primary else None
            secondary = sorted(row_domains - ({row_primary} if row_primary else set()))
            route_domains = ([row_primary] if row_primary else []) + secondary
            required_invariants = sorted({item for domain in route_domains for item in requirements.get(domain, {}).get("invariants", [])})
            acceptance_evidence = sorted({item for domain in route_domains for item in requirements.get(domain, {}).get("evidence", [])})
            routes.append({
                "delta_id": row["id"],
                "feature_group": group_id,
                "grouping_basis": "source-group-id",
                "primary_domain": row_primary,
                "secondary_domains": secondary,
                "handler_skill": handler,
                "secondary_handlers": {domain: coverage_domains.get(domain, {}).get("handler_skill") for domain in secondary},
                "recommended_specialist_skill": specialists.get(row_primary),
                "action": row.get("effective_action"),
                "category": row.get("category"),
                "reason": "group category and risk-label routing",
                "required_invariants": required_invariants,
                "acceptance_evidence": acceptance_evidence,
                "review_required": bool(unknown_categories) or row.get("effective_action") == "REDESIGN" or len(intrinsic_domains - observers) > 1,
            })
            if row_primary:
                domain_counts[row_primary] += 1

    used_domains = sorted({domain for route in routes for domain in ([route["primary_domain"]] + route["secondary_domains"]) if domain})
    for domain in used_domains:
        entry = coverage_domains.get(domain)
        if entry and entry.get("status") == "covered" and entry.get("handler_skill"):
            continue
        reason = "domain is absent from the skill coverage manifest" if entry is None else f"domain coverage is {entry.get('status')}"
        key = f"domain-coverage:{domain}"
        affected = sorted(route["delta_id"] for route in routes if domain == route["primary_domain"] or domain in route["secondary_domains"])
        gaps.append({
            "gap_id": stable_gap_id("domain-coverage", domain),
            "gap_class": "reusable-pattern",
            "blocking": True if entry is None else bool(entry.get("blocking")),
            "status": "skill-update-required",
            "detecting_phase": "skill-coverage-audit",
            "current_domain": domain,
            "reason": reason,
            "affected_delta_ids": affected,
            "proposed_target_skill": specialists.get(domain) or "mg-fl-upstream-sync",
            "proposed_change": "implement the domain method, artifact contract, and regression scenarios; then mark coverage as covered",
            "existing_evidence": [] if entry is None else entry.get("evidence", []),
            "rerun_phases": [domain, "domain-routing", "classifier-intake"],
        })

    routes.sort(key=lambda row: row["delta_id"])
    route_ids = [row["delta_id"] for row in routes]
    unrouted = sum(not row["primary_domain"] or not row["handler_skill"] for row in routes)
    review_required = sum(bool(row["review_required"]) for row in routes)
    specialization = []
    rows_by_id = {row["id"]: row for row in rows}
    participation_counts = Counter()
    for domain in used_domains:
        affected_routes = [route for route in routes if domain == route["primary_domain"] or domain in route["secondary_domains"]]
        participation_counts[domain] = len(affected_routes)
        affected_rows = [rows_by_id[route["delta_id"]] for route in affected_routes]
        specialization.append({
            "domain": domain,
            "primary_delta_count": domain_counts[domain],
            "affected_delta_count": len(affected_routes),
            "feature_group_count": len({route["feature_group"] for route in affected_routes}),
            "both_changed_count": sum(bool(row.get("both_changed")) for row in affected_rows),
            "redesign_count": sum(row.get("effective_action") == "REDESIGN" for row in affected_rows),
            "recommended_skill": specialists.get(domain),
            "coverage_status": coverage_domains.get(domain, {}).get("status", "missing"),
        })

    routing = {
        "schema_version": "1.0",
        "refs": refs,
        "skill_set_version": coverage.get("skill_set_version"),
        "review_status": "needs-review" if review_required or gaps else "candidate-clean",
        "routes": routes,
        "specialization_candidates": specialization,
        "summary": {
            "route_count": len(routes),
            "unrouted": unrouted,
            "conflicting_routes": 0,
            "review_required": review_required,
            "primary_domain_counts": dict(sorted(domain_counts.items())),
            "domain_participation_counts": dict(sorted(participation_counts.items())),
        },
    }
    ledger = {
        "schema_version": "1.0",
        "refs": refs,
        "skill_set_version": coverage.get("skill_set_version"),
        "gaps": sorted(gaps, key=lambda row: row["gap_id"]),
        "summary": {
            "gap_count": len(gaps),
            "blocking_open": sum(row["blocking"] and row["status"] != "resolved" for row in gaps),
            "skill_update_required": sum(row["status"] == "skill-update-required" for row in gaps),
        },
    }
    if len(route_ids) != len(ids) or set(route_ids) != set(ids):
        raise SystemExit("routing changed effective-decision ordering or coverage")
    write_json(args.output / "domain-routing.json", routing)
    write_json(args.output / "skill-gap-ledger.json", ledger)
    print(json.dumps({"routing": routing["summary"], "gaps": ledger["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
