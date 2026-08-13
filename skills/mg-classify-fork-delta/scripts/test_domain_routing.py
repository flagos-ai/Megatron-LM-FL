#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_domain_routing.py")


class DomainRoutingTests(unittest.TestCase):
    def run_case(self, rows, categories, coverage):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = {"fork": "f" * 40, "target": "t" * 40}
            decisions = {"refs": refs, "decisions": rows}
            policy = {
                "schema_version": "1.0",
                "category_domains": categories,
                "risk_secondary_domains": {"both_changed": "upstream-conflict"},
                "coordinator_priority": ["plugin-override", "runtime-feature", "test-hardware", "upstream-conflict"],
                "observer_domains": ["test-hardware"],
                "recommended_specialists": {"runtime-feature": "mg-integrate-runtime-features"},
            }
            skill_coverage = {"schema_version": "1.0", "skill_set_version": "test", "domains": coverage}
            for name, value in [("decisions.json", decisions), ("policy.json", policy), ("coverage.json", skill_coverage)]:
                (root / name).write_text(json.dumps(value))
            proc = subprocess.run([
                "python", str(SCRIPT), "--decisions", str(root / "decisions.json"),
                "--policy", str(root / "policy.json"), "--coverage", str(root / "coverage.json"),
                "--output", str(root / "out"),
            ], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return (
                json.loads((root / "out/domain-routing.json").read_text()),
                json.loads((root / "out/skill-gap-ledger.json").read_text()),
            )

    def test_implementation_domain_coordinates_tests(self):
        rows = [
            {"id": "MGD-1", "group_id": "g", "category": "runtime", "risk_labels": [], "effective_action": "REPLAY_FL", "both_changed": False},
            {"id": "MGD-2", "group_id": "g", "category": "tests", "risk_labels": [], "effective_action": "REPLAY_FL", "both_changed": False},
        ]
        covered = {
            "runtime-feature": {"status": "covered", "handler_skill": "runtime-skill", "blocking": True},
            "test-hardware": {"status": "covered", "handler_skill": "test-skill", "blocking": True},
        }
        routing, gaps = self.run_case(rows, {"runtime": "runtime-feature", "tests": "test-hardware"}, covered)
        self.assertEqual([row["primary_domain"] for row in routing["routes"]], ["runtime-feature", "runtime-feature"])
        self.assertEqual(gaps["summary"]["gap_count"], 0)

    def test_output_is_stably_sorted(self):
        rows = [
            {"id": "MGD-2", "group_id": "b", "category": "runtime", "risk_labels": [], "effective_action": "REPLAY_FL", "both_changed": False},
            {"id": "MGD-1", "group_id": "a", "category": "runtime", "risk_labels": [], "effective_action": "REPLAY_FL", "both_changed": False},
        ]
        covered = {"runtime-feature": {"status": "covered", "handler_skill": "runtime-skill", "blocking": True}}
        routing, _ = self.run_case(rows, {"runtime": "runtime-feature"}, covered)
        self.assertEqual([row["delta_id"] for row in routing["routes"]], ["MGD-1", "MGD-2"])

    def test_risk_domain_stays_secondary(self):
        rows = [{"id": "MGD-1", "group_id": "g", "category": "runtime", "risk_labels": ["both_changed"], "effective_action": "REPLAY_FL", "both_changed": True}]
        covered = {
            "runtime-feature": {"status": "covered", "handler_skill": "runtime-skill", "blocking": True},
            "upstream-conflict": {"status": "covered", "handler_skill": "conflict-skill", "blocking": True},
        }
        routing, gaps = self.run_case(rows, {"runtime": "runtime-feature"}, covered)
        self.assertEqual(routing["routes"][0]["primary_domain"], "runtime-feature")
        self.assertEqual(routing["routes"][0]["secondary_domains"], ["upstream-conflict"])
        self.assertEqual(routing["routes"][0]["secondary_handlers"]["upstream-conflict"], "conflict-skill")
        self.assertEqual(gaps["summary"]["gap_count"], 0)

    def test_unknown_category_is_blocking_gap(self):
        rows = [{"id": "MGD-1", "group_id": "g", "category": "new-kind", "risk_labels": [], "effective_action": "REPLAY_FL", "both_changed": False}]
        routing, gaps = self.run_case(rows, {}, {})
        self.assertEqual(routing["summary"]["unrouted"], 1)
        self.assertEqual(gaps["summary"]["blocking_open"], 1)
        self.assertIn("category has no domain route", gaps["gaps"][0]["reason"])

    def test_partial_coverage_requests_skill_update(self):
        rows = [{"id": "MGD-1", "group_id": "g", "category": "runtime", "risk_labels": [], "effective_action": "REDESIGN", "both_changed": True}]
        partial = {
            "runtime-feature": {"status": "partial", "handler_skill": "orchestrator", "blocking": True, "evidence": []},
            "upstream-conflict": {"status": "covered", "handler_skill": "conflict-skill", "blocking": True, "evidence": []},
        }
        routing, gaps = self.run_case(rows, {"runtime": "runtime-feature"}, partial)
        self.assertEqual(routing["summary"]["review_required"], 1)
        self.assertEqual(gaps["summary"]["skill_update_required"], 1)
        self.assertEqual(gaps["gaps"][0]["current_domain"], "runtime-feature")


if __name__ == "__main__":
    unittest.main()
