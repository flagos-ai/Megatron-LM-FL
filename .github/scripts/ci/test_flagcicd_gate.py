#!/usr/bin/env python3
"""Tests for the FlagCICD plan/evidence gate."""

from __future__ import annotations

import copy
import pathlib
import unittest

import flagcicd_gate
import generate_test_plan

ROOT = pathlib.Path(__file__).resolve().parents[3]


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = generate_test_plan.build_plan(
            event_name="schedule",
            stage="nightly",
            selection="configured",
            level="L2",
            changed=[],
            unknown_diff=False,
            platforms=["cuda"],
            catalog=generate_test_plan.unit_group_routing.load_catalog(
                ROOT / ".flagcicd/project.yaml"
            ),
            platform_units={
                "cuda": generate_test_plan.unit_group_routing.load_platform(
                    ROOT / ".github/configs/cuda.yml"
                )
            },
            source_sha="source",
            base_sha="base",
            project_hash="sha256:project",
            gate_policy={"required_plan_status": ["ready"], "allow_catalog_partial": True},
        )

    def test_plan_and_attestation_pass(self) -> None:
        plan = copy.deepcopy(self.plan)
        groups = plan["expected_unit_groups"]
        errors = flagcicd_gate.verify_plan(
            plan,
            expected_hash=plan["plan_hash"],
            expected_groups=groups,
            required_statuses=["ready"],
            allow_catalog_partial=True,
            matrix=generate_test_plan.execution_matrix(plan),
        )
        self.assertEqual(errors, [])
        attestations = [
            {
                "plan_hash": plan["plan_hash"],
                "platform": "cuda",
                "device": "a100",
                "expected_groups": groups["cuda"],
                "executed_groups": groups["cuda"],
                "missing_groups": [],
                "unexpected_groups": [],
                "status": "passed",
                "groups": [
                    {
                        "name": name,
                        "exit_code": 0,
                        "evidence": {
                            "schema_version": 1,
                            "reports": {
                                report: {
                                    "tests": 1,
                                    "passed": 1,
                                    "skipped": 0,
                                    "failures": 0,
                                    "errors": 0,
                                }
                                for report in plan["platforms"][0]["junit_reports"]
                            },
                        },
                    }
                    for name in groups["cuda"]
                ],
            }
        ]
        self.assertEqual(flagcicd_gate.verify_attestations(plan, attestations), [])

    def test_plan_policy_and_hash_are_checked(self) -> None:
        plan = copy.deepcopy(self.plan)
        errors = flagcicd_gate.verify_plan(
            plan,
            expected_hash="sha256:wrong",
            expected_groups={},
            required_statuses=["ready"],
            allow_catalog_partial=False,
            matrix=[{"platform": "cuda", "groups": ["wrong"]}],
        )
        self.assertIn("plan hash output does not match the uploaded plan", errors)
        self.assertIn("catalog is partial but the stage policy does not allow it", errors)
        self.assertIn("expected execution groups differ from the uploaded plan", errors)
        self.assertIn("unit matrix does not match expected groups", errors)

    def test_blocked_plan_is_never_accepted(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["plan_status"] = "blocked"
        errors = flagcicd_gate.verify_plan(
            plan,
            expected_hash=plan["plan_hash"],
            expected_groups=plan["expected_unit_groups"],
            required_statuses=["ready"],
            matrix=generate_test_plan.execution_matrix(plan),
        )
        self.assertTrue(any("blocked plans" in error for error in errors))

    def test_matrix_rejects_duplicates_and_malformed_entries(self) -> None:
        plan = copy.deepcopy(self.plan)
        errors = flagcicd_gate.verify_matrix(
            plan["expected_unit_groups"],
            [
                {"platform": "cuda", "groups": plan["expected_unit_groups"]["cuda"]},
                {"platform": "cuda", "groups": []},
                "invalid",
            ],
        )
        self.assertIn("duplicate platform in unit matrix: cuda", errors)
        self.assertIn("unit matrix contains a non-object entry", errors)

    def test_attestation_set_and_execution_are_exact(self) -> None:
        plan = copy.deepcopy(self.plan)
        errors = flagcicd_gate.verify_attestations(
            plan,
            [
                {
                    "plan_hash": "sha256:wrong",
                    "platform": "cuda",
                    "device": "a100",
                    "expected_groups": [],
                    "executed_groups": [],
                    "missing_groups": ["core"],
                    "unexpected_groups": [],
                    "status": "failed",
                }
            ],
        )
        self.assertTrue(any("plan hash mismatch" in error for error in errors))
        self.assertTrue(any("expected group mismatch" in error for error in errors))
        self.assertTrue(any("group execution mismatch" in error for error in errors))
        self.assertTrue(any("did not pass" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
