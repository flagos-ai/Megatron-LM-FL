#!/usr/bin/env python3
"""Tests for the repository-facing FlagCICD contract."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import flagcicd_contract
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]


class ContractTests(unittest.TestCase):
    def test_project_builds_independent_group_index(self) -> None:
        project, digest = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        catalog = flagcicd_contract.project_to_catalog(project)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(catalog["levels"], project["levels"])
        for group in project["test_groups"]:
            self.assertEqual(
                catalog["groups"][group["name"]],
                {key: value for key, value in group.items() if key != "name"},
            )
        catalog["groups"]["models"]["paths"].append("fixture/")
        models = next(group for group in project["test_groups"] if group["name"] == "models")
        self.assertNotIn("fixture/", models["paths"])

    def test_project_has_explicit_stage_semantics(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        self.assertEqual(project["stages"]["pr"]["selection"], "affected")
        self.assertEqual(project["stages"]["nightly"]["selection"], "configured")
        self.assertTrue(project["stages"]["nightly"]["fixed_level"])
        self.assertEqual(set(project["stages"]), {"pr", "branch", "nightly", "manual"})
        self.assertEqual(set(project["levels"]), {"L1", "L2"})
        self.assertEqual(project["gate_policy"]["pr"]["required_plan_status"], ["ready", "partial"])

    def test_removed_configuration_is_rejected(self) -> None:
        cases = [
            ("project", "resource_classes", {}),
            ("global", "default_timeout_minutes", 1),
            ("group", "resource_class", "single_device"),
            ("group", "parallel_safe", True),
            ("group", "timeout_minutes", 1),
            ("group", "kind", "unit"),
            ("group", "evidence", "junit"),
            ("stage", "full_configured_matrix", True),
            ("stage", "status", "planned"),
            ("level", "status", "planned"),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key), tempfile.TemporaryDirectory() as directory:
                project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
                target = {
                    "project": project,
                    "global": project["global"],
                    "group": project["test_groups"][0],
                    "stage": project["stages"]["pr"],
                    "level": project["levels"]["L1"],
                }[section]
                target[key] = value
                path = pathlib.Path(directory) / "project.yaml"
                path.write_text(yaml.safe_dump(project, sort_keys=False))
                with self.assertRaisesRegex(flagcicd_contract.ContractError, "unsupported fields"):
                    flagcicd_contract.load_project(path)

    def test_consumers_require_project_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = str(pathlib.Path(directory) / "missing-project.yaml")
            for script in (
                "generate_test_plan.py",
                "validate_flagcicd_contract.py",
                "validate_unit_group_catalog.py",
            ):
                with self.subTest(script=script):
                    command = [
                        sys.executable,
                        str(ROOT / ".github/scripts/ci" / script),
                        "--project",
                        missing,
                    ]
                    if script == "generate_test_plan.py":
                        command.extend(
                            [
                                "--event-name",
                                "pull_request",
                                "--source-sha",
                                "source",
                                "--base-sha",
                                "base",
                            ]
                        )
                    result = subprocess.run(
                        command, cwd=ROOT, capture_output=True, text=True, timeout=30
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "Cannot read FlagCICD project contract", result.stdout + result.stderr
                    )

    def test_stage_cannot_reference_unknown_level(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        project["stages"]["pr"]["default_level"] = "L4"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            path.write_text(yaml.safe_dump(project, sort_keys=False))
            with self.assertRaisesRegex(flagcicd_contract.ContractError, "unknown level"):
                flagcicd_contract.load_project(path)


if __name__ == "__main__":
    unittest.main()
