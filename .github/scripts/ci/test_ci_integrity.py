"""Regression probes for routing, candidate configuration, and real test evidence."""

from __future__ import annotations

import copy
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import flagcicd_contract as contract
import flagcicd_evidence as evidence
import flagcicd_gate as gate
import generate_test_plan as planner
import yaml
from test_unit_plan import ROOT, build_plan, execution_names

sys.path.insert(0, str(ROOT / ".github/scripts"))
import run_unit_groups as runner


class RoutingIntegrityTests(unittest.TestCase):
    def test_mixed_changes_preserve_every_blocker_and_hash(self) -> None:
        paths = [
            ".github/workflows/ci.yml",
            "unknown_file.py",
            "megatron/core/optimizer/optimizer.py",
        ]
        plans = [
            build_plan(list(order), platforms=["cuda"]) for order in itertools.permutations(paths)
        ]
        self.assertEqual(len({plan["plan_hash"] for plan in plans}), 1)
        for plan in plans:
            self.assertEqual(plan["plan_status"], "blocked")
            self.assertEqual(plan["direct_groups"], ["optimizer"])
            self.assertEqual(plan["global_groups"]["unmapped_paths"], ["unknown_file.py"])
            self.assertEqual([item["group"] for item in plan["direct_gaps"]], ["optimizer"])

    def test_core_changes_include_known_consumers(self) -> None:
        for path in ("utils.py", "parallel_state.py", "activations.py"):
            plan = build_plan([f"megatron/core/{path}"], platforms=["cuda"], level="L2")
            self.assertTrue(
                {"tensor_parallel", "transformer", "models"} <= execution_names(plan, "cuda")
            )
            self.assertEqual(plan["direct_groups"], ["core"])
            self.assertEqual(plan["plan_status"], "ready")

    def test_pre_commit_config_is_shared_without_hiding_unknown_changes(self) -> None:
        plan = build_plan([".pre-commit-config.yaml"])
        self.assertEqual(plan["plan_status"], "partial")
        self.assertFalse(plan["global_groups"]["unmapped_paths"])
        for platform in plan["platforms"]:
            self.assertEqual(
                plan["expected_unit_groups"][platform["name"]], platform["configured_groups"]
            )
        mixed = build_plan([".pre-commit-config.yaml", "unmapped_config.yaml"])
        self.assertEqual(mixed["plan_status"], "blocked")
        self.assertEqual(mixed["global_groups"]["unmapped_paths"], ["unmapped_config.yaml"])

    def test_known_test_infrastructure_is_not_an_unknown_unit_path(self) -> None:
        for path in (
            "tests/test_utils/runners/run_ci_unit_tests.sh",
            "tests/functional_tests/test_cases/gpt/qwen3_0p6b_mcore_te_tp2_pp2_benchmark_s60/golden_values_dev_s60.json",
        ):
            plan = build_plan([path], platforms=["enflame"])
            self.assertFalse(plan["global_groups"]["unmapped_paths"])
            self.assertNotEqual(plan["plan_status"], "blocked")
            self.assertEqual(list(plan["expected_unit_groups"]), ["enflame"])

    def test_configured_nightly_still_selects_every_configured_group(self) -> None:
        plan = build_plan([], selection="configured", level="L2", event="schedule")
        self.assertEqual(plan["plan_status"], "ready")
        self.assertEqual(len(plan["platforms"]), 8)
        self.assertEqual(sum(map(len, plan["expected_unit_groups"].values())), 71)
        for platform in plan["platforms"]:
            self.assertEqual(
                plan["expected_unit_groups"][platform["name"]], platform["configured_groups"]
            )


class EvidenceIntegrityTests(unittest.TestCase):
    def test_real_pytest_success_skip_failure_and_empty(self) -> None:
        for source, success in (
            ("def test_ok(): assert 1 + 1 == 2", True),
            (
                "import pytest\n@pytest.mark.skip(reason='unavailable')\ndef test_skip(): pass",
                False,
            ),
            ("def test_fail(): assert False", False),
            ("value = 1", False),
        ):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                test = root / "test_probe.py"
                test.write_text(source + "\n")
                env = {
                    **os.environ,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "CI_JUNIT_DIRECTORY": str(root / "junit"),
                    "CI_REQUIRE_TEST_EVIDENCE": "1",
                    "CI_JUNIT_REPORTS": '["normal-0.xml"]',
                }
                code = runner.execute(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-c",
                        "/dev/null",
                        str(test),
                        f"--junitxml={root}/junit/normal-0.xml",
                    ],
                    env,
                    root / "logs",
                    timeout=15,
                )
                result = json.loads((root / "logs/result.json").read_text())
                self.assertEqual(code == 0, success)
                self.assertEqual(not evidence.validate(result.get("evidence")), success)

    def test_missing_rank_corrupt_report_and_empty_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "normal-0.xml").write_text('<testsuite tests="1"><testcase/></testsuite>')
            with self.assertRaisesRegex(ValueError, "report set"):
                evidence.collect(root, ["normal-0.xml", "normal-1.xml"])
            (root / "normal-0.xml").write_text(
                '<testsuite tests="2" skipped="0" failures="0" errors="0">'
                "<testcase/></testsuite>"
            )
            report = evidence.collect(root, ["normal-0.xml"])
            self.assertEqual(report["reports"]["normal-0.xml"]["tests"], 2)
            self.assertEqual(report["reports"]["normal-0.xml"]["passed"], 2)
            (root / "normal-0.xml").write_text(
                '<testsuite tests="1" skipped="2" failures="0" errors="0">'
                "<testcase/></testsuite>"
            )
            with self.assertRaisesRegex(ValueError, "inconsistent JUnit counts"):
                evidence.collect(root, ["normal-0.xml"])
            (root / "normal-0.xml").write_text(
                '<testsuite tests="1" skipped="0" failures="0" errors="0">'
                "<testcase><failure/></testcase></testsuite>"
            )
            with self.assertRaisesRegex(ValueError, "failures count mismatch"):
                evidence.collect(root, ["normal-0.xml"])
            (root / "normal-0.xml").write_text('<testsuite><testcase/></testsuite>')
            (root / "experimental-0.xml").write_text('<testsuite tests="0"/>')
            self.assertFalse(
                evidence.validate(evidence.collect(root, ["normal-0.xml", "experimental-0.xml"]))
            )

    def test_gate_rejects_changed_content_and_missing_group_detail(self) -> None:
        plan = build_plan([], selection="configured", platforms=["cuda"])
        kwargs = dict(
            expected_hash=plan["plan_hash"],
            expected_groups=plan["expected_unit_groups"],
            allow_catalog_partial=False,
            required_statuses=["ready"],
        )
        for field, value in (
            ("source_sha", "other"),
            ("config_hash", "other"),
            ("schema_version", 99),
        ):
            altered = copy.deepcopy(plan)
            altered[field] = value
            self.assertIn(
                "plan content does not match its hash", gate.verify_plan(altered, **kwargs)
            )
        self.assertEqual(
            contract.plan_hash(json.loads(json.dumps(plan, sort_keys=True))), plan["plan_hash"]
        )
        matrix = copy.deepcopy(planner.execution_matrix(plan))
        matrix[0]["config_identity"]["platform_hash"] = "different"
        self.assertIn(
            "matrix configuration identity differs from plan",
            gate.verify_plan(plan, matrix=matrix, **kwargs),
        )
        groups = plan["expected_unit_groups"]["cuda"]
        report = dict(
            plan_hash=plan["plan_hash"],
            platform="cuda",
            device="a100",
            expected_groups=groups,
            executed_groups=groups,
            status="passed",
        )
        self.assertTrue(gate.verify_attestations(plan, [report]))


class CandidateConfigTests(unittest.TestCase):
    def test_requested_plan_uses_candidate_data_and_executor_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            shutil.copytree(ROOT / ".flagcicd", root / ".flagcicd")
            shutil.copytree(ROOT / ".github/configs", root / ".github/configs")
            project_path = root / ".flagcicd/project.yaml"
            project = yaml.safe_load(project_path.read_text())
            trusted_policy = copy.deepcopy(project["gate_policy"]["manual"])
            project["gate_policy"]["manual"]["allow_catalog_partial"] = not trusted_policy[
                "allow_catalog_partial"
            ]
            for group in project["test_groups"]:
                if group["name"] == "optimizer":
                    group["status"] = "active"
                    group.pop("reason", None)
            project_path.write_text(yaml.safe_dump(project))
            config_path = root / ".github/configs/cuda.yml"
            config = yaml.safe_load(config_path.read_text())
            config["test_matrix"]["unit"]["groups"].append(
                {"name": "optimizer", "path": "tests/unit_tests/optimizer/"}
            )
            config_path.write_text(yaml.safe_dump(config))
            for args in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=CI Test",
                    "-c",
                    "user.email=ci@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True, timeout=10
                )
            sha = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            output = root / "plan.json"
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/ci/generate_test_plan.py"),
                    "--event-name",
                    "workflow_dispatch",
                    "--stage",
                    "manual",
                    "--config-root",
                    str(root),
                    "--source-sha",
                    sha,
                    "--platforms",
                    '["cuda"]',
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["gate_policy"], trusted_policy)
            self.assertIn("optimizer", plan["expected_unit_groups"]["cuda"])
            self.assertEqual(len(plan["expected_unit_groups"]["cuda"]), 11)
            identity = planner.execution_matrix(plan)[0]["config_identity"]
            workflow = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
            step = next(
                step
                for step in workflow["jobs"]["checkout_and_config"]["steps"]
                if step.get("name") == "Verify planned configuration identity"
            )
            script = step["run"].split("<<'PYCODE'\n")[1].rsplit("PYCODE", 1)[0]
            env = {**os.environ, "CONFIG_IDENTITY": json.dumps(identity), "PLATFORM": "cuda"}

            def verify() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            self.assertEqual(verify().returncode, 0)
            config_path.write_text(config_path.read_text() + "\n# drift\n")
            self.assertIn("configuration differs", verify().stderr)
            identity["source_sha"] = "different"
            env["CONFIG_IDENTITY"] = json.dumps(identity)
            self.assertIn("source SHA differs", verify().stderr)
