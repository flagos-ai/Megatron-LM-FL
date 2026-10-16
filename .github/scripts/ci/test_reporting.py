"""Check coverage upload configuration and secret forwarding."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


class ReportWorkflowTests(unittest.TestCase):
    def test_upload_prerequisites_require_report_and_token(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/unit_tests_common.yml").read_text())
        steps = workflow["jobs"]["unit_test"]["steps"]
        check = next(step for step in steps if step.get("id") == "flagcicd")
        upload = next(
            step for step in steps if step["name"] == "Upload Coverage Report to FlagCICD"
        )
        self.assertIn("!cancelled()", check["if"])
        self.assertNotIn("success()", check["if"])
        self.assertNotIn("success()", upload["if"])
        self.assertIn("steps.flagcicd.outputs.upload_ready == 'true'", upload["if"])
        self.assertEqual(upload["timeout-minutes"], 2)
        self.assertEqual(check["env"]["COVERAGE_REPORT_PATH"], upload["with"]["report_path"])
        for has_report, has_token in ((False, False), (False, True), (True, False), (True, True)):
            with (
                self.subTest(report=has_report, token=has_token),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = Path(directory) / "coverage.json"
                output = Path(directory) / "output"
                if has_report:
                    report.write_text('{"totals": {}}')
                result = subprocess.run(
                    ["bash", "-e", "-u", "-c", check["run"]],
                    env={
                        **os.environ,
                        "COVERAGE_REPORT_PATH": str(report),
                        "FLAGCICD_API_TOKEN": "dummy-token" if has_token else "",
                        "GITHUB_OUTPUT": str(output),
                    },
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ready = str(has_report and has_token).lower()
                self.assertEqual(output.read_text(), f"upload_ready={ready}\n")
                self.assertNotIn("dummy-token", result.stdout + result.stderr)
                if not (has_report and has_token):
                    self.assertIn("Skipping FlagCICD upload", result.stdout)

    def test_secret_reaches_coverage_uploader(self):
        workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / ".github/workflows").glob("*.yml")
        }
        callers = {"_platform_pipeline.yml", "all_tests_common.yml", "unit_tests_common.yml"}
        for name in callers:
            self.assertIn("FLAGCICD_API_TOKEN", workflows[name][True]["workflow_call"]["secrets"])
        for name, workflow in workflows.items():
            for job in workflow.get("jobs", {}).values():
                target = job.get("uses", "").removeprefix("./.github/workflows/")
                if target in callers and name != "ci-requested.yml":
                    self.assertEqual(
                        job["secrets"], {"FLAGCICD_API_TOKEN": "${{ secrets.FLAGCICD_API_TOKEN }}"}
                    )
        upload = next(
            step
            for job in workflows["unit_tests_common.yml"]["jobs"].values()
            for step in job.get("steps", [])
            if step.get("uses") == "flagos-ai/FlagOps/actions/post-pytest-report@v2"
        )
        self.assertTrue(upload["continue-on-error"])
        self.assertEqual(upload["with"]["api_token"], "${{ secrets.FLAGCICD_API_TOKEN }}")
        self.assertEqual(
            upload["with"]["backend_url"],
            "https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/",
        )
        self.assertNotIn("user_id", upload["with"])
        self.assertIn("${{ inputs.platform }}", upload["with"]["job_name"])
        self.assertIn("${{ inputs.device }}", upload["with"]["job_name"])

    def test_functional_training_has_no_performance_reporting(self):
        data = yaml.safe_load((ROOT / ".github/workflows/functional_tests_common.yml").read_text())
        self.assertNotIn("secrets", data[True]["workflow_call"])
        for job in data["jobs"].values():
            self.assertNotIn("PERFORMANCE_REPORT_NAME", job.get("env", {}))
            for step in job.get("steps", []):
                self.assertNotEqual(
                    step.get("uses"), "flagos-ai/FlagOps/actions/post-pytest-report@v2"
                )
                self.assertNotIn("performance_report.py", step.get("run", ""))


if __name__ == "__main__":
    unittest.main()
