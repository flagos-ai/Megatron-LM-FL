"""Hardware-independent regression tests for CI routing, commands, and retries."""

from __future__ import annotations

import copy
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import cache
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import commands
import generate_matrix as matrix
import requests
import yaml

from tests.test_utils.python_scripts import download_unit_tests_dataset as downloads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_unit_groups as runner

ROOT = Path(__file__).resolve().parents[3]
STATUSES = ("success", "failure", "cancelled", "skipped")


@cache
def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


class PlatformRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = matrix._load_watch_rules(ROOT / ".github/configs")
        cls.platforms = sorted(cls.rules)

    def select(self, *paths, event="pull_request"):
        return matrix.select_platforms(self.rules, list(paths), event)[0]

    def test_platform_discovery_excludes_metadata(self):
        self.assertTrue(self.platforms, "At least one platform configuration is required")
        self.assertTrue(matrix._NON_PLATFORM_STEMS.isdisjoint(self.platforms))

    def test_each_configured_watch_path_selects_its_platform(self):
        for platform, watch in self.rules.items():
            with self.subTest(platform=platform):
                self.assertTrue(watch, f"Declare paths_watch for {platform}")
                self.assertIn(f".github/configs/{platform}.yml", watch)
            for entry in watch:
                # Probe directory ownership without depending on a particular
                # test-script name or requiring every platform to share a layout.
                path = entry + "routing_probe.py" if entry.endswith("/") else entry
                with self.subTest(platform=platform, path=path):
                    self.assertEqual(self.select(path), [platform])

    def test_functional_watch_paths_include_every_configured_consumer(self):
        for platform in self.platforms:
            config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
            for case in config.get("test_matrix", {}).get("functional", {}).get("train", []):
                path = f"tests/functional_tests/test_cases/{case['model']}/{case['test_case']}/"
                # An unmapped case runs all platforms; a mapped case must include
                # every platform that consumes it, even after future config edits.
                with self.subTest(platform=platform, path=path):
                    self.assertIn(platform, self.select(path + "model_config.yaml"))

    def test_platform_addition_and_removal_are_config_driven(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            for platform in ("existing", "new-accelerator"):
                (configs / f"{platform}.yml").write_text(
                    f"paths_watch:\n  - .github/configs/{platform}.yml\n"
                    f"  - megatron/plugin/{platform}/\n"
                )
            # Metadata must be ignored before attempting to parse it as a
            # platform config, regardless of its contents.
            for stem in matrix._NON_PLATFORM_STEMS:
                (configs / f"{stem}.yml").write_text("[]\n")
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"existing", "new-accelerator"})
            self.assertEqual(
                matrix.select_platforms(
                    rules, ["megatron/plugin/new-accelerator/test.py"], "pull_request"
                )[0],
                ["new-accelerator"],
            )
            (configs / "existing.yml").unlink()
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"new-accelerator"})

    def test_metadata_only_directory_is_not_a_platform_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            (configs / "template.yml").write_text("paths_watch: []\n")
            with self.assertRaisesRegex(ValueError, "No platform configurations"):
                matrix._load_watch_rules(configs)

    def test_vendor_aliases(self):
        for path, platform in (
            ("megatron/plugin/Ascend/optimizer.py", "ascend"),
            ("megatron/plugin/platform/platform_npu.py", "ascend"),
            ("megatron/plugin/mthreads/transformer/ops.py", "musa"),
            ("megatron/plugin/kunlunxin/debug.py", "kunlunxin"),
            ("megatron/plugin/metax/distributed/ops.py", "metax"),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.select(path), [platform])

    def test_multiple_platforms_and_documentation(self):
        self.assertEqual(
            self.select(
                "megatron/plugin/Ascend/test.py",
                "megatron/plugin/platform/platform_cuda.py",
                "README.md",
            ),
            ["ascend", "cuda"],
        )

    def test_shared_or_unknown_path_in_mixed_pr_runs_all(self):
        for path in (
            "megatron/core/transformer/attention.py",
            "megatron/plugin/override_registry.py",
            "megatron/plugin/platform/platform_manager.py",
            "megatron/plugin/tests/test_kunlunxin_debug.py",
            ".github/workflows/all_tests_common.yml",
            ".github/scripts/ci/generate_matrix.py",
            "tests/test_utils/runners/run_ci_unit_tests.sh",
            "examples/pytorch/new_example.py",
            "setup.py",
            "unknown/new.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select("megatron/plugin/platform/platform_cuda.py", path), self.platforms
                )

    def test_documentation_only(self):
        self.assertEqual(self.select("README.rst", "docs/conf.py", "examples/README.md"), [])

    def test_missing_diff_and_non_pr_events_run_all(self):
        self.assertEqual(self.select(), self.platforms)
        for event in ("push", "schedule", "workflow_dispatch"):
            self.assertEqual(self.select("README.md", event=event), self.platforms)

    def test_path_boundaries(self):
        self.assertFalse(
            matrix._matches("other/.github/configs/cuda.yml", ".github/configs/cuda.yml")
        )
        self.assertFalse(matrix._matches("tests/cuda_extra/test.py", "tests/cuda/"))

    def test_missing_watch_rules_run_all(self):
        rules = dict(self.rules, cuda=[])
        self.assertEqual(
            matrix.select_platforms(rules, ["megatron/plugin/Ascend/test.py"], "pull_request")[0],
            self.platforms,
        )

    def test_invalid_configs_fail(self):
        for content in ("[]", "paths_watch: cuda", "paths_watch: [null]", "invalid: ["):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                Path(directory, "cuda.yml").write_text(content)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    matrix._load_watch_rules(Path(directory))

    def test_rename_keeps_source_and_destination(self):
        with patch.object(matrix.subprocess, "run") as run:
            run.return_value.stdout = (
                "megatron/plugin/platform/platform_cuda.py\0megatron/plugin/Ascend/test.py\0"
            )
            paths = matrix._changed_files("main")
            self.assertIn("--no-renames", run.call_args.args[0])
            self.assertEqual(self.select(*paths), ["ascend", "cuda"])


class WorkflowGateTests(unittest.TestCase):
    def check_gate(self, workflow_name, job, expected, **env):
        command = workflow(workflow_name)["jobs"][job]["steps"][0]["run"]
        with self.subTest(env=env):
            result = subprocess.run(
                ["bash", "-e", "-c", command], env=dict(os.environ, **env), capture_output=True
            )
            self.assertEqual(result.returncode == 0, expected, result.stderr)

    def test_top_level_gate_status_combinations(self):
        gate = workflow("ci.yml")["jobs"]["ci-complete"]
        self.assertEqual(gate["if"], "always()")
        self.assertEqual(set(gate["needs"]), {"lint", "generate-matrix", "platform-pipeline"})
        cases = list(itertools.product(STATUSES, repeat=3))
        for platforms in ('["cuda"]', "[]", "", "null", "{}", "[1]"):
            for lint, generated, pipeline in (
                cases if platforms in ('["cuda"]', "[]") else [("success",) * 3]
            ):
                expected = lint == generated == "success" and (
                    (platforms == '["cuda"]' and pipeline == "success")
                    or (platforms == "[]" and pipeline == "skipped")
                )
                self.check_gate(
                    "ci.yml",
                    "ci-complete",
                    expected,
                    LINT_RESULT=lint,
                    MATRIX_RESULT=generated,
                    PIPELINE_RESULT=pipeline,
                    PLATFORMS=platforms,
                )

    def test_platform_gate_checks_enabled_tests_and_te_fl(self):
        self.assertEqual(
            workflow("all_tests_common.yml")["jobs"]["all_tests_complete"]["if"], "always()"
        )
        env = dict(
            CHECKOUT_RESULT="success",
            TE_FL_ARTIFACT="prepared-wheel",
            RUN_UNIT_TESTS="true",
            UNIT_RESULT="success",
            RUN_FUNCTIONAL_TESTS="true",
            FUNCTIONAL_RESULT="success",
        )
        for unit_enabled, functional_enabled in itertools.product((False, True), repeat=2):
            for checkout, unit, functional in itertools.product(STATUSES, repeat=3):
                expected = (
                    checkout == "success"
                    and (not unit_enabled or unit == "success")
                    and (not functional_enabled or functional == "success")
                )
                env.update(
                    CHECKOUT_RESULT=checkout,
                    RUN_UNIT_TESTS=str(unit_enabled).lower(),
                    UNIT_RESULT=unit,
                    RUN_FUNCTIONAL_TESTS=str(functional_enabled).lower(),
                    FUNCTIONAL_RESULT=functional,
                )
                self.check_gate("all_tests_common.yml", "all_tests_complete", expected, **env)
        env.update(
            CHECKOUT_RESULT="success",
            UNIT_RESULT="success",
            FUNCTIONAL_RESULT="success",
            TE_FL_ARTIFACT="",
        )
        self.check_gate("all_tests_common.yml", "all_tests_complete", False, **env)

    def test_common_checks_gate_every_entrypoint_without_platform_lint(self):
        for name in ("all_tests_common.yml", "_platform_pipeline.yml"):
            data = workflow(name)
            self.assertNotIn("lint", data["jobs"])
            self.assertNotIn("run_common_checks", data[True]["workflow_call"]["inputs"])
        entrypoints = [("ci.yml", "platform-pipeline"), ("ci-requested.yml", "platform-pipeline")]
        entrypoints.extend(
            (path.name, "run_tests")
            for path in (ROOT / ".github/workflows").glob("all_tests_*.yml")
            if path.name != "all_tests_common.yml"
        )
        for name, job_name in entrypoints:
            with self.subTest(workflow=name):
                jobs = workflow(name)["jobs"]
                self.assertEqual(jobs["lint"]["uses"], "./.github/workflows/lint_common.yml")
                if name != "ci-requested.yml":
                    self.assertNotIn("if", jobs["lint"])

                # Walk prerequisites: no platform can start before common checks succeed.
                def ancestors(job):
                    needs = jobs[job].get("needs", [])
                    needs = [needs] if isinstance(needs, str) else needs
                    return set(needs).union(*(ancestors(parent) for parent in needs))

                self.assertIn("lint", ancestors(job_name))
                self.assertNotIn("always()", jobs[job_name].get("if", ""))
                self.assertNotIn("run_common_checks", jobs[job_name]["with"])

    def test_platform_pipeline_requires_prepared_te_fl(self):
        jobs = workflow("_platform_pipeline.yml")["jobs"]
        self.assertEqual(jobs["test"]["needs"], "te_fl_prepare")
        self.assertEqual(
            jobs["test"]["with"]["te_fl_artifact"],
            "${{ needs.te_fl_prepare.outputs.wheel_artifact }}",
        )


SHA = "a" * 40
OTHER = "b" * 40
PR = {"state": "open", "head": {"sha": SHA}, "base": {"ref": "main"}, "merge_commit_sha": OTHER}
EVENT = {
    "action": "created",
    "issue": {"number": 7, "pull_request": {}},
    "comment": {"body": "/ci run ascend", "user": {"login": "maintainer"}},
}
RULES = {"ascend": ["ascend/"], "cuda": ["cuda/"]}


class LabelTests(unittest.TestCase):
    def test_labels_override_path_routing_like_te(self):
        for paths, labels, expected in (
            (["ascend/ops.py"], ["platform:cuda"], ["cuda"]),
            (["common.py"], ["platform:ascend"], ["ascend"]),
            (["common.py"], ["platform:cuda", "platform:ascend"], ["ascend", "cuda"]),
            (["README.md"], ["platform:ascend"], ["ascend"]),
            (["ascend/ops.py"], ["platform:all"], ["ascend", "cuda"]),
            (["ascend/ops.py"], [], ["ascend"]),
            (["ascend/ops.py"], ["Run tests"], ["ascend"]),
        ):
            with self.subTest(paths=paths, labels=labels):
                self.assertEqual(
                    matrix.select_platforms(RULES, paths, "pull_request", labels)[0], expected
                )
        with self.assertRaises(ValueError):
            matrix.select_platforms(RULES, ["README.md"], "pull_request", ["platform:typo"])


class CommentTests(unittest.TestCase):
    def setUp(self):
        context = patch.dict(
            os.environ,
            GITHUB_RUN_ID="123",
            GITHUB_REPOSITORY="owner/repo",
            GITHUB_SERVER_URL="https://github.com",
        )
        context.start()
        self.addCleanup(context.stop)

    def api(self, permission="write", runs=None, checks=None, owner_status="in_progress"):
        def call(method, path, body=None):
            if path.endswith("/permission"):
                return {"permission": permission}
            if path == "/pulls/7":
                return copy.deepcopy(PR)
            if path.startswith("/actions/workflows/ci.yml/runs?"):
                return {"workflow_runs": runs or []}
            if path.startswith("/commits/"):
                return {"check_runs": checks or []}
            if path.startswith("/actions/runs/") and method == "GET":
                return {"status": owner_status}
            if path == "/check-runs":
                return {"id": 42}
            return None

        return Mock(request=Mock(side_effect=call))

    def test_commands_are_exact_and_cannot_contain_shell_code(self):
        for body in (
            "/ci run ascend; echo bad",
            "hello /ci run all",
            "/ci run all\nextra",
            "/ci run ../main",
            "`/ci run all`",
        ):
            self.assertIsNone(commands.parse_command(body))
        self.assertEqual(commands.parse_command("/ci rerun failed"), ("rerun", "failed"))
        for command in ("/ci run cuda,ascend", "/test ascend cuda", "/ci run cuda, cuda ascend"):
            parsed = commands.parse_command(command)
            self.assertEqual(parsed[0], "run")
            self.assertEqual(commands.requested_platforms(parsed[1], RULES), ["ascend", "cuda"])

    def test_unauthorized_comment_never_mutates_github(self):
        for permission in ("read", "triage", "none"):
            api = self.api(permission)
            with self.assertRaises(ValueError):
                commands.handle_comment(EVENT, api, RULES)
            self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))

    def test_comment_starts_tests_for_current_sha(self):
        api = self.api()
        outputs = commands.handle_comment(EVENT, api, RULES)
        self.assertEqual(outputs["source_ref"], SHA)
        self.assertEqual(outputs["pr_number"], "7")
        api.request.assert_any_call("POST", "/check-runs", unittest.mock.ANY)

    def test_unknown_platform_is_rejected_before_dispatch(self):
        event = copy.deepcopy(EVENT)
        event["comment"]["body"] = "/ci run unknown"
        api = self.api()
        with self.assertRaises(ValueError):
            commands.handle_comment(event, api, RULES)
        self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))

    def test_multi_platform_alias_creates_a_canonical_request(self):
        event = copy.deepcopy(EVENT)
        event["comment"]["body"] = "/test cuda,ascend,cuda"
        api = self.api()
        outputs = commands.handle_comment(event, api, RULES)
        self.assertEqual(json.loads(outputs["platforms"]), ["ascend", "cuda"])
        created = next(
            call.args[2]
            for call in api.request.call_args_list
            if call.args[:2] == ("POST", "/check-runs")
        )
        self.assertIn(":ascend,cuda:", created["external_id"])
        with self.assertRaises(ValueError):
            commands.requested_platforms("all,unknown", RULES)

    def test_issue_comments_and_edits_do_not_dispatch(self):
        for event in ({**EVENT, "action": "edited"}, {**EVENT, "issue": {"number": 7}}):
            api = self.api()
            commands.handle_comment(event, api, RULES)
            api.request.assert_not_called()

    def test_rerun_filters_pr_sha_and_latest_conclusion(self):
        event = copy.deepcopy(EVENT)
        event["comment"]["body"] = "/ci rerun failed"
        run = {
            "id": 12,
            "head_sha": SHA,
            "pull_requests": [{"number": 7}],
            "event": "pull_request",
            "status": "completed",
            "conclusion": "failure",
        }
        api = self.api(
            runs=[
                run,
                {**run, "id": 99, "head_sha": "c" * 40},
                {**run, "id": 98, "pull_requests": [{"number": 8}]},
            ]
        )
        commands.handle_comment(event, api, RULES)
        api.request.assert_any_call("POST", "/actions/runs/12/rerun-failed-jobs", {})
        for latest in (
            {**run, "id": 13, "conclusion": "success"},
            {**run, "id": 13, "status": "in_progress"},
        ):
            api = self.api(runs=[run, latest])
            with self.assertRaises(ValueError):
                commands.handle_comment(event, api, RULES)
            self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))

    def test_sha_mismatch_rejects_dispatch_preparation(self):
        api = self.api()
        with self.assertRaises(ValueError):
            commands.prepare_request(api, 7, OTHER, "all", RULES)
        self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))

    def test_supplemental_check_is_bound_to_head_and_separate_from_gate(self):
        api = self.api()
        outputs = commands.prepare_request(api, 7, SHA, "all", RULES)
        self.assertEqual(outputs["source_ref"], SHA)
        check = next(
            call.args[2]
            for call in api.request.call_args_list
            if call.args[:2] == ("POST", "/check-runs")
        )
        self.assertEqual(check["head_sha"], SHA)
        self.assertEqual(check["name"], "CI Requested Tests")

    def test_duplicate_request_reuses_active_run_and_closes_cancelled_check(self):
        check = {
            "id": 40,
            "status": "in_progress",
            "name": "CI Requested Tests",
            "app": {"slug": "github-actions"},
            "external_id": f"ci-requested:7:{SHA}:ascend:100",
            "details_url": "https://github.com/owner/repo/actions/runs/100",
        }
        api = self.api(checks=[check])
        self.assertIn("Reusing", commands.prepare_request(api, 7, SHA, "ascend", RULES))
        self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))
        api = self.api(checks=[check], owner_status="completed")
        outputs = commands.prepare_request(api, 7, SHA, "ascend", RULES)
        self.assertEqual(outputs["source_ref"], SHA)
        api.request.assert_any_call(
            "PATCH", "/check-runs/40", {"status": "completed", "conclusion": "cancelled"}
        )

    def test_cancelled_request_finishes_its_check(self):
        api = Mock()
        api.request.side_effect = [
            {
                "head_sha": SHA,
                "external_id": f"ci-requested:7:{SHA}:ascend:123",
                "name": "CI Requested Tests",
            },
            PR,
            None,
        ]
        self.assertEqual(commands.report_request(api, 7, SHA, 42, "cancelled"), "cancelled")
        self.assertEqual(
            api.request.call_args.args[2], {"status": "completed", "conclusion": "cancelled"}
        )

    def test_old_commit_results_cannot_report_success_on_new_commit(self):
        api = Mock()
        api.request.side_effect = [
            {
                "head_sha": SHA,
                "external_id": f"ci-requested:7:{SHA}:ascend:123",
                "name": "CI Requested Tests",
            },
            {**PR, "head": {"sha": OTHER}},
            None,
        ]
        self.assertEqual(commands.report_request(api, 7, SHA, 42, "success"), "cancelled")
        self.assertEqual(api.request.call_args.args[2]["conclusion"], "cancelled")


class WorkflowIsolationTests(unittest.TestCase):
    def test_every_megatron_checkout_uses_the_requested_sha(self):
        for name in (
            "prepare_te_fl.yml",
            "all_tests_common.yml",
            "unit_tests_common.yml",
            "functional_tests_common.yml",
            "lint_common.yml",
        ):
            data = workflow(name)
            self.assertIn("source_ref", data[True]["workflow_call"]["inputs"])
            for job in data["jobs"].values():
                for step in job.get("steps", []):
                    if step.get("uses") != "actions/checkout@v4":
                        continue
                    options = step["with"]
                    if options.get("repository") == "flagos-ai/TransformerEngine-FL":
                        self.assertEqual(options["ref"], "main")
                        continue
                    self.assertEqual(options["ref"], "${{ inputs.source_ref || github.sha }}")
                    self.assertIs(options["persist-credentials"], False)

    def test_sha_is_forwarded_through_all_reusable_calls(self):
        for name in ("_platform_pipeline.yml", "all_tests_common.yml"):
            for job in workflow(name)["jobs"].values():
                if job.get("uses", "").startswith("./.github/workflows/"):
                    self.assertEqual(job["with"]["source_ref"], "${{ inputs.source_ref }}")

    def test_privileged_control_jobs_only_checkout_default_branch(self):
        requested = workflow("ci-requested.yml")
        for job in (requested["jobs"]["validate"], requested["jobs"]["report"]):
            for step in job["steps"]:
                if step.get("uses") == "actions/checkout@v4":
                    self.assertEqual(
                        step["with"]["ref"], "${{ github.event.repository.default_branch }}"
                    )
                    self.assertIs(step["with"]["persist-credentials"], False)
        pipeline = requested["jobs"]["platform-pipeline"]
        self.assertEqual(pipeline["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", pipeline)
        self.assertNotIn("ci-complete", requested["jobs"])

    def test_retry_artifacts_include_hidden_attempt_directories(self):
        for name in ("unit_tests_common.yml", "functional_tests_common.yml"):
            uploads = [
                step
                for job in workflow(name)["jobs"].values()
                for step in job.get("steps", [])
                if step.get("with", {}).get("path") == ".retry-logs/"
            ]
            self.assertEqual(len(uploads), 1)
            self.assertIs(uploads[0]["with"]["include-hidden-files"], True)

    def test_request_locks_queue_without_dropping_other_platform_requests(self):
        jobs = workflow("ci-requested.yml")["jobs"]
        for name in ("validate", "platform-pipeline"):
            self.assertFalse(jobs[name]["concurrency"]["cancel-in-progress"])
            self.assertEqual(jobs[name]["concurrency"]["queue"], "max")
        self.assertNotIn("comment.id", jobs["validate"]["concurrency"]["group"])
        self.assertIn("source_ref", jobs["platform-pipeline"]["concurrency"]["group"])
        self.assertEqual(set(jobs["platform-pipeline"]["needs"]), {"validate", "lint"})

    def test_label_changes_recompute_the_automatic_matrix(self):
        triggers = workflow("ci.yml")[True]["pull_request"]["types"]
        self.assertIn("labeled", triggers)
        self.assertIn("unlabeled", triggers)


class DownloadTests(unittest.TestCase):
    def test_interrupted_stream_retries_only_download_and_replaces_partial_archive(self):
        def interrupted():
            yield b"partial"
            raise requests.ReadTimeout("slow stream")

        first, second = MagicMock(), MagicMock()
        for response in (first, second):
            response.__enter__.return_value = response
        first.iter_content.return_value = interrupted()
        second.iter_content.return_value = [b"complete archive"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloads.requests, "get", side_effect=[first, second]) as get,
            patch.object(downloads.time, "sleep") as sleep,
        ):
            path = Path(directory) / "data.zip"
            downloads._download_asset("https://example.com/data.zip", path)
            self.assertEqual(path.read_bytes(), b"complete archive")
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args.kwargs["timeout"], (10, 60))
            sleep.assert_called_once_with(2)
            first.__exit__.assert_called_once()

    def test_download_retries_are_bounded_and_do_not_retry_http_404(self):
        for error, attempts in (
            (requests.ConnectTimeout("connect"), 2),
            (requests.exceptions.ChunkedEncodingError("broken stream"), 2),
            (requests.HTTPError(response=Mock(status_code=503)), 2),
            (requests.HTTPError(response=Mock(status_code=404)), 1),
        ):
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as directory,
                patch.object(downloads.requests, "get", side_effect=error) as get,
                patch.object(downloads.time, "sleep"),
            ):
                path = Path(directory) / "data.zip"
                path.write_bytes(b"partial")
                with self.assertRaises(type(error)):
                    downloads._download_asset("https://example.com/data.zip", path)
                self.assertEqual(get.call_count, attempts)
                self.assertFalse(path.exists())


class RetryTests(unittest.TestCase):
    def test_cancellation_exit_codes_and_retry_limits(self):
        for code in (-15, 0, 124, 130, 137, 143):
            self.assertFalse(runner._retryable_exit(code))
        for value in (-1, 2, True, "anything"):
            with self.assertRaises(ValueError):
                runner.retry_limit(value)

    def test_transient_retry_archives_failed_output_and_records_flaky_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "checkpoints").mkdir()
            (output / "checkpoints/large.pt").write_bytes(b"checkpoint")
            (output / "weights.safetensors").write_bytes(b"weights")
            script = (
                "from pathlib import Path; import sys; "
                f"p=Path({str(root / 'count')!r}); "
                "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
                f"Path({str(output / 'coverage')!r}).write_text(str(n)); "
                "print('ReadTimeout' if n==1 else 'passed'); sys.exit(1 if n==1 else 0)"
            )
            self.assertEqual(
                runner.execute(
                    [sys.executable, "-c", script],
                    dict(os.environ),
                    root / "logs",
                    1,
                    cleanup=output,
                ),
                0,
            )
            self.assertEqual((output / "coverage").read_text(), "2")
            self.assertEqual((root / "logs/attempt-1-outputs/coverage").read_text(), "1")
            self.assertFalse((root / "logs/attempt-1-outputs/checkpoints").exists())
            self.assertFalse((root / "logs/attempt-1-outputs/weights.safetensors").exists())
            self.assertFalse((output / "checkpoints").exists())
            result = json.loads((root / "logs/result.json").read_text())
            self.assertEqual(result["status"], "FLAKY_PASS")
            self.assertEqual([item["exit_code"] for item in result["attempts"]], [1, 0])

    def test_retries_are_bounded_and_disabled_by_default(self):
        for retries, message, attempts in (
            (0, "ReadTimeout", 1),
            (1, "ReadTimeout", 2),
            (1, "AssertionError ReadTimeout", 1),
            (1, "golden values mismatch ReadTimeout", 1),
            (1, "ordinary test failure", 1),
            (1, "ImportError ReadTimeout", 1),
        ):
            with (
                self.subTest(retries=retries, message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.assertEqual(
                    runner.execute(
                        [sys.executable, "-c", f"print({message!r}); raise SystemExit(7)"],
                        dict(os.environ),
                        root,
                        retries,
                    ),
                    7,
                )
                self.assertEqual(
                    len(json.loads((root / "result.json").read_text())["attempts"]), attempts
                )

    def test_early_assertion_in_large_log_prevents_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = "print('AssertionError'); print('x'*300000); print('ReadTimeout'); raise SystemExit(1)"
            with patch("builtins.print"):
                self.assertEqual(
                    runner.execute([sys.executable, "-c", script], dict(os.environ), root, 1), 1
                )
            self.assertEqual(len(json.loads((root / "result.json").read_text())["attempts"]), 1)

    def test_real_short_traceback_and_junit_failures(self):
        for body, expected_attempts in (
            ("print('ReadTimeout'); assert 1 == 2", 1),
            ("raise ReadTimeout('slow')", 2),
            ("print('ReadTimeout'); assert False, 'ReadTimeout'", 1),
            ("import pytest; pytest.fail('ReadTimeout')", 1),
        ):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                test = root / "test_example.py"
                test.write_text(
                    "class ReadTimeout(Exception): pass\ndef test_example():\n    " + body + "\n"
                )
                reports = root / "junit"
                env = {**os.environ, "CI_JUNIT_DIRECTORY": str(reports), "CI_JUNIT_EXPECTED": "1"}
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-c",
                    "/dev/null",
                    "--tb=short",
                    "-q",
                    f"--junitxml={reports / 'rank-0.xml'}",
                    str(test),
                ]
                with patch("builtins.print"):
                    self.assertEqual(runner.execute(command, env, root / "logs", 1), 1)
                result = json.loads((root / "logs/result.json").read_text())
                self.assertEqual(len(result["attempts"]), expected_attempts)

    def test_junit_must_be_complete_and_all_failures_transient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"CI_JUNIT_DIRECTORY": directory, "CI_JUNIT_EXPECTED": "2"}
            self.assertFalse(runner.junit_allows_retry(env))
            network = (
                '<testsuite><testcase><failure message="ReadTimeout: slow"/></testcase></testsuite>'
            )
            (root / "rank0.xml").write_text(network)
            self.assertFalse(runner.junit_allows_retry(env))
            (root / "rank1.xml").write_text('<testsuite><testcase/></testsuite>')
            self.assertTrue(runner.junit_allows_retry(env))
            for invalid in (
                '<bad',
                '<unknown/>',
                '<testsuite><testcase><failure message="assert ReadTimeout"/></testcase></testsuite>',
                '<testsuite><testcase><failure message="Failed: ReadTimeout"/></testcase></testsuite>',
            ):
                (root / "rank1.xml").write_text(invalid)
                self.assertFalse(runner.junit_allows_retry(env))

    def test_pytest_worker_keeps_rank_and_phase_reports_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "junit"
            reports.mkdir()
            (root / ".coveragerc").write_text(
                f"[run]\nparallel = true\ndata_file = {root}/.coverage\n"
            )
            test = root / "test_example.py"
            test.write_text("def test_example(): assert True\n")
            for phase, rank in itertools.product(("normal", "experimental"), ("0", "1")):
                env = {
                    **os.environ,
                    "CI_JUNIT_DIRECTORY": str(reports),
                    "CI_COVERAGE_DIRECTORY": str(root),
                    "CI_JUNIT_PHASE": phase,
                    "RANK": rank,
                }
                command = [
                    sys.executable,
                    str(ROOT / ".github/scripts/run_unit_groups.py"),
                    "pytest-worker",
                    str(Path(sys.executable).with_name("pytest")),
                    "-c",
                    "/dev/null",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    str(test),
                ]
                result = subprocess.run(command, env=env, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in reports.glob("*.xml")},
                {
                    f"{phase}-{rank}.xml"
                    for phase, rank in itertools.product(("normal", "experimental"), ("0", "1"))
                },
            )

    def test_timeout_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                runner.execute(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('ReadTimeout', flush=True); time.sleep(20)",
                    ],
                    dict(os.environ),
                    root,
                    1,
                    timeout=0.1,
                ),
                124,
            )
            self.assertEqual(len(json.loads((root / "result.json").read_text())["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
