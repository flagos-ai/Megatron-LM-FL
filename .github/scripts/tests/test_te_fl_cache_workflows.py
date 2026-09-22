import itertools
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    with (ROOT / "workflows" / name).open() as stream:
        return yaml.safe_load(stream)


class WheelCacheWorkflowTests(unittest.TestCase):
    def test_backup_identity_is_forwarded_from_producer_to_consumers(self):
        prepare = workflow("prepare_te_fl.yml")
        self.assertEqual(
            prepare[True]["workflow_call"]["outputs"]["local_cache_key"]["value"],
            "${{ jobs.prepare.outputs.local_cache_key }}",
        )
        self.assertEqual(
            prepare["jobs"]["prepare"]["outputs"]["local_cache_key"],
            "${{ steps.cache_key.outputs.local_key }}",
        )
        caller = workflow("all_tests_cuda.yml")
        self.assertEqual(
            caller["jobs"]["run_tests"]["with"]["te_fl_local_cache_key"],
            "${{ needs.te_fl_prepare.outputs.local_cache_key }}",
        )
        common = workflow("all_tests_common.yml")
        for name in ("unit_tests", "functional_tests"):
            self.assertEqual(
                common["jobs"][name]["with"]["te_fl_local_cache_key"],
                "${{ inputs.te_fl_local_cache_key }}",
            )
        for job in common["jobs"].values():
            for step in job.get("steps", []):
                self.assertNotIn("GITHUB_RUN_ATTEMPT", step.get("run", ""))

    def test_backup_identity_is_unique_per_producer_attempt(self):
        steps = workflow("prepare_te_fl.yml")["jobs"]["prepare"]["steps"]
        script = next(step["run"] for step in steps if step.get("id") == "cache_key")
        for identity in ("GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            self.assertIn(f'"${identity}"', script)
        publish = next(
            step for step in steps if step["name"] == "Publish local TE-FL wheel fallback"
        )
        self.assertEqual(
            publish["env"]["TE_FL_CACHE_KEY"], "${{ steps.cache_key.outputs.local_key }}"
        )
        self.assertTrue(publish["continue-on-error"])

    def test_remote_cache_remains_primary(self):
        steps = workflow("prepare_te_fl.yml")["jobs"]["prepare"]["steps"]
        restore = next(step for step in steps if step.get("id") == "cache")
        self.assertNotIn("if", restore)
        save = next(step for step in steps if step["name"] == "Save TE-FL wheel cache")
        self.assertEqual(save["if"], "steps.cache.outputs.cache-hit != 'true'")

    def test_consumer_fallback_and_failure_gate(self):
        for name in ("unit_tests_common.yml", "functional_tests_common.yml"):
            document = workflow(name)
            steps = next(job["steps"] for job in document["jobs"].values() if "steps" in job)
            by_id = {step["id"]: step for step in steps if "id" in step}
            fallback = by_id["local_cache"]
            self.assertEqual(
                fallback["env"]["TE_FL_CACHE_KEY"], "${{ inputs.te_fl_local_cache_key }}"
            )
            for number in range(1, 4):
                remote = by_id[f"te_fl_cache_{number}"]
                self.assertNotIn("cache_directory", remote["if"])
                self.assertTrue(remote["continue-on-error"])
                self.assertLess(steps.index(remote), steps.index(fallback))
            for enabled, directory, local_key, *hits in itertools.product((False, True), repeat=6):
                expression = fallback["if"]
                values = {
                    "inputs.te_fl_prepare": enabled,
                    "inputs.te_fl_cache_key": "remote-key",
                    "inputs.te_fl_cache_directory": "/cache" if directory else "",
                    "inputs.te_fl_local_cache_key": "backup-key" if local_key else "",
                    **{
                        f"steps.te_fl_cache_{i}.outputs.cache-hit": "true" if hit else ""
                        for i, hit in enumerate(hits, 1)
                    },
                }
                for key, value in values.items():
                    expression = expression.replace(key, repr(value))
                actual = eval(expression.replace("&&", " and "), {"__builtins__": {}}, {})
                self.assertEqual(actual, enabled and directory and local_key and not any(hits))
            gate = next(step for step in steps if step["name"] == "Check TE-FL wheel cache")
            for hits in itertools.product((False, True), repeat=4):
                env = {f"CACHE_HIT_{i}": str(hit).lower() for i, hit in enumerate(hits[:3], 1)}
                env["LOCAL_CACHE_HIT"] = str(hits[3]).lower()
                result = subprocess.run(
                    ["bash", "-c", gate["run"]], env=env, capture_output=True, timeout=5
                )
                self.assertEqual(result.returncode == 0, any(hits))
            install = next(step for step in steps if step["name"] == "Install TE-FL wheel")
            self.assertIn("steps.local_cache.outputs.cache-hit", install["env"]["TE_FL_WHEEL_DIR"])


if __name__ == "__main__":
    unittest.main()
