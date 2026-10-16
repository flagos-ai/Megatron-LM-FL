import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / "workflows" / name).read_text())


class WheelCacheWorkflowTests(unittest.TestCase):
    def test_preparation_uses_cache_only_for_acceleration(self):
        prepare = workflow("prepare_te_fl.yml")
        self.assertEqual(set(prepare[True]["workflow_call"]["outputs"]), {"wheel_artifact"})
        self.assertEqual(
            prepare[True]["workflow_call"]["outputs"]["wheel_artifact"]["value"],
            "${{ jobs.prepare.outputs.wheel_artifact }}",
        )
        steps = prepare["jobs"]["prepare"]["steps"]
        restore = next(step for step in steps if step.get("id") == "cache")
        self.assertNotIn("if", restore)
        save = next(step for step in steps if step["name"] == "Save TE-FL wheel cache")
        self.assertEqual(save["if"], "steps.cache.outputs.cache-hit != 'true'")
        self.assertTrue(save["continue-on-error"])
        upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertNotIn("continue-on-error", upload)
        self.assertNotIn("if", upload)

    def test_every_test_entrypoint_forwards_the_prepared_artifact(self):
        pipeline = workflow("_platform_pipeline.yml")
        self.assertEqual(
            pipeline["jobs"]["test"]["with"]["te_fl_artifact"],
            "${{ needs.te_fl_prepare.outputs.wheel_artifact }}",
        )
        for platform in ("ascend", "cuda", "enflame", "hygon", "kunlunxin", "metax", "musa", "ppu"):
            name = f"all_tests_{platform}.yml"
            data = workflow(name)
            self.assertEqual(set(data["jobs"]), {"lint", "run_tests"})
            call = data["jobs"]["run_tests"]
            self.assertEqual(call["uses"], "./.github/workflows/_platform_pipeline.yml")
            self.assertEqual(call["with"]["platform"], platform)
            dispatch = data[True]["workflow_dispatch"] or {}
            for option in dispatch.get("inputs", {}):
                self.assertEqual(call["with"][option], "${{ inputs." + option + " }}")
            self.assertNotIn("run_common_checks", call["with"])
        for option in ("image_override", "run_unit_tests", "run_functional_tests"):
            self.assertEqual(
                pipeline["jobs"]["test"]["with"][option], "${{ inputs." + option + " }}"
            )
        self.assertEqual(
            pipeline["jobs"]["te_fl_prepare"]["with"]["image_override"],
            "${{ inputs.image_override }}",
        )
        common = workflow("all_tests_common.yml")
        for name in ("unit_tests", "functional_tests"):
            self.assertEqual(
                common["jobs"][name]["with"]["te_fl_artifact"], "${{ inputs.te_fl_artifact }}"
            )

    def test_consumers_require_artifacts_without_cache_fallbacks(self):
        for name in ("unit_tests_common.yml", "functional_tests_common.yml"):
            data = workflow(name)
            inputs = data[True]["workflow_call"]["inputs"]
            self.assertTrue(inputs["te_fl_artifact"]["required"])
            self.assertNotIn("te_fl_prepare", inputs)
            self.assertFalse(
                any(
                    key.startswith("te_fl_cache") or key == "te_fl_local_cache_key"
                    for key in inputs
                )
            )
            steps = next(job["steps"] for job in data["jobs"].values() if "steps" in job)
            download = next(
                step for step in steps if step.get("uses") == "actions/download-artifact@v4"
            )
            self.assertNotIn("continue-on-error", download)
            self.assertNotIn("if", download)
            self.assertEqual(download["with"]["name"], "${{ inputs.te_fl_artifact }}")
            self.assertFalse(
                any(step.get("id") in ("te_fl_cache", "local_cache") for step in steps)
            )
            install = next(step for step in steps if step.get("name") == "Install TE-FL wheel")
            self.assertNotIn("if", install)
            self.assertEqual(
                install["env"]["TE_FL_WHEEL_DIR"], "${{ github.workspace }}/te-fl-wheel"
            )


if __name__ == "__main__":
    unittest.main()
