"""Hardware-independent tests for logical unit-group routing and plans."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import flagcicd_contract
import generate_matrix
import generate_test_plan as planner
import unit_group_routing as routing
import validate_unit_group_catalog as validator

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ROOT / ".github/configs"
PROJECT = ROOT / ".flagcicd/project.yaml"


def load_catalog() -> routing.Catalog:
    return routing.load_catalog(PROJECT)


def write_project_fixture(path: pathlib.Path, catalog: dict) -> None:
    """Create a complete project contract around focused routing test data."""
    project, _ = flagcicd_contract.load_project(PROJECT)
    project["global"] = catalog.get("global", {})
    project["levels"] = {"L1": {"required_groups": []}}
    project["stages"] = {"pr": {"selection": "affected", "default_level": "L1"}}
    project["gate_policy"] = {"pr": project["gate_policy"]["pr"]}
    project["test_groups"] = [dict(name=name, **entry) for name, entry in catalog["groups"].items()]
    path.write_text(json.dumps(project))


def load_platforms() -> dict[str, routing.PlatformUnit]:
    rules = generate_matrix._load_watch_rules(CONFIGS)
    return {name: routing.load_platform(CONFIGS / f"{name}.yml") for name in rules}


def build_plan(
    changed: list[str],
    *,
    platforms: list[str] | None = None,
    selection: str = "affected",
    level: str = "L1",
    event: str = "pull_request",
    catalog: routing.Catalog | None = None,
    units: dict[str, routing.PlatformUnit] | None = None,
) -> dict:
    catalog = catalog or load_catalog()
    units = units or load_platforms()
    return planner.build_plan(
        event_name=event,
        stage="pr" if event == "pull_request" else "nightly",
        selection=selection,
        level=level,
        changed=changed,
        unknown_diff=False,
        platforms=sorted(units) if platforms is None else platforms,
        catalog=catalog,
        platform_units=units,
        source_sha="source",
        base_sha="base",
    )


def platform_plan(plan: dict, name: str) -> dict:
    return next(item for item in plan["platforms"] if item["name"] == name)


def selected_names(plan: dict, name: str) -> list[str]:
    return [group["name"] for group in platform_plan(plan, name)["selected_groups"]]


def execution_names(plan: dict, name: str) -> set[str]:
    return {
        group["execution_group"]
        for group in platform_plan(plan, name)["selected_groups"]
        if group["execution_group"]
    }


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_non_platform_configs_are_not_platforms(self):
        self.assertEqual(
            set(self.platforms),
            {"ascend", "cuda", "enflame", "hygon", "kunlunxin", "metax", "musa", "ppu"},
        )

    def test_catalog_paths_and_dependencies_are_valid(self):
        self.assertEqual(self.catalog.validate(ROOT), [])

    def test_every_unit_test_directory_has_a_logical_group(self):
        for directory in sorted((ROOT / "tests/unit_tests").iterdir()):
            if not directory.is_dir() or directory.name == "__pycache__":
                continue
            probe = f"tests/unit_tests/{directory.name}/"
            with self.subTest(directory=probe):
                self.assertIsNotNone(self.catalog.matches(probe))

    def test_every_platform_group_exists_in_the_catalog(self):
        for name, platform in self.platforms.items():
            for group_name in platform.groups:
                with self.subTest(platform=name, group=group_name):
                    self.assertIn(group_name, self.catalog.groups)

    def test_required_groups_are_executable_everywhere(self):
        for group_name in self.catalog.required_groups("L1"):
            effective = self.catalog.groups[group_name].execution_group or group_name
            for name, platform in self.platforms.items():
                with self.subTest(platform=name, group=group_name):
                    self.assertIn(effective, platform.groups)

    def test_dependency_graph_is_acyclic(self):
        errors = [error for error in self.catalog.validate(ROOT) if "cycle" in error]
        self.assertEqual(errors, [])

    def test_catalog_defines_the_coverage_boundary(self):
        self.assertEqual(self.catalog.coverage_sources, ("megatron/plugin/", "megatron/core/"))

    def test_active_groups_have_a_platform_owner(self):
        for name, group in self.catalog.groups.items():
            if group.status != "active":
                continue
            effective = group.execution_group or name
            with self.subTest(group=name):
                self.assertTrue(
                    any(effective in platform.groups for platform in self.platforms.values())
                )

    def test_planned_groups_are_not_executable(self):
        for name, group in self.catalog.groups.items():
            if group.status != "planned":
                continue
            effective = group.execution_group or name
            with self.subTest(group=name):
                self.assertTrue(group.reason)
                self.assertFalse(
                    any(effective in platform.groups for platform in self.platforms.values())
                )

    def test_every_active_group_is_mapped_or_declared_unsupported(self):
        for name, group in self.catalog.groups.items():
            if group.status != "active":
                continue
            effective = group.execution_group or name
            for platform_name, platform in self.platforms.items():
                with self.subTest(group=name, platform=platform_name):
                    self.assertTrue(
                        effective in platform.groups or name in platform.unsupported_groups
                    )

    def test_platform_unsupported_groups_are_valid(self):
        for platform_name, platform in self.platforms.items():
            for name, reason in platform.unsupported_groups.items():
                group = self.catalog.groups[name]
                effective = group.execution_group or name
                with self.subTest(platform=platform_name, group=name):
                    self.assertTrue(reason)
                    self.assertEqual(group.status, "active")
                    if effective in platform.groups:
                        executed = platform.groups[effective]["path"]
                        test_paths = [
                            path
                            for path in group.paths
                            if path.startswith("tests/") or "/tests/" in path
                        ]
                        self.assertTrue(test_paths)
                        self.assertFalse(
                            all(routing.execution_covers(executed, path) for path in test_paths),
                            "an unsupported declaration must cover at least one "
                            "unexecuted test path",
                        )

    def test_validator_accepts_the_catalog_and_platforms(self):
        self.assertEqual(
            validator.collect_errors(ROOT, CONFIGS, self.catalog, self.platforms, "L1"), []
        )

    def test_validator_rejects_fully_covered_unsupported_declaration(self):
        platform = dataclasses.replace(
            self.platforms["cuda"], unsupported_groups={"models": "temporarily disabled"}
        )
        errors = validator.collect_errors(
            ROOT, CONFIGS, self.catalog, {**self.platforms, "cuda": platform}, "L1"
        )
        self.assertTrue(
            any(
                "unsupported group 'models'" in error and "already covers every test path" in error
                for error in errors
            ),
            errors,
        )

    def test_validator_rejects_required_unsupported_declaration(self):
        platform = dataclasses.replace(
            self.platforms["cuda"], unsupported_groups={"core": "temporarily disabled"}
        )
        errors = validator.collect_errors(
            ROOT, CONFIGS, self.catalog, {**self.platforms, "cuda": platform}, "L1"
        )
        self.assertTrue(
            any("declares required group 'core' as unsupported" in error for error in errors),
            errors,
        )

    def test_validator_rejects_execution_group_narrowed_to_one_file(self):
        platform = self.platforms["cuda"]
        groups = dict(platform.groups)
        groups["models"] = {**groups["models"], "path": "tests/unit_tests/models/test_gpt_model.py"}
        narrowed = dataclasses.replace(platform, groups=groups)
        errors = validator.collect_errors(
            ROOT, CONFIGS, self.catalog, {**self.platforms, "cuda": narrowed}, "L1"
        )
        self.assertTrue(
            any(
                "Active group 'models' test path 'tests/unit_tests/models/' "
                "is not executed by platform cuda" in error
                for error in errors
            ),
            errors,
        )

    def test_validator_rejects_overlapping_catalog_paths(self):
        groups = dict(self.catalog.groups)
        groups["overlap"] = dataclasses.replace(groups["core"], name="overlap")
        catalog = dataclasses.replace(self.catalog, groups=groups)
        errors = validator.collect_errors(ROOT, CONFIGS, catalog, {}, "L1")
        self.assertTrue(
            any(
                "Tracked path matches multiple logical groups (core, overlap)" in error
                for error in errors
            ),
            errors,
        )

    def test_validator_resolves_required_group_alias(self):
        levels = dict(self.catalog.levels)
        levels["alias"] = routing.Level(
            name="alias", description="alias test", required_groups=("plugin",)
        )
        catalog = dataclasses.replace(self.catalog, levels=levels)
        platforms = {name: platform for name, platform in self.platforms.items() if name != "musa"}
        self.assertEqual(validator.collect_errors(ROOT, CONFIGS, catalog, platforms, "alias"), [])

    def test_validator_rejects_required_alias_without_execution_group(self):
        platform = self.platforms["cuda"]
        groups = dict(platform.groups)
        groups.pop("core", None)
        missing = dataclasses.replace(platform, groups=groups)
        levels = dict(self.catalog.levels)
        levels["alias"] = routing.Level(
            name="alias", description="alias test", required_groups=("plugin",)
        )
        catalog = dataclasses.replace(self.catalog, levels=levels)
        errors = validator.collect_errors(ROOT, CONFIGS, catalog, {"cuda": missing}, "alias")
        self.assertTrue(
            any(
                "Required group 'plugin' (execution group 'core') "
                "is not executable on: cuda" in error
                for error in errors
            ),
            errors,
        )

    def test_glob_rules_do_not_cross_directories(self):
        self.assertEqual(self.catalog.matches("tests/unit_tests/test_basic.py"), "core")
        self.assertEqual(
            self.catalog.matches("tests/unit_tests/models/test_gpt_model.py"), "models"
        )
        self.assertIsNone(self.catalog.matches("tests/unit_tests/test_data/fixture.py"))

    def test_matches_all_reports_every_matching_group(self):
        self.assertEqual(self.catalog.matches_all("megatron/core/config.py"), ["core"])
        self.assertEqual(self.catalog.matches_all("megatron/training/utils/foo.py"), ["others"])
        self.assertEqual(self.catalog.matches_all("megatron/training/datasets/foo.py"), ["data"])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            write_project_fixture(
                path, {"groups": {"first": {"paths": ["pkg/"]}, "second": {"paths": ["pkg/"]}}}
            )
            catalog = routing.load_catalog(path)
        self.assertEqual(catalog.matches_all("pkg/file.py"), ["first", "second"])

    def test_catalog_has_no_tracked_path_conflicts(self):
        conflicts = {}
        for path in routing.tracked_paths(ROOT) or ():
            groups = self.catalog.matches_all(path)
            if len(groups) > 1:
                conflicts[path] = groups
        self.assertEqual(conflicts, {})

    def test_catalog_rejects_alias_to_unknown_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            write_project_fixture(
                path,
                {
                    "groups": {
                        "core": {"paths": ["megatron/core/"]},
                        "plugin": {"paths": ["megatron/plugin/"], "execution_group": "missing"},
                    }
                },
            )
            with self.assertRaises(routing.CatalogError) as raised:
                routing.load_catalog(path)
        self.assertIn("execution_group targets unknown group", str(raised.exception))

    def test_catalog_rejects_alias_to_non_active_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            write_project_fixture(
                path,
                {
                    "groups": {
                        "core": {
                            "status": "planned",
                            "reason": "not wired yet",
                            "paths": ["megatron/core/"],
                        },
                        "plugin": {"paths": ["megatron/plugin/"], "execution_group": "core"},
                    }
                },
            )
            with self.assertRaises(routing.CatalogError) as raised:
                routing.load_catalog(path)
        self.assertIn("execution_group targets planned group", str(raised.exception))

    def test_execution_coverage_matches_directories_and_globs(self):
        # A directory covers files, subdirectories, and globs beneath it.
        self.assertTrue(
            routing.execution_covers("tests/unit_tests/models/", "tests/unit_tests/models/")
        )
        self.assertTrue(
            routing.execution_covers(
                "tests/unit_tests/models/", "tests/unit_tests/models/test_gpt.py"
            )
        )
        self.assertTrue(
            routing.execution_covers(
                "tests/unit_tests/models/", "tests/unit_tests/models/sub/test_gpt.py"
            )
        )
        self.assertTrue(
            routing.execution_covers(
                "tests/unit_tests/models/", "tests/unit_tests/models/test_*.py"
            )
        )
        # A glob covers files it matches in the same directory, but not a directory.
        self.assertTrue(
            routing.execution_covers(
                "tests/unit_tests/models/test_*.py", "tests/unit_tests/models/test_gpt.py"
            )
        )
        self.assertTrue(
            routing.execution_covers("tests/unit_tests/test_*.py", "tests/unit_tests/test_*.py")
        )
        self.assertFalse(
            routing.execution_covers(
                "tests/unit_tests/models/test_*.py", "tests/unit_tests/models/"
            )
        )
        # A single file does not cover its parent directory or a sibling glob.
        self.assertFalse(
            routing.execution_covers(
                "tests/unit_tests/models/test_gpt.py", "tests/unit_tests/models/"
            )
        )
        self.assertFalse(
            routing.execution_covers(
                "tests/unit_tests/models/test_gpt.py", "tests/unit_tests/models/test_*.py"
            )
        )
        # A subdirectory does not cover its parent.
        self.assertFalse(
            routing.execution_covers("tests/unit_tests/models/sub/", "tests/unit_tests/models/")
        )
        self.assertFalse(
            routing.execution_covers("tests/unit_tests/test_*.py", "tests/unit_tests/plugin/")
        )


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_transformer_change_selects_transformer_models_and_core(self):
        plan = build_plan(["megatron/core/transformer/attention.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models", "transformer"])
        groups = {group["name"]: group for group in platform_plan(plan, "cuda")["selected_groups"]}
        self.assertEqual(
            groups["transformer"]["reasons"],
            [{"type": "direct", "path": "megatron/core/transformer/attention.py"}],
        )
        self.assertIn(
            {"type": "dependency", "source_group": "transformer"}, groups["models"]["reasons"]
        )
        self.assertTrue(groups["core"]["required"])

    def test_distributed_change_selects_parallel_dependencies(self):
        plan = build_plan(
            ["megatron/core/distributed/param_and_grad_buffer.py"], platforms=["cuda"]
        )
        self.assertEqual(
            selected_names(plan, "cuda"),
            ["core", "distributed", "models", "pipeline_parallel", "tensor_parallel"],
        )

    def test_dist_checkpointing_change_does_not_select_models(self):
        plan = build_plan(["megatron/core/dist_checkpointing/serialization.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "dist_checkpointing"])

    def test_test_directory_maps_to_its_group(self):
        for path, expected in (
            ("tests/unit_tests/models/test_gpt_model.py", "models"),
            ("tests/unit_tests/distributed/test_data_parallel.py", "distributed"),
            ("tests/unit_tests/tensor_parallel/test_layers.py", "tensor_parallel"),
            ("tests/unit_tests/inference/test_engine.py", "inference"),
            ("tests/unit_tests/rl/test_agent.py", "rl"),
            ("tests/unit_tests/training/test_training.py", "training"),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.catalog.matches(path), expected)

    def test_others_test_directories_map_to_others(self):
        for directory in ("post_training", "tokenizers", "utils"):
            with self.subTest(directory=directory):
                self.assertEqual(
                    self.catalog.matches(f"tests/unit_tests/{directory}/test_x.py"), "others"
                )

    def test_plugin_tests_select_plugin_executed_by_core(self):
        plan = build_plan(["megatron/plugin/tests/test_override.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "plugin"])
        self.assertEqual(execution_names(plan, "cuda"), {"core"})

    def test_dsa_kernel_tests_are_explicitly_planned(self):
        self.assertEqual(
            self.catalog.matches("tests/unit_tests/plugin/dsa_kernel/test_fused_dsa.py"),
            "dsa_kernel",
        )
        plan = build_plan(
            ["tests/unit_tests/plugin/dsa_kernel/test_fused_dsa.py"], platforms=["cuda"]
        )
        platform = platform_plan(plan, "cuda")
        self.assertEqual([group["name"] for group in platform["planned"]], ["dsa_kernel"])
        self.assertEqual(execution_names(plan, "cuda"), set(platform["configured_groups"]))

    def test_shared_path_falls_back_to_every_group(self):
        plan = build_plan([".github/workflows/ci.yml"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(set(plan["global_groups"]["selected"]), set(self.catalog.groups))
        self.assertEqual(
            execution_names(plan, "cuda"), set(platform_plan(plan, "cuda")["configured_groups"])
        )

    def test_build_inputs_fall_back_to_every_group(self):
        for path in ("uv.lock", "docker/Dockerfile", "pyproject.toml", "setup.py"):
            with self.subTest(path=path):
                plan = build_plan([path], platforms=["cuda"])
                self.assertTrue(plan["global_groups"]["fallback"])

    def test_unknown_path_falls_back_and_is_reported(self):
        plan = build_plan(["megatron/core/new_shared_module.py"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(
            plan["global_groups"]["unmapped_paths"], ["megatron/core/new_shared_module.py"]
        )
        self.assertEqual(plan["plan_status"], "blocked")
        self.assertEqual(plan["direct_groups"], [])
        # The full fallback plan is diagnostic only; it is still materialized.
        self.assertTrue(plan["platforms"])
        self.assertTrue(plan["expected_unit_groups"]["cuda"])

    def test_route_conflict_blocks_the_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            write_project_fixture(
                path,
                {
                    "groups": {
                        "first": {"paths": ["megatron/core/"]},
                        "second": {"paths": ["megatron/core/"]},
                    }
                },
            )
            catalog = routing.load_catalog(path)
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=1,
            groups={
                "first": {"name": "first", "path": "megatron/core/"},
                "second": {"name": "second", "path": "megatron/core/"},
            },
        )
        path = "megatron/core/config.py"
        plan = build_plan(
            [path], platforms=["temp"], level="L1", catalog=catalog, units={"temp": unit}
        )
        self.assertEqual(plan["plan_status"], "blocked")
        self.assertEqual(
            plan["global_groups"]["route_conflicts"],
            [{"path": path, "groups": ["first", "second"]}],
        )
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(plan["direct_groups"], [])

    def test_direct_planned_group_blocks_the_plan(self):
        plan = build_plan(
            ["tests/unit_tests/plugin/dsa_kernel/test_fused_dsa.py"], platforms=["cuda"]
        )
        self.assertEqual(plan["plan_status"], "blocked")
        self.assertEqual(plan["direct_groups"], ["dsa_kernel"])
        self.assertEqual(plan["global_groups"]["direct_groups"], ["dsa_kernel"])
        gap = plan["direct_gaps"][0]
        self.assertEqual(gap["group"], "dsa_kernel")
        self.assertEqual(gap["status"], "planned")
        self.assertEqual(gap["platforms"][0]["status"], "planned")
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "cuda")["planned"]], ["dsa_kernel"]
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = pathlib.Path(directory) / "summary.md"
            planner.write_summary(summary, plan)
            text = summary.read_text()
        self.assertIn("The directly affected group is planned and was not executed.", text)
        self.assertIn("**Plan blocked: no unit matrix was emitted", text)
        self.assertIn("**Diagnostic fallback materialization (not executed):**", text)
        self.assertIn("| Platform | Configured | Executable (diagnostic) |", text)
        self.assertIn("Diagnostic fallback reasons (not executed):", text)

    def test_direct_unsupported_group_blocks_the_plan(self):
        plan = build_plan(["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["musa"])
        self.assertEqual(plan["plan_status"], "blocked")
        self.assertEqual(plan["direct_groups"], ["plugin"])
        gap = plan["direct_gaps"][0]
        self.assertEqual(gap["group"], "plugin")
        self.assertEqual(gap["status"], "active")
        self.assertEqual(gap["platforms"][0]["status"], "unsupported")
        self.assertIn("MUSA smoke", gap["platforms"][0]["reason"])

    def test_shared_fallback_is_partial(self):
        plan = build_plan([".github/workflows/ci.yml"], platforms=["cuda"])
        self.assertEqual(plan["plan_status"], "partial")
        self.assertEqual(plan["direct_groups"], [])

    def test_summary_headings_follow_plan_status(self):
        plans = {
            "ready": build_plan(["megatron/core/config.py"], platforms=["cuda"]),
            "partial": build_plan([".github/workflows/ci.yml"], platforms=["cuda"]),
            "blocked": build_plan(
                ["tests/unit_tests/plugin/dsa_kernel/test_fused_dsa.py"], platforms=["cuda"]
            ),
        }
        for status, plan in plans.items():
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                summary = pathlib.Path(directory) / "summary.md"
                planner.write_summary(summary, plan)
                text = summary.read_text()
                self.assertEqual(plan["plan_status"], status)
                self.assertEqual(
                    "Diagnostic fallback reasons (not executed):" in text, status == "blocked"
                )
                self.assertEqual("Selected executable group reasons:" in text, status != "blocked")

    def test_selection_all_is_partial(self):
        plan = build_plan([], selection="all", event="schedule", platforms=["cuda"])
        self.assertEqual(plan["selection"], "all")
        self.assertEqual(plan["scope"], "all_catalog")
        self.assertEqual(plan["plan_status"], "partial")

    def test_nightly_uses_configured_selection(self):
        plan = build_plan([], selection="configured", event="schedule", platforms=["cuda"])
        self.assertEqual(plan["stage"], "nightly")
        self.assertEqual(plan["selection"], "configured")
        self.assertEqual(plan["scope"], "configured_matrix")
        self.assertEqual(plan["plan_status"], "ready")
        self.assertEqual(plan["catalog_completeness"], "partial")
        self.assertEqual(len(plan["selected_logical_groups"]), 11)
        self.assertEqual(len(plan["expected_unit_groups"]["cuda"]), 10)

    def test_configured_selection_excludes_planned_groups(self):
        plan = build_plan([], selection="configured", event="schedule", platforms=["cuda"])
        planned = {name for name, group in self.catalog.groups.items() if group.status == "planned"}
        self.assertTrue(planned)
        self.assertEqual(set(plan["selected_logical_groups"]) & planned, set())
        self.assertEqual(platform_plan(plan, "cuda")["planned"], [])

    def test_configured_selection_ignores_changed_files(self):
        plan = build_plan(
            ["megatron/core/inference/engine.py"],
            selection="configured",
            event="schedule",
            platforms=["cuda"],
        )
        self.assertEqual(plan["direct_groups"], [])
        self.assertNotIn("inference", plan["selected_logical_groups"])
        self.assertEqual(plan["plan_status"], "ready")

    def test_nightly_reports_unwired_planned_groups(self):
        plan = build_plan([], selection="configured", event="schedule", platforms=["cuda"])
        expected = sorted(
            name for name, group in self.catalog.groups.items() if group.status == "planned"
        )
        self.assertEqual(plan["unwired_planned_groups"], expected)
        self.assertEqual(len(plan["unwired_planned_groups"]), 12)

    def test_nightly_plan_status_ready_with_partial_catalog(self):
        plan = build_plan([], selection="configured", event="schedule", platforms=["cuda"])
        self.assertEqual(plan["plan_status"], "ready")
        self.assertEqual(plan["catalog_completeness"], "partial")
        self.assertTrue(plan["unwired_planned_groups"])

    def test_nightly_selected_groups_include_all_configured_groups(self):
        plan = build_plan(
            [], selection="configured", event="schedule", platforms=sorted(self.platforms)
        )
        for name in sorted(self.platforms):
            platform = platform_plan(plan, name)
            self.assertEqual(execution_names(plan, name), set(platform["configured_groups"]))
        self.assertEqual(plan["plan_status"], "ready")

    def test_level_required_groups_are_subset_of_nightly_groups(self):
        plan = build_plan(
            [], selection="configured", event="schedule", level="L2", platforms=["cuda"]
        )
        required = set(plan["global_groups"]["required"])
        selected = set(plan["selected_logical_groups"])
        self.assertTrue(required)
        self.assertLessEqual(required, selected)
        self.assertGreater(len(selected), len(required))

    def test_documentation_only_plan_is_ready_and_has_no_platforms(self):
        plan = build_plan(["docs/guide.md", "README.md"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["documentation_only"])
        self.assertEqual(plan["platforms"], [])
        self.assertEqual(plan["expected_unit_groups"], {})
        self.assertEqual(plan["plan_status"], "ready")

    def test_project_contract_change_uses_shared_fallback(self):
        plan = build_plan([".flagcicd/project.yaml"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(plan["plan_status"], "partial")
        self.assertEqual(plan["direct_groups"], [])

    def test_mixed_pr_takes_the_union(self):
        plan = build_plan(
            [
                "megatron/core/transformer/attention.py",
                "megatron/core/dist_checkpointing/serialization.py",
            ],
            platforms=["cuda"],
        )
        selected = selected_names(plan, "cuda")
        for name in ("transformer", "models", "dist_checkpointing", "core"):
            self.assertIn(name, selected)

    def test_documentation_only_selects_nothing(self):
        for changed in (["docs/guide.md"], ["README.md"], ["docs/guide.rst", "notes.md"]):
            with self.subTest(changed=changed):
                plan = build_plan(changed, platforms=[])
                self.assertTrue(plan["global_groups"]["documentation_only"])
                self.assertEqual(plan["global_groups"]["selected"], [])
                self.assertEqual(plan["platforms"], [])

    def test_documentation_mixed_with_code_runs_code(self):
        plan = build_plan(
            ["docs/guide.md", "megatron/core/models/gpt/gpt_model.py"], platforms=["cuda"]
        )
        self.assertFalse(plan["global_groups"]["documentation_only"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models"])

    def test_all_selection_selects_all_logical_groups(self):
        plan = build_plan([], selection="all", event="schedule", platforms=["cuda"])
        self.assertEqual(set(plan["global_groups"]["selected"]), set(self.catalog.groups))
        self.assertEqual(
            execution_names(plan, "cuda"), set(platform_plan(plan, "cuda")["configured_groups"])
        )
        self.assertEqual(
            set(plan["global_groups"]["planned"]),
            {name for name, group in self.catalog.groups.items() if group.status == "planned"},
        )

    def test_rename_considers_old_and_new_paths(self):
        plan = build_plan(
            ["megatron/core/models/legacy.py", "megatron/core/transformer/renamed.py"],
            platforms=["cuda"],
        )
        self.assertIn("transformer", selected_names(plan, "cuda"))
        self.assertIn("models", selected_names(plan, "cuda"))

    def test_vendor_plugin_maps_to_plugin_and_core(self):
        plan = build_plan(["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["ascend"])
        self.assertEqual(selected_names(plan, "ascend"), ["core", "plugin"])
        groups = {
            group["name"]: group for group in platform_plan(plan, "ascend")["selected_groups"]
        }
        self.assertEqual(groups["plugin"]["execution_group"], "core")
        self.assertEqual(platform_plan(plan, "ascend")["unsupported"], [])

    def test_level_two_adds_models(self):
        plan = build_plan(
            ["tests/unit_tests/test_parallel_state.py"], platforms=["cuda"], level="L2"
        )
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models"])


class MaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_platform_without_transformer_reports_unsupported(self):
        plan = build_plan(["megatron/core/transformer/attention.py"], platforms=["ascend"])
        self.assertEqual(selected_names(plan, "ascend"), ["core", "models"])
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "ascend")["unsupported"]],
            ["transformer"],
        )

    def test_hygon_missing_groups_are_reported(self):
        plan = build_plan(
            ["megatron/core/distributed/param_and_grad_buffer.py"], platforms=["hygon"]
        )
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "hygon")["unsupported"]],
            ["distributed"],
        )
        self.assertIn("pipeline_parallel", selected_names(plan, "hygon"))

    def test_missing_required_group_is_blocked(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={"models": {"name": "models", "path": "tests/unit_tests/models/"}},
        )
        plan = build_plan(
            ["megatron/core/models/gpt/gpt_model.py"], platforms=["temp"], units={"temp": unit}
        )
        blocked = [group["name"] for group in platform_plan(plan, "temp")["blocked"]]
        self.assertEqual(blocked, ["core"])
        self.assertEqual(selected_names(plan, "temp"), ["models"])

    def test_required_group_declared_unsupported_is_blocked(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={"models": {"name": "models", "path": "tests/unit_tests/models/"}},
            unsupported_groups={"core": "core is unavailable on this test platform"},
        )
        plan = build_plan(
            ["megatron/core/models/gpt/gpt_model.py"], platforms=["temp"], units={"temp": unit}
        )
        platform = platform_plan(plan, "temp")
        self.assertEqual([group["name"] for group in platform["blocked"]], ["core"])
        self.assertEqual([group["name"] for group in platform["unsupported"]], [])
        self.assertEqual(selected_names(plan, "temp"), ["models"])

    def test_platform_partial_unsupported_keeps_alias_execution(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={"core": {"name": "core", "path": "tests/unit_tests/test_*.py"}},
            unsupported_groups={"plugin": "plugin tests are not run on this platform"},
        )
        plan = build_plan(
            ["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["temp"], units={"temp": unit}
        )
        platform = platform_plan(plan, "temp")
        self.assertEqual([group["name"] for group in platform["unsupported"]], ["plugin"])
        self.assertEqual(selected_names(plan, "temp"), ["core"])
        self.assertEqual(platform["blocked"], [])

    def test_planned_fallback_skips_platform_unsupported_groups(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={
                "core": {"name": "core", "path": "tests/unit_tests/test_*.py"},
                "models": {"name": "models", "path": "tests/unit_tests/models/"},
            },
            unsupported_groups={"plugin": "plugin tests are not run on this platform"},
        )
        plan = build_plan(
            ["megatron/core/inference/engine.py"], platforms=["temp"], units={"temp": unit}
        )
        self.assertEqual(execution_names(plan, "temp"), {"core", "models"})
        self.assertNotIn(
            "plugin", [group["name"] for group in platform_plan(plan, "temp")["selected_groups"]]
        )

    def test_planned_fallback_records_excluded_groups(self):
        plan = build_plan(["megatron/core/inference/engine.py"], platforms=["musa"])
        platform = platform_plan(plan, "musa")
        excluded = {group["name"]: group for group in platform["fallback_excluded"]}
        self.assertEqual(sorted(excluded), ["plugin", "transformer"])
        self.assertEqual(excluded["plugin"]["status"], "unsupported")
        self.assertEqual(excluded["plugin"]["platform"], "musa")
        self.assertIn("MUSA smoke", excluded["plugin"]["reason"])
        self.assertIn("CUDA", excluded["transformer"]["reason"])
        self.assertEqual(plan["plan_status"], "blocked")
        with tempfile.TemporaryDirectory() as directory:
            summary = pathlib.Path(directory) / "summary.md"
            planner.write_summary(summary, plan)
            text = summary.read_text()
        self.assertIn("**Fallback excluded groups:**", text)
        self.assertIn("`musa/plugin`: status `unsupported`", text)
        self.assertIn("`musa/transformer`: status `unsupported`", text)

    def test_planned_group_expands_to_every_executable_group(self):
        plan = build_plan(["megatron/core/inference/engine.py"], platforms=["cuda"])
        platform = platform_plan(plan, "cuda")
        self.assertEqual([group["name"] for group in platform["planned"]], ["inference"])
        self.assertEqual(execution_names(plan, "cuda"), set(platform["configured_groups"]))
        self.assertTrue(
            all(group["reasons"][0]["type"] == "fallback" for group in platform["selected_groups"])
        )

    def test_active_group_missing_without_declaration_is_unresolved(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={
                "core": {"name": "core", "path": "tests/unit_tests/test_*.py"},
                "models": {"name": "models", "path": "tests/unit_tests/models/"},
            },
        )
        plan = build_plan(
            ["megatron/core/datasets/foo.py"], platforms=["temp"], units={"temp": unit}
        )
        platform = platform_plan(plan, "temp")
        self.assertEqual([group["name"] for group in platform["unresolved"]], ["data"])
        self.assertEqual(selected_names(plan, "temp"), ["core"])

    def test_execution_matrix_deduplicates_aliases(self):
        plan = build_plan(["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["cuda"])
        self.assertEqual(
            planner.execution_matrix(plan),
            [
                {
                    "platform": "cuda",
                    "groups": ["core"],
                    "config_identity": plan["platforms"][0]["config_identity"],
                }
            ],
        )

    def test_summary_reduction_counts_unique_execution_groups(self):
        plan = build_plan(["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["cuda"])
        platform = platform_plan(plan, "cuda")
        expected = len(platform["configured_groups"]) - len(execution_names(plan, "cuda"))
        with tempfile.TemporaryDirectory() as directory:
            summary = pathlib.Path(directory) / "summary.md"
            planner.write_summary(summary, plan)
            text = summary.read_text()
        self.assertIn("plugin (executes core)", text)
        self.assertIn(f"| {expected} groups |", text)

    def test_plan_contains_identity_and_hash_fields(self):
        plan = build_plan(["megatron/core/models/gpt/gpt_model.py"], platforms=["cuda"])
        for field in (
            "schema_version",
            "source_sha",
            "base_sha",
            "project_hash",
            "catalog_hash",
            "config_hash",
            "plan_hash",
            "plan_status",
            "gate_policy",
            "direct_groups",
            "direct_gaps",
            "expected_unit_groups",
        ):
            self.assertIn(field, plan)
        self.assertTrue(plan["plan_hash"].startswith("sha256:"))
        self.assertEqual(plan["direct_groups"], ["models"])
        self.assertEqual(
            plan["expected_unit_groups"]["cuda"],
            [entry["groups"] for entry in planner.execution_matrix(plan)][0],
        )

    def test_plan_hash_is_stable_and_order_independent(self):
        changed = [
            "megatron/core/transformer/attention.py",
            "megatron/core/models/gpt/gpt_model.py",
        ]
        first = build_plan(changed, platforms=["cuda"])
        second = build_plan(list(reversed(changed)), platforms=["cuda"])
        self.assertEqual(first["plan_hash"], second["plan_hash"])
        third = build_plan(changed + ["megatron/core/config.py"], platforms=["cuda"])
        self.assertNotEqual(first["plan_hash"], third["plan_hash"])

    def test_selected_and_gap_buckets_are_sorted(self):
        plan = build_plan(
            ["megatron/core/models/gpt/gpt_model.py", "megatron/core/transformer/attention.py"],
            platforms=["cuda"],
        )
        for platform in plan["platforms"]:
            for bucket in ("selected_groups", "planned", "unsupported", "unresolved", "blocked"):
                names = [group["name"] for group in platform[bucket]]
                self.assertEqual(names, sorted(names))


class PathExistenceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_untracked_paths_do_not_count_as_repository_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            (root / "tracked.txt").write_text("tracked\n")
            (root / "untracked.txt").write_text("untracked\n")
            (root / "tests").mkdir()
            (root / "tests/test_plan.py").write_text("pass\n")
            subprocess.run(
                ["git", "-C", str(root), "add", "tracked.txt", "tests/test_plan.py"], check=True
            )

            self.assertTrue(routing.path_exists(root, "tracked.txt"))
            self.assertTrue(routing.path_exists(root, "tests/test_*.py"))
            self.assertFalse(routing.path_exists(root, "untracked.txt"))
            self.assertFalse(routing.path_exists(root, "missing/"))


class PlatformSelectionTests(unittest.TestCase):
    def test_unknown_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            planner._parse_platforms('["not-a-platform"]', {"cuda"})

    def test_platform_json_must_be_a_string_array(self):
        with self.assertRaises(ValueError):
            planner._parse_platforms('["cuda", 1]', {"cuda"})


class PlatformLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = generate_matrix._load_watch_rules(CONFIGS)

    def test_documentation_only_ignores_platform_labels(self):
        platforms, reasons = generate_matrix.select_platforms(
            self.rules, ["docs/guide.md"], "pull_request", ["platform:cuda"]
        )
        self.assertEqual(platforms, [])
        self.assertEqual(reasons, ["Documentation only: no hardware tests required"])

    def test_documentation_with_code_honors_platform_labels(self):
        platforms, reasons = generate_matrix.select_platforms(
            self.rules,
            ["docs/guide.md", "megatron/plugin/Ascend/ascend_plugin.py"],
            "pull_request",
            ["platform:cuda"],
        )
        self.assertEqual(platforms, ["cuda"])
        self.assertEqual(reasons, ["Explicit platform labels override changed-file routing"])

    def test_code_only_uses_changed_file_routing(self):
        platforms, reasons = generate_matrix.select_platforms(
            self.rules, ["megatron/plugin/Ascend/ascend_plugin.py"], "pull_request", []
        )
        self.assertEqual(platforms, ["ascend"])
        self.assertIn("megatron/plugin/Ascend/ascend_plugin.py", reasons[0])


class PlannerCliTests(unittest.TestCase):
    def run_planner(
        self,
        directory,
        changed,
        *,
        platforms='["cuda"]',
        labels=None,
        event="pull_request",
        selection=None,
    ):
        output = pathlib.Path(directory) / "unit-test-plan.json"
        summary = pathlib.Path(directory) / "summary.md"
        github_output = pathlib.Path(directory) / "github-output"
        environment = dict(os.environ)
        if labels is not None:
            event_path = pathlib.Path(directory) / "event.json"
            event_path.write_text(
                json.dumps({"pull_request": {"labels": [{"name": label} for label in labels]}})
            )
            environment["GITHUB_EVENT_PATH"] = str(event_path)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / ".github/scripts/ci/generate_test_plan.py"),
                "--event-name",
                event,
                "--base-ref",
                "main",
                "--changed-files",
                "\n".join(changed),
                *(["--selection", selection] if selection is not None else []),
                *(["--platforms", platforms] if platforms is not None else []),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--github-output",
                str(github_output),
                "--source-sha",
                "source",
                "--base-sha",
                "base",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result, output, summary, github_output

    def test_unknown_path_writes_a_blocked_diagnostic_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, summary, github_output = self.run_planner(
                directory, ["megatron/core/new_shared_module.py"]
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["plan_status"], "blocked")
            self.assertEqual(
                plan["global_groups"]["unmapped_paths"], ["megatron/core/new_shared_module.py"]
            )
            # The diagnostic fallback is materialized for debugging but not run.
            self.assertTrue(plan["platforms"])
            text = summary.read_text()
            self.assertIn("**Plan blocked: no unit matrix was emitted", text)
            self.assertIn("fallback plan was not executed", text)
            self.assertIn("**Diagnostic fallback materialization (not executed):**", text)
            self.assertIn("Diagnostic fallback reasons (not executed):", text)
            self.assertIn("core/plugin coverage only", text)
            self.assertIn("not full Megatron unit-group coverage", text)
            captured = github_output.read_text()
            self.assertIn("plan_status=blocked", captured)
            self.assertNotIn("unit_matrix=", captured)
            self.assertNotIn("plan_hash=", captured)

    def test_partial_plan_emits_identity_and_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, summary, github_output = self.run_planner(
                directory, [".github/workflows/ci.yml"]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["plan_status"], "partial")
            captured = github_output.read_text()
            self.assertIn("plan_status=partial", captured)
            self.assertIn("unit_matrix=", captured)
            self.assertIn(f"plan_hash={plan['plan_hash']}", captured)
            self.assertIn("expected_unit_groups=", captured)
            self.assertIn("not proof of a complete logical-group regression", summary.read_text())

    def test_nightly_cli_defaults_to_configured_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, summary, github_output = self.run_planner(
                directory, [], event="schedule"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["selection"], "configured")
            self.assertEqual(plan["scope"], "configured_matrix")
            self.assertEqual(plan["level"], "L2")
            self.assertEqual(plan["gate_policy"]["required_plan_status"], ["ready"])
            self.assertTrue(plan["gate_policy"]["allow_catalog_partial"])
            self.assertEqual(plan["plan_status"], "ready")
            self.assertEqual(plan["catalog_completeness"], "partial")
            self.assertEqual(len(plan["unwired_planned_groups"]), 12)
            text = summary.read_text()
            self.assertIn("Configured matrix: complete", text)
            self.assertIn("Logical catalog completeness: partial", text)
            self.assertIn("Planned catalog groups are not part of the configured matrix", text)
            self.assertNotIn("not proof of a complete logical-group regression", text)
            captured = github_output.read_text()
            self.assertIn("plan_status=ready", captured)
            self.assertIn("unit_matrix=", captured)

    def test_shared_fallback_is_allowed_by_pr_gate_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, _, _ = self.run_planner(
                directory, [".github/workflows/ci.yml"], event="pull_request"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["plan_status"], "partial")
            self.assertEqual(plan["gate_policy"]["required_plan_status"], ["ready", "partial"])

    def test_documentation_only_with_label_skips_hardware_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, summary, github_output = self.run_planner(
                directory, ["docs/guide.md"], platforms=None, labels=["platform:cuda"]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertTrue(plan["global_groups"]["documentation_only"])
            self.assertEqual(plan["platforms"], [])
            self.assertEqual(plan["plan_status"], "ready")
            self.assertIn(
                "Documentation-only change: no hardware tests proposed.", summary.read_text()
            )
            self.assertIn("unit_matrix=[]", github_output.read_text())

    def test_project_contract_rejects_unimplemented_release_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            # Manual requests use the implemented manual stage; an explicit
            # release request must fail before emitting an executable matrix.
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/ci/generate_test_plan.py"),
                    "--event-name",
                    "workflow_dispatch",
                    "--stage",
                    "release",
                    "--platforms",
                    '["cuda"]',
                    "--output",
                    str(pathlib.Path(directory) / "release-plan.json"),
                    "--source-sha",
                    "source",
                    "--base-sha",
                    "base",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid choice", result.stderr)

    def test_project_contract_rejects_unimplemented_level(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/ci/generate_test_plan.py"),
                    "--event-name",
                    "pull_request",
                    "--level",
                    "L3",
                    "--changed-files",
                    "megatron/core/config.py",
                    "--platforms",
                    '["cuda"]',
                    "--output",
                    str(pathlib.Path(directory) / "l3-plan.json"),
                    "--source-sha",
                    "source",
                    "--base-sha",
                    "base",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Level 'L3' is not declared in the project contract", result.stderr)

    def test_documentation_with_code_and_label_runs_the_labeled_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, _, _ = self.run_planner(
                directory,
                ["docs/guide.md", "megatron/plugin/Ascend/ascend_plugin.py"],
                platforms=None,
                labels=["platform:cuda"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertFalse(plan["global_groups"]["documentation_only"])
            self.assertEqual([item["name"] for item in plan["platforms"]], ["cuda"])
            self.assertEqual(plan["plan_status"], "ready")


if __name__ == "__main__":
    unittest.main()
