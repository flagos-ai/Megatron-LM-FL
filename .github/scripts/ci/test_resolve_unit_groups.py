"""Hardware-independent tests for catalog-owned unit-test paths."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import unittest

import flagcicd_contract
import generate_matrix
import resolve_unit_groups as resolver
import unit_group_routing as routing

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ROOT / ".github/configs"
PROJECT = ROOT / ".flagcicd/project.yaml"

# A tracked file that belongs to a planned group, used as a PR-selected test.
PLANNED_TEST = "tests/unit_tests/tools/checkpoint/test_distributed_round_trip.py"


def load(name: str) -> routing.PlatformUnit:
    return routing.load_platform(CONFIGS / f"{name}.yml")


class ExecutionPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = routing.load_catalog(PROJECT)

    def test_alias_group_contributes_its_test_targets(self) -> None:
        paths = routing.execution_paths(self.catalog, load("cuda"), "core")
        self.assertEqual(paths, ("tests/unit_tests/test_*.py", "megatron/plugin/tests/"))

    def test_unsupported_group_is_dropped_from_the_alias(self) -> None:
        paths = routing.execution_paths(self.catalog, load("musa"), "core")
        self.assertEqual(paths, ("tests/unit_tests/test_*.py",))

    def test_unsupported_groups_are_never_part_of_an_execution_group(self) -> None:
        for name in generate_matrix._load_watch_rules(CONFIGS):
            platform = load(name)
            for group in platform.unsupported_groups:
                with self.subTest(platform=name, group=group):
                    effective = self.catalog.groups[group].execution_group or group
                    executed = " ".join(routing.execution_paths(self.catalog, platform, effective))
                    for path in self.catalog.groups[group].test_targets:
                        self.assertFalse(
                            routing.execution_covers(executed, path),
                            f"{path} still executes on {name}",
                        )

    def test_every_platform_group_resolves_to_a_test_path(self) -> None:
        for name in generate_matrix._load_watch_rules(CONFIGS):
            platform = load(name)
            for group, paths in routing.execution_groups(self.catalog, platform).items():
                with self.subTest(platform=name, group=group):
                    self.assertTrue(paths)

    def test_trigger_paths_are_not_executed(self) -> None:
        for group in self.catalog.groups.values():
            for path in group.paths:
                self.assertFalse(
                    routing.is_test_path(path),
                    f"{group.name} still mixes test code into its trigger paths: {path}",
                )
            self.assertTrue(group.test_targets, group.name)


class ContractSplitTests(unittest.TestCase):
    """The catalog keeps trigger scope and execution scope apart."""

    def _project(self, directory: str, groups: dict) -> pathlib.Path:
        project, _ = flagcicd_contract.load_project(PROJECT)
        project["global"] = {}
        project["levels"] = {"L1": {"required_groups": []}}
        project["stages"] = {"pr": {"selection": "affected", "default_level": "L1"}}
        project["gate_policy"] = {"pr": project["gate_policy"]["pr"]}
        project["test_groups"] = [dict(name=name, **entry) for name, entry in groups.items()]
        path = pathlib.Path(directory) / "project.yaml"
        path.write_text(json.dumps(project))
        return path

    def test_a_test_path_in_paths_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._project(
                directory,
                {
                    "core": {
                        "paths": ["megatron/core/", "tests/unit_tests/"],
                        "test_targets": ["tests/unit_tests/"],
                    }
                },
            )
            with self.assertRaises(ValueError) as raised:
                routing.load_catalog(path)
        self.assertIn("declare it in test_targets", str(raised.exception))

    def test_a_source_path_in_test_targets_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._project(
                directory,
                {"core": {"paths": ["megatron/core/"], "test_targets": ["megatron/core/"]}},
            )
            with self.assertRaises(ValueError) as raised:
                routing.load_catalog(path)
        self.assertIn("is not test code", str(raised.exception))


class ResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = routing.load_catalog(PROJECT)
        cls.platform = load("cuda")

    def test_resolves_every_derived_group_in_catalog_order(self) -> None:
        groups = resolver.resolve(self.catalog, self.platform)
        self.assertEqual(
            [group["name"] for group in groups],
            list(routing.execution_groups(self.catalog, self.platform)),
        )
        self.assertTrue(all(group["path"] for group in groups))

    def test_expected_subset_keeps_catalog_order(self) -> None:
        groups = resolver.resolve(self.catalog, self.platform, ["models", "core"])
        self.assertEqual([group["name"] for group in groups], ["core", "models"])

    def test_planned_test_files_get_an_ephemeral_group(self) -> None:
        identity = {"planned_execution": [{"name": "planned-tools", "paths": [PLANNED_TEST]}]}
        groups = resolver.resolve(self.catalog, self.platform, ["planned-tools"], identity)
        self.assertEqual(groups, [{"name": "planned-tools", "path": PLANNED_TEST}])

    def test_a_new_catalog_group_needs_no_platform_edit(self) -> None:
        groups = dict(self.catalog.groups)
        groups["brand_new"] = dataclasses.replace(
            self.catalog.groups["models"],
            name="brand_new",
            paths=("megatron/core/models/",),
            test_targets=("tests/unit_tests/brand_new/",),
            depends_on=(),
        )
        catalog = dataclasses.replace(self.catalog, groups=groups)
        resolved = {
            group["name"]: group["path"] for group in resolver.resolve(catalog, self.platform)
        }
        self.assertEqual(resolved["brand_new"], "tests/unit_tests/brand_new/")

    def test_opting_out_only_hides_the_named_group(self) -> None:
        platform = dataclasses.replace(
            self.platform,
            unsupported_groups={**self.platform.unsupported_groups, "fusions": "no fused kernels"},
        )
        resolved = {group["name"] for group in resolver.resolve(self.catalog, platform)}
        self.assertNotIn("fusions", resolved)
        self.assertIn("models", resolved)

    def test_unknown_expected_group_fails_loudly(self) -> None:
        with self.assertRaises(resolver.ResolutionError) as raised:
            resolver.resolve(self.catalog, self.platform, ["not-a-group"])
        self.assertIn("absent from the catalog execution groups", str(raised.exception))

    def test_temporary_group_rejects_a_source_path(self) -> None:
        identity = {
            "planned_execution": [{"name": "planned-tools", "paths": ["megatron/core/utils.py"]}]
        }
        with self.assertRaises(resolver.ResolutionError) as raised:
            resolver.resolve(self.catalog, self.platform, None, identity)
        self.assertIn("Invalid temporary test path", str(raised.exception))

    def test_temporary_group_may_not_shadow_a_configured_group(self) -> None:
        identity = {"planned_execution": [{"name": "core", "paths": [PLANNED_TEST]}]}
        with self.assertRaises(resolver.ResolutionError) as raised:
            resolver.resolve(self.catalog, self.platform, None, identity)
        self.assertIn("Invalid or duplicate temporary group", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
