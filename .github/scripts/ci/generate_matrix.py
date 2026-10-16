#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select affected PR platforms; run all platforms for other events.

paths_watch entries are exact repository paths or directory prefixes ending in
"/". Unmapped code, unavailable diffs, and missing watch rules run all platforms.
Only explicitly recognized documentation changes can produce an empty matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import yaml

_NON_PLATFORM_STEMS = frozenset(("template", "image_sources", "platforms"))


def _changed_files(base_ref: str) -> list[str]:
    # Disabling rename detection includes both old and new paths for moves.
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z", f"origin/{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def _matches(path: str, watch: str) -> bool:
    return path.startswith(watch) if watch.endswith("/") else path == watch


def _is_documentation(path: str) -> bool:
    return path.startswith("docs/") or pathlib.PurePosixPath(path).suffix in (".md", ".rst")


def _load_watch_rules(configs_dir: pathlib.Path) -> dict[str, list[str]]:
    rules = {}
    for config in sorted(configs_dir.glob("*.yml")):
        if config.stem in _NON_PLATFORM_STEMS:
            continue
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid platform configuration: {config}")
        watch = data.get("paths_watch", [])
        if not isinstance(watch, list) or any(
            not isinstance(path, str) or not path for path in watch
        ):
            raise ValueError(f"paths_watch must be a list of nonempty paths: {config}")
        rules[config.stem] = watch
    if not rules:
        raise ValueError(f"No platform configurations found: {configs_dir}")
    return rules


def _select_paths(
    rules: dict[str, list[str]], changed: list[str], event_name: str
) -> tuple[list[str], list[str]]:
    all_platforms = sorted(rules)
    if event_name != "pull_request":
        return all_platforms, [f"{event_name}: full matrix"]
    if not changed:
        return all_platforms, ["No changed-file information: full matrix"]

    code_paths = [path for path in changed if not _is_documentation(path)]
    if not code_paths:
        return [], ["Documentation only: no hardware tests required"]
    if any(not watch for watch in rules.values()):
        return all_platforms, ["Missing platform watch rules: full matrix"]

    selected = set()
    reasons = []
    for path in code_paths:
        affected = {
            platform
            for platform, watch in rules.items()
            if any(_matches(path, entry) for entry in watch)
        }
        # A mixed PR must not hide an unmapped/common change behind a matched
        # vendor path. Check each path before taking the union.
        if not affected:
            return all_platforms, [f"Shared or unmapped path {path}: full matrix"]
        selected.update(affected)
        reasons.append(f"{path}: {', '.join(sorted(affected))}")
    return sorted(selected), reasons


def select_platforms(
    rules: dict[str, list[str]],
    changed: list[str],
    event_name: str,
    labels: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Let explicit PR platform labels override automatic path selection."""
    if event_name != "pull_request":
        return _select_paths(rules, changed, event_name)
    extra = set()
    for label in labels or []:
        if not label.startswith("platform:"):
            continue
        platform = label.removeprefix("platform:")
        if platform == "all":
            extra.update(rules)
        elif platform in rules:
            extra.add(platform)
        else:
            raise ValueError(f"Unknown platform label: {label}")
    if extra:
        return sorted(extra), ["Explicit platform labels override changed-file routing"]
    return _select_paths(rules, changed, event_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--changed-files", default=None, help="Newline-separated paths for testing")
    args = parser.parse_args()
    try:
        rules = _load_watch_rules(pathlib.Path(args.configs_dir))
        changed = []
        if args.event_name == "pull_request":
            if args.changed_files is not None:
                changed = args.changed_files.splitlines()
            else:
                try:
                    changed = _changed_files(args.base_ref)
                except subprocess.CalledProcessError as exc:
                    print(
                        f"::warning::Cannot read diff; running all platforms: {exc}",
                        file=sys.stderr,
                    )
        labels = []
        if args.event_name == "pull_request" and os.environ.get("GITHUB_EVENT_PATH"):
            event = json.loads(pathlib.Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            labels = [label["name"] for label in event.get("pull_request", {}).get("labels", [])]
        platforms, reasons = select_platforms(rules, changed, args.event_name, labels)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"platforms={json.dumps(platforms)}")
    for reason in reasons:
        print(reason, file=sys.stderr)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as summary:
            summary.write("### Platform selection\n\n")
            summary.write(f"Selected: {', '.join(platforms) or 'none (documentation only)'}\n\n")
            for reason in reasons:
                # Keep repository paths from injecting markup into the summary.
                summary.write(f"- {reason.replace('<', '&lt;').replace('>', '&gt;')}\n")


if __name__ == "__main__":
    main()
