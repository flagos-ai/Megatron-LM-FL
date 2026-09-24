#!/usr/bin/env python3
"""Run configured unit groups in one prepared container, preserving group failures."""

import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path


def run_group(environment: dict, timeout: float = 3600) -> int:
    command = [
        "bash",
        "-euo",
        "pipefail",
        "-c",
        'source "$CI_SETUP_SCRIPT"\n'
        'exec bash "$GITHUB_WORKSPACE/tests/test_utils/runners/run_ci_unit_tests.sh"',
    ]
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"::error::Unit group {environment['CI_TEST_GROUP']} exceeded {timeout}s", flush=True)
        return 124
    finally:
        # Kill only this group's process tree, including ranks left by failed torchrun.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def combine_coverage(directories: list, output: Path, environment: dict) -> None:
    data_files = [
        str(path)
        for directory in directories
        for path in directory.glob(".coverage*")
        if path.name == ".coverage" or path.name.startswith(".coverage.")
    ]
    configs = [directory / ".coveragerc" for directory in directories]
    config = next((path for path in configs if path.exists()), None)
    if not data_files or config is None:
        return
    python = environment.get("CI_PYTHON_BIN", "python3")
    options = [f"--rcfile={config}", f"--data-file={output.parent / '.coverage'}"]
    try:
        subprocess.run(
            [python, "-m", "coverage", "combine", "--keep", *options, *data_files],
            env=environment,
            cwd=environment.get("GITHUB_WORKSPACE"),
            check=True,
            timeout=120,
        )
        subprocess.run(
            [
                python,
                "-m",
                "coverage",
                "json",
                *options,
                "--show-contexts",
                "--include=megatron/training/*,megatron/plugin/*",
                "-o",
                str(output),
            ],
            env=environment,
            cwd=environment.get("GITHUB_WORKSPACE"),
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        # Match the existing runner: reporting must not replace the test exit status.
        print(f"::warning::Could not combine unit coverage: {error}", flush=True)


def restore_results(path: Path, identity: dict, attempt: int) -> dict:
    if attempt <= 1 or not identity["GITHUB_RUN_ID"] or not identity["GITHUB_SHA"]:
        return {}
    try:
        state = json.loads(path.read_text())
        if state["identity"] != identity or not 0 < state["attempt"] < attempt:
            return {}
        results = state["results"]
        names = {group["name"] for group in json.loads(identity["CI_TEST_GROUPS"])}
        if not isinstance(results, dict) or any(
            name not in names or result["name"] != name or type(result["exit_code"]) is not int
            for name, result in results.items()
        ):
            raise ValueError("Invalid group results")
        return {name: result for name, result in results.items() if result["exit_code"] == 0}
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"::warning::No usable unit checkpoint; running all groups: {error}", flush=True)
        return {}


def save_results(path: Path, identity: dict, attempt: int, results: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "attempt": attempt, "results": results}))
    temporary.replace(path)


def main() -> int:
    groups = json.loads(os.environ["CI_TEST_GROUPS"])
    if not isinstance(groups, list) or not groups:
        raise ValueError("CI_TEST_GROUPS must be a non-empty array")
    names = set()
    for group in groups:
        name = group["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate unit group: {name!r}")
        if not isinstance(group["path"], str) or not group["path"].strip():
            raise ValueError(f"Missing test path for unit group: {name}")
        names.add(name)

    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    reports = workspace / "coverage-report"
    reports.mkdir(exist_ok=True)
    prefix = f"coverage-{os.environ['CI_PLATFORM']}-{os.environ['CI_DEVICE']}"
    for name in [group["name"] for group in groups] + ["all"]:
        (reports / f"{prefix}-{name}.json").unlink(missing_ok=True)
    checkpoint = workspace / ".unit-checkpoint"
    checkpoint.mkdir(exist_ok=True)
    state_path = checkpoint / "state.json"
    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    identity = {
        key: os.environ.get(key, "")
        for key in (
            "GITHUB_RUN_ID",
            "GITHUB_SHA",
            "GITHUB_WORKSPACE",
            "CI_PLATFORM",
            "CI_DEVICE",
            "CI_TEST_GROUPS",
            "CI_SETUP_SCRIPT",
            "CI_NPROC_PER_NODE",
            "CI_IGNORED_TESTS",
            "CI_PYTEST_EXTRA_ARGS",
            "CI_EXPERIMENTAL_PYTEST_EXTRA_ARGS",
        )
    }
    completed = (
        restore_results(state_path, identity, attempt)
        if os.environ.get("CI_UNIT_RESUME") == "true"
        else {}
    )
    # Persist all previous successes before running anything, including groups later in the list.
    save_results(state_path, identity, attempt, completed)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(
                "| Unit group | Status | Exit code | Execution |\n| --- | --- | --- | --- |\n"
            )
    results = []
    directories = []
    for group in groups:
        name = group["name"]
        directory = checkpoint / name
        directories.append(directory)
        reused = name in completed
        print(f"::group::Unit tests: {name}", flush=True)
        if reused:
            code = 0
            print("Previously passed in this workflow run; skipping.", flush=True)
        else:
            # A retry must replace the failed attempt's partial coverage, not merge into it.
            if directory.exists():
                shutil.rmtree(directory)
            environment = {
                **os.environ,
                "CI_TEST_SUITE": "unit_group",
                "CI_TEST_GROUP": name,
                "CI_TEST_PATH": group["path"],
                "CI_COVERAGE_DIRECTORY": str(directory),
                "CI_SETUP_SCRIPT": str(workspace / os.environ["CI_SETUP_SCRIPT"]),
                # A group's hook must not change subsequent groups or Actions steps.
                "GITHUB_ENV": os.devnull,
                "GITHUB_PATH": os.devnull,
            }
            try:
                code = run_group(environment)
            except BaseException:
                print("::endgroup::", flush=True)
                print(f"::warning::INCOMPLETE: Unit tests: {name}", flush=True)
                if summary:
                    with open(summary, "a") as stream:
                        stream.write(f"| {name} | INCOMPLETE | - | Interrupted |\n")
                raise
        print("::endgroup::", flush=True)
        status = "PASS" if code == 0 else "TIMEOUT" if code == 124 else "FAIL"
        execution = "Previously passed" if reused else "Executed"
        if code:
            print(f"::error::{status}: Unit group {name} failed with exit code {code}", flush=True)
        else:
            print(f"PASS: Unit tests: {name} ({execution})", flush=True)
        result = {"name": name, "exit_code": code}
        if reused:
            result["reused"] = True
        completed[name] = result
        save_results(state_path, identity, attempt, completed)
        results.append(result)
        (reports / "unit-results.json").write_text(json.dumps(results, indent=2) + "\n")
        report = directory / f"{prefix}-{name}.json"
        if report.exists():
            (reports / report.name).write_bytes(report.read_bytes())
        if summary:
            with open(summary, "a") as stream:
                stream.write(f"| {name} | {status} | {code} | {execution} |\n")
    combine_coverage(directories, reports / f"{prefix}-all.json", dict(os.environ))

    return int(any(result["exit_code"] for result in results))


def terminate_group(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate_group)
    raise SystemExit(main())
