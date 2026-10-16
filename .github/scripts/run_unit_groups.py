#!/usr/bin/env python3
"""Run unit groups or a single test command with bounded transient retries."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

TRANSIENT = re.compile(
    r"ReadTimeout|ConnectTimeout|ConnectionResetError|Connection aborted|"
    r"Temporary failure in name resolution|Network is unreachable|"
    r"HTTP(?:Error| error| status)?[: ]+(?:502|503|504)\b",
    re.IGNORECASE,
)
DETERMINISTIC = re.compile(
    r"AssertionError|golden.{0,40}mismatch|loss.{0,40}mismatch|"
    r"SyntaxError|ModuleNotFoundError|ImportError|unrecognized arguments|"
    r"invalid configuration|invalid platform|out of memory|^(?:E\s+)?assert\b",
    re.IGNORECASE | re.MULTILINE,
)


def retry_limit(value: object) -> int:
    """Validate the limit: at most one additional execution is permitted."""
    if isinstance(value, bool) or str(value) not in ("0", "1"):
        raise ValueError("transient_retries must be 0 or 1")
    return int(value)


def _retryable_exit(code: int) -> bool:
    return code > 0 and code not in (124, 130, 137, 143)


def junit_allows_retry(environment: dict) -> bool:
    """Require complete reports whose failure messages all identify network errors."""
    directory = environment.get("CI_JUNIT_DIRECTORY")
    if not directory:
        return True  # Non-pytest commands retain the log-based policy.
    reports = list(Path(directory).glob("*.xml"))
    expected = int(environment.get("CI_JUNIT_EXPECTED", "0"))
    if not reports or (expected and len(reports) != expected):
        return False
    failures = []
    try:
        for report in reports:
            root = ET.parse(report).getroot()
            if root.tag not in ("testsuite", "testsuites") or not list(root.iter("testcase")):
                return False
            for node in root.iter():
                if node.tag in ("failure", "error"):
                    message = node.get("message", "")
                    kind = node.get("type") or message.split(":", 1)[0]
                    failures.append((kind.strip().rsplit(".", 1)[-1], message))
    except (OSError, ET.ParseError):
        return False
    return bool(failures) and all(
        kind
        in {
            "ReadTimeout",
            "ReadTimeoutError",
            "ConnectTimeout",
            "ConnectTimeoutError",
            "ConnectionResetError",
            "ConnectionError",
            "HTTPError",
            "URLError",
            "gaierror",
        }
        and TRANSIENT.search(f"{kind}: {message}")
        and not DETERMINISTIC.search(message)
        for kind, message in failures
    )


def pytest_worker() -> None:
    """Give each distributed pytest rank and phase its own JUnit report."""
    directory = Path(os.environ["CI_JUNIT_DIRECTORY"])
    report = directory / f"{os.environ['CI_JUNIT_PHASE']}-{os.environ.get('RANK', '0')}.xml"
    args = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        f"--rcfile={os.environ['CI_COVERAGE_DIRECTORY']}/.coveragerc",
        *sys.argv[2:],
        f"--junitxml={report}",
    ]
    os.execv(sys.executable, args)


def execute(
    command: list[str],
    environment: dict,
    logs: Path,
    retries: int = 0,
    timeout: float = 3600,
    cleanup: Path | None = None,
) -> int:
    """Run a process tree, preserving output and cleaning failed retry outputs."""
    retries = retry_limit(retries)
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    for attempt in range(1, retries + 2):
        if directory := environment.get("CI_JUNIT_DIRECTORY"):
            shutil.rmtree(directory, ignore_errors=True)
            Path(directory).mkdir(parents=True, exist_ok=True)
        log = logs / f"attempt-{attempt}.log"
        log.touch()
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        signals = [False, False]

        def copy_output() -> None:
            tail = ""
            with log.open("wb") as stream:
                while chunk := os.read(process.stdout.fileno(), 8192):
                    stream.write(chunk)
                    decoded = chunk.decode(errors="replace")
                    text = tail + decoded
                    for i, pattern in enumerate((TRANSIENT, DETERMINISTIC)):
                        signals[i] |= bool(pattern.search(text))
                    tail = text[-256:]
                    print(decoded, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                code = 124
                print(f"::error::Test attempt exceeded {timeout}s", flush=True)
        finally:
            # Also remove distributed workers left behind by a failed parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            reader.join(timeout=10)
            process.stdout.close()
        transient = (
            _retryable_exit(code)
            and signals[0]
            and not signals[1]
            and junit_allows_retry(environment)
        )
        if directory := environment.get("CI_JUNIT_DIRECTORY"):
            destination = logs / f"attempt-{attempt}-junit"
            shutil.rmtree(destination, ignore_errors=True)
            shutil.move(directory, str(destination))
        records.append({"attempt": attempt, "exit_code": code, "transient": transient})
        status = "FLAKY_PASS" if code == 0 and attempt > 1 else "PASS" if code == 0 else "FAIL"
        (logs / "result.json").write_text(json.dumps({"status": status, "attempts": records}))
        if code == 0 or not transient or attempt > retries:
            return code
        print("::warning::Transient infrastructure failure; retrying once", flush=True)
        if cleanup is not None and cleanup.exists():
            # Keep diagnostics, excluding model weights and checkpoint directories.
            destination = logs / f"attempt-{attempt}-outputs"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(
                cleanup,
                destination,
                ignore=shutil.ignore_patterns(
                    "checkpoints", "iter_*", "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin"
                ),
            )
            shutil.rmtree(cleanup)
            cleanup.mkdir(parents=True, exist_ok=True)
    raise AssertionError("Unreachable retry state")


def run_command() -> int:
    """Expose the same retry policy to the functional-test shell workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--cleanup-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A test command is required")
    config = json.loads(os.environ.get("CI_TEST_CONFIG", "{}"))
    retries = retry_limit(config.get("transient_retries", os.environ.get("CI_TEST_RETRIES", "0")))
    environment = {**os.environ, "CI_JUNIT_DIRECTORY": str(args.cleanup_dir.resolve() / "junit")}
    code = execute(command, environment, args.logs, retries, args.timeout, args.cleanup_dir)
    result = json.loads((args.logs / "result.json").read_text())
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as stream:
            stream.write(
                f"\nFunctional result: {result['status']} ({len(result['attempts'])} attempts)\n"
            )
    return code if code >= 0 else 128 - code


def run_group(environment: dict, timeout: float = 3600) -> int:
    command = [
        "bash",
        "-euo",
        "pipefail",
        "-c",
        'source "$CI_SETUP_SCRIPT"\n'
        'exec bash "$GITHUB_WORKSPACE/tests/test_utils/runners/run_ci_unit_tests.sh"',
    ]
    return execute(
        command,
        environment,
        Path(environment["GITHUB_WORKSPACE"])
        / ".retry-logs"
        / (
            f"unit-{environment.get('CI_PLATFORM', 'local')}-"
            f"{environment.get('CI_DEVICE', 'local')}-{environment['CI_TEST_GROUP']}"
        ),
        retries=retry_limit(environment.get("CI_TEST_RETRIES", "0")),
        timeout=timeout,
        cleanup=(
            Path(environment["CI_COVERAGE_DIRECTORY"])
            if environment.get("CI_COVERAGE_DIRECTORY")
            else None
        ),
    )


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
            "CI_TEST_RETRIES",
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
                "CI_TEST_RETRIES": str(
                    retry_limit(
                        group.get("transient_retries", os.environ.get("CI_TEST_RETRIES", "0"))
                    )
                ),
                "CI_COVERAGE_DIRECTORY": str(directory),
                "CI_JUNIT_DIRECTORY": str(directory / "junit"),
                "CI_JUNIT_EXPECTED": str(
                    int(os.environ.get("CI_NPROC_PER_NODE", "1"))
                    * (
                        2
                        if json.loads(os.environ.get("CI_EXPERIMENTAL_PYTEST_EXTRA_ARGS", "[]"))
                        else 1
                    )
                ),
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
        retry_result = (
            workspace
            / ".retry-logs"
            / f"unit-{os.environ['CI_PLATFORM']}-{os.environ['CI_DEVICE']}-{name}"
            / "result.json"
        )
        flaky = (
            bool(completed[name].get("flaky"))
            if reused
            else code == 0
            and retry_result.exists()
            and json.loads(retry_result.read_text())["status"] == "FLAKY_PASS"
        )
        status = (
            "FLAKY_PASS" if flaky else "PASS" if code == 0 else "TIMEOUT" if code == 124 else "FAIL"
        )
        execution = "Previously passed" if reused else "Executed"
        if code:
            print(f"::error::{status}: Unit group {name} failed with exit code {code}", flush=True)
        else:
            print(f"{status}: Unit tests: {name} ({execution})", flush=True)
        result = {"name": name, "exit_code": code}
        if flaky:
            result["flaky"] = True
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
    if sys.argv[1:2] == ["pytest-worker"]:
        pytest_worker()
    raise SystemExit(run_command() if len(sys.argv) > 1 else main())
