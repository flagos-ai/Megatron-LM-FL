"""Trusted control-plane operations for PR CI comments and supplemental checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from generate_matrix import _load_watch_rules


class GitHub:
    """Minimal repository-scoped REST client; never execute comment text."""

    def __init__(self) -> None:
        self.root = f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/{os.environ['GITHUB_REPOSITORY']}"

    def request(self, method: str, path: str, body: dict | None = None):
        """Make a bounded authenticated API request without printing credentials."""
        request = urllib.request.Request(
            self.root + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return json.loads(payload) if payload else None


def parse_command(body: str) -> tuple[str, str] | None:
    """Accept a single complete command, not quoted or embedded instructions."""
    command = body.strip()
    if match := re.fullmatch(r"/(?:ci run|test) ([a-z0-9_-]+(?:[ ,]+[a-z0-9_-]+)*)", command):
        return "run", match[1]
    if command == "/ci rerun failed":
        return "rerun", "failed"
    return None


def current_pr(api, number: int, expected_sha: str | None = None) -> dict:
    """Validate an open PR and bind the operation to its current head commit."""
    pr = api.request("GET", f"/pulls/{number}")
    if pr["state"] != "open" or pr["base"]["ref"] not in ("main", "devops"):
        raise ValueError("CI commands require an open PR targeting main or devops")
    sha = pr["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid PR head SHA")
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("PR changed after the request; request tests for the new commit")
    return pr


def requested_platforms(target: str, known: dict) -> list[str]:
    """Resolve comma/space-separated platforms or the complete platform matrix."""
    selected = set(re.split(r"[ ,]+", target))
    if not selected or selected - known.keys() - {"all"}:
        raise ValueError(f"Unknown platform: {target}")
    if "all" in selected:
        return sorted(known)
    return sorted(selected)


def handle_comment(event: dict, api, known: dict) -> dict | str:
    """Authorize maintainers before dispatching tests or rerunning failed jobs."""
    if event.get("action") != "created" or "pull_request" not in event.get("issue", {}):
        return "Not a new PR comment"
    command = parse_command(event["comment"]["body"])
    if command is None:
        return "Not a supported CI command"
    actor = urllib.parse.quote(event["comment"]["user"]["login"], safe="")
    permission = api.request("GET", f"/collaborators/{actor}/permission")["permission"]
    if permission not in ("admin", "maintain", "write"):
        raise ValueError("CI commands require write, maintain, or admin repository permission")
    number = int(event["issue"]["number"])
    if command[0] == "run":
        return prepare_request(api, number, None, command[1], known)
    pr = current_pr(api, number)

    # Find the latest automatic run for THIS PR and head, never an older failure
    # superseded by a successful/in-progress run. Pagination includes busy repos.
    candidates = []
    for page in range(1, 11):
        runs = api.request(
            "GET", f"/actions/workflows/ci.yml/runs?event=pull_request&per_page=100&page={page}"
        )["workflow_runs"]
        for run in runs:
            if run["event"] != "pull_request":
                continue
            if run["head_sha"] not in (pr["head"]["sha"], pr.get("merge_commit_sha")):
                continue
            if any(item["number"] == number for item in run.get("pull_requests", [])):
                candidates.append(run)
        if len(runs) < 100:
            break
    if not candidates:
        raise ValueError("No automatic CI run found for this PR's current commit")
    run = max(candidates, key=lambda item: item["id"])
    if run["status"] != "completed" or run["conclusion"] not in ("failure", "timed_out"):
        raise ValueError("The latest CI run is not a completed failed run")
    # Recheck after listing runs in case the contributor pushed while we queried.
    current_pr(api, number, pr["head"]["sha"])
    api.request("POST", f"/actions/runs/{run['id']}/rerun-failed-jobs", {})
    return f"Rerunning failed jobs from CI run {run['id']}"


def prepare_request(api, number: int, sha: str | None, target: str, known: dict) -> dict | str:
    """Validate dispatch inputs and create a separate check on the PR head SHA."""
    pr = current_pr(api, number, sha)
    sha = pr["head"]["sha"]
    platforms = requested_platforms(target, known)
    prefix = f"ci-requested:{number}:{sha}:{','.join(platforms)}:"
    # The validation job is serialized per PR: check and claim happen together.
    for page in range(1, 11):
        checks = api.request(
            "GET",
            f"/commits/{sha}/check-runs?check_name=CI%20Requested%20Tests&filter=all&per_page=100&page={page}",
        )["check_runs"]
        for check in checks:
            external = check.get("external_id") or ""
            if (
                check.get("app", {}).get("slug") != "github-actions"
                or check["status"] == "completed"
                or not external.startswith(prefix)
            ):
                continue
            owner = external[len(prefix) :]
            if not owner.isdigit() or check.get("name") != "CI Requested Tests":
                continue
            run = api.request("GET", f"/actions/runs/{owner}")
            if owner != os.environ["GITHUB_RUN_ID"] and run["status"] != "completed":
                return f"Reusing requested tests: {check['details_url']}"
            # A cancelled/interrupted reporter must not leave its check pending.
            api.request(
                "PATCH",
                f"/check-runs/{check['id']}",
                {"status": "completed", "conclusion": "cancelled"},
            )
        if len(checks) < 100:
            break
    run_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    check = api.request(
        "POST",
        "/check-runs",
        {
            "name": "CI Requested Tests",
            "head_sha": sha,
            "status": "in_progress",
            "external_id": prefix + os.environ["GITHUB_RUN_ID"],
            "details_url": run_url,
        },
    )
    return {
        "base_ref": pr["base"]["ref"],
        "pr_number": str(number),
        "platforms": json.dumps(platforms),
        "source_ref": sha,
        "check_id": str(check["id"]),
    }


def report_request(api, number: int, sha: str, check_id: int, result: str) -> str:
    """Finish only this run's supplemental check; never update CI Complete."""
    check = api.request("GET", f"/check-runs/{check_id}")
    if (
        check["head_sha"] != sha
        or not (check.get("external_id") or "").startswith(f"ci-requested:{number}:{sha}:")
        or not (check.get("external_id") or "").endswith(":" + os.environ["GITHUB_RUN_ID"])
        or check.get("name") != "CI Requested Tests"
    ):
        raise ValueError("Supplemental check does not belong to this workflow run")
    pr = api.request("GET", f"/pulls/{number}")
    obsolete = pr["state"] != "open" or pr["head"]["sha"] != sha
    conclusion = (
        "cancelled"
        if obsolete or result == "cancelled"
        else "success" if result == "success" else "failure"
    )
    api.request(
        "PATCH", f"/check-runs/{check_id}", {"status": "completed", "conclusion": conclusion}
    )
    return conclusion


def main() -> None:
    """Read inputs via environment variables so event text never becomes code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("comment", "prepare", "report"))
    args = parser.parse_args()
    api = GitHub()
    if args.operation == "report":
        print(
            report_request(
                api,
                int(os.environ["PR_NUMBER"]),
                os.environ["HEAD_SHA"],
                int(os.environ["CHECK_ID"]),
                os.environ["PIPELINE_RESULT"],
            )
        )
        return
    known = _load_watch_rules(Path(".github/configs"))
    if args.operation == "comment":
        result = handle_comment(
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), api, known
        )
    else:
        result = prepare_request(
            api, int(os.environ["PR_NUMBER"]), os.environ["HEAD_SHA"], os.environ["PLATFORM"], known
        )
    if isinstance(result, dict):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
