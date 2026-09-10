# Classifier to Sync Artifact Contract

## Identity

Set `schema_version` and full SHAs for `history_base`, `sync_tree_base`, `release_base`, `fork`, and `target`. Record repository identity, generation time, tool version, worktree state, and the exact command.

## Coverage gates

Before handoff require:

- fork delta count equals the raw Git name-status count;
- each delta ID occurs exactly once;
- `unclassified=0`;
- marker health is reported without blocking classification;
- all both-changed paths are present;
- each delta has category, priority, risk labels, owner, provisional action, and validation evidence;
- each delta has exactly one primary problem domain, zero or more secondary domains, and a named handling skill or an explicit blocking skill gap;
- `unrouted=0`, `conflicting_routes=0`, and `unresolved_blocking_skill_gaps=0`;
- JSON, Markdown, and TSV totals agree;
- provenance is complete or explicitly `MANUAL`;
- the user approval records artifact hashes and immutable refs.
- the user approval binds `domain-routing.json` and `skill-gap-ledger.json` hashes as well as inventory and decisions.

## Two-axis classification

Keep disposition and solution method independent. The replay action answers what to retain; the domain route answers how to analyze, integrate, and verify it. Never assume all `REDESIGN` or all `REPLAY_FL` items share one method.

Route by feature group and semantic dependency, not merely by file. Permit secondary domains when one feature crosses plugin, platform, runtime, CI, build, or test boundaries. Record one coordinating primary domain to prevent duplicate ownership.

## Coverage-gap gate

Emit a structured gap whenever the current skills cannot supply a method, invariant, or credible acceptance test. Classify it as a one-off anomaly, reusable missing pattern, or correctness-critical gap. Require a skill change and rerun for reusable or correctness-critical gaps. Do not turn an unknown into a non-blocking note merely to complete handoff.

## Freshness

The sync skill must reject artifacts when repository identity, fork SHA, target SHA, schema version, coverage checks, or approval hashes differ. Re-run classification instead of patching stale artifacts.

## Mutation boundary

Classifier artifacts must live outside the source repository unless the user explicitly chooses a tracked audit directory. Generating artifacts never authorizes source changes, branch creation, commits, pushes, PRs, or submodule updates.

## Candidate overrides

Keep generic candidates immutable. Store reviewed changes in `decision-overrides.json` with delta ID, old action, new action, evidence type, evidence references, reviewer, and reason. Reject overrides that mention only intuition or clean-merge status. The effective decision matrix is the deterministic composition of generic candidates plus approved overrides.
