---
name: mg-integrate-upstream-conflicts
description: Inventory, analyze, approve, and validate Megatron-LM fork paths changed by both the exact previous NVIDIA sync tree and a target NVIDIA release. Use for three-way textual and semantic conflict analysis, clean-merge risk review, target-first replay planning, early-backport provenance handling, cross-domain conflict coordination, or proving every both-changed delta has an invariant, owner, strategy, approval, and observing test before an upstream upgrade.
---

# Integrate Megatron Upstream Conflicts

Build an auditable conflict plan before editing source. Use exact classifier refs and approved artifacts. Operate read-only until the user separately authorizes an isolated integration branch.

## Intake

Require `inventory.json`, `effective-decisions.json`, `domain-routing.json`, and the repository. Verify their full refs agree. Use `sync_tree_base` as the explicit three-way base; do not replace it with a historical merge-base. Reject unclassified, unrouted, duplicate, or missing both-changed delta IDs.

Read [conflict-policy.md](references/conflict-policy.md) before proposing resolutions and [artifact-contract.md](references/artifact-contract.md) before handoff.

## Workflow

1. Run `scripts/build_conflict_ledger.py` to execute read-only `git merge-tree <sync-tree-base> <fork> <target>` and build the candidate ledger.
2. Require every classifier `both_changed` delta exactly once, including paths that merge textually without markers.
3. Inspect base, fork, and target at symbol or hunk granularity. Record the fork invariant, upstream change, affected symbols, provenance relationship, owner, strategy, and observing tests in a separate resolution override file.
4. Treat plugin, platform, runtime, training, build, CI, and test domains as required secondary reviewers from `domain-routing.json`; do not let this skill invent their domain semantics.
5. Start every resolution from target code. Replay only the approved FL behavior. Never copy a stale fork file wholesale over target.
6. Require explicit user approval for P0 strategies and every `REDESIGN`. Keep clean textual merges classified as semantic risks until their invariants are observed.
7. Compose overrides with `scripts/apply_conflict_resolutions.py`. Reject changed action, path, refs, or evidence identity.
8. Validate the effective ledger with `scripts/validate_conflict_ledger.py`. Stop on any unresolved row, missing evidence, unapproved P0, or uncovered both-changed delta.
9. During later source integration, update implementation commit and test result without rewriting the approved strategy. Create a skill gap when a new conflict shape lacks a rule, invariant, or credible observing test.

```bash
python scripts/build_conflict_ledger.py \
  --repo /path/to/Megatron-LM-FL \
  --inventory /artifacts/inventory.json \
  --decisions /artifacts/effective-decisions.json \
  --routing /artifacts/domain-routing.json \
  --output /artifacts/upstream-conflicts
```

## Non-negotiable rules

- Distinguish textual conflicts, automatic textual merges, and semantic conflicts.
- Treat clean application as evidence only, never acceptance.
- Preserve target architecture unless an approved decision explicitly replaces it.
- Require provenance evidence before dropping an early NVIDIA backport; replay later FL deltas separately.
- Never use blanket `ours` or `theirs`, whole-tree replacement, direct changes to `main`, or unapproved P0 resolution.
- Treat `FlagScale Begin/End/Add` markers only as ownership hints; recover invariants from deltas, symbols, call chains, and tests.
- Keep unavailable hardware evidence blocked with an owner; never mark it passed.
- Keep candidate output immutable and place human decisions in overrides.

## Completion

Deliver `merge-tree.txt`, `candidate-conflict-ledger.json`, reviewed resolution overrides, `effective-conflict-ledger.json`, and `conflict-validation.json`. Require full both-changed coverage, zero unresolved strategies, all required approvals, focused evidence for P0 invariants, and zero open conflict-specific skill gaps before allowing the orchestrator to integrate these paths.
