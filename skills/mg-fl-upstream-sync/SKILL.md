---
name: mg-fl-upstream-sync
description: Integrate an approved Megatron-LM-FL fork delta onto a pinned NVIDIA Megatron-LM release while preserving plugin contracts, multi-vendor overrides, platform abstraction, invasive FlagScale features, CI/CD, tests, and packaging. Use only after mg-classify-fork-delta has produced complete, current, user-approved artifacts; use for replaying approved changes, resolving semantic conflicts, validating vendors and overrides, testing the upgraded tree, and preparing a reviewed final merge. Do not use this skill to invent or bypass fork classification.
---

# Sync Megatron-LM-FL with NVIDIA Upstream

Consume an approved classifier bundle and coordinate domain-specific integration in auditable batches. Do not begin when classification, domain routing, skill coverage, or approval is missing, stale, incomplete, or unapproved.

## Intake gate

Read [artifact-intake.md](references/artifact-intake.md), [domain-routing.md](references/domain-routing.md), and [skill-gap-protocol.md](references/skill-gap-protocol.md). Run `scripts/validate_classifier_bundle.py` against the repository and bundle before creating a branch. Require matching repository identity, fork SHA, target SHA, schema, zero classification and routing blockers, complete decisions, zero unresolved blocking skill gaps, and an approval record bound to artifact hashes.

If validation fails, stop and invoke `mg-classify-fork-delta`. Do not reconstruct the inventory with ad hoc Git commands.

## Integration workflow

1. Record the approved immutable refs and the clean starting worktree.
2. Create a uniquely named integration branch from the approved target SHA only after user authorization.
3. Dispatch each feature group through its primary domain method and consult every secondary domain. Apply decisions in dependency order: plugin contract; platform; vendor implementations; override hooks; plugin features; invasive runtime/training changes; build/package; CI/CD; tests; then audit support files with  and [repository-support.md](references/repository-support.md).
4. For `UPSTREAM_COVERS`, adopt target code and prove no later FL delta remains.
5. For `UPSTREAM_PLUS_FL_DELTA`, start from target code and replay only the recorded FL delta.
6. For `REPLAY_FL`, adapt the recorded feature and preserve its invariant; never copy an entire stale file over target.
7. Stop on every `REDESIGN` or unresolved `MANUAL` decision and request user direction.
8. When implementation exposes a new conflict shape, missing invariant, unsupported domain, incompatible methods, or an unobservable behavior, add it to `skill-gap-ledger.json` and apply the gap protocol. Never silently improvise reusable policy.
9. After each batch, run its observing tests and update the execution ledger. Commit only at approved stage boundaries.
10. Run the layered validation matrix. Mark unavailable hardware suites as external gates, never as passes.
11. Re-run intake validation and every affected domain phase after a skill changes. Generate the final coverage report and request approval before merge, push, or PR creation.

Read the applicable references before each batch:

- [integration-policy.md](references/integration-policy.md) for replay semantics and conflict handling.
- [domain-routing.md](references/domain-routing.md) for selecting the problem-specific method and coordinating cross-domain work.
- [skill-gap-protocol.md](references/skill-gap-protocol.md) whenever existing instructions do not safely cover a discovered case.
- [plugin-platform-policy.md](references/plugin-platform-policy.md) for overrides, markers, and devices.
- [cicd-verification.md](references/cicd-verification.md) for CI and layered testing.
- [repository-support.md](references/repository-support.md) for cross-cutting documentation, maintenance, recipe, golden-value, and helper-script closure.

## Repository support closure

Audit repository-support routes with scripts/audit_repository_support.py and read references/repository-support.md. Preserve documentation, contribution metadata, ignore rules, recipes, golden values, and maintenance helpers according to their downstream consumers and every secondary domain. Require an owner, purpose, target relationship, strategy, consumers, and validation evidence.

## Invariants

- Preserve target upstream functionality unless an approved decision explicitly replaces it.
- Preserve plugin dispatch contracts and validate every vendor implementation against the new base signature and behavior.
- Treat FlagScale markers as incomplete, non-uniform ownership hints, not block grammar or completeness proof; validate Git deltas, symbols, call chains, and tests.
- Convert device assumptions only when the platform API has a compatible capability; maintain a reviewed CUDA-specific allowlist.
- Never treat a clean patch application, equal decorator count, import success, or syntax success as semantic validation.
- Never force an unknown case into the nearest domain to achieve zero open items. Surface the coverage gap and decide whether the skill must evolve.
- Never merge directly into `main`, overwrite a dirty worktree, use broad `ours/theirs`, or use tree replacement before the integrated tree is independently complete.

## Completion

Deliver an execution ledger mapping every classifier delta ID to its applied action, domain method, resulting commit, validation evidence, and status. Also deliver the final skill-gap ledger and a skill-adequacy verdict. Require zero unresolved IDs, zero unresolved blocking gaps, no conflicting routes, all locally available gates passing, external gates enumerated with owners, and user approval before final publication.
