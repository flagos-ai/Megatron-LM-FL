---
name: mg-classify-fork-delta
description: Inventory and classify all Megatron-LM-FL fork-owned changes before an NVIDIA Megatron-LM upstream upgrade. Use when establishing historical, sync-tree, release, fork, and target refs; discovering plugin, override, platform, FlagScale marker, CI/CD, test, packaging, and vendor surfaces; identifying both-changed paths and upstream backports; building a replay decision matrix; or proving an upgrade plan covers every fork delta. This skill is read-only and must run before mg-fl-upstream-sync.
---

# Classify Megatron-LM-FL Fork Delta

Create the authoritative, read-only input for an upstream upgrade. Do not create branches, edit source, apply patches, update submodules, commit, stash, reset, or merge.

## Required inputs

Require the repository path, sync-tree base, release base, fork ref, target ref, and an output directory outside the repository. Accept an optional historical base; otherwise compute it only as historical evidence. Resolve every ref to a full commit SHA.

Distinguish these refs:

- `history_base`: Git ancestry evidence; never assume it is the fork delta base.
- `sync_tree_base`: exact NVIDIA tree used by the previous fork sync; use this for `base..fork` inventory.
- `release_base`: prior official NVIDIA release; use it to explain release history.
- `fork`: current Megatron-LM-FL state.
- `target`: intended NVIDIA release.

Stop if the sync-tree base is uncertain. Read [classification.md](references/classification.md) before resolving uncertain categories and [artifact-contract.md](references/artifact-contract.md) before publishing results. Read `references/domain-routing-policy.json` and `references/skill-coverage.json` before generating domain routes.

## Workflow

1. Record worktree state, refs, ancestry, merge-bases, tags, remotes, and relevant commit evidence without modification.
2. Run `scripts/classify_fork_delta.py` with explicit refs and an approved output directory.
3. Run `scripts/derive_replay_decisions.py` after inventory/provenance generation to create generic evidence-based candidates.
4. Resolve every `unclassified` path with a narrow rule. Never hide it in `other`.
5. Review every `both_changed` path and record its invariant, observing test, downstream owner, and provisional action.
6. Investigate upstream provenance for fork commits that reference NVIDIA PRs, commits, cherry-picks, backports, or upstream syncs. Distinguish merged-to-dev from included-in-target-release.
7. Review AST-derived override sites, FlagScale blocks, device assumptions, plugin/platform/vendor surfaces, CI/CD, tests, build, packaging, and repository metadata.
8. Review generic candidates and record evidence-backed changes in `decision-overrides.json`; compose them with `scripts/apply_decision_overrides.py`.
9. Assign exactly one effective action to every delta: `UPSTREAM_ONLY`, `UPSTREAM_COVERS`, `UPSTREAM_PLUS_FL_DELTA`, `REPLAY_FL`, `REDESIGN`, or `MANUAL`.
10. Route every delta to one primary problem domain and zero or more secondary domains. Treat the action as disposition and the domain as solution method; never substitute one for the other.
11. Create `skill-gap-ledger.json`. Add a gap whenever no existing domain method, invariant, or acceptance test can safely handle an item. Do not force unknown work into the closest category.
12. Produce the design baseline and decision inputs. Present them for user approval. Do not begin integration.

```bash
python scripts/classify_fork_delta.py \
  --repo /path/to/Megatron-LM-FL \
  --history-base <historical-ref> \
  --sync-tree-base <exact-previous-upstream-tree> \
  --release-base <previous-release-tag> \
  --fork <fork-ref> \
  --target <target-release-tag> \
  --output /approved/artifact/directory
```

Then derive generic candidates:

```bash
python scripts/derive_replay_decisions.py \
  --repo /path/to/Megatron-LM-FL \
  --inventory /artifact/directory/inventory.json \
  --provenance /artifact/directory/upstream-provenance.json \
  --output /artifact/directory/decisions
```

Compose reviewed overrides without editing candidates:

```bash
python scripts/apply_decision_overrides.py \
  --candidates /artifact/directory/decisions/candidate-decisions.json \
  --overrides /artifact/directory/decision-overrides.json \
  --output /artifact/directory/effective-decisions.json
```

Build domain routes against an explicit, versioned skill capability manifest:

```bash
python scripts/build_domain_routing.py \
  --decisions /artifact/directory/effective-decisions.json \
  --policy references/domain-routing-policy.json \
  --coverage references/skill-coverage.json \
  --output /artifact/directory/routing
```

Review `domain-routing.json`, specialization candidates, and every skill gap. Update the coverage manifest only after the responsible skill method and regression scenarios exist.

Use `--allow-divergent-upstream` only after reviewing and recording the release merge-base.

## Non-negotiable checks

- Include every `git diff --name-status sync_tree_base..fork` path exactly once.
- Leave zero unclassified paths. Inventory FlagScale Begin, End, Add, legacy variants, and unmatched markers as non-blocking ownership hints.
- List every both-changed path even when Git predicts a clean merge.
- Discover vendors and CI/test surfaces dynamically; do not use a fixed acceptance list.
- Treat path classification as an index, not proof of semantic equivalence.
- Require `unrouted=0`. Unknown categories or missing handlers must create blocking skill gaps rather than fall through to `other`.
- Keep one-off repository anomalies distinct from reusable missing patterns. Require a skill update for any gap that affects decision correctness, acceptance credibility, or a pattern likely to recur.
- Require commit/patch/symbol evidence before declaring `UPSTREAM_COVERS`. Generic automation may remain more conservative than a reviewed decision.
- Never encode release tags, PR numbers, vendor names, file IDs, or one-off paths in decision rules.
- Record every human adjustment to a generic candidate in a decision override ledger with evidence, reviewer, and reason; never patch the generic rule to match one upgrade.
- Mark uncertain provenance or product choices `MANUAL`; never infer that a dev PR is in a release tag.
- Keep JSON, Markdown, and TSV totals consistent.
- Record dirty worktree state and verify the source worktree is unchanged after analysis.

## Required outputs

Require `inventory.json`, `inventory.md`, `fork-changes.tsv`, `both-changed.tsv`, `override-manifest.json`, `flagscale-block-manifest.json`, `platform-manifest.json`, `upstream-provenance.json`, `replay-decision-matrix.md`, `manual-decisions.md`, `domain-routing.json`, `skill-gap-ledger.json`, and `mg-fl-design-baseline.md` before handing off to the sync skill.

The initial classifier may report provenance and decisions as incomplete. That is a blocker for handoff, not permission to omit the files.

## Acceptance

Stop on unresolved refs, uncertain sync-tree base, inconsistent totals, unclassified or unrouted paths, incomplete provenance, missing owners/tests/actions, unresolved blocking skill gaps, or any source worktree mutation. Obtain user approval over the immutable ref set, decision artifacts, domain routing, and skill-gap disposition before invoking `mg-fl-upstream-sync`.
