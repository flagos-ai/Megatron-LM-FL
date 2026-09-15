---
name: mg-integrate-runtime-features
description: Inventory, analyze, redesign, and validate invasive Megatron-LM-FL runtime features across an NVIDIA upstream upgrade. Use when fork changes span configuration, construction, dispatch, execution, output/checkpoint behavior, tests, FlagScale marker blocks, or multiple files; when clean merges may break call chains; or when REPLAY_FL/REDESIGN work needs feature-level invariants and observing tests rather than path-level patch replay.
---

# Integrate Megatron Runtime Features

Treat runtime changes as semantic feature chains, not independent files or marker blocks. Audit before source edits and consume exact classifier decisions, domain routing, conflict, plugin, and platform evidence.

## Workflow

1. Run `scripts/build_runtime_feature_ledger.py` for every route involving `runtime-feature`.
2. Group paths by classifier feature group, then review group cohesion. Split or merge groups only through an explicit decision ledger with evidence; never encode current feature names in generic rules.
3. Reconstruct stages: configuration/arguments, construction/import, dispatch/scheduling, execution/communication, output/checkpoint/metrics, and observing tests.
4. Record symbols and cross-file definition/import/call relationships. Treat FlagScale Begin/End/Add as ownership hints only.
5. For each feature, define invariant, target change, approved strategy, owner, dependencies, failure modes, and tests. Require explicit approval for `REDESIGN`.
6. Start from target code and replay the narrowest approved behavior. Preserve upstream architecture and later FL deltas.
7. Coordinate override, platform, conflict, build, training, and test secondary domains through their handler skills.
8. Compose reviewed decisions with `scripts/apply_feature_decisions.py` and run `scripts/validate_runtime_features.py`.
9. Create a skill gap when the feature cannot be represented as an observable lifecycle, grouping is ambiguous, or a new generated/dynamic mechanism defeats static discovery.

Read [feature-contract.md](references/feature-contract.md) and [artifact-contract.md](references/artifact-contract.md).

## Invariants

- Every runtime-related delta belongs to exactly one reviewed feature group.
- Every feature records at least one implementation stage and one observing test or owned external gate.
- Every configuration flag reaches its consumer; every added definition has import/call evidence; every output/checkpoint change has compatibility evidence.
- Clean application and marker pairing never prove semantic preservation.
- Hardware-unavailable evidence remains blocked with an owner.
- Source edits occur only later on an authorized isolated branch.

## Completion

Deliver `candidate-runtime-features.json`, `runtime-feature-matrix.tsv`, reviewed decisions, `effective-runtime-features.json`, and `runtime-validation.json`. Require complete delta coverage, reviewed grouping, lifecycle/invariant/test closure, explicit redesign approval, and zero runtime-specific skill gaps before integration.
