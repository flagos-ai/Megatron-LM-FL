---
name: mg-audit-training-integration
description: Audit and integrate Megatron-LM-FL training-lifecycle changes across an NVIDIA upstream upgrade, including public imports, argument registration and validation, config propagation, model/optimizer construction, train/eval loops, checkpoint and output behavior, pipeline scheduling, precision parameters, and observing tests. Use when training entry points or shared training configuration changed in both fork and target and semantic compatibility must be proven before replay.
---

# Audit Megatron Training Integration

Build an immutable lifecycle ledger before editing training code.

## Workflow

1. Run `scripts/audit_training_integration.py` with classifier inventory and domain routing.
2. Cover every primary or secondary `training-integration` route.
3. Extract fork and target symbols, public imports, argument flags, annotated config fields, calls, and FlagScale ownership markers directly from Git objects.
4. Map each delta through lifecycle stages: public API, argument declaration, validation, config propagation, construction, execution, checkpoint/output, and observing test.
5. Treat matching names as evidence only. Review default changes, renamed/removed symbols, value flow, control-flow ownership, distributed semantics, and side effects.
6. Compose invariant, strategy, owner, target relationship, and observing tests with `scripts/apply_training_decisions.py`; never edit immutable facts.
7. Run `scripts/validate_training_audit.py`. Hand executable rows to `mg-run-upgrade-test-matrix`.
8. Create a skill gap for an unmodeled launcher, scheduler, configuration system, checkpoint protocol, or result oracle.

Read [lifecycle-contract.md](references/lifecycle-contract.md) and [artifact-contract.md](references/artifact-contract.md).

## Completion

Deliver `training-audit.json`, `training-routes.tsv`, reviewed decisions, and `training-validation.json`. Require all training routes covered and every row to have an invariant, target relationship, integration strategy, owner, and observing test or owned external gate.

Do not modify source, infer compatibility from a clean merge, or run training during audit.
