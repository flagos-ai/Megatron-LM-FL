---
name: mg-audit-plugin-overrides
description: Audit Megatron-LM-FL plugin override compatibility across an exact sync-tree base, fork, and target NVIDIA release. Use when overridable decorators, centralized or eager registrations, registry keys, vendors, target or implementation signatures/defaults, async/static/class behavior, lazy imports, fallback behavior, parameter forwarding, or target bodies may drift during an upstream upgrade; also use to prove every dynamically discovered override has an owner, disposition, and focused contract test.
---

# Audit Megatron Plugin Overrides

Audit before editing source. Consume immutable classifier refs and domain routing. Do not import accelerator implementations merely to discover them; use AST and Git objects so unavailable vendor runtimes do not hide coverage.

## Workflow

1. Require exact `sync_tree_base`, `fork`, and `target` SHAs from classifier artifacts.
2. Run `scripts/audit_plugin_overrides.py`. Dynamically parse centralized `register(...)`, eager `@override(...)`, and core `@overridable` sites; never use a fixed vendor list.
3. Resolve every target and implementation against Git objects. Compare object kind, sync/fork/target signatures, parameter names/order/kinds/defaults, async behavior, decorators, and target existence.
4. Preserve registry identity as `(method_key, vendor)`. Reject duplicate identities, unresolved dotted paths, missing implementations, stale targets, and unexplained signature drift.
5. For every mismatch, inspect the complete target → wrapper/registry → implementation → forwarding call → observing test chain. Record a reviewed disposition in a separate decision ledger.
6. Distinguish intentional implementation-body differences from contract drift. Equal decorator or registry counts are not proof.
7. Coordinate raw CUDA/device behavior with `mg-preserve-platform-patches`; coordinate both-changed target bodies with `mg-integrate-upstream-conflicts`.
8. Validate decisions with `scripts/validate_override_audit.py`. Create a skill gap when generated registrations, dynamic paths, callable factories, or forwarding mechanisms are not represented by the audit model.
9. Re-run after plugin or target API edits and require stable symbol identities.

Read [override-contract.md](references/override-contract.md) for compatibility rules and [artifact-contract.md](references/artifact-contract.md) for required outputs.

```bash
python scripts/audit_plugin_overrides.py \
  --repo /path/to/Megatron-LM-FL \
  --inventory /artifacts/inventory.json \
  --routing /artifacts/domain-routing.json \
  --output /artifacts/plugin-overrides
```

## Non-negotiable rules

- Discover vendors, registrations, targets, and implementations from source artifacts.
- Audit default and vendor-specific implementations independently.
- Preserve lazy-import fallback semantics and vendor selection behavior.
- Compare defaults and optionality, not only parameter count.
- Treat class overrides through constructor/inheritance contracts as well as registration.
- Treat methods, static/class methods, async functions, and callable classes distinctly.
- Require an observing test for each required registration and each intentional incompatibility.
- Keep unavailable hardware tests blocked with owners, never passed.
- Never change the generic rule to match one release, PR, vendor, or path.

## Completion

Deliver `override-audit.json`, `override-matrix.tsv`, reviewed decisions, and `override-validation.json`. Require zero unresolved registry identities, no unexplained missing/stale/duplicate/signature mismatches, all discovered vendors represented, every exception owned and tested, and zero open override-specific skill gaps before the orchestrator integrates plugin changes.
