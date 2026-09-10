# Domain Routing

Use two independent axes for every delta. Record a replay action for what survives and a domain route for how to integrate and verify it. Group related paths by feature and call chain before routing; a path count is not a problem count.

## Domain methods

| Domain | Required method | Core evidence |
|---|---|---|
| `upstream-conflict` | Compare base, fork, and target semantics at symbol or hunk level; preserve target architecture | overlap ledger, provenance, invariant test |
| `plugin-override` | Audit registry keys, decorators, signatures, forwarding, defaults, and every discovered implementation | override manifest, API chain, contract tests |
| `platform` | Audit capability abstractions and device assumptions without assuming a fixed vendor list | platform manifest, capability matrix, reviewed allowlist |
| `runtime-feature` | Reconstruct configuration, construction, dispatch, execution, and output paths | feature ledger, call chain, observing test |
| `upstream-provenance` | Prove target inclusion or later FL delta before dropping an early backport | commit, patch, symbol, or semantic evidence |
| `build-packaging` | Verify dependency, build, install, import, image, and artifact graphs | build/package manifest and smoke tests |
| `cicd` | Audit workflow graph, triggers, runners, permissions, references, matrices, and ignored coverage | CI manifest and static validation |
| `test-hardware` | Build capability-by-backend-by-hardware coverage; keep unavailable rows blocked | test matrix, logs, external-gate owners |
| `training-integration` | Verify Megatron API consumers, argument propagation, launch path, checkpoint and runtime contracts | downstream contract ledger and E2E evidence |
| `finalization` | Reconcile immutable refs, all ledgers, tests, external gates, and publication approval | final coverage index |

Do not create one domain per feature or vendor. Add a domain only when it requires a genuinely different reusable method. Discover concrete vendors and features dynamically through manifests.

## Routing contract

Generate candidate routes with `scripts/build_domain_routing.py` using the versioned policy and skill coverage manifest. Write `domain-routing.json` with:

- `schema_version`, immutable `refs`, `skill_set_version`, and candidate review status;
- one `routes` row per delta ID;
- exactly one `primary_domain`, optional distinct `secondary_domains`, a primary `handler_skill`, and one `secondary_handlers` entry per secondary domain for every row;
- `feature_group`, `reason`, required invariants, and acceptance evidence;
- summary counts for `unrouted` and `conflicting_routes`.

Use a primary domain as coordinator and secondary domains as mandatory reviewers. A route conflicts when handlers prescribe incompatible ownership, ordering, or acceptance. Resolve the method conflict explicitly or create a blocking skill gap.

## Routing adequacy

Reject a route when it is based only on filename proximity, action equality, or the desire to avoid an unknown. Route again when implementation exposes a cross-file dependency, generated registry, new build mechanism, new hardware capability, or downstream consumer absent from the original feature group.
