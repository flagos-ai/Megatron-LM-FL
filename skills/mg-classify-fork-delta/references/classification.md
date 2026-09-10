# Classification Policy

## Categories

| Category | Scope | Downstream owner |
|---|---|---|
| plugin-contract | Decorators, registry, initialization, public plugin contract | Plugin API audit |
| plugin-platform | Platform abstraction and platform registry | Platform capability audit |
| plugin-vendor | Vendor implementations and registrations | Vendor owner |
| plugin-feature | dualpipev, hetero, DSA kernels, plugin optimizer features | Feature owner |
| invasive-runtime | Modified upstream `megatron/core` files | Semantic integration |
| training | Training packages and root training entrypoints | Training integration |
| build-packaging | Build, wheel, dependencies, manifests, containers | Build/package audit |
| cicd-common | Shared automation | CI/CD audit |
| cicd-vendor | Vendor workflows, configs, and environment setup | Vendor CI owner |
| tests-common | Shared tests and test utilities | Test matrix audit |
| tests-vendor | Plugin/device/vendor-specific tests | Vendor test owner |
| examples-tools-docs | Executable examples, tools, docs, benchmarks | Compatibility audit |
| repository-metadata | Lint, license, ignore, contribution metadata | Finalization audit |

Use exactly one category per path and any number of orthogonal risk labels. Add a narrow rule when a real path is unclassified; never add a catch-all `other` category.

## Priorities

- P0: plugin contracts, invasive runtime/device behavior, numerical behavior, public API, or any both-changed path.
- P1: training, build, packaging, tests, dependencies, or surfaces gating P0 behavior.
- P2: CI/CD, executable examples, docs, and repository metadata.

Never lower priority because Git predicts a clean merge.

## Required review for both-changed paths

Record fork behavior, target change, invariant, observing test, downstream owner, provenance, and provisional replay action. A file-level record is insufficient when the file contains multiple independent fork symbols or marker blocks.

## Design baseline

Group every fork-owned path into plugin contract, platform/vendor, feature, invasive runtime, training, build/package, CI/CD, tests, or support surfaces. For each group record its purpose, public entry points, call-chain owner, invariants, dependencies, and `preserve/adapt/drop/manual` placeholder.
