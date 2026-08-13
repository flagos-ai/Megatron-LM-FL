---
name: mg-integrate-build-packaging
description: Audit and integrate Megatron-LM-FL build and packaging changes across an NVIDIA upstream upgrade, including pyproject metadata and dependencies, MANIFEST/package inclusion, plugin import surfaces, Docker images, shell entrypoints, test installation guidance, and wheel/editable-install gates. Use when build-packaging routes or fork-only build assets must be preserved without confusing static metadata checks with a successful build.
---

# Integrate Megatron Build and Packaging

Audit immutable Git objects before editing or building.

## Workflow

1. Run `scripts/audit_build_packaging.py` with classifier inventory and domain routing.
2. Cover every primary or secondary `build-packaging` route and compare fork/target existence, type, mode, and content hash.
3. Parse `pyproject.toml` build system, project metadata, dependencies, extras, and tool sections; inventory MANIFEST rules, Docker bases/stages/args, shell syntax, and plugin package/import surfaces dynamically.
4. Distinguish source inclusion, runtime dependency, optional accelerator dependency, build dependency, image/runtime provisioning, and test-only assets.
5. Compose owner, requirement, target relationship, strategy, and validation gates with `scripts/apply_build_decisions.py`; never edit immutable facts.
6. Validate with `scripts/validate_build_audit.py`. Later run editable install, wheel build/content inspection, clean import, and image checks only in an approved isolated environment.
7. Create a skill gap for an unmodeled build backend, native extension system, artifact registry, or package oracle.

Read [build-contract.md](references/build-contract.md) and [validation-gates.md](references/validation-gates.md).

## Completion

Deliver `build-audit.json`, `build-routes.tsv`, reviewed decisions, and `build-validation.json`. Require all routes covered and every row to have an owned requirement, target relationship, strategy, and validation gate.

Do not install dependencies, build wheels/images, update lockfiles, or modify source during audit. A successful import does not prove wheel contents or accelerator packaging.
