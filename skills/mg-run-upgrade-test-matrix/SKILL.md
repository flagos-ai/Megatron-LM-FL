---
name: mg-run-upgrade-test-matrix
description: Build, authorize, execute, and validate a capability- and hardware-aware Megatron-LM-FL test matrix for an NVIDIA upstream upgrade. Use after fork-delta routing and CI/runtime/platform audits to discover unit suites, functional recipes, golden environments, plugin/platform tests, FlagScale integration, and external accelerator gates; never treat unavailable hardware, an unrun command, or a skipped test as passed.
---

# Run Megatron Upgrade Test Matrix

Separate discovery, authorization, execution, and result validation. Building the matrix is static; never run tests unless the user authorizes the exact environment and scope.

## Workflow

1. Run `scripts/build_test_matrix.py` against immutable classifier inventory and domain routing.
2. Discover unit-test groups, functional model families and recipes, golden environments, CI hardware configs, platform registrations, and test-related routes from the fork Git object.
3. Group by capability identity, not only filename. Preserve member counts and representative paths.
4. Assign tiers in order: static, import/CPU, focused unit, single-accelerator, distributed, functional/golden, FlagScale E2E.
5. Record hardware, process count, datasets/checkpoints/services, command template, and evidence source. Mark unavailable or unknown prerequisites as `external-gate`, never pass.
6. Compose reviewed authorization with `scripts/apply_test_decisions.py`. Decisions may set owner, environment, command, disposition, and reason but cannot alter row identity or discovered facts.
7. Execute only explicitly authorized rows against the recorded commit. Capture start/end, exit code, log, environment, and artifacts; retain the first failure.
8. Validate the final ledger with `scripts/validate_test_matrix.py`. Coordinate workflow coverage with `mg-audit-cicd`.
9. Create a skill gap when a new test framework, scheduler, result oracle, or hardware prerequisite cannot be represented.

Read [matrix-contract.md](references/matrix-contract.md) before composing decisions and [execution-policy.md](references/execution-policy.md) before any execution.

## Completion

Deliver `test-matrix.json`, `test-matrix.tsv`, reviewed decisions, `test-validation.json`, environment evidence, and per-row logs for executed rows. Require every current test-hardware route to be covered, every discovered group represented, every non-pass visible, and every external gate owned.

Do not edit tests or fixtures to obtain a pass. Do not trigger CI, allocate hardware, install dependencies, or execute test commands during matrix discovery.
