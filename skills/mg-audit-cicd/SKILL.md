---
name: mg-audit-cicd
description: Statically audit Megatron-LM-FL GitHub Actions, CI configs, reusable workflows, local actions, scripts, runner labels, permissions, triggers, backend matrices, ignored tests, and local references across an NVIDIA upstream upgrade. Use before relying on remote CI or integrating fork-owned .github changes; dynamically discover backends and test surfaces, detect stale/missing references and failure masking, and produce owned coverage decisions without triggering workflows.
---

# Audit Megatron CI/CD

Audit exact Git refs without editing workflows or triggering remote jobs.

## Workflow

1. Run `scripts/audit_cicd.py` with classifier inventory and domain routing.
2. Parse every fork workflow/config, record triggers, permissions, jobs, runners, reusable workflows, matrices, secrets, and failure controls.
3. Resolve local workflow/action/script/config/Dockerfile/test references against the same Git tree.
4. Discover backend configs and platform registrations dynamically; require CI coverage or an owned exclusion.
5. Compare fork and target presence. Preserve fork-only CI assets unless an approved removal decision exists.
6. Run shell syntax checks through stdin for discovered shell files; never execute them.
7. Review `continue-on-error`, `|| true`, disabled/ignored tests, path filters, and conditional jobs as possible failure masking.
8. Compose owner/exclusion decisions and run `scripts/validate_cicd_audit.py`. Coordinate executable coverage with `mg-run-upgrade-test-matrix`.
9. Create a skill gap for generated workflows, external templates, custom expression semantics, or CI providers outside the model.

Read [cicd-contract.md](references/cicd-contract.md) and [artifact-contract.md](references/artifact-contract.md).

## Completion

Deliver `cicd-audit.json`, `workflow-matrix.tsv`, `missing-references.tsv`, reviewed decisions, and `cicd-validation.json`. Require all 36 current CI routes covered, YAML and shell syntax accounted for, local references resolved, backend/config coverage closed, failure masking reviewed, and external gates owned.
