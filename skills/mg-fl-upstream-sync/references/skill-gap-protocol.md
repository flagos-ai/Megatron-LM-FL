# Skill Coverage-Gap Protocol

Create a gap whenever existing skills cannot provide a safe method, invariant, ownership decision, or credible acceptance test. Continue discovering unrelated work, but stop the affected route before source integration or final acceptance.

## Required ledger

Write `skill-gap-ledger.json` with `schema_version`, immutable `refs`, and `gaps`. Give every gap a stable `MG-GAP-NNNN` ID and record:

- affected delta IDs, feature groups, paths, and detecting phase;
- current domain or `unknown`;
- the exact missing rule, invariant, evidence, or method conflict;
- `gap_class`: `one-off-anomaly`, `reusable-pattern`, or `correctness-critical`;
- `blocking`, temporary disposition, owner, evidence, and status;
- proposed target skill, proposed change, regression fixture, and rerun phases.

Use status `open`, `accepted-one-off`, `skill-update-required`, or `resolved`. Never use a passing test alone to close a gap when the behavior was not observed.

## Decision policy

- Permit `accepted-one-off` only for a repository-specific anomaly with documented evidence, reviewer, bounded impact, and no reusable pattern.
- Require a skill update for `reusable-pattern`, even if this upgrade can be manually completed.
- Set `blocking=true` and stop the affected route for `correctness-critical`, or whenever the gap undermines classification, replay decisions, or acceptance credibility.
- Treat conflicting domain methods and missing observing tests as blocking until resolved.

## Skill-update loop

1. Preserve the raw evidence and register the gap before changing instructions.
2. Update the smallest responsible skill or introduce a domain skill only when the method is genuinely distinct.
3. Add a generic regression fixture; never encode the current release tag, PR, vendor, or path as the rule.
4. Validate and forward-test the changed skill when practical.
5. Re-run every phase named by the gap and regenerate affected artifacts.
6. Obtain renewed approval when any approved artifact hash or decision changes.
7. Close the gap with evidence and retain it in the final ledger.

At finalization report two independent verdicts: whether the repository upgrade is complete and whether the skill set adequately covered every case discovered by that upgrade.
