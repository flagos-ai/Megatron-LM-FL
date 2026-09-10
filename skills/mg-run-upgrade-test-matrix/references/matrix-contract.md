# Matrix contract

Use schema version 1.0. Pin the classifier `refs`; reject mismatched inventory, routing, decisions, or results.

Each row has an immutable ID derived from kind, capability, hardware, and source identity. Preserve tier, member paths/count, required resources, command template, and route IDs as discovered facts.

Decision fields are `owner`, `disposition`, `environment`, `command`, `reason`, and `evidence`. Valid dispositions are `authorized`, `external-gate`, `not-applicable`, and `superseded`. Execution statuses are `not-run`, `pass`, `fail`, `blocked`, and `skipped`. Only an exit-code-backed result may be pass.

Group large functional recipe sets by model family and hardware/golden environment. Retain counts and representative paths so grouping never hides discovery.
