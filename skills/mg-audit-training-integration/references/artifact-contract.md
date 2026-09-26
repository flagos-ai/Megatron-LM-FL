# Artifact contract

Use schema version 1.0 and immutable classifier refs. Stable row IDs derive from delta ID and path.

Immutable facts include Git status/path, lifecycle stages, fork/target existence, symbols, argument flags, annotated fields, calls, markers, and route domains.

Reviewed fields are owner, invariant, target_relationship, strategy, observing_tests, external_gate, reason, and evidence. Strategies are `preserve`, `adapt`, `upstream`, `redesign`, and `drop`. Never change path, route ID, or extracted facts through a decision file.
