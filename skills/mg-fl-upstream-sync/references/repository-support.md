# Repository support closure

Treat repository-support as an orchestrator-owned cross-cutting domain for documentation, contribution templates, ignore rules, formatting/maintenance scripts, test recipes, golden values, and other support assets that do not justify a standalone integration method.

Run `scripts/audit_repository_support.py` against immutable inventory/routing. Compare fork and target existence/hash and parse format-specific facts: Markdown links/fences, gitignore rules, YAML/JSON validity, shell syntax, executable mode, and referenced test artifacts. Compose owner, purpose, target relationship, strategy, downstream consumers, and validation evidence. Do not remove a support asset merely because runtime imports do not reference it; consult all secondary domains.
