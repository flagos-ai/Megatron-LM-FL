# Conflict Artifact Contract

Bind every artifact to the same full `sync_tree_base`, `fork`, and `target` SHAs used by classifier inventory, effective decisions, and domain routing.

Require one ledger row per `inventory.fork_changes[].both_changed == true`, sorted by delta ID. Preserve delta ID, path, priority, action, primary/secondary domains, blob evidence, and merge-tree classification as immutable candidate fields.

Store human resolutions separately. Each override must name the delta ID and provide owner, fork invariant, upstream change, affected symbols, resolution strategy, acceptance tests, evidence references, reviewer, approval status, and reason. Reject unknown or duplicate IDs and attempts to change immutable fields.

An effective row is complete only when:

- its strategy is allowed and compatible with the approved replay action;
- owner, invariant, upstream change, affected symbols, and at least one observing test exist;
- P0 or redesign approval is explicit;
- textual conflict status is retained;
- blocked test evidence names an external owner;
- new unsupported conflict shapes are linked to a skill-gap ID.
