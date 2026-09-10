# Classifier Bundle Intake

Require a complete bundle produced by `mg-classify-fork-delta` and a separate approval JSON. Bind approval to the full fork and target SHAs plus the SHA-256 of `inventory.json`, effective decisions, `domain-routing.json`, and `skill-gap-ledger.json`.

Run `scripts/validate_classifier_bundle.py` before branch creation and again before finalization. Reject missing files, unsupported schema, SHA drift, unclassified or unrouted paths, incomplete decisions, conflicting routes, unresolved blocking skill gaps, and approval mismatches.

Never edit a stale bundle to make validation pass. Re-run classification and ask for renewed approval.
