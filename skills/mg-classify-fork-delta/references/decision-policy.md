# Generic Replay Decision Policy

## Automatic evidence hierarchy

1. Identical fork and target blob: `UPSTREAM_COVERS` with high confidence.
2. Fork-owned plugin path absent from target: `REPLAY_FL`.
3. Fork-only changed path: `REPLAY_FL`, subject to dependency review.
4. Both-changed runtime with uncovered upstream-derived provenance: `REDESIGN`.
5. Both-changed runtime without equivalence proof: conservatively `REDESIGN`.
6. Both-changed CI, tests, build, packaging, docs, or metadata: `UPSTREAM_PLUS_FL_DELTA`.
7. No safe rule: `MANUAL`.

Do not infer equivalence from a matching PR number alone. A reviewer may promote a provenance candidate to `UPSTREAM_COVERS` only after patch or semantic equivalence review and after proving there are no independent fork commits on the path.

## Generality constraints

Do not encode tag names, PR numbers, vendors, repository-specific delta IDs, or current file lists. Discover evidence from the provided refs and artifacts. Treat a real-upgrade decision matrix as regression data, not executable policy.
