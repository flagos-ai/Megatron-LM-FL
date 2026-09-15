# Conflict Resolution Policy

Use the target file as the implementation base for every both-changed path.

| Strategy | Meaning | Required proof |
|---|---|---|
| `TARGET_COVERS` | Target already preserves the approved fork invariant | provenance or semantic equivalence plus observing test |
| `TARGET_PLUS_FL_DELTA` | Target is retained and only a later FL delta is replayed | isolated FL delta, affected symbols, focused test |
| `REPLAY_FL_NARROW` | Target lacks the feature and it is replayed at symbol/block granularity | invariant, dependency chain, focused test |
| `REDESIGN_APPROVED` | Target architecture invalidates the old implementation | alternatives, user decision, new invariant and test |

Never use a resolution strategy to change the classifier action silently. Flag incompatible action/strategy pairs for review.

Prioritize P0 for plugin contracts, platform/device behavior, invasive runtime semantics, training correctness, checkpoint compatibility, and security-sensitive CI permissions. Use P1 for build/package and test infrastructure; use P2 for support metadata. Require explicit approval for P0 and all redesigns.

A `git merge-tree` block containing conflict markers is a textual conflict. A both-changed path without markers remains a semantic-conflict candidate. Deleted, renamed, binary, or unparsable cases require manual evidence and may expose a new skill gap.
