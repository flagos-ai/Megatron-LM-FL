---
name: mg-fl-upstream-sync
description: >
  Manages the upstream sync workflow for the Megatron-LM-FL fork (flagos-ai/Megatron-LM-FL
  forked from NVIDIA/Megatron-LM). Megatron-LM-FL is the full Megatron-LM with enhanced core
  and enhanced plugin — upgrading it means bringing the entire repo forward to align with a new
  upstream release, not just core and plugin. Handles creating dev branches aligned with upstream
  core releases, merging into main with conflict resolution, validating the custom plugin system
  (platform mechanism via cur_platform, override mechanism via @overridable/@override decorators,
  and new features like dualpipev/hetero), and running build/import verification. Use this skill
  whenever the user mentions syncing upstream, updating from NVIDIA/Megatron-LM, merging upstream
  releases (e.g. core_v0.16.1), fork sync, repo update, pulling upstream changes, plugin
  validation, or any upstream integration workflow for Megatron-LM-FL. Also trigger when the user
  references conflict resolution for megatron.core or megatron.plugin files, verifying @overridable
  decorators survive a merge, cur_platform replacements, or running FlagScale training validation
  after an upstream merge.
---

# Megatron-LM-FL Upstream Sync Workflow

You are guiding a developer through syncing their fork `flagos-ai/Megatron-LM-FL` with upstream
`NVIDIA/Megatron-LM`. The fork's core value is a custom plugin system — every decision you make
must protect plugin functionality above all else.

## Fork Architecture

Megatron-LM-FL is the full NVIDIA/Megatron-LM with enhanced core and enhanced plugin layered on
top. Think of it as: **Megatron-LM-FL = Megatron-LM + modified core + modified plugin**. When
upgrading to align with a new upstream Megatron-LM version, all components need to be updated —
including training, tests, examples, and everything else in the repo — not just core and plugin.

The fork's enhancements on top of upstream, all at the Python level:

- **Platform mechanism**: `megatron/plugin/platform/` — Multi-chip support. All `torch.cuda` API
  calls in `megatron/core/` and `megatron/plugin/` are replaced with `cur_platform` equivalents,
  enabling the same codebase to run on diverse accelerators (NVIDIA, MetaX, Hygon, etc.).

- **Override mechanism**: `megatron/plugin/decorators/` — Dynamic function replacement via
  `@overridable` and `@override` decorators. Selected Python-level functions in `megatron/core/`
  are decorated with `@overridable`, and alternative implementations live in `megatron/plugin/`
  using `@override`. This allows runtime swapping of behavior without modifying core code.

- **New features**: `megatron/plugin/dualpipev/`, `megatron/plugin/hetero/` — Fork-specific
  features built on top of the plugin system.

- **CI/CD toolchain**: GitHub workflows, testing infrastructure, and build configurations
  adapted for the fork's multi-platform needs.

**Key architectural constraint — FlagScale decoupling and the full upgrade scope:**
Megatron-LM-FL is consumed as a library via `pip install` + `import megatron.core` / `import
megatron.plugin`. Only `megatron.core` and `megatron.plugin` are pip-packaged (via
`pyproject.toml`). Other packages like `megatron.training`, `megatron.legacy`, `megatron.rl`,
and `megatron.post_training` exist in the repo but are NOT pip-installed — they resolve at
runtime via PYTHONPATH when used by downstream consumers like FlagScale. The training code
lives in a separate repo at `FlagScale/flagscale/train/megatron/`, derived from upstream
NVIDIA/Megatron-LM training code. A complete upstream upgrade has two co-equal parts:

1. **Megatron-LM-FL** (this skill) — upgrades the entire Megatron-LM-FL repo to align with the
   new upstream release. This includes `megatron.core` (with plugin enhancements), `megatron.plugin`,
   `megatron/training/`, and all other components (tests, examples, configs, etc.)
2. **FlagScale training** (companion skill: `flagscale-train-upstream-sync`) — upgrades
   `flagscale/train/megatron/` to align with the upstream training code changes

Both must be completed for a full upgrade. This means:
- Megatron-LM-FL must remain pip-installable after every sync
- The entire repo should be consistent with the target upstream version (plus fork enhancements)
- Public API surface of `megatron.core` must not break (FlagScale depends on it)
- FlagScale training code must be upgraded in tandem (not just validated)

## Repo Detection Preamble

Run this **at the start of every stage** to set `MG_FL_DIR` and `FLAGSCALE_DIR`. Both variables
are used throughout the workflow. If a repo is not found locally, offer to `git clone` it.

```bash
# --- Detect Megatron-LM-FL ---
MG_FL_DIR=""
for candidate in \
  "$(pwd)" \
  "$(pwd)/Megatron-LM-FL" \
  "$(pwd)/../Megatron-LM-FL" \
  "$(dirname $(pwd))/Megatron-LM-FL"; do
  if [ -d "$candidate/megatron" ] && [ -f "$candidate/setup.py" -o -f "$candidate/pyproject.toml" ]; then
    MG_FL_DIR=$(cd "$candidate" && pwd)
    break
  fi
done

if [ -z "$MG_FL_DIR" ]; then
  echo "Megatron-LM-FL not found locally. Cloning..."
  git clone https://github.com/flagos-ai/Megatron-LM-FL.git
  MG_FL_DIR=$(cd Megatron-LM-FL && pwd)
fi
echo "Megatron-LM-FL: $MG_FL_DIR"

# --- Detect FlagScale ---
# FlagScale must live as a sibling of Megatron-LM-FL in the same workspace directory.
WORKSPACE_DIR="$(dirname "$MG_FL_DIR")"
FLAGSCALE_DIR=""
for candidate in \
  "$WORKSPACE_DIR/FlagScale" \
  "$WORKSPACE_DIR/flagscale"; do
  if [ -d "$candidate/flagscale" ] && [ "$(cd "$candidate" && pwd)" != "$MG_FL_DIR" ]; then
    FLAGSCALE_DIR=$(cd "$candidate" && pwd)
    break
  fi
done

# FlagScale is only required from Stage 10 onward (and in the companion skill).
# If not found, set to empty and warn — do not block earlier stages.
if [ -z "$FLAGSCALE_DIR" ]; then
  echo "WARNING: FlagScale not found in $WORKSPACE_DIR. Set FLAGSCALE_DIR manually or clone when needed:"
  echo "  git clone <flagscale-repo-url> $WORKSPACE_DIR/FlagScale"
else
  echo "FlagScale: $FLAGSCALE_DIR"
fi

# --- Switch into Megatron-LM-FL for this skill's stages ---
cd "$MG_FL_DIR"
echo "Working directory: $(pwd)"
```

**Usage in each stage:** Every stage script should start with the preamble above (or source it
from a shared script). The variables `MG_FL_DIR` and `FLAGSCALE_DIR` are then available for
all subsequent commands. Stages 1-9 operate inside `$MG_FL_DIR`; Stage 10 hands off to
`$FLAGSCALE_DIR`.

---

## The Multi-Stage Sync Workflow

Each stage is sequential. Guide the user through the full flow when they ask to "sync upstream."

**Commit discipline:** After every stage that modifies code, run `pre-commit` and then `git commit`.
This creates clean bisect points and ensures code quality gates are enforced incrementally.
Every stage below ends with a **Checkpoint** section — follow it before moving to the next stage.
The commit message convention is: `Stage N: <brief description>`. This makes `git log` a readable
audit trail of the sync and enables clean rollback to any stage boundary.

---


## Phase Index

This workflow is split into phases. Read each phase file when you reach that point in the workflow.

| Phase | Stages | File | Description |
|-------|--------|------|-------------|
| 1 | Stage 1-2 | `phases/01-setup-and-analysis.md` | Repo setup, branch preparation, identify plugin changes |
| 2 | Stage 3-5 | `phases/02-merge-and-integrate.md` | Patch-based fork integration, plugin verification, stale reference fixes |
| 3 | Stage 6-7 | `phases/03-verify-and-test.md` | Build & import verification, unit test verification |
| 4 | Stage 8-10 | `phases/04-finalize.md` | Merge to main, sync report, hand off to FlagScale |

## Critical Rules

These are non-negotiable because the fork's entire value is the plugin system:

1. **Plugin files are sacred.** Never auto-resolve a P0 conflict toward upstream. Always keep the
   fork version and manually integrate upstream changes.
2. **cur_platform replacements must survive.** If upstream adds new `torch.cuda` calls in
   `megatron/core/`, they must be converted to `cur_platform` equivalents.
3. **@overridable decorators must survive.** If upstream modifies a decorated function, preserve
   the decorator and reconcile the function body.
4. **pip installability is mandatory.** Every stage that modifies code should be followed by a
   quick `pip install -e .` sanity check. Only `megatron.core` and `megatron.plugin` are
   pip-packaged — verify both with `import megatron.core` and `import megatron.plugin`.
5. **FlagScale training upgrade is the other half.** This skill covers the library (megatron.core
   + megatron.plugin). The full upgrade also requires syncing FlagScale's training code via the
   companion `flagscale-train-upstream-sync` skill.
6. **Rollback is always an option.** `git revert -m 1 <merge-commit>` should always be ready.
