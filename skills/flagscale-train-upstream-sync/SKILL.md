---
name: flagscale-train-upstream-sync
description: >
  Manages the upstream sync workflow for FlagScale's training and legacy code at
  FlagScale/flagscale/train/megatron/, which is derived from NVIDIA/Megatron-LM but contains
  FlagScale's own modifications that must be preserved during upgrades. When Megatron-LM-FL
  upgrades to a new upstream core release (e.g. core_v0.16.1), the FlagScale code must also be
  updated to align with the corresponding upstream training and legacy code. Use this skill
  whenever the user mentions upgrading FlagScale training or legacy code, syncing
  flagscale/train/megatron with upstream, updating FlagScale after a Megatron-LM-FL upgrade,
  aligning FlagScale with a new core release, fixing FlagScale import errors after an upstream
  merge, or any workflow involving FlagScale/flagscale/train/megatron/ and upstream
  NVIDIA/Megatron-LM compatibility. Also trigger when the user mentions broken training after a
  Megatron-LM-FL sync, or needs to propagate upstream API changes into FlagScale's training or
  legacy layer.
---

# FlagScale Training & Legacy Code Upstream Sync Workflow

You are guiding a developer through the second half of a full upstream upgrade. A complete
upgrade from one NVIDIA/Megatron-LM release to another has two parts:

1. **Megatron-LM-FL** (companion skill: `mg-fl-upstream-sync`) — syncs the fork with upstream.
   For `megatron.core` and `megatron.plugin`, preserves fork enhancements while merging upstream
   changes. For other packages (training/, legacy/, etc.), simply keeps them consistent with
   upstream (no fork modifications). Only core + plugin are pip-installed.
2. **FlagScale** (this skill) — upgrades `flagscale/train/megatron/` (training and legacy code)
   to align with upstream, while preserving FlagScale's own modifications.

Both must be completed for a full upgrade. This skill handles part 2.

## Architecture Context

Megatron-LM-FL is a fork of NVIDIA/Megatron-LM. The fork has modifications only in `megatron.core`
and `megatron.plugin` — other packages (training/, legacy/, rl/, post_training/) have no
fork-specific changes and simply stay consistent with upstream. Only core and plugin are
pip-installed.

```
NVIDIA/Megatron-LM (upstream)
├── megatron/core/              ← core library
├── megatron/training/          ← training code (upstream's version)
├── megatron/legacy/            ← legacy code (upstream's version)
├── tests/, examples/, ...      ← everything else
└── pretrain_*.py               ← entry points

Megatron-LM-FL (fork with modifications only in core + plugin)
├── megatron/core/              ← forked core with plugin system (cur_platform, @overridable)
├── megatron/plugin/            ← platform + override mechanisms + new features
├── megatron/training/          ← no fork modifications, consistent with upstream
├── megatron/legacy/            ← no fork modifications, consistent with upstream
├── megatron/rl/                ← no fork modifications, consistent with upstream
├── tests/, examples/, ...      ← consistent with upstream
└── installed via: pip install -e .
    → only installs: megatron.core, megatron.core.*, megatron.plugin, megatron.plugin.*

FlagScale (separate repo, has its own modifications to training + legacy)
└── flagscale/train/megatron/   ← derived from upstream, with FlagScale's own changes
    ├── training/               ← FlagScale's version of megatron/training/ (has modifications)
    ├── legacy/                 ← FlagScale's version of megatron/legacy/ (has modifications)
    └── ...
    → added to PYTHONPATH at runtime by FlagScale's runner
    → pkgutil.extend_path merges pip-installed megatron.core/plugin into namespace
```

**Install scope rule:** Megatron-LM-FL's pyproject.toml must only include `megatron.core` and
`megatron.plugin` packages. Follow upstream's sync target (e.g. core_v0.16.1) for what to
pip-install — if upstream's sync target doesn't install `megatron.training`, neither should we.
Even if upstream `main` adds new packages later, only match the sync target version.

`megatron.training`, `megatron.legacy`, `megatron.rl`, and `megatron.post_training` are resolved
at runtime via PYTHONPATH (set by FlagScale's runner backend). FlagScale's
`flagscale/train/megatron/__init__.py` uses `pkgutil.extend_path` to merge the pip-installed
`megatron.core` and `megatron.plugin` into the same namespace. FlagScale has its own modifications
to training and legacy code — these are the customizations that must be preserved during the sync.

**Lazy import rule:** Modules that may not be on PYTHONPATH (like `megatron.rl`) must use lazy
imports (inside functions) or try/except guards at the top level. Never use unconditional
top-level imports for non-pip-installed packages.

**Three-way merge strategy:** FlagScale's training/legacy code upgrade follows the same pattern
as Megatron-LM-FL's core/plugin upgrade:
- `base` = upstream training/legacy at the old version (the version corresponding to Megatron-LM-FL's base)
- `main` = FlagScale's current training/legacy (with FlagScale's customizations)
- `dev` = upstream training/legacy at the new target version
- `diff = main - base` = FlagScale's customizations only
- Apply diff to dev = new upstream + FlagScale customizations

This correctly handles upstream deletions (like `--config-logger-dir`): if it existed in base
but FlagScale didn't modify it, it's NOT in the diff. When upstream removes it in dev, it stays
removed. The direct comparison approach was incorrect because it couldn't distinguish between
"upstream removed this" vs "FlagScale added this".

## Prerequisites

Before starting this workflow:
- Megatron-LM-FL has already been synced to the target upstream release (e.g. core_v0.16.1)
- Megatron-LM-FL is pip-installed in the current environment
- `import megatron.core` works with the updated library

If these aren't done yet, complete the `mg-fl-upstream-sync` workflow first.

---

## The Multi-Stage Sync Workflow

**Commit discipline:** After every stage that modifies code, run `pre-commit` and then `git commit`.
This creates clean bisect points and ensures code quality gates are enforced incrementally.

---


## Phase Index

This workflow is split into phases. Read each phase file when you reach that point in the workflow.

| Phase | Stages | File | Description |
|-------|--------|------|-------------|
| 1 | Stage 1-2 | `phases/01-setup-and-branch.md` | Environment setup, branch structure for three-way merge |
| 2 | Stage 3-4 | `phases/02-diff-and-apply.md` | Identify FlagScale customizations, apply to dev-train |
| 3 | Stage 5-6 | `phases/03-fix-and-verify.md` | Fix stale references, verification |
| 4 | Stage 7-9 | `phases/04-alignment-and-report.md` | Precision alignment, feature verification, sync report |

---

## Critical Rules

1. **FlagScale customizations are sacred.** FlagScale-specific training and legacy logic, config
   handling, and custom features must survive the upgrade — they are the reason FlagScale
   maintains its own copy of training and legacy code rather than using Megatron-LM-FL's directly.
2. **Branch consistency is mandatory.** FlagScale and Megatron-LM-FL must always be on matching
   branch pairs: `main`+`main` or `dev-train-*`+`dev-*`. A mismatch causes import errors and
   runtime failures that masquerade as real bugs. Always verify both branches before launching
   any training run, especially after switching branches. This applies to Stage 6 verification,
   Stage 7 precision alignment, and any ad-hoc debugging runs.
3. **Megatron-LM-FL must be synced first.** This is the second half of a two-part upgrade.
   The library (megatron.core/plugin) must be updated first via `mg-fl-upstream-sync`. If
   `import megatron.core` or `import megatron.plugin` fails, complete that skill first.
3. **Three-way merge is mandatory.** Always use `diff = main - base` to isolate FlagScale's
   customizations, then apply that diff to dev. Never compare FlagScale directly against the
   target upstream — that approach cannot distinguish upstream deletions from FlagScale additions.
   The base version is the upstream version corresponding to Megatron-LM-FL's base.
4. **Respect upstream deletions.** If code existed in base and FlagScale didn't modify it, and
   upstream removed it in dev, it must stay removed. The `--config-logger-dir` pattern is the
   canonical example: it originated from upstream, not FlagScale, so when upstream removes it,
   FlagScale should too.
5. **Test with real training.** Static import checks catch most issues, but some API changes
   only surface at runtime. A short training run is the definitive validation.
6. **Error triage — fix in the right repo.** When training fails after the upgrade:
   - **megatron.core / megatron.plugin issues** (ImportError, AttributeError on megatron
     symbols, missing core APIs) → fix in **Megatron-LM-FL**, reinstall, then re-test.
   - **Training/legacy issues** (FlagScale config errors, training loop bugs, FlagScale-specific
     logic failures) → fix in **FlagScale** (`flagscale/train/megatron/`).
   This distinction is critical — do not patch megatron.core issues in FlagScale with shims,
   and do not modify Megatron-LM-FL to accommodate FlagScale-specific training or legacy patterns.
7. **Commit incrementally.** Commit after each stage so you can bisect if something breaks later.
8. **ALL files under `flagscale/train/megatron/` must be consistent with upstream.** The sync
   scope is NOT limited to `flagscale/train/megatron/training/` and
   `flagscale/train/megatron/legacy/`. **Every** file in `flagscale/train/megatron/xxx` that has
   an upstream counterpart (e.g. `gpt_builders.py` ↔ upstream `gpt_builders.py`,
   `model_provider.py` ↔ upstream `model_provider.py`, `train_gpt.py` ↔ upstream
   `pretrain_gpt.py`) must also be checked and updated to match upstream, preserving only
   FlagScale-specific additions (wrapped in `######### FlagScale Begin/End ########` markers or
   clearly identifiable as FlagScale features like `get_parallel_context()`, `extra_valid`,
   `spiky_loss`, etc.). When upstream changes function signatures, adds new arguments, modifies
   logic flow, or updates imports in these top-level files, FlagScale's copies MUST be updated
   accordingly. Failing to do so causes runtime errors like `AttributeError: 'Namespace' object
   has no attribute 'xxx'` when FlagScale code uses stale patterns (e.g. `args.moe_grouped_gemm`
   instead of `config.moe_grouped_gemm`).
9. **FlagScale-specific train scripts must keep common infrastructure patterns consistent with
   upstream.** Files like `train_aquila_sft.py`, `train_engram.py`, `train_llava.py`,
   `train_qwen*_vl.py`, `train_robobrain_x0.py`, `train_rwkv.py` etc. have no direct upstream
   counterpart, but they are built on top of Megatron's training infrastructure. Their common
   patterns — ModelOpt import/check (`getattr(args, 'modelopt_enabled', False)` not
   `modelopt_args_enabled(args)`), `loss_func` signature (`model` parameter), `pretrain` import
   (`from megatron.training.training import pretrain`), `inprocess_restart`, and `set_startup_timestamps`
   — must stay aligned with the gold standard (`train_gpt.py`). Model-specific logic (custom
   builders, VL get_batch, multimodal forward) is exempt. Exception:
   `train_robobrain_x0.5_qwengroot.py` uses raw DDP (no Megatron pretrain loop) and is exempt
   from this rule entirely.
10. **Megatron-LM-FL FlagScale patches must survive upstream merges.** When Megatron-LM-FL merges
   a new upstream release, FlagScale-specific patches in `megatron/core/` and `megatron/plugin/`
   can be silently dropped if they conflict with upstream changes or if the merge resolution
   doesn't preserve them. After every upstream merge, audit all files that had FlagScale markers
   (`######### FlagScale Begin/End #########` or `########## FlagScale Add ##########`) in the
   pre-merge branch. Compare the FlagScale marker count between the old branch and the merged
   result. Any reduction indicates a dropped patch that must be re-applied.

   **Known critical patches that must be preserved:**
   - `megatron/core/pipeline_parallel/schedules.py`:
     - `get_forward_backward_func()`: dualpipev dispatch — checks
       `get_dualpipev_pipeline_model_parallel_world_size()` before the standard vp_size check.
     - `forward_backward_pipelining_without_interleaving()`: `p2p_communicator.warm_up_comm_group()`
       call before the warmup forward loop. Without this, hetero training hangs because
       `batch_isend_irecv` requires all ranks in a process group to participate on the first
       collective call. This is the most common cause of silent hangs in hetero PP training.
   - `megatron/core/pipeline_parallel/p2p_communication.py`:
     - `_communicate()`: `group` parameter added to allow hetero PP to pass explicit process groups.
     - Hetero send/recv methods and `warm_up_comm_group()` / `warm_up_comm_group_hetero()`.
   - `megatron/core/distributed/distributed_data_parallel.py`:
     - Engram embedding buffer separation: `engram_dp_group`, `engram_embedding_params` split,
       `engram_embedding_buffers`/`engram_embedding_bucket_groups` allocation, and inclusion in
       all buffer/bucket iteration loops. Without this, engram params land in expert buffers and
       cause `KeyError` in `DistributedOptimizer._build_optimizer_group_ranges`.
   - `megatron/core/optimizer/__init__.py`:
     - `_get_param_groups()`: three-way key `(override, is_expert_parallel, is_engram_parallel)`.
     - `get_megatron_optimizer()`: three mutually exclusive filter_fn calls for dense, MoE, engram
       param groups and buffers.

   **Audit procedure:** `git show main:<file> | grep -c "FlagScale"` vs `grep -c "FlagScale" <file>`
   for every file under `megatron/core/pipeline_parallel/`. Any mismatch means a patch was dropped.

11. **TE-FL version compatibility.** TransformerEngine-FL (TE-FL) may lag behind the upstream
   Megatron-LM API. When upstream adds new parameters to ops that TE-FL implements (e.g.
   `fused_rope_backward` gaining a 9th argument), the TE-FL backend will raise `TypeError: takes
   N positional arguments but N+1 were given`. The fix is either:
   - (Preferred) Update TE-FL to match the new signature.
   - (Workaround) Disable the fusion in the YAML config (e.g. `no_rope_fusion: true` for rope
     fusion). This is acceptable for test YAMLs but should not be the long-term solution for
     production configs.
   These errors only appear during backward pass and only on ranks that execute the affected layer,
   making them easy to misdiagnose as a hang (other ranks wait indefinitely for the crashed rank).

12. **FlagScale model builders must track upstream API changes.** FlagScale-specific models
   (engram, rwkv, llava, etc.) reuse and extend upstream Megatron functions — layer specs,
   transformer configs, optimizer infrastructure. When upstream removes, renames, or changes
   the signature of these functions (e.g. `moe_use_legacy_grouped_gemm` removed from
   `TransformerConfig`, `get_gpt_layer_*_spec()` signature changes), every FlagScale model
   builder that calls those functions must be updated too. The symptom is typically
   `AttributeError: '<Config>' object has no attribute '<removed_field>'` or `TypeError`
   on changed signatures. Audit procedure: after every upstream merge, grep all files under
   `FlagScale/flagscale/models/` and `FlagScale/flagscale/train/megatron/train_*.py` for
   references to the changed symbols.

13. **DDP buffer separation for engram parallel params.** Engram embedding params have
   `allreduce=False` (like expert params) but need their own communication group
   (`engram_dp_group`) and separate grad buffers (`engram_embedding_buffers`). Without the
   DDP patch in `megatron/core/distributed/distributed_data_parallel.py`, engram params land
   in `expert_parallel_buffers`, causing a `KeyError` in `DistributedOptimizer._build_optimizer_group_ranges`
   because the optimizer's param group filter excludes engram params from the MoE optimizer
   but the MoE buffers still contain them. The DDP patch must:
   - Add `engram_dp_group` from `process_group_dict`
   - Split `engram_embedding_params` (where `is_engram_embedding=True`) out of `expert_parallel_params`
   - Allocate separate `engram_embedding_buffers` and `engram_embedding_bucket_groups`
   - Include engram bucket groups/buffers in all iteration loops (grad sync, param sync,
     zero_grad_buffer, scale_gradients, no_sync context manager)
   - Route engram params to `engram_dp_group` in `broadcast_params()`
   The corresponding optimizer patch in `megatron/core/optimizer/__init__.py` must use
   three mutually exclusive filters: dense (`not expert and not engram`), MoE (`expert and
   not engram`), engram (`engram`).
