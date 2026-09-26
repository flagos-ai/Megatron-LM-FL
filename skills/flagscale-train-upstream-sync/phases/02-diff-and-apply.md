# Phase 2: Diff & Apply Customizations

Stages 3-4 of the FlagScale training upstream sync workflow.

---

### Stage 3: Identify FlagScale Customizations (diff base-train..main)

Now that the branch structure is in place, use git's own diff to identify FlagScale's
customizations. The diff `base-train..main` on `flagscale/train/megatron/training/` and
`flagscale/train/megatron/legacy/` captures exactly what FlagScale changed relative to the
old upstream.

#### Step 1: Catalogue FlagScale's changes

```bash
cd "$FLAGSCALE_DIR"

echo "=== FlagScale customizations (base-train..main) ==="
git diff --stat base-train..main -- \
  flagscale/train/megatron/training/ \
  flagscale/train/megatron/legacy/

echo ""
echo "=== File-level classification ==="

# Modified files (FlagScale customized these)
git diff --name-only --diff-filter=M base-train..main -- \
  flagscale/train/megatron/training/ \
  flagscale/train/megatron/legacy/ > /tmp/fs_modified_files.txt
echo "Modified (customized): $(wc -l < /tmp/fs_modified_files.txt)"

# Added files (FlagScale additions, not in upstream)
git diff --name-only --diff-filter=A base-train..main -- \
  flagscale/train/megatron/training/ \
  flagscale/train/megatron/legacy/ > /tmp/fs_added_files.txt
echo "Added (FlagScale-only): $(wc -l < /tmp/fs_added_files.txt)"

# Deleted files (FlagScale removed these from upstream)
git diff --name-only --diff-filter=D base-train..main -- \
  flagscale/train/megatron/training/ \
  flagscale/train/megatron/legacy/ > /tmp/fs_deleted_files.txt
echo "Deleted (FlagScale removed): $(wc -l < /tmp/fs_deleted_files.txt)"
```

#### Step 2: Generate the full customization patch

```bash
cd "$FLAGSCALE_DIR"

# Full patch of FlagScale's customizations
git diff base-train..main -- \
  flagscale/train/megatron/training/ \
  flagscale/train/megatron/legacy/ > /tmp/flagscale_customizations.patch

echo "Full customization patch: $(wc -l < /tmp/flagscale_customizations.patch) lines"

# Also generate per-file patches for conflict resolution
mkdir -p /tmp/flagscale_patches
while IFS= read -r f; do
  patch_name=$(echo "$f" | tr '/' '_').patch
  git diff base-train..main -- "$f" > "/tmp/flagscale_patches/$patch_name"
  echo "PATCH: $f ($(wc -l < /tmp/flagscale_patches/$patch_name) lines)"
done < /tmp/fs_modified_files.txt
```

#### Step 3: Summarize

Save to `FLAGSCALE_CUSTOMIZATIONS.md`:
```markdown
# FlagScale Customizations (diff base-train..main)

## Modified files (FlagScale customized upstream code)
<list with summary of what changed>

## Added files (FlagScale-only, not in upstream)
<list — these are pure FlagScale additions>

## Deleted files (FlagScale removed from upstream)
<list — FlagScale intentionally removed these>
```

The critical distinction this enables:
- `--config-logger-dir` exists in base-train AND in main identically → NOT in the diff →
  dev-train (which removed it) stays as-is → correctly removed
- `cur_platform` replacements exist in main but NOT in base-train → IN the diff → patch
  applied to dev-train → correctly preserved

#### Checkpoint

```bash
cd "$FLAGSCALE_DIR"
git checkout dev-train
git add FLAGSCALE_CUSTOMIZATIONS.md 2>/dev/null || true
git commit -m "Stage 3: catalogue FlagScale customizations" --allow-empty
```

---

### Stage 4: Apply FlagScale Customizations to dev-train

Now apply the customization patches (base-train..main) onto dev-train. This is the same
pattern as Megatron-LM-FL's Stage 3 — apply fork patches onto the new upstream base.

#### Step 1: Ensure we're on dev-train

```bash
cd "$FLAGSCALE_DIR"
git checkout dev-train
echo "Working on dev-train: $(git rev-parse --short HEAD)"
```

#### Step 2: Apply Category A — FlagScale-added files (zero-conflict)

These files don't exist in upstream, so copy them directly from main:

```bash
while IFS= read -r f; do
  mkdir -p "$(dirname "$f")"
  git show main:"$f" > "$f"
  git add "$f"
  echo "Copied: $f"
done < /tmp/fs_added_files.txt

git commit -m "patch(train): add FlagScale-only files" --allow-empty
```

#### Step 3: Apply Category B — Modified files (per-file patches)

For each file FlagScale customized, apply the per-file patch onto dev-train:

```bash
CONFLICT_FILES=""

while IFS= read -r f; do
  patch_name=$(echo "$f" | tr '/' '_').patch
  PATCH_FILE="/tmp/flagscale_patches/$patch_name"

  if [ ! -s "$PATCH_FILE" ]; then continue; fi

  # Try to apply with 3-way merge fallback
  if git apply --3way "$PATCH_FILE" 2>/dev/null; then
    git add "$f"
    echo "OK: $f"
  else
    echo "CONFLICT: $f"
    CONFLICT_FILES="$CONFLICT_FILES $f"
  fi
done < /tmp/fs_modified_files.txt

if [ -n "$CONFLICT_FILES" ]; then
  echo ""
  echo "=== Files needing manual resolution ==="
  echo "$CONFLICT_FILES" | tr ' ' '\n' | grep -v '^$'
fi
```

#### Step 4: Resolve conflicts

For each conflicted file, the patch failed because upstream changed the same area FlagScale
customized. Resolve using the three-way context:

```bash
# For a specific conflicted file:
f="flagscale/train/megatron/training/<filename>.py"

# View all three versions:
echo "=== Base (old upstream) ==="
git show base-train:"$f" | head -50
echo "=== Main (FlagScale current) ==="
git show main:"$f" | head -50
echo "=== Dev (new upstream, current working copy) ==="
head -50 "$f"
```

| Conflict type | Resolution strategy |
|--------------|---------------------|
| FlagScale added cur_platform, upstream modified same function | Start from dev-train version, re-apply cur_platform replacements |
| FlagScale added custom logic, upstream refactored | Start from dev-train version, re-integrate FlagScale's custom logic |
| Upstream removed code FlagScale customized | If FlagScale's customization is still needed, keep it; otherwise follow upstream's removal |
| FlagScale modified API call, upstream changed API signature | Update FlagScale's call to match new API |

**For subagent-assisted conflict resolution**, use this prompt:

```
You are resolving a three-way merge conflict for FlagScale's training code upgrade.

## Context — use git show to read each version:
- Base (old upstream): git show base-train:"$f"
- Main (FlagScale current): git show main:"$f"
- Dev (new upstream, target): the current working copy of $f
- Output: write the resolved version to $f

## Strategy
1. Start from dev-train (new upstream) as the base
2. Identify FlagScale's customizations by diffing main vs base-train
3. Re-apply those customizations to dev-train
4. Preserve ALL FlagScale-specific patterns (see list below)

## Important Rules
- Upstream deletions should be respected (if code existed in base but not in dev, and
  FlagScale didn't customize it, don't add it back)
- FlagScale additions should be preserved (if code exists in main but not in base, keep it)
- cur_platform replacements must survive
- FlagScale-specific imports must survive
```

After resolving each file:
```bash
git add "$f"
```

#### Step 5: Commit and verify

```bash
cd "$FLAGSCALE_DIR"
git commit -m "patch(train): apply FlagScale customizations to $TARGET_VERSION" --allow-empty

# Verify no conflict markers remain
grep -rn "<<<<<<<\|=======\|>>>>>>>" flagscale/train/megatron/ --include="*.py" \
  && echo "CONFLICT MARKERS FOUND" || echo "OK: clean"

# Syntax check all modified Python files
find flagscale/train/megatron/ -name "*.py" ! -path "*__pycache__*" | while read f; do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```

#### Checkpoint: pre-commit & commit

```bash
cd "$FLAGSCALE_DIR"
pre-commit run --all-files
git add -A
git commit -m "Stage 4: apply FlagScale customizations to $TARGET_VERSION upstream"
```

#### Comprehensive FlagScale Customization Patterns

These are the patterns that identify FlagScale-specific code. All must be preserved during merges:

**Platform abstraction (cur_platform):**
- `from megatron.plugin.platform import get_platform` / `cur_platform = get_platform()`
- `cur_platform.device()` replaces `torch.cuda.current_device()`
- `cur_platform.device_name()` replaces `'cuda'` string literals
- `cur_platform.synchronize()` replaces `torch.cuda.synchronize()`
- `cur_platform.empty_cache()` replaces `torch.cuda.empty_cache()`
- `cur_platform.memory_allocated()` / `max_memory_allocated()` / `memory_reserved()` / `max_memory_reserved()`
- `cur_platform.device_memory_used()` replaces `torch.cuda.device_memory_used()`
- `cur_platform.get_device_properties()` replaces `torch.cuda.get_device_properties()`
- `cur_platform.get_device_capability()` replaces `torch.cuda.get_device_capability()`
- `cur_platform.device_count()` replaces `torch.cuda.device_count()`
- `cur_platform.set_device()` replaces `torch.cuda.set_device()`
- `cur_platform.Stream()` / `cur_platform.stream()` replaces `torch.cuda.Stream()` / `torch.cuda.stream()`
- `cur_platform.Event()` replaces `torch.cuda.Event()`
- `cur_platform.get_rng_state()` / `cur_platform.set_rng_state()` replaces `torch.cuda.get_rng_state/set_rng_state`
- `cur_platform.range_push()` / `cur_platform.range_pop()` replaces `nvtx.range_push/pop`
- `cur_platform.FloatTensor` / `cur_platform.HalfTensor` / `cur_platform.BFloat16Tensor` / `cur_platform.LongTensor`
- `cur_platform.current_device()` replaces `torch.cuda.current_device()`
- `cur_platform.manual_seed()` replaces `torch.cuda.manual_seed()`
- `.to(device=cur_platform.device(), non_blocking=True)` replaces `.cuda(non_blocking=True)`

**torch.cuda APIs that should NOT be replaced** (new PyTorch APIs with no cur_platform equivalent):
- `torch.cuda.use_mem_pool` — UVM memory pool management
- `torch.cuda.cudart()` — CUDA runtime profiler start/stop (use `cur_platform.cudart()` if available)

**Heterogeneous communication:**
- `from megatron.plugin.utils import get_device_type_for_comm`
- `get_device_type_for_comm(group)` for determining communication device (CPU fallback for gloo)
- `model_parallel_groups` handled as lists (not single group)

**FlagScale marker blocks:**
- `######### FlagScale Begin ########` / `######### FlagScale End ########` — new code blocks
- `######### FlagScale Modify ########` — modified upstream code

**DualPipeV pipeline parallelism:**
- `mpu.get_dualpipev_pipeline_model_parallel_world_size()`
- Modified `is_last_rank()` with dualpipev awareness
- Extra broadcasts in `get_batch_on_this_tp_rank()` for first pipeline stage

**Engram embedding:**
- `engram_embedding_data` list, `engram_mp_group` reduction
- `getattr(param, "is_engram_embedding", False)` checks

**Spiky loss detection:**
- `get_spiky_loss_detector()` / `set_spiky_loss_detector()`
- Spiky loss detection block in training loop

**Extra valid datasets:**
- `get_extra_valid_datasets()` / `set_extra_valid_datasets()`
- `extra_valid_dataset_provider` parameter in `pretrain()`
- Extra validation evaluation block in training loop

**Custom tokenizers:**
- AquilaTokenizerFS, HFTokenizerFS, and other FlagScale-specific tokenizer types

**Training-specific:**
- `get_megatron_optimizer_config()` in utils.py (FlagScale keeps this locally)
- `set_parallel_context()`, `FSTrainArguments` in initialize.py
- `set_global_writers()` in global_vars.py
- `decoupled_learning_rate` logging in training.py
- `auto_tune` guard in training_log()
- `fs_report_theoretical_memory` call

**Retro support:**
- FlagScale keeps retro code that upstream removed
- Retro imports should be guarded with try/except (see Stage 5)

#### Category 4: FlagScale-only files

These are FlagScale's custom additions with no upstream counterpart. Keep them as-is,
but check that they don't reference stale megatron.core APIs (handled in Stage 5).

#### Checkpoint: pre-commit & commit

```bash
cd "$FLAGSCALE_DIR"
pre-commit run --all-files
git add -A
git commit -m "upgrade(train): apply upstream training code changes for $TARGET_VERSION"
```

Fix any pre-commit failures before committing. This creates a clean bisect point before stale reference scanning.

---

