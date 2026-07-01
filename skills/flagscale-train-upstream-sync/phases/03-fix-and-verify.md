# Phase 3: Fix & Verify

Stages 5-6 of the FlagScale training upstream sync workflow.

---

### Stage 5: Fix Stale References

After upgrading, scan all FlagScale training code for references to megatron.core and
megatron.plugin symbols that may have been renamed/removed in the target upstream version.

**Important:** Only `megatron.core` and `megatron.plugin` are pip-installed from Megatron-LM-FL.
Other packages like `megatron.training`, `megatron.legacy`, `megatron.rl`, and
`megatron.post_training` are NOT pip-installed — they resolve at runtime via PYTHONPATH or
FlagScale's own path setup. Import checks here should only verify `megatron.core.*` and
`megatron.plugin.*` modules.

```bash
cd "$FLAGSCALE_DIR"

# Extract megatron.core and megatron.plugin imports from FlagScale training code
# (skip megatron.training/legacy/rl/post_training — those are not pip-installed)
grep -rhn "from megatron\.core\|from megatron\.plugin" \
  flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | \
  sed 's/.*from \(megatron\.[^ ]*\).*/\1/' | sed 's/ import.*//' | sort -u > /tmp/flagscale_imports.txt

# Test each import in a single Python process for speed
python3 -c "
import importlib
with open('/tmp/flagscale_imports.txt') as f:
    modules = [l.strip() for l in f if l.strip()]
failed = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append((mod, type(e).__name__, str(e)))
if failed:
    print('FAILED imports:')
    for mod, etype, msg in failed:
        print(f'  FAIL: {mod} -> {etype}: {msg}')
else:
    print(f'All {len(modules)} megatron.core/plugin imports: OK')
"
```

For each failed import:
1. Search the updated megatron.core for the symbol's new location
2. Update the import in FlagScale
3. Update all usages of the symbol

```bash
# Find where a missing symbol moved to
SYMBOL="<missing_symbol>"
grep -rn "def $SYMBOL\|class $SYMBOL" \
  $(python3 -c "import megatron.core; import os; print(os.path.dirname(megatron.core.__file__))") \
  --include="*.py"
```

#### Guard removed upstream modules with try/except

When upstream removes an entire module that FlagScale still uses (e.g., upstream removed
`megatron.core.models.retro` but FlagScale's legacy code still references it), guard the
import with try/except rather than removing the FlagScale code:

```python
# Before (breaks if upstream removed the module):
from megatron.core.models.retro.utils import (
    get_config_path as get_retro_config_path,
    get_gpt_data_dir as get_retro_data_dir,
)

# After (graceful degradation):
try:
    from megatron.core.models.retro.utils import (
        get_config_path as get_retro_config_path,
        get_gpt_data_dir as get_retro_data_dir,
    )
except ImportError:
    get_retro_config_path = None
    get_retro_data_dir = None
```

This pattern is appropriate when:
- The module was removed upstream but FlagScale has code paths that use it
- The code paths are conditional (only activated when certain args are set)
- Removing the FlagScale code entirely would break backward compatibility

#### Scan ALL training code for CUDA-specific references (multi-platform compliance)

After all merges, scan the entire `flagscale/train/megatron/` tree — including `training/` and
`legacy/` subdirectories — for three categories of CUDA-specific code that must be replaced
with `cur_platform` equivalents to support multiple platforms (MetaX, Hygon, etc.):

1. **`torch.cuda.*` API calls** — e.g., `torch.cuda.current_device()`, `torch.cuda.synchronize()`
2. **`.cuda()` tensor moves** — e.g., `data.cuda(non_blocking=True)` → `.to(device=cur_platform.current_device(), non_blocking=True)`
3. **Hardcoded `device='cuda'`** — e.g., `torch.tensor(..., device='cuda')` → `torch.tensor(..., device=cur_platform.device_name())`

```bash
cd "$FLAGSCALE_DIR"

echo "=== torch.cuda.* API calls ==="
grep -rn "torch\.cuda" flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform" | grep -v "#.*torch\.cuda" | grep -v '"""' | \
  grep -v "torch\.cuda\.use_mem_pool" | grep -v "torch\.cuda\.cudart" | \
  grep -v "torch\.cuda\.memory\._"

echo "=== .cuda() tensor moves ==="
grep -rn '\.cuda(' flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform" | grep -v "#" | grep -v '"""' | \
  grep -v "torch\.cuda"

echo "=== Hardcoded device='cuda' ==="
grep -rn "device=['\"]cuda['\"]" flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform"
```

For each file with violations:
1. Ensure `cur_platform` is imported (add if missing):
   ```python
   from megatron.plugin.platform import get_platform
   cur_platform = get_platform()
   ```
2. Apply replacements:
   - `torch.cuda.current_device()` → `cur_platform.current_device()`
   - `.cuda(non_blocking=True)` → `.to(device=cur_platform.current_device(), non_blocking=True)`
   - `device='cuda'` → `device=cur_platform.device_name()`
   - See the full replacement table in Stage 4's "Comprehensive FlagScale Customization Patterns"
3. Syntax check each modified file

**Legitimate exceptions** (do NOT replace):
- `torch.cuda.memory._record_memory_history` / `_snapshot` — CUDA debug profiling APIs
- `torch.cuda.use_mem_pool` — UVM memory pool management
- `torch.cuda.cudart()` — CUDA runtime profiler
- References in comments and docstrings

#### Checkpoint: pre-commit & commit

If any stale references were fixed, commit them:

```bash
cd "$FLAGSCALE_DIR"
pre-commit run --all-files
git add -A
git commit -m "fix(train): update stale megatron.core references for $TARGET_VERSION"
```

Skip if no stale references were found.

---

### Stage 6: Verification

#### Step 0: Branch consistency gate

Before any verification step, confirm that both repos are on matching branch pairs. A mismatch
(e.g. FlagScale on `dev-train-*` but Megatron-LM-FL still on `main`) causes misleading import
errors and runtime failures that look like real bugs but are just branch misalignment.

The two valid pairs are:
| FlagScale branch | Megatron-LM-FL branch |
|------------------|-----------------------|
| `main` | `main` |
| `$FS_DEV_BRANCH` | `$MG_DEV_BRANCH` |

```bash
FS_BRANCH=$(cd "$FLAGSCALE_DIR" && git branch --show-current)
MG_BRANCH=$(cd "$MG_FL_DIR" && git branch --show-current)
echo "FlagScale: $FS_BRANCH"
echo "Megatron-LM-FL: $MG_BRANCH"

# Validate pairing
MISMATCH=false
if [[ "$FS_BRANCH" == "main" && "$MG_BRANCH" != "main" ]]; then
  MISMATCH=true
elif [[ "$FS_BRANCH" == dev-train-* && "$MG_BRANCH" != dev-* ]]; then
  MISMATCH=true
elif [[ "$MG_BRANCH" == "main" && "$FS_BRANCH" != "main" ]]; then
  MISMATCH=true
elif [[ "$MG_BRANCH" == dev-* && "$FS_BRANCH" != dev-train-* ]]; then
  MISMATCH=true
fi

if $MISMATCH; then
  echo "ERROR: Branch mismatch! FlagScale=$FS_BRANCH, Megatron-LM-FL=$MG_BRANCH"
  echo "Both repos must be on matching branch pairs. Fix before proceeding."
  exit 1
else
  echo "OK: Branches are consistent ($FS_BRANCH + $MG_BRANCH)"
fi
```

If mismatched, switch the lagging repo to the correct branch and reinstall if needed:
```bash
cd "$MG_FL_DIR" && git checkout <correct_branch> && pip install -e .
```

Do NOT proceed with any training or import checks until this gate passes.

#### Step 1: Syntax check all modified files

```bash
cd "$FLAGSCALE_DIR"
find flagscale/train/megatron/ -name "*.py" ! -path "*__pycache__*" | while read f; do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```

#### Step 2: Import verification

Verify that all `megatron.core` and `megatron.plugin` imports used by FlagScale training code
resolve correctly. Only these two packages are pip-installed from Megatron-LM-FL. Other packages
(`megatron.training`, `megatron.legacy`, etc.) resolve at runtime via FlagScale's path setup
and should NOT be checked here.

```bash
cd "$FLAGSCALE_DIR"

# Extract megatron.core and megatron.plugin imports only
grep -rhn "from megatron\.core\|from megatron\.plugin" \
  flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | \
  sed 's/.*from \(megatron\.[^ ]*\).*/\1/' | sed 's/ import.*//' | sort -u > /tmp/flagscale_imports_verify.txt

# Test all in a single Python process
python3 -c "
import importlib
with open('/tmp/flagscale_imports_verify.txt') as f:
    modules = [l.strip() for l in f if l.strip()]
failed = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append((mod, type(e).__name__, str(e)))
if failed:
    print('FAILED imports:')
    for mod, etype, msg in failed:
        print(f'  FAIL: {mod} -> {etype}: {msg}')
else:
    print(f'All {len(modules)} megatron.core/plugin imports: OK')
"
```

#### Step 2b: Multi-platform compliance check

Verify no CUDA-specific code remains in `flagscale/train/megatron/` (including `training/` and
`legacy/` subdirectories). This catches `.cuda()` tensor moves, hardcoded `device='cuda'`, and
`torch.cuda.*` API calls that should use `cur_platform`.

```bash
cd "$FLAGSCALE_DIR"

echo "=== torch.cuda.* API calls ==="
grep -rn "torch\.cuda" flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform" | grep -v "#.*torch\.cuda" | grep -v '"""' | \
  grep -v "torch\.cuda\.use_mem_pool" | grep -v "torch\.cuda\.cudart" | \
  grep -v "torch\.cuda\.memory\._"

echo "=== .cuda() tensor moves ==="
grep -rn '\.cuda(' flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform" | grep -v "#" | grep -v '"""' | \
  grep -v "torch\.cuda"

echo "=== Hardcoded device='cuda' ==="
grep -rn "device=['\"]cuda['\"]" flagscale/train/megatron/ --include="*.py" | \
  grep -v "__pycache__" | grep -v "cur_platform"
```

All three commands should produce empty output. Any hits are bugs unless they match the
legitimate exceptions listed in Stage 5.

#### Step 3: End-to-end training validation

Launch a real training run to verify end-to-end functionality. This is the definitive validation
— static checks catch most issues, but some API changes only surface at runtime.

Ask the user for their training config (supports user-provided YAML file paths):

```
To validate, I need:
1. Path to your training YAML config file
   (e.g., flagscale/examples/aquila/conf/train.yaml, or any custom YAML path)
2. Number of steps to run (default: 20 — just enough to verify the loop works)
```

```bash
cd "$FLAGSCALE_DIR"

# USER_CONFIG_PATH: the user-provided YAML config path (absolute or relative to FlagScale root)
# Examples:
#   flagscale/examples/aquila/conf/train.yaml
#   /home/user/my_custom_config/train.yaml
#   flagscale/examples/llama/conf/train.yaml

python run.py \
  --config-path=$(dirname $USER_CONFIG_PATH) \
  --config-name=$(basename $USER_CONFIG_PATH .yaml) \
  action=run \
  2>&1 | tee /tmp/flagscale_training_test.log
```

Check results:
```bash
# Verify training ran
grep -E "iteration\s+[0-9]+.*lm loss" /tmp/flagscale_training_test.log | tail -5

# Check for errors
grep -iE "error|traceback|exception" /tmp/flagscale_training_test.log | \
  grep -v "No errors" | head -10
```

#### Step 4: Fix errors promptly using error triage

If training fails, diagnose and fix immediately — do not defer to a later stage:

1. **Read the traceback** and identify the root cause
2. **Triage the error to the correct repo:**
   - `ImportError` / `ModuleNotFoundError` / `AttributeError` on `megatron.*` symbols
     → The symbol was renamed/moved/removed in the upstream merge.
     → **Fix in Megatron-LM-FL**, then `pip install -e .` and re-run training.
   - `TypeError` (wrong arguments to `megatron.core` functions)
     → Function signature changed upstream.
     → **Fix in Megatron-LM-FL** (update the function or add compatibility), reinstall, re-test.
   - Training loop errors, config errors, FlagScale-specific logic failures
     → **Fix in FlagScale** (`flagscale/train/megatron/`).
3. **Re-run training** after each fix to verify
4. **Repeat** until training completes successfully

**Success criteria:** Training completes requested steps, loss values present, no import/API errors.

#### Checkpoint: pre-commit & commit

After all errors are fixed and training passes, commit the fixes:

```bash
cd "$FLAGSCALE_DIR"
pre-commit run --all-files
git add -A
git commit -m "fix(train): resolve training errors after upstream upgrade to $TARGET_VERSION"
```

If fixes were also made in Megatron-LM-FL during error triage, commit those separately:

```bash
cd "$MG_FL_DIR"
pre-commit run --all-files
git add -A
git commit -m "fix(core): resolve FlagScale training compatibility issues for $TARGET_VERSION"
```

Skip if training passed on the first attempt with no fixes needed.

---

