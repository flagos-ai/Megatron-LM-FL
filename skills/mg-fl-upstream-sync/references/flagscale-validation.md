# FlagScale Integration Validation Reference

Detailed procedures for Stage 7 of the Megatron-LM-FL upstream sync workflow.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Pre-Validation Setup](#pre-validation-setup)
- [Import Compatibility Check](#import-compatibility-check)
- [Training Validation](#training-validation)
- [Diagnosis Checklist](#diagnosis-checklist)

---

## Architecture Overview

Megatron-LM-FL and FlagScale are decoupled:

```
Megatron-LM-FL (this repo)
├── megatron/core/          ← library code, pip-installed
├── megatron/plugin/        ← fork-specific plugin system
└── setup.py / pyproject.toml

FlagScale (separate repo)
└── flagscale/train/megatron/  ← training code, imports megatron.core
```

FlagScale consumes Megatron-LM-FL as a library:
- `pip install megatron-lm-fl` (or editable install from source)
- `import megatron.core` in training scripts
- No direct file-level coupling — only Python import-level dependency

This means the sync can break FlagScale in two ways:
1. **Import errors** — symbols renamed/removed in `megatron.core`
2. **API changes** — function signatures changed, new required parameters, changed return types

---

## Pre-Validation Setup

### Step 1: Locate FlagScale

```bash
MG_FL_DIR=$(pwd)  # Should be in Megatron-LM-FL
cd ..

# Find FlagScale directory
if [ -d "FlagScale" ]; then
  FLAGSCALE_DIR=$(pwd)/FlagScale
elif [ -d "flagscale" ]; then
  FLAGSCALE_DIR=$(pwd)/flagscale
else
  echo "ERROR: FlagScale directory not found as sibling of Megatron-LM-FL"
  echo "Please clone FlagScale or provide the correct path."
  exit 1
fi
echo "FlagScale directory: $FLAGSCALE_DIR"
echo "Megatron-LM-FL directory: $MG_FL_DIR"
```

### Step 2: Ensure Megatron-LM-FL is installed

```bash
cd "$MG_FL_DIR"
pip install -e . --no-build-isolation

# Verify installation
python3 -c "import megatron.core; print('megatron.core imported successfully')"
python3 -c "from megatron.plugin.platform import cur_platform; print('cur_platform:', cur_platform)"
```

---

## Import Compatibility Check

Before running training, do a static check of FlagScale's imports against the updated
megatron.core API surface.

### Step 1: Extract all megatron imports from FlagScale

```bash
# Find all import statements referencing megatron in FlagScale training code
grep -rn "from megatron" "$FLAGSCALE_DIR/flagscale/train/megatron/" --include="*.py" | \
  grep -v "__pycache__" | \
  sort -u > /tmp/flagscale_megatron_imports.txt

grep -rn "import megatron" "$FLAGSCALE_DIR/flagscale/train/megatron/" --include="*.py" | \
  grep -v "__pycache__" | \
  sort -u >> /tmp/flagscale_megatron_imports.txt

echo "FlagScale megatron imports:"
cat /tmp/flagscale_megatron_imports.txt
```

### Step 2: Verify each import resolves

```bash
# Extract unique import paths and test each one
grep -oP 'from \K[a-zA-Z0-9_.]+' /tmp/flagscale_megatron_imports.txt | sort -u | while read module; do
  python3 -c "import $module" 2>&1 && echo "OK: $module" || echo "FAIL: $module"
done
```

### Step 3: Check for renamed/removed symbols

If any imports fail, cross-reference with the upstream diff to find what changed:

```bash
# For each failed import, check if the symbol was renamed
cd "$MG_FL_DIR"
for symbol in <failed_symbols>; do
  echo "=== Searching for '$symbol' ==="
  # Check if it exists in the current codebase
  grep -rn "def $symbol\|class $symbol" megatron/ --include="*.py" | grep -v "__pycache__"
  # Check what happened to it between base and dev
  git diff base..dev -- '*.py' | grep -B5 -A5 "$symbol"
done
```

---

## Training Validation

### Step 1: Collect parameters from the user

Before running, ask the user for:

1. **Path to training config** — the FlagScale training YAML config file
   Example: `flagscale/examples/aquila/conf/train.yaml`

2. **Number of validation steps** — how many training steps to run (default: 20)
   Enough to verify the training loop works, not for convergence.

### Step 2: Run training

```bash
cd "$FLAGSCALE_DIR"

# Run training with the user's config
python run.py \
  --config-path=<user-provided-config-path> \
  --config-name=train \
  action=run \
  2>&1 | tee /tmp/flagscale_training.log
```

### Step 3: Verify training output

Check that training ran successfully:

```bash
# Look for loss values in the log (indicates training loop is working)
grep -E "iteration\s+[0-9]+.*lm loss" /tmp/flagscale_training.log | tail -5

# Count completed steps
STEPS=$(grep -cE "iteration\s+[0-9]+.*lm loss" /tmp/flagscale_training.log)
echo "Completed steps: $STEPS"

# Check for errors
grep -iE "error|traceback|exception" /tmp/flagscale_training.log | head -10
```

**Success criteria:**
- Training completes the requested number of steps
- Loss values are present and decreasing (model is learning)
- No import errors, attribute errors, or type errors

### Step 4: Stop training

```bash
python run.py \
  --config-path=<user-provided-config-path> \
  --config-name=train \
  action=stop
```

---

## Diagnosis Checklist

When FlagScale training fails after a sync, categorize the error:

### 1. ImportError / ModuleNotFoundError

**Cause:** A module or symbol was renamed/moved/removed in the upstream merge.

**Fix:**
```bash
# Find where the symbol moved to
cd "$MG_FL_DIR"
git diff base..dev -- '*.py' | grep -B5 -A5 "<missing_symbol>"
# Or search the current codebase
grep -rn "<missing_symbol>" megatron/ --include="*.py" | grep -v "__pycache__"
```

Update the import in FlagScale, or add a compatibility shim in Megatron-LM-FL.

### 2. TypeError / Missing Positional Argument

**Cause:** A function signature changed in the upstream merge.

**Fix:**
```bash
# Compare the function signature between base and current
git show base:<file> | grep "def <function_name>"
grep "def <function_name>" <current_file>
```

Update the caller in FlagScale to match the new signature, or add a compatibility
wrapper in Megatron-LM-FL.

### 3. AttributeError

**Cause:** A class attribute or method was renamed/removed.

**Fix:** Same approach as ImportError — find the new name and update references.

### 4. RuntimeError: Device Mismatch

**Cause:** A new `torch.cuda` call was introduced by upstream and not converted to
`cur_platform`. This causes failures on non-NVIDIA hardware.

**Fix:** Go back to Stage 4 (Plugin Integrity Verification) and patch the missed
`torch.cuda` call.

### 5. Plugin Dispatch Errors

**Cause:** The plugin system's `@overridable`/`@override` chain is broken — either
a decorator was lost during the merge, or a signature mismatch between the overridable
and override functions.

**Fix:** Go back to Stage 4 and verify the decorator chain. Check that all `@override`
functions match their `@overridable` counterparts' signatures.

### General Fix Workflow

For each fix:
1. Identify the root cause from the traceback
2. Fix the code in Megatron-LM-FL (preferred) or FlagScale
3. Syntax check: `python3 -c "import ast; ast.parse(open('<file>').read())"`
4. Reinstall: `cd "$MG_FL_DIR" && pip install -e . --no-build-isolation`
5. Re-run training to verify the fix
6. Commit the fix with a descriptive message
