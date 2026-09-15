# Plugin Verification Reference

Detailed procedures for Stage 4 of the Megatron-LM-FL upstream sync workflow.

## Table of Contents
- [Platform Mechanism Verification](#platform-mechanism-verification)
- [Override Mechanism Verification](#override-mechanism-verification)
- [New Feature Integrity](#new-feature-integrity)
- [New torch.cuda Calls from Upstream](#new-torchcuda-calls-from-upstream)
- [Decorator Signature Compatibility](#decorator-signature-compatibility)

---

## Platform Mechanism Verification

The platform mechanism replaces `torch.cuda` API calls with `cur_platform` equivalents,
enabling multi-chip support. After the merge, upstream may have introduced new `torch.cuda`
calls that bypass this mechanism.

### Step 1: Scan for unpatched torch.cuda calls

```bash
# Find raw torch.cuda calls in megatron/core/ that should use cur_platform
grep -rn "torch\.cuda" megatron/core/ --include="*.py" | \
  grep -v "__pycache__" | \
  grep -v "cur_platform" | \
  grep -v "#.*torch\.cuda" | \
  grep -v '"""' | \
  grep -v "'torch\.cuda'" > /tmp/unpatched_cuda_calls.txt

if [ -s /tmp/unpatched_cuda_calls.txt ]; then
  echo "WARNING: Found unpatched torch.cuda calls:"
  cat /tmp/unpatched_cuda_calls.txt
else
  echo "OK: No unpatched torch.cuda calls found"
fi
```

### Step 2: Identify which calls are new from upstream

```bash
# Cross-reference with the upstream diff to find only NEW torch.cuda calls
git diff base..dev -- 'megatron/core/' | \
  grep '^+' | grep -v '^+++' | \
  grep 'torch\.cuda' | \
  grep -v 'cur_platform' > /tmp/new_cuda_calls_from_upstream.txt

echo "New torch.cuda calls introduced by upstream:"
cat /tmp/new_cuda_calls_from_upstream.txt
```

### Step 3: Apply cur_platform replacements

Common replacement patterns:

| Original (upstream) | Replacement (fork) |
|---|---|
| `torch.cuda.current_device()` | `cur_platform.current_device()` |
| `torch.cuda.device_count()` | `cur_platform.device_count()` |
| `torch.cuda.set_device(x)` | `cur_platform.set_device(x)` |
| `torch.cuda.synchronize()` | `cur_platform.synchronize()` |
| `torch.cuda.Stream()` | `cur_platform.Stream()` |
| `torch.cuda.Event()` | `cur_platform.Event()` |
| `torch.cuda.is_available()` | `cur_platform.is_available()` |
| `torch.cuda.get_device_properties(x)` | `cur_platform.get_device_properties(x)` |
| `torch.cuda.memory_allocated()` | `cur_platform.memory_allocated()` |
| `torch.cuda.max_memory_allocated()` | `cur_platform.max_memory_allocated()` |
| `torch.cuda.reset_peak_memory_stats()` | `cur_platform.reset_peak_memory_stats()` |
| `torch.cuda.empty_cache()` | `cur_platform.empty_cache()` |
| `device="cuda"` | `device=cur_platform.device_type` |
| `torch.device("cuda")` | `torch.device(cur_platform.device_type)` |
| `.to("cuda")` | `.to(cur_platform.device_type)` |

For each file with new `torch.cuda` calls:
1. Add the import if not present: `from megatron.plugin.platform import cur_platform`
2. Replace each `torch.cuda` call with the `cur_platform` equivalent
3. Syntax check: `python3 -c "import ast; ast.parse(open('<file>').read())"`

**What to leave as-is:**
- `torch.cuda` in comments and docstrings
- `torch.cuda.amp` (autocast) — check if cur_platform has an equivalent first
- `torch.cuda.nvtx` — profiling, may not have a cur_platform equivalent
- `torch.version.cuda` — build-time version query
- `torch.cuda` in CUDA-specific guard blocks (e.g., `if torch.cuda.is_available():`)
- `torch.cuda.use_mem_pool` — new PyTorch API for UVM memory pools, no cur_platform equivalent
- `torch.cuda.cudart()` — low-level CUDA runtime API, may not have cur_platform equivalent
- `torch.cuda.nccl` — NCCL-specific operations
---

## Override Mechanism Verification

The override mechanism uses `@overridable` decorators on functions in `megatron/core/` and
corresponding `@override` implementations in `megatron/plugin/`. After the merge, verify
the decorator chain is intact.

### Step 1: Count and list all @overridable decorators

```bash
echo "=== @overridable decorators in megatron/core/ ==="
grep -rn "@overridable" megatron/core/ --include="*.py" | grep -v "__pycache__"
OVERRIDABLE_COUNT=$(grep -rn "@overridable" megatron/core/ --include="*.py" | grep -v "__pycache__" | wc -l)
echo "Total: $OVERRIDABLE_COUNT"
```

### Step 2: List all @override implementations

```bash
echo "=== @override implementations in megatron/plugin/ ==="
grep -rn "@override" megatron/plugin/ --include="*.py" | grep -v "__pycache__"
OVERRIDE_COUNT=$(grep -rn "@override" megatron/plugin/ --include="*.py" | grep -v "__pycache__" | wc -l)
echo "Total: $OVERRIDE_COUNT"
```

### Step 3: Cross-reference overridable ↔ override pairs

Extract function names from both sides and verify they match:

```bash
# Extract overridable function names
grep -rn "@overridable" megatron/core/ --include="*.py" -A1 | \
  grep "def " | sed 's/.*def \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/' | sort -u > /tmp/overridable_funcs.txt

# Extract override function names
grep -rn "@override" megatron/plugin/ --include="*.py" -A1 | \
  grep "def " | sed 's/.*def \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/' | sort -u > /tmp/override_funcs.txt

# Find overridable functions without a corresponding override
echo "=== @overridable without @override (may be intentional) ==="
comm -23 /tmp/overridable_funcs.txt /tmp/override_funcs.txt

# Find override functions without a corresponding overridable
echo "=== @override without @overridable (potential bug) ==="
comm -13 /tmp/overridable_funcs.txt /tmp/override_funcs.txt
```

### Step 4: Verify signature compatibility

If upstream changed the signature of an `@overridable` function, the `@override`
implementation must be updated to match:

```bash
# For each overridable function, compare signatures between base and current
while IFS= read -r func_name; do
  echo "--- $func_name ---"
  # Current signature
  grep -rn "def $func_name" megatron/core/ --include="*.py" | grep -v "__pycache__" | head -1
  # Override signature
  grep -rn "def $func_name" megatron/plugin/ --include="*.py" | grep -v "__pycache__" | head -1
  echo ""
done < /tmp/overridable_funcs.txt
```

If signatures diverge, update the `@override` implementation to match the new `@overridable`
signature. The override must accept the same parameters (it can ignore some internally, but
the signature must be compatible).

---

## New Feature Integrity

Verify fork-specific features (dualpipev, hetero, etc.) are intact after the merge.

```bash
# Syntax check all plugin Python files
find megatron/plugin/ -name "*.py" ! -path "*__pycache__*" | while read f; do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done

# Verify plugin directory structure is intact
echo "=== Plugin directory structure ==="
find megatron/plugin/ -type f -name "*.py" ! -path "*__pycache__*" | sort

# Verify __init__.py files exist for all plugin subpackages
find megatron/plugin/ -type d ! -path "*__pycache__*" | while read d; do
  if [ ! -f "$d/__init__.py" ]; then
    echo "WARNING: Missing __init__.py in $d"
  fi
done
```

---

## New torch.cuda Calls from Upstream

When upstream introduces new Python files or significantly modifies existing ones, they may
contain `torch.cuda` calls that the fork needs to convert. This is a superset of the
platform mechanism check — it covers ALL Python files, not just `megatron/core/`.

```bash
# Comprehensive scan: all new torch.cuda calls introduced by the merge
git diff base..dev -- '*.py' | \
  grep '^+' | grep -v '^+++' | \
  grep -E 'torch\.cuda\.|device.*=.*"cuda"|torch\.device\("cuda"\)' | \
  grep -v '#' | grep -v '"""' > /tmp/all_new_cuda_refs.txt

echo "All new torch.cuda references from upstream:"
wc -l /tmp/all_new_cuda_refs.txt
cat /tmp/all_new_cuda_refs.txt
```

Triage each reference:
- In `megatron/core/` or `megatron/plugin/` → must convert to `cur_platform`
- In test files → convert if tests should be multi-platform, leave if CUDA-specific
- In example scripts → convert if they demonstrate plugin usage, leave otherwise
- In documentation → leave as-is

---

## Decorator Signature Compatibility

When upstream modifies a function that has `@overridable`, the function body changes but the
decorator must be preserved. Additionally, if the function signature changes (new parameters,
removed parameters, changed defaults), the `@override` implementation must be updated.

### Detection

```bash
# Find @overridable functions whose signatures changed between base and dev
while IFS= read -r func_name; do
  BASE_SIG=$(git show base:$(grep -rl "@overridable" megatron/core/ --include="*.py" | \
    xargs grep -l "def $func_name" | head -1) 2>/dev/null | \
    grep -A1 "@overridable" | grep "def $func_name" | head -1)
  DEV_SIG=$(git show dev:$(grep -rl "def $func_name" megatron/core/ --include="*.py" | head -1) 2>/dev/null | \
    grep "def $func_name" | head -1)
  if [ "$BASE_SIG" != "$DEV_SIG" ] && [ -n "$BASE_SIG" ] && [ -n "$DEV_SIG" ]; then
    echo "SIGNATURE CHANGED: $func_name"
    echo "  Base: $BASE_SIG"
    echo "  Dev:  $DEV_SIG"
  fi
done < /tmp/overridable_funcs.txt
```

### Resolution

For each function with a changed signature:
1. Update the `@overridable` function in `megatron/core/` to have the new signature + decorator
2. Update the `@override` function in `megatron/plugin/` to match the new signature
3. If the override implementation uses the changed parameters, update the implementation logic
4. If the override ignores the changed parameters, just update the signature for compatibility
