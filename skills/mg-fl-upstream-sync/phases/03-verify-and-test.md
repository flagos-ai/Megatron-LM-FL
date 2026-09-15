# Phase 3: Verify & Test

Stages 6-7 of the Megatron-LM-FL upstream sync workflow.

---

### Stage 6: Build & Import Verification

Because Megatron-LM-FL is consumed via `pip install` + `import megatron.core` / `import
megatron.plugin`, this stage validates the library is installable and both packages are
importable.

**Important:** Only `megatron.core` and `megatron.plugin` are pip-packaged (defined in
`pyproject.toml` under `[tool.setuptools.packages.find]`). Other packages like
`megatron.training`, `megatron.legacy`, `megatron.rl`, and `megatron.post_training` exist in
the repo but are NOT pip-installed — they resolve at runtime via PYTHONPATH. Do not attempt
to verify imports for those packages here.

```bash
# Syntax check all Python files
find megatron/ -name "*.py" ! -path "*__pycache__*" -exec python3 -c \
  "import ast, sys; ast.parse(open(sys.argv[1]).read()); print(f'{sys.argv[1]} OK')" {} \;

# Verify pyproject.toml includes both megatron.core and megatron.plugin
python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)
includes = config['tool']['setuptools']['packages']['find']['include']
assert 'megatron.core' in includes, 'megatron.core missing from pyproject.toml includes'
assert 'megatron.core.*' in includes, 'megatron.core.* missing from pyproject.toml includes'
assert 'megatron.plugin' in includes, 'megatron.plugin missing from pyproject.toml includes'
assert 'megatron.plugin.*' in includes, 'megatron.plugin.* missing from pyproject.toml includes'
print('pyproject.toml includes: OK')
print('  includes:', includes)
"

# Install in editable mode
pip install -e . --no-build-isolation

# Core and plugin import verification
python3 -c "
import megatron.core
from megatron.plugin.platform import cur_platform
import megatron.plugin
print('megatron.core version:', getattr(megatron.core, '__version__', 'N/A'))
print('cur_platform:', cur_platform)
print('megatron.core import: PASSED')
print('megatron.plugin import: PASSED')
"
```

#### Native Megatron-LM unit tests

Because non-plugin components (e.g., `megatron/training/`) are kept directly consistent with
upstream, the repo's native unit tests should still pass. Run a subset to confirm:

```bash
# Run native Megatron-LM unit tests (subset — focus on training and core)
pytest tests/ -x -v --timeout=120 -k "not distributed" 2>&1 | tail -50

# If specific test directories exist:
pytest tests/unit_tests/ -x -v --timeout=120 2>&1 | tail -50
```

If tests fail on non-plugin code, the upstream version was likely not applied cleanly — revisit
Stage 3 and ensure those files match upstream exactly. If tests fail on plugin-modified code,
debug the plugin integration (Stage 4).

**Success criteria:** `pip install -e .` succeeds, `import megatron.core` and `import megatron.plugin` both work, native unit tests pass.

#### Checkpoint: pre-commit & commit

If any fixes were applied during build/import verification, commit them:

```bash
pre-commit run --all-files
git add -A
git commit -m "Stage 6: build and import verification passed" --allow-empty
```

---

### Stage 7: Unit Test Verification (Grouped Execution)

Run Megatron-LM-FL unit tests in groups to verify the synced code doesn't break existing
functionality. Tests are executed via `torchrun` + `pytest` with 8 GPUs. Before running, parse
`.github/configs/cuda.yml` to build the `--deselect` list from `test_matrix.unit.ignored_tests`.

#### Step 1: Build the ignored-tests deselect list

```bash
cd "$MG_FL_DIR"

# Extract ignored_tests from cuda.yml into --deselect args
DESELECT_ARGS=""
while IFS= read -r line; do
  test_path=$(echo "$line" | sed 's/^[[:space:]]*-[[:space:]]*//' | sed 's/[[:space:]]*##.*//')
  [ -n "$test_path" ] && DESELECT_ARGS="$DESELECT_ARGS --deselect=$test_path"
done < <(grep -A 200 'ignored_tests:' .github/configs/cuda.yml | tail -n +2 | grep '^\s*-' | sed '/^[[:space:]]*$/d')

echo "Deselect args: $DESELECT_ARGS"
```

#### Step 2: Run tests in groups

Split unit tests into logical groups to isolate failures. Each group runs as a separate
`torchrun` invocation. If a group fails, record it and continue with the next group.

Common environment setup for all groups:

```bash
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0
export MASTER_ADDR=localhost
export MASTER_PORT=6000
NPROC=8
```

**Group 1 — tensor_parallel:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/tensor_parallel
```

**Group 2 — transformer (excluding moe/):**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/transformer --ignore=tests/unit_tests/transformer/moe \
  --ignore=tests/unit_tests/transformer/experimental_attention_variant
```

**Group 3 — transformer/moe:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/transformer/moe
```

**Group 4 — dist_checkpointing:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/dist_checkpointing
```

**Group 5 — models:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/models
```

**Group 6 — distributed:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/distributed
```

**Group 7 — pipeline_parallel:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/pipeline_parallel
```

**Group 8 — optimizer & data & fusions:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/optimizer tests/unit_tests/data tests/unit_tests/fusions
```

**Group 9 — root-level test files (basic, imports, utilities, etc.):**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/test_basic.py \
  tests/unit_tests/test_imports.py \
  tests/unit_tests/test_utilities.py \
  tests/unit_tests/test_argument_utils.py \
  tests/unit_tests/test_parallel_state.py \
  tests/unit_tests/test_training.py \
  tests/unit_tests/test_checkpointing.py \
  tests/unit_tests/test_typed_torch.py \
  tests/unit_tests/test_num_microbatches_calculator.py \
  tests/unit_tests/test_process_groups_config.py
```

**Group 10 — remaining subdirectories:**
```bash
torchrun --nproc_per_node=$NPROC -m pytest -xvs $DESELECT_ARGS \
  tests/unit_tests/ssm tests/unit_tests/export tests/unit_tests/inference \
  tests/unit_tests/tokenizers tests/unit_tests/utils tests/unit_tests/resharding \
  tests/unit_tests/a2a_overlap tests/unit_tests/extension tests/unit_tests/rl \
  tests/unit_tests/post_training
```

#### Step 3: Collect results

```bash
# Summarize which groups passed/failed
echo "=== Unit Test Summary ==="
echo "Group 1 (tensor_parallel):    $G1_STATUS"
echo "Group 2 (transformer):        $G2_STATUS"
echo "Group 3 (transformer/moe):    $G3_STATUS"
echo "Group 4 (dist_checkpointing): $G4_STATUS"
echo "Group 5 (models):             $G5_STATUS"
echo "Group 6 (distributed):        $G6_STATUS"
echo "Group 7 (pipeline_parallel):  $G7_STATUS"
echo "Group 8 (optimizer/data/fusions): $G8_STATUS"
echo "Group 9 (root-level tests):   $G9_STATUS"
echo "Group 10 (remaining):         $G10_STATUS"
```

**Success criteria:** All groups pass (after accounting for `ignored_tests` deselections).
Failures in groups that touch `megatron.core` or `megatron.plugin` code (Groups 1–7) are
blockers. Failures in other groups should be investigated but may be pre-existing upstream
issues — cross-check against the upstream CI status before blocking the sync.

---

