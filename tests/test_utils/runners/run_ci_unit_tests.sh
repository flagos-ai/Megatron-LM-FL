#!/usr/bin/env bash

# Platform-neutral distributed unit-test runner used by GitHub Actions.
set -euo pipefail

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "::error::Required environment variable is empty: $name"
    exit 1
  fi
}

for name in \
  CI_PLATFORM \
  CI_DEVICE \
  CI_TEST_GROUP \
  CI_TEST_PATH \
  CI_NPROC_PER_NODE \
  CI_IGNORED_TESTS \
  CI_PYTEST_EXTRA_ARGS; do
  require_env "$name"
done

if ! [[ "$CI_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CI_NPROC_PER_NODE must be a positive integer"
  exit 1
fi

python3 -c \
  "import json, os; value = json.loads(os.environ['CI_IGNORED_TESTS']); assert isinstance(value, list) and all(isinstance(item, str) for item in value)"
python3 -c \
  "import json, os; value = json.loads(os.environ['CI_PYTEST_EXTRA_ARGS']); assert isinstance(value, list) and all(isinstance(item, str) for item in value)"

TEST_PATHS=()
while IFS= read -r item; do
  [ -n "$item" ] && TEST_PATHS+=("$item")
done < <(
  python3 -c '
import glob
import json
import os
import shlex

ignored = json.loads(os.environ["CI_IGNORED_TESTS"])
ignored_files = {item for item in ignored if "::" not in item}
selected = []
for item in shlex.split(os.environ["CI_TEST_PATH"]):
    matches = sorted(glob.glob(item)) if glob.has_magic(item) else [item]
    selected.extend(path for path in matches if path not in ignored_files)
print("\n".join(selected))
'
)

IGNORE_OPTS=()
while IFS= read -r item; do
  [ -n "$item" ] && IGNORE_OPTS+=("$item")
done < <(
  python3 -c '
import json
import os

for item in json.loads(os.environ["CI_IGNORED_TESTS"]):
    prefix = "--deselect=" if "::" in item else "--ignore="
    print(prefix + item)
'
)

EXTRA_ARGS=()
while IFS= read -r item; do
  [ -n "$item" ] && EXTRA_ARGS+=("$item")
done < <(
  python3 -c '
import json
import os

print("\n".join(json.loads(os.environ["CI_PYTEST_EXTRA_ARGS"])))
'
)

if [ "${#TEST_PATHS[@]}" -eq 0 ]; then
  echo "::error::No tests selected for $CI_TEST_GROUP from CI_TEST_PATH='$CI_TEST_PATH'"
  exit 1
fi

PYTEST_BIN="${CI_PYTEST_BIN:-$(command -v pytest)}"
TORCHRUN_BIN="${CI_TORCHRUN_BIN:-$(command -v torchrun)}"
if [ ! -x "$PYTEST_BIN" ]; then
  echo "::error::pytest executable not found: $PYTEST_BIN"
  exit 1
fi
if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "::error::torchrun executable not found: $TORCHRUN_BIN"
  exit 1
fi

export PYTHONPATH="$GITHUB_WORKSPACE:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/.torch_inductor_cache}"
export MEGATRON_TEST_PLATFORM="$CI_PLATFORM"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "Running unit tests: $CI_TEST_GROUP"
echo "Test path: $CI_TEST_PATH"
echo "Platform: $CI_PLATFORM"
echo "Device: $CI_DEVICE"
echo "Distributed processes: $CI_NPROC_PER_NODE"
printf 'Selected test arguments:'
printf ' %q' "${TEST_PATHS[@]}"
printf '\n'

COVERAGE_DIR="$GITHUB_WORKSPACE/coverage-report"
COVERAGE_TRAINING_SOURCE="$GITHUB_WORKSPACE/megatron/training"
COVERAGE_PLUGIN_SOURCE="$GITHUB_WORKSPACE/megatron/plugin"
COVERAGE_INCLUDE="megatron/training/*,megatron/plugin/*"
COVERAGE_JSON="$COVERAGE_DIR/coverage-$CI_PLATFORM-$CI_DEVICE-$CI_TEST_GROUP.json"
mkdir -p "$COVERAGE_DIR"
printf '[run]\nparallel = true\ndynamic_context = test_function\nsource =\n    %s\n    %s\ndata_file = %s/.coverage\n' \
  "$COVERAGE_TRAINING_SOURCE" \
  "$COVERAGE_PLUGIN_SOURCE" \
  "$COVERAGE_DIR" \
  > "$COVERAGE_DIR/.coveragerc"

PYTEST_ARGS=(
  -v
  "${TEST_PATHS[@]}"
  --ignore=tests/functional_tests
)
if [ "${#IGNORE_OPTS[@]}" -gt 0 ]; then
  PYTEST_ARGS+=("${IGNORE_OPTS[@]}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  PYTEST_ARGS+=("${EXTRA_ARGS[@]}")
fi
PYTEST_ARGS+=(
  -p
  no:randomly
  -o
  addopts="--durations=15 -s -rA"
)

set +e
"$TORCHRUN_BIN" --nproc_per_node="$CI_NPROC_PER_NODE" \
  -m coverage run \
  --rcfile="$COVERAGE_DIR/.coveragerc" \
  "$PYTEST_BIN" \
  "${PYTEST_ARGS[@]}"
test_exit_code=$?
set -e

python3 -m coverage combine \
  --rcfile="$COVERAGE_DIR/.coveragerc" \
  "$COVERAGE_DIR" 2>/dev/null || true
python3 -m coverage json \
  --rcfile="$COVERAGE_DIR/.coveragerc" \
  --show-contexts \
  -o "$COVERAGE_JSON" \
  --include="$COVERAGE_INCLUDE" 2>/dev/null || true

exit "$test_exit_code"
