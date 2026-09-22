#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

validate_enflame_torch() {
  python3 -c \
    "import torch, torch_gcu; print(f'Torch: {torch.__version__}, torch_gcu: {torch_gcu.__version__}')"
}

validate_enflame_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch, torch_gcu; print(torch.gcu.device_count())" |
    awk '/^[0-9]+$/ { count = $0 } END { print count }')
  ci_validate_device_capacity "$device_count"

  python3 -c \
    "import torch, torch_gcu; assert torch.gcu.is_available(); print(f'GCU devices: {torch.gcu.device_count()}')"
}

patch_coverage_for_torch_gcu() {
  # torch_gcu registers _OpNamespace objects in sys.modules whose __path__ is
  # not a sequence. coverage.py scans sys.modules inside coverage.start() and
  # calls len() on every module's __path__, which raises TypeError and kills
  # every rank before pytest is even imported. Because the crash happens in the
  # `coverage run` CLI before pytest loads, a pytest plugin cannot fix it and a
  # .pth hook is fragile (it runs before coverage is installed). Patching the
  # installed source is the only interception point that is guaranteed to be in
  # effect when coverage.start() runs. ascend/metax do not need this: their
  # torch backends keep __path__ a real sequence.
  python3 - <<'PYEOF'
import coverage.inorout as m

path = m.__file__
src = open(path).read()
old = 'if len(getattr(mod, "__path__", ())) > 1:'
guard = '_enflame_gcu_patch'

if guard in src:
    print(f"coverage already patched: {path}")
elif old in src:
    # Coerce __path__ to a real sequence before len(); _OpNamespace is truthy
    # but has no __len__, so isinstance is the safe test rather than `or ()`.
    new = (
        'if len(getattr(mod, "__path__", ())'
        ' if isinstance(getattr(mod, "__path__", ()), (list, tuple)) else ()'
        ') > 1:  # _enflame_gcu_patch'
    )
    open(path, "w").write(src.replace(old, new, 1))
    print(f"Patched coverage for torch_gcu: {path}")
else:
    raise SystemExit(
        f"::error::coverage.inorout source changed; torch_gcu patch needs an "
        f"update: {path}"
    )
PYEOF
}

configure_enflame_runtime() {
  validate_enflame_torch
  validate_enflame_capacity
}

disable_unavailable_test_asset_downloads() {
  local data_dir=/opt/data
  mkdir -p "$data_dir"

  # The Enflame unit runner does not mount the NVIDIA unit-test release assets.
  # Asset-dependent tests are excluded in enflame.yml; this marker prevents the
  # session fixture from downloading the same archives in every matrix job.
  if [ -z "$(find "$data_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    touch "$data_dir/.enflame-ci-assets-unavailable"
  fi
}

setup_unit_environment() {
  ci_activate_python_environment
  ci_ensure_curl
  validate_enflame_torch

  local test_dependencies=(
    mock
    pytest-mock
    coverage
    pytest-asyncio
    anyio
    wandb
    openai
    httpx
    nltk
    msgpack
  )
  local pip_index_args=(
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    --timeout 300
    --retries 10
    --no-cache-dir
    --break-system-packages
  )

  # boto3 is intentionally omitted: S3 unit tests provide a local mock, while
  # botocore downloads have been unreliable through the CI proxy.
  python3 -m pip install ninja "${test_dependencies[@]}" "${pip_index_args[@]}"
  echo "Ninja: $(ninja --version)"

  # Collection-only dependencies are installed without dependencies to preserve
  # the torch, protobuf, and numpy versions validated in the Enflame image.
  python3 -m pip install fastapi starlette uvicorn griffe \
    --no-deps "${pip_index_args[@]}"

  echo "Skipping NVIDIA CUPTI dependencies and Emerging-Optimizers on Enflame."

  patch_coverage_for_torch_gcu

  ci_install_project --break-system-packages
  configure_enflame_runtime
  disable_unavailable_test_asset_downloads
}

setup_build_environment() {
  ci_activate_python_environment
  validate_enflame_torch
  ci_install_project --break-system-packages
  configure_enflame_runtime
}

ci_require_env CI_TEST_SUITE

case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    validate_enflame_torch
    ci_setup_functional_environment
    configure_enflame_runtime
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
