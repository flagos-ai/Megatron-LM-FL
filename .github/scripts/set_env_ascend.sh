#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

validate_ascend_torch() {
  python3 -c \
    "import torch, torch_npu; assert torch.__version__ == '2.7.1+cpu', torch.__version__; assert torch_npu.__version__ == '2.7.1.post2', torch_npu.__version__"
}

validate_ascend_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch, torch_npu; print(torch_npu.npu.device_count())" |
    awk '/^[0-9]+$/ { count = $0 } END { print count }')
  ci_validate_device_capacity "$device_count"

  python3 -c \
    "import torch, torch_npu; assert torch.npu.is_available(); print(f'Torch: {torch.__version__}, torch_npu: {torch_npu.__version__}, NPU devices: {torch.npu.device_count()}')"
}

configure_ascend_runtime() {
  validate_ascend_torch
  validate_ascend_capacity
  ci_export_env HCCL_NPU_SOCKET_PORT_RANGE "16666-16766"

  # Suppress "CPU RNG state changed within GPU RNG context" log spam.
  # On Ascend the NPU RNG path touches CPU RNG state as a side-effect,
  # so this warning fires on every fork() context entry.  It is harmless
  # noise (not an error indicator) but can produce tens of thousands of
  # lines in dist_checkpointing, inflating logs past 60k lines.
  mkdir -p /tmp/ascend-ci-site
  cat > /tmp/ascend-ci-site/sitecustomize.py <<'SITEEOF'
import logging
logging.getLogger('megatron.core.tensor_parallel.random').setLevel(logging.ERROR)
SITEEOF
  ci_export_env PYTHONPATH "/tmp/ascend-ci-site:${PYTHONPATH:-}"
}

install_ascend_uv_system_python_shim() {
  local shim_dir=/tmp/ascend-ci-bin
  mkdir -p "$shim_dir"

  cat > "$shim_dir/uv" <<'UVEOF'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "run" ]; then
  echo "Ascend CI uv shim only supports 'uv run': $*" >&2
  exit 1
fi

shift
if [ "${1:-}" = "--no-sync" ]; then
  shift
fi

if [ "${1:-}" = "python" ]; then
  shift
  exec python3 "$@"
fi
if [ "${1:-}" = "pytest" ]; then
  shift
  exec python3 -m pytest "$@"
fi

exec "$@"
UVEOF
  chmod 0755 "$shim_dir/uv"

  export PATH="$shim_dir:$PATH"
  if [ -n "${GITHUB_PATH:-}" ]; then
    printf '%s\n' "$shim_dir" >> "$GITHUB_PATH"
  fi
  ci_export_env PATH "$PATH"
  uv run --no-sync python -c "import sys; print('Ascend functional Python:', sys.executable)"
}

install_ascend_functional_dependencies() {
  local pip_index_args=(
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    --timeout 300
    --retries 10
    --no-cache-dir
  )

  # The image already contains TensorBoard's runtime dependencies. Installing
  # only the frontend package avoids changing torch, protobuf, or numpy.
  python3 -m pip install "tensorboard==2.17.1" --no-deps "${pip_index_args[@]}"
  python3 -c \
    "import numpy, pytest; from torch.utils.tensorboard import SummaryWriter; print('Ascend functional dependencies validated')"
}

install_python_config_shim() {
  if command -v python3-config >/dev/null 2>&1; then
    return
  fi

  cat > /usr/local/bin/python3-config <<'PYTHON_CONFIG'
#!/usr/bin/env python3
import sys
import sysconfig

if sys.argv[1:] != ["--extension-suffix"]:
    raise SystemExit("python3-config shim only supports --extension-suffix")
suffix = sysconfig.get_config_var("EXT_SUFFIX")
if not suffix:
    raise SystemExit("Python EXT_SUFFIX is unavailable")
print(suffix)
PYTHON_CONFIG
  chmod 0755 /usr/local/bin/python3-config
}

install_flash_attn_collection_stub() {
  if [ "${CI_TEST_GROUP:-}" != "models" ]; then
    return
  fi

  local stub_dir=/tmp/megatron-ci-stubs
  mkdir -p "$stub_dir/flash_attn"
  printf '__version__ = "0.0.0"\n' > "$stub_dir/flash_attn/__init__.py"
  ci_export_env PYTHONPATH "$stub_dir:${PYTHONPATH:-}"
  python3 -c "import flash_attn; assert flash_attn.__version__ == '0.0.0'"
}

disable_unavailable_test_asset_downloads() {
  local data_dir=/opt/data
  mkdir -p "$data_dir"

  # The Ascend unit runner does not mount the NVIDIA unit-test release assets.
  # Asset-dependent tests are excluded in ascend.yml; this marker prevents the
  # session fixture from downloading the same archives in every matrix job.
  if [ -z "$(find "$data_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    touch "$data_dir/.ascend-ci-assets-unavailable"
  fi
}

setup_unit_environment() {
  ci_activate_python_environment
  ci_ensure_curl
  validate_ascend_torch
  install_python_config_shim
  echo "Python extension suffix: $(python3-config --extension-suffix)"

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
  )

  # boto3 is intentionally omitted: S3 unit tests provide a local mock, while
  # botocore downloads have been unreliable through the CI proxy.
  python3 -m pip install ninja "${test_dependencies[@]}" "${pip_index_args[@]}"
  echo "Ninja: $(ninja --version)"

  # Collection-only dependencies are installed without dependencies to preserve
  # the torch, protobuf, and numpy versions validated in the Ascend image.
  python3 -m pip install "tensorboard<2.18" fastapi starlette uvicorn griffe \
    --no-deps "${pip_index_args[@]}"

  echo "Skipping NVIDIA CUPTI dependencies and Emerging-Optimizers on Ascend."
  ci_install_project
  configure_ascend_runtime
  disable_unavailable_test_asset_downloads
  install_flash_attn_collection_stub
}

setup_build_environment() {
  ci_activate_python_environment
  validate_ascend_torch
  ci_install_project
  configure_ascend_runtime
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    validate_ascend_torch
    ci_setup_functional_environment
    install_ascend_uv_system_python_shim
    install_ascend_functional_dependencies
    configure_ascend_runtime
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
