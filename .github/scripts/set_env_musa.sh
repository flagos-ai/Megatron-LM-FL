#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_musa_runtime() {
  ci_export_env DISTRIBUTED_BACKEND mccl
  ci_export_env TORCHDYNAMO_DISABLE 1
  ci_export_env TORCH_COMPILE_DISABLE 1
  ci_export_env LD_LIBRARY_PATH "/usr/local/musa-4.3.4/lib:${LD_LIBRARY_PATH:-}"

  # Derive a unique MASTER_PORT per test group to avoid TCP store collisions
  # when multiple groups run in parallel on the same runner.
  if [[ -n "${CI_TEST_GROUP:-}" ]]; then
    local _port
    case "$CI_TEST_GROUP" in
      core)               _port=29600 ;;
      models)             _port=29601 ;;
      distributed)        _port=29602 ;;
      dist_checkpointing) _port=29603 ;;
      tensor_parallel)    _port=29604 ;;
      pipeline_parallel)  _port=29605 ;;
      data)               _port=29606 ;;
      fusions)            _port=29607 ;;
      others)             _port=29608 ;;
      *)                  _port=29500 ;;
    esac
    ci_export_env MASTER_PORT "${_port}"
  fi
}

validate_musa_capacity() {
  local device_count
  device_count=$(python3 -c "import torch; print(torch.musa.device_count())" |
    awk '/^[0-9]+$/ { count = $0 } END { if (count == "") exit 1; print count }')
  ci_validate_device_capacity "$device_count"
}

install_musa_compatibility_layer() {
  # Keep the image-provided torch/torch_musa pair intact while redirecting
  # legacy CUDA APIs and device strings used by Megatron tests to MUSA.
  python3 -m pip install \
    torchada==0.1.40 \
    --no-deps \
    --no-cache-dir

  # Create this only after installing the project so pip cannot import Megatron
  # through the compatibility layer while it is still building editable metadata.
  local site_dir=/tmp/musa-ci-site
  mkdir -p "$site_dir"

  if [ "${CI_TEST_SUITE:-}" = "unit" ]; then
    # coverage must start before torch is imported. Loading the compatibility
    # layer from sitecustomize imports torch too early and breaks coverage's
    # module scan on the MUSA torch build.
    rm -f "$site_dir/sitecustomize.py"
    cat > "$site_dir/musa_ci_pytest.py" <<'PYTESTEOF'
import contextlib
import io


def pytest_configure(config):
    with contextlib.redirect_stdout(io.StringIO()):
        import torchada  # noqa: F401
        import torch
        from megatron.plugin.platform import get_platform

        is_musa = get_platform().device_name() == "musa"

    if is_musa:
        torch.cuda.is_available = torch.musa.is_available
PYTESTEOF
    ci_export_env PYTEST_ADDOPTS "${PYTEST_ADDOPTS:-} -p musa_ci_pytest"
  else
    cat > "$site_dir/sitecustomize.py" <<'SITEEOF'
import torchada  # noqa: F401
import torch

from megatron.plugin.platform import get_platform

if get_platform().device_name() == "musa":
    torch.cuda.is_available = torch.musa.is_available
SITEEOF
  fi

  ci_export_env PYTHONPATH "$site_dir:${PYTHONPATH:-}"
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_musa_runtime
  ci_ensure_curl

  local test_dependencies=(
    boto3
    mock
    pytest-mock
    coverage
    pytest-asyncio
    anyio
    wandb
    openai
    httpx
    nltk
  )
  python3 -m pip install "${test_dependencies[@]}" --no-cache-dir
  python3 -m pip install fastapi uvicorn --no-cache-dir

  echo "Skipping NVIDIA CUPTI and Emerging-Optimizers dependencies on MUSA."
  ci_install_project --ignore-requires-python
  install_musa_compatibility_layer
  validate_musa_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  configure_musa_runtime
  ci_install_project --ignore-requires-python
  validate_musa_capacity
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    configure_musa_runtime

    # Functional test toolchain and Python 3.10-compatible project install.
    ci_setup_functional_environment --ignore-requires-python
    ci_install_local_tokenizer_dependencies
    ci_validate_qwen_assets /opt/data/datasets /opt/data/tokenizers
    install_musa_compatibility_layer
    validate_musa_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
