#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

validate_metax_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

setup_metax_toolchain() {
  local bridge_bin=/opt/maca/tools/cu-bridge/bin
  local cucc_path="$bridge_bin/cucc"
  local nvcc_path="$bridge_bin/nvcc"

  if [ ! -x "$cucc_path" ]; then
    echo "::error::MetaX compiler bridge not found at $cucc_path"
    exit 1
  fi

  ln -sf "$cucc_path" "$nvcc_path"
  ci_export_env PATH "$bridge_bin:$PATH"
  ls -l "$cucc_path" "$nvcc_path"
  "$nvcc_path" -V
}

setup_unit_environment() {
  ci_activate_python_environment
  ci_ensure_curl

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
  python3 -m pip install torch boto3 "${test_dependencies[@]}" --no-cache-dir
  python3 -m pip install fastapi uvicorn --no-cache-dir

  echo "Skipping NVIDIA CUPTI dependencies on MetaX."
  ci_install_project
  validate_metax_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  ci_install_project
  validate_metax_capacity
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    ci_setup_functional_environment
    setup_metax_toolchain
    validate_metax_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
