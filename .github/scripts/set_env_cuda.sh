#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

validate_cuda_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
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

  git clone --branch v0.6.0-main \
    https://github.com/NVIDIA/nvidia-resiliency-ext.git \
    /tmp/nvidia-resiliency-ext
  python3 -m pip install -e /tmp/nvidia-resiliency-ext --no-cache-dir
  python3 -m pip install protobuf==6.33.1
  python3 -m pip install \
    git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0 \
    --no-cache-dir

  ci_install_project
  validate_cuda_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  ci_install_project
  validate_cuda_capacity
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    ci_setup_functional_environment
    validate_cuda_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
