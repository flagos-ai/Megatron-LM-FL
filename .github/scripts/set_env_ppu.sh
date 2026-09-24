#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_ppu_runtime() {
  ci_export_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
  ci_export_env CUDA_DEVICE_MAX_CONNECTIONS 1
  ci_export_env OMP_NUM_THREADS 1
  ci_export_env NCCL_DEBUG WARN
  ci_export_env NCCL_MAX_NCHANNELS 1
  ci_export_env NCCL_NVLS_ENABLE 0
  # The manual tests use the vendor NCCL/PCCL interface, not the FlagCX plugin.
  ci_export_env DISTRIBUTED_BACKEND nccl

  # The snapshot pins TE, but prepare owns its version now. Keep vendor pins.
  if [ "${PIP_CONSTRAINT:-}" = /opt/megatron-ppu-ci/constraints.txt ]; then
    sed -i '/^[[:space:]]*transformer-engine==/d' "$PIP_CONSTRAINT"
  fi
}

validate_ppu_capacity() {
  "$CI_PYTHON_BIN" -c \
    'import torch; assert torch.cuda.is_available(); print(f"Torch: {torch.__version__}")'
  local device_count
  device_count=$("$CI_PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())' |
    awk '/^[0-9]+$/ { count = $0 } END { print count }')
  ci_validate_device_capacity "$device_count"
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_ppu_runtime
  validate_ppu_capacity

  # Asset-dependent tests validate their fixtures; unrelated groups need no tokenizers.
  # Install only the checked-out Megatron source. Never resolve the vendor stack here.
  ci_install_project
}

setup_functional_environment() {
  configure_ppu_runtime
  ci_setup_functional_environment
  ci_install_local_tokenizer_dependencies
  ci_validate_qwen_assets /home/gitlab-runner/data /home/gitlab-runner/tokenizers
  validate_ppu_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  configure_ppu_runtime
  validate_ppu_capacity
  ci_install_project
}

ci_require_env CI_TEST_SUITE
ci_require_env CI_NPROC_PER_NODE
if ! [[ "$CI_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CI_NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi

case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    setup_functional_environment
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
