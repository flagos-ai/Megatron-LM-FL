#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_iluvatar_runtime() {
  # Keep FlagOS as the default and route IXTE-specific ops to the native backend.
  # Routing does not supply symbols missing from the installed IXTE extension.
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env TE_FL_PREFER flagos
  ci_export_env TE_FL_PER_OP 'rmsnorm_bwd_add=impl:vendor.iluvatar;te_general_grouped_gemm=impl:vendor.iluvatar'
  ci_export_env CUDA_DEVICE_MAX_CONNECTIONS 1
  ci_export_env NCCL_DEBUG WARN
}

validate_iluvatar_capacity() {
  local device_count
  device_count=$("$CI_PYTHON_BIN" -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

verify_iluvatar_software_stack() {
  "$CI_PYTHON_BIN" - <<'PY'
import flag_gems
import torch

print(f"FlagGems: {flag_gems.__file__}")
print(f"Torch: {torch.__version__}")
if torch.cuda.is_available() and torch.cuda.device_count():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
PY
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_iluvatar_runtime
  ci_ensure_curl
  verify_iluvatar_software_stack
  validate_iluvatar_capacity
  ci_install_project
}

setup_functional_environment() {
  configure_iluvatar_runtime
  ci_setup_functional_environment
  ci_install_local_tokenizer_dependencies
  ci_validate_qwen_assets /home/gitlab-runner/data /home/gitlab-runner/tokenizers
  verify_iluvatar_software_stack
  validate_iluvatar_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  configure_iluvatar_runtime
  verify_iluvatar_software_stack
  validate_iluvatar_capacity
  ci_install_project
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit_group)
    ;;
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
