#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_iluvatar_runtime() {
  # Keep FlagOS as the default and route IXTE-specific ops to the native backend.
  # Routing does not supply symbols missing from the installed IXTE extension.
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env TE_FL_PREFER flagos
  # generic_gemm goes to the native backend as well: the flagos path dispatches
  # to FlagGems, whose Triton autotune hangs in _init_handles on CoreX 4.5 and
  # takes the whole group to the step timeout. With the native route the
  # pipeline_parallel group goes from timing out to passing and the models
  # group stops hanging in test_llava_model.
  local per_op='rmsnorm_bwd_add=impl:vendor.iluvatar;te_general_grouped_gemm=impl:vendor.iluvatar;generic_gemm=impl:vendor.iluvatar'
  # Prepared TE wheels may replace the image's Python files. Reapply the
  # validated reference fix after installation, using the image's version guard.
  if [ -f /opt/megatron-image/apply_te_fl_patches.py ]; then
    "$CI_PYTHON_BIN" /opt/megatron-image/apply_te_fl_patches.py
    per_op+=';multi_tensor_sgd=impl:reference.torch'
  fi
  ci_export_env TE_FL_PER_OP "$per_op"
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

install_iluvatar_unit_test_plugins() {
  # The config passes --timeout to bound each test, which needs this plugin.
  # ci_install_project runs with --no-deps and the image does not ship it, so
  # install it here rather than relying on the test dependency group.
  "$CI_PYTHON_BIN" -m pip install pytest-timeout \
    --index-url https://pypi.org/simple \
    --timeout 300 \
    --retries 10 \
    --no-cache-dir \
    --quiet
  "$CI_PYTHON_BIN" -c "import pytest_timeout"
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_iluvatar_runtime
  ci_ensure_curl
  verify_iluvatar_software_stack
  validate_iluvatar_capacity
  ci_install_project
  install_iluvatar_unit_test_plugins
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
    configure_iluvatar_runtime
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
