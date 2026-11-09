#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_iluvatar_runtime() {
  # Only home paths are mounted. Preserve the legacy paths used by unit tests.
  # -T fails on an existing real directory rather than creating a link inside it.
  mkdir -p /opt/data
  ln -sfnT /home/gitlab-runner/data /opt/data/datasets
  ln -sfnT /home/gitlab-runner/tokenizers /opt/data/tokenizers

  # Keep FlagOS as the default and route IXTE-specific ops to the native backend.
  # Routing does not supply symbols missing from the installed IXTE extension.
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env TE_FL_PREFER flagos
  # Do not add generic_gemm here. 3d6e71594 routed it to vendor.iluvatar on the
  # strength of its entry in vendor/iluvatar/register_ops.py, but the installed
  # ixte 5.0.0 exposes only allgather_gemm, cublas_gemm_2d, grouped_gemm and
  # reducescatter_gemm - there is no generic_gemm symbol, so every TE gemm call
  # raised "module ...ixte_torch has no attribute 'generic_gemm'" (5602 times in
  # job 127711739931). Measured on test_transformer_block.py in the CI image:
  # unset 8 failed / 44 passed, vendor.iluvatar 15 failed / 37 passed,
  # reference.torch 8 failed / 44 passed. Routing buys nothing here, so leave it
  # on the default path; the FlagGems Triton autotune hang it was meant to avoid
  # is now bounded by the per-test timeout instead.
  local per_op='rmsnorm_bwd_add=impl:vendor.iluvatar;te_general_grouped_gemm=impl:vendor.iluvatar'
  # Prepared TE wheels may replace the image's Python files. Reapply the
  # validated reference fix after installation, using the image's version guard.
  if [ -f /opt/megatron-image/apply_te_fl_patches.py ]; then
    "$CI_PYTHON_BIN" /opt/megatron-image/apply_te_fl_patches.py
    per_op+=';multi_tensor_sgd=impl:reference.torch'
  fi
  ci_export_env TE_FL_PER_OP "$per_op"
  ci_export_env CUDA_DEVICE_MAX_CONNECTIONS 1
  ci_export_env NCCL_DEBUG WARN
  # Pin the visible devices to the eight ranks the suite launches. Runners with
  # 16 cards otherwise fail the tests that compare device_count() against
  # world_size, e.g. test_optimizer.py asserting 16 == 8.
  ci_export_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
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
  # dist_checkpointing needs more than the 3600s default. It measures 2484-2754s
  # across runs, so the default leaves under 25% headroom and job 127750191995
  # hit the limit at 82% with zero failures. test_layer_wise_optimizer.py alone
  # accounts for 862s of that: test_layer_wise_optimizer_resharding takes ~156s
  # per parameter here against 4.5s for the slowest case in the same file on
  # CUDA A100. That 35x gap is worth its own investigation; until then the
  # budget has to cover the measured runtime. Per-test --timeout still catches
  # real deadlocks, so this only governs the group's total wall clock.
  ci_export_env CI_GROUP_TIMEOUT_SECONDS 5400
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
