#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_hygon_runtime() {
  # DTK exposes BW1000 through PyTorch's CUDA-compatible API. FlagGems uses
  # the AMD Triton backend in the documented image.
  local dtk_home="${DTK_HOME:-/opt/dtk}"
  local hip_path="${HIP_PATH:-$dtk_home/hip}"
  local hsa_path="${HSA_PATH:-$dtk_home/hsa}"
  local dtk_path="$dtk_home/bin:$hip_path/bin:$dtk_home/llvm/bin"
  local dtk_library_path="/opt/hyhal/lib/criu:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib:$hip_path/lib:$dtk_home/lib:$dtk_home/llvm/lib:$dtk_home/dcc/lib:$dtk_home/aillvm/lib:$hsa_path/lib"
  local -a visible_devices=()
  local device_index
  local cuda_visible_devices

  ci_export_env GEMS_VENDOR amd
  ci_require_env CI_NPROC_PER_NODE
  if ! [[ "$CI_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::CI_NPROC_PER_NODE must be a positive integer"
    exit 1
  fi
  # The host has more devices, so expose exactly the configured process count.
  for ((device_index = 0; device_index < CI_NPROC_PER_NODE; device_index += 1)); do
    visible_devices+=("$device_index")
  done
  local IFS=,
  cuda_visible_devices="${visible_devices[*]}"
  ci_export_env CUDA_VISIBLE_DEVICES "$cuda_visible_devices"
  # RCCL warns about this at init and hangs in ncclCommInitRank without it,
  # so every multi-device collective needs it set before torchrun starts.
  ci_export_env HSA_FORCE_FINE_GRAIN_PCIE 1
  # Preserve warnings and errors without printing every process-group setup.
  ci_export_env NCCL_DEBUG WARN
  ci_export_env TORCH_CPP_LOG_LEVEL WARNING
  # PyTorch otherwise starts 32 Inductor workers per rank in this image.
  ci_export_env TORCHINDUCTOR_COMPILE_THREADS 1
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env TE_FL_PREFER flagos
  ci_export_env NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD 1
  ci_export_env NVTE_FUSED_ATTN 0
  ci_export_env DTK_HOME "$dtk_home"
  ci_export_env ROCM_PATH "$dtk_home"
  ci_export_env HIP_PATH "$hip_path"
  ci_export_env HSA_PATH "$hsa_path"
  ci_export_env HIP_CLANG_PATH "$dtk_home/llvm/bin"
  ci_export_env DEVICE_LIB_PATH "$dtk_home/amdgcn/bitcode"
  ci_export_env PATH "$dtk_path:$PATH"
  ci_export_env LD_LIBRARY_PATH "$dtk_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  test -d /opt/hyhal
  test -e /dev/kfd
  test -e /dev/mkfd
  test -d /dev/dri
}

configure_hygon_unit_safety() {
  local python_bin
  python_bin=$(command -v python3)
  local timeout_seconds=2700
  local timeout_wrapper=/tmp/hygon-unit-python-with-timeout

  # Abort a unit-test job when one rank stops making collective progress
  # instead of leaving the remaining ranks blocked until the Actions timeout.
  ci_export_env TORCH_NCCL_ASYNC_ERROR_HANDLING 1
  ci_export_env TORCH_NCCL_ENABLE_MONITORING 1
  ci_export_env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 180
  ci_export_env TORCH_NCCL_DUMP_ON_TIMEOUT 1
  ci_export_env TORCH_NCCL_TRACE_BUFFER_SIZE 2000

  if ! command -v timeout >/dev/null 2>&1; then
    echo "::error::GNU timeout is required for Hygon unit tests"
    exit 1
  fi

  cat > "$timeout_wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec timeout \
  --signal=TERM \
  --kill-after=30s \
  "${timeout_seconds}s" \
  "$python_bin" "\$@"
EOF
  chmod 0755 "$timeout_wrapper"
  ci_export_env CI_PYTHON_BIN "$timeout_wrapper"
  echo "Hygon unit-test hard timeout: ${timeout_seconds}s"
}

# Catch a stale or mis-tagged vendor image before TE-FL is installed.
verify_hygon_software_stack() {
  python3 - <<'PY'
import flag_gems

print(f"FlagGems import passed: {flag_gems.__file__}")
PY
}

remove_broken_hygon_cupy() {
  # DTK ships a cupy that imports but has no ndarray attribute. Einops then
  # selects it as a backend and fails instead of using torch.
  python3 -m pip uninstall cupy -y
}

install_hygon_project() {
  cd "$CI_PROJECT_ROOT"
  git config --global --add safe.directory "$CI_PROJECT_ROOT"
  # The validated DTK image currently provides Python 3.10 while the project
  # metadata requires 3.12. The documented training stack is source-compatible
  # with 3.10, so bypass only the metadata gate and preserve the image runtime.
  python3 -m pip install -e . \
    --ignore-requires-python \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir
}

validate_hygon_capacity() {
  python3 - <<'PY'
import torch

print(f"Torch: {torch.__version__}")
print(f"HIP runtime: {torch.version.hip}")
print(f"CUDA-compatible API available: {torch.cuda.is_available()}")
print(f"Visible Hygon devices: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
PY

  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

prepare_hygon_functional_assets() {
  local data_root=/opt/data/datasets
  local tokenizer_root=/opt/data/tokenizers
  local hf_home=/tmp/hygon-huggingface

  ci_validate_qwen_assets "$data_root" "$tokenizer_root"

  mkdir -p "$hf_home/modules"
  ci_export_env HF_HOME "$hf_home"
  ci_export_env HF_MODULES_CACHE "$hf_home/modules"
  ci_export_env HF_HUB_OFFLINE 1
  ci_export_env TRANSFORMERS_OFFLINE 1

  # Populate the dynamic-module cache before torchrun starts multiple ranks.
  python3 - <<'PY'
from transformers import AutoTokenizer

path = "/opt/data/tokenizers/qwentokenizer"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print(f"Hygon tokenizer prewarm passed: {type(tokenizer)}; vocab={tokenizer.vocab_size}")
PY
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_hygon_runtime
  configure_hygon_unit_safety
  ci_ensure_curl

  echo "Preserving the PyTorch and DTK packages supplied by the BW1000 image."
  echo "Skipping NVIDIA CUPTI, NVRx, and Emerging Optimizers dependencies."
  python3 -m pip install multi-storage-client \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-cache-dir
  remove_broken_hygon_cupy
  verify_hygon_software_stack
  validate_hygon_capacity
  install_hygon_project
}

setup_functional_environment() {
  ci_activate_python_environment
  configure_hygon_runtime

  local python_bin
  python_bin=$(command -v python3)
  if [ "$python_bin" != "/usr/bin/python3" ] && \
     [ "$python_bin" != "/usr/local/bin/python3" ]; then
    ln -sf "$python_bin" /usr/local/bin/python3
  fi

  ci_install_yq
  ci_install_envsubst
  # Keep uv on the Python 3.10 interpreter supplied with the DTK stack.
  ci_install_uv_compatibility_shim true
  uv run --no-sync python -c \
    "import sys, torch; print(f'Hygon functional Python: {sys.executable}; Torch: {torch.__version__}')"
  remove_broken_hygon_cupy
  verify_hygon_software_stack
  install_hygon_project
  ci_install_local_tokenizer_dependencies
  validate_hygon_capacity
  prepare_hygon_functional_assets
}

setup_build_environment() {
  ci_activate_python_environment
  configure_hygon_runtime
  verify_hygon_software_stack
  install_hygon_project
  validate_hygon_capacity
}

ci_require_env CI_TEST_SUITE
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
