#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

# Override the common install helper: pyproject.toml declares
# requires-python>=3.12, but the KunLunXin torch environment ships
# Python 3.10. --ignore-requires-python bypasses that version gate;
# the code runs correctly with the XPU stack on 3.10.
ci_install_project() {
  cd "$CI_PROJECT_ROOT"
  python3 -m pip install -e . --no-deps --no-build-isolation \
      --no-cache-dir --ignore-requires-python
}

activate_kunlunxin_python_environment() {
  # The KunLunXin CI image installs conda under /root/miniconda, not
  # /opt/conda, so the common ci_activate_python_environment() guard
  # never fires. Activate the PyTorch/XPU environment directly.
  source /root/miniconda/etc/profile.d/conda.sh
  conda activate python310_torch29_cuda
  ci_export_env PATH "$PATH"
  echo "Python: $(command -v python3) ($(python3 --version 2>&1))"
}

# Override the common activation helper so every shared entry point picks up
# the XPU environment. ci_setup_functional_environment() calls
# ci_activate_python_environment() internally; without this override the
# functional suite would silently fall through to the system interpreter,
# where torch and torch_xmlir are not importable.
ci_activate_python_environment() {
  activate_kunlunxin_python_environment
}

ci_install_uv_compatibility_shim() {
  # The common shim skips installation if a real `uv` is already present
  # in PATH. On the vendor runner the image ships a real uv that manages
  # its own virtual environment — it will resolve to the base conda Python
  # (/root/miniconda/bin/python3, no torch/numpy) rather than the activated
  # XPU environment. Override unconditionally so every `uv run python ...`
  # call in _run_training.sh reaches the correct interpreter.
  local python_bin
  python_bin=$(command -v python3)

  local shim_dir=/tmp/kunlunxin-ci-bin
  mkdir -p "$shim_dir"

  cat > "$shim_dir/uv" <<UVEOF
#!/usr/bin/env bash
set -euo pipefail

if [ "\${1:-}" != "run" ]; then
  echo "KunLunXin CI uv shim only supports 'uv run': \$*" >&2
  exit 1
fi

shift
if [ "\${1:-}" = "--no-sync" ]; then
  shift
fi

if [ "\${1:-}" = "python" ]; then
  shift
  exec "$python_bin" "\$@"
fi
if [ "\${1:-}" = "pytest" ]; then
  shift
  exec "$python_bin" -m pytest "\$@"
fi

exec "\$@"
UVEOF
  chmod 0755 "$shim_dir/uv"

  export PATH="$shim_dir:$PATH"
  if [ -n "${GITHUB_PATH:-}" ]; then
    printf '%s\n' "$shim_dir" >> "$GITHUB_PATH"
  fi
  ci_export_env PATH "$PATH"
  uv run --no-sync python -c "import sys; print('KunLunXin functional Python:', sys.executable)"
}

install_kunlunxin_python_config_shim() {
  # megatron/core/datasets/Makefile takes its include flags from
  # `python3 -m pybind11 --includes` but derives the module suffix from
  # `python3-config --extension-suffix`. The activated XPU environment ships
  # no python3-config, so PATH falls through to the conda base interpreter
  # (Python 3.13) and stamps a cpython-313 suffix onto a module compiled
  # against Python 3.10 headers. The 3.10 training process cannot import it
  # and compile_helpers() reports "Failed to compile the C++ dataset helper
  # functions". Shim python3-config onto the active interpreter so both
  # Makefile lines agree on one Python.
  local python_bin
  python_bin=$(command -v python3)

  local shim_dir=/tmp/kunlunxin-ci-bin
  mkdir -p "$shim_dir"

  cat > "$shim_dir/python3-config" <<PYCFGEOF
#!/usr/bin/env bash
set -euo pipefail

case "\${1:-}" in
  --extension-suffix)
    exec "$python_bin" -c \
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    ;;
  --includes|--cflags)
    exec "$python_bin" -c \
      "import sysconfig; print('-I' + sysconfig.get_paths()['include'])"
    ;;
  --prefix)
    exec "$python_bin" -c "import sys; print(sys.prefix)"
    ;;
  *)
    echo "KunLunXin CI python3-config shim only supports --extension-suffix, --includes, --cflags, --prefix: \$*" >&2
    exit 1
    ;;
esac
PYCFGEOF
  chmod 0755 "$shim_dir/python3-config"

  export PATH="$shim_dir:$PATH"
  if [ -n "${GITHUB_PATH:-}" ]; then
    printf '%s\n' "$shim_dir" >> "$GITHUB_PATH"
  fi
  ci_export_env PATH "$PATH"
  echo "KunLunXin python3-config extension suffix: $(python3-config --extension-suffix)"
}

install_kunlunxin_functional_dependencies() {
  local pip_index_args=(
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    --timeout 300
    --retries 10
    --no-cache-dir
  )

  # --tensorboard-dir needs SummaryWriter, and the regular pipeline reads the
  # emitted event files back through get_test_results_from_tensorboard_logs.py.
  # Install the frontend package only so torch, numpy, and protobuf pinned by
  # the XPU image stay untouched.
  python3 -m pip install "tensorboard==2.17.1" --no-deps "${pip_index_args[@]}"
  python3 -c \
    "from torch.utils.tensorboard import SummaryWriter; print('KunLunXin functional dependencies validated')"
}

configure_kunlunxin_runtime() {
  # KunLunXin P800 uses XMLIR to expose XPU as a CUDA-compatible device.
  # FlagCx is the collective communication library (KunLunXin's equivalent
  # of NCCL).  TE_FL_SKIP_CUDA tells TransformerEngine-FL not to probe the
  # CUDA vendor backend so it falls through to the kunlunxin vendor path.
  ci_export_env XPU 1
  # DISTRIBUTED_BACKEND is NOT exported: XMLIR's mock_torch intercepts NCCL
  # calls at the C++ layer and redirects them to FlagCX, but PyTorch sees
  # backend='nccl' at the Python layer. This allows PyTorch's Gloo helper
  # process group logic to work correctly. Explicitly setting backend=flagcx
  # causes gather()/gather_object() failures because FlagCX does not fully
  # implement these operations yet (backend_flagcx.cpp:1035 error).
  # ci_export_env DISTRIBUTED_BACKEND flagcx
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env KLX_USE_AUTOTUNE 0
  # TE_FL_PREFER=vendor routes ops to vendor.kunlunxin (hydrax/XDNN tuned
  # kernels, 21 ops registered). However vendor.kunlunxin does not support
  # BF16 in softmax_with_mask (XDNN_PYTORCH error "scalar type of output:
  # kbfloat16 is unsupported"), and the error does not propagate to Python
  # so TE-FL policy cannot fall back to FlagGems. The training hangs silently
  # at the first attention forward. Disable vendor routing until kunlunxin
  # fixes the BF16 support or raises NotImplementedError correctly.
  # ci_export_env TE_FL_PREFER vendor
}

validate_kunlunxin_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

setup_unit_environment() {
  activate_kunlunxin_python_environment
  ci_install_project
  configure_kunlunxin_runtime
  validate_kunlunxin_capacity
}

setup_build_environment() {
  activate_kunlunxin_python_environment
  ci_install_project
  configure_kunlunxin_runtime
  validate_kunlunxin_capacity
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    ci_setup_functional_environment
    install_kunlunxin_python_config_shim
    install_kunlunxin_functional_dependencies
    configure_kunlunxin_runtime
    validate_kunlunxin_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
