#!/usr/bin/env bash

# Shared, platform-neutral helpers for CI environment setup scripts.
set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_PROJECT_ROOT="$(cd "$CI_SETUP_DIR/../.." && pwd)"

ci_require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "::error::Required environment variable is empty: $name"
    exit 1
  fi
}

ci_export_env() {
  local name="$1"
  local value="$2"

  export "$name=$value"
  if [ -n "${GITHUB_ENV:-}" ]; then
    printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
  fi
}

ci_activate_python_environment() {
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
  fi

  ci_export_env PATH "$PATH"
  echo "Python: $(command -v python3) ($(python3 --version 2>&1))"
}

ci_ensure_curl() {
  if command -v curl >/dev/null 2>&1; then
    command -v curl
    return
  fi

  apt-get update -qq
  apt-get install -y --no-install-recommends curl
}

ci_install_yq() {
  if command -v yq >/dev/null 2>&1; then
    yq --version
    return
  fi

  if ! command -v wget >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends wget
  fi

  local architecture
  case "$(uname -m)" in
    x86_64|amd64) architecture=amd64 ;;
    aarch64|arm64) architecture=arm64 ;;
    *)
      echo "::error::Unsupported architecture for yq: $(uname -m)"
      exit 1
      ;;
  esac

  wget -qO /usr/local/bin/yq \
    "https://github.com/mikefarah/yq/releases/download/v4.45.1/yq_linux_${architecture}"
  chmod 0755 /usr/local/bin/yq
  yq --version
}

ci_install_envsubst() {
  if command -v envsubst >/dev/null 2>&1; then
    return
  fi

  if apt-get update -qq && apt-get install -y --no-install-recommends gettext-base; then
    return
  fi
  if command -v conda >/dev/null 2>&1 && conda install -y -q gettext; then
    return
  fi

  cat > /usr/local/bin/envsubst <<'ENVEOF'
#!/usr/bin/env python3
import os
import re
import sys

text = sys.stdin.read()
print(
    re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)",
        lambda match: os.environ.get(
            match.group(1) or match.group(2), match.group(0)
        ),
        text,
    ),
    end="",
)
ENVEOF
  chmod 0755 /usr/local/bin/envsubst
}

ci_install_uv_compatibility_shim() {
  if command -v uv >/dev/null 2>&1; then
    uv --version
    return
  fi

  local python_bin
  python_bin=$(command -v python3)
  cat > /usr/local/bin/uv <<UVEOF
#!/usr/bin/env bash
set -euo pipefail
if [ "\${1:-}" = "run" ]; then
  shift
  if [ "\${1:-}" = "--no-sync" ]; then
    shift
  fi
  if [ "\${1:-}" = "python" ]; then
    shift
    exec "$python_bin" "\$@"
  fi
  if [ "\${1:-}" = "pytest" ] && ! command -v pytest >/dev/null 2>&1; then
    shift
    exec "$python_bin" -m pytest "\$@"
  fi
  exec "\$@"
fi
echo "uv shim: unsupported command: \$*" >&2
exit 1
UVEOF
  chmod 0755 /usr/local/bin/uv
}

ci_setup_functional_environment() {
  ci_activate_python_environment

  # Dataset build helpers call python3 directly.
  local python_bin
  python_bin=$(command -v python3)
  if [ "$python_bin" != "/usr/bin/python3" ] && \
     [ "$python_bin" != "/usr/local/bin/python3" ]; then
    ln -sf "$python_bin" /usr/local/bin/python3
  fi

  python3 -c "import torch; print('Torch:', torch.__version__)"
  ci_install_yq
  ci_install_envsubst
  ci_install_uv_compatibility_shim
  python3 -m pip install pybind11 --no-cache-dir
  ci_install_project
}

ci_validate_device_capacity() {
  local available="$1"

  ci_require_env CI_NPROC_PER_NODE
  if ! [[ "$available" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::Invalid device count: '$available'"
    exit 1
  fi
  if [ "$available" -lt "$CI_NPROC_PER_NODE" ]; then
    echo "::error::Configured for $CI_NPROC_PER_NODE processes, but only $available devices are available"
    exit 1
  fi

  echo "Available devices: $available; distributed processes: $CI_NPROC_PER_NODE"
}

ci_install_project() {
  cd "$CI_PROJECT_ROOT"
  python3 -m pip install -e . --no-deps --no-build-isolation --no-cache-dir
}
