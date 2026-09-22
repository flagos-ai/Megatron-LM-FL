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

ci_apply_env_json() {
  local environment_json="$1"
  local python_bin="${CI_PYTHON_BIN:-python3}"
  local entries
  local name
  local value

  if ! entries=$("$python_bin" - "$environment_json" <<'PY'
import json
import re
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, dict) or not values:
    raise SystemExit("environment must be a non-empty JSON object")
for key, value in values.items():
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"invalid environment variable name: {key!r}")
    if not isinstance(value, (str, int, float, bool)):
        raise SystemExit(f"environment value must be scalar: {key}")
    text = str(value)
    if any(character in text for character in ("\n", "\r", "\t")):
        raise SystemExit(f"environment value contains unsupported control characters: {key}")
    print(f"__CI_ENV__\t{key}\t{text}")
PY
  ); then
    echo "::error::Invalid environment JSON" >&2
    return 1
  fi

  while IFS=$'\t' read -r record_type name value; do
    if [ "$record_type" != "__CI_ENV__" ]; then
      [ -z "$record_type" ] || printf '%s\n' "$record_type" >&2
      continue
    fi
    [ -n "$name" ] || continue
    ci_export_env "$name" "$value"
  done <<< "$entries"
}

ci_activate_python_environment() {
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
  fi

  local python_bin
  python_bin=$(command -v python3)
  ci_export_env PATH "$PATH"
  ci_export_env CI_PYTHON_BIN "$python_bin"
  echo "Python: $python_bin ($($python_bin --version 2>&1))"
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

ci_install_runtime_packages() {
  local python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
  local packages_json="${CI_RUNTIME_PIP_PACKAGES_JSON:-[]}"
  local install_args_json="${CI_RUNTIME_PIP_INSTALL_ARGS_JSON:-[]}"
  local parsed
  local kind
  local value
  local -a packages=()
  local -a install_args=()

  if ! parsed=$("$python_bin" - "$packages_json" "$install_args_json" <<'PY'
import json
import sys

packages = json.loads(sys.argv[1])
install_args = json.loads(sys.argv[2])
def valid_value(item):
    return isinstance(item, str) and item and not any(
        character in item for character in ("\n", "\r", "\t")
    )

if not isinstance(packages, list) or not all(valid_value(item) for item in packages):
    raise SystemExit("runtime pip packages must be a JSON string array")
if not isinstance(install_args, list) or not all(valid_value(item) for item in install_args):
    raise SystemExit("runtime pip install args must be a JSON string array")

for item in packages:
    print(f"__CI_RUNTIME_PIP__\tpackage\t{item}")
for item in install_args:
    print(f"__CI_RUNTIME_PIP__\targ\t{item}")
PY
  ); then
    echo "::error::Invalid runtime pip package configuration" >&2
    return 1
  fi

  # Filter out only lines that start with our marker to avoid pollution from
  # vendor Python startup messages (e.g., torch_gcu auto-injection warnings).
  local filtered_lines
  filtered_lines=$(grep '^__CI_RUNTIME_PIP__' <<< "$parsed" || true)

  while IFS=$'\t' read -r record_type kind value; do
    case "$record_type" in
      __CI_RUNTIME_PIP__)
        case "$kind" in
          package) packages+=("$value") ;;
          arg) install_args+=("$value") ;;
          *) echo "::error::Invalid runtime pip package record: $kind" >&2; return 1 ;;
        esac
        ;;
      '') ;;
      *)
        # Should never reach here after grep filtering
        echo "::warning::Unexpected line in runtime pip config: $record_type" >&2
        ;;
    esac
  done <<< "$filtered_lines"

  if [ "${#packages[@]}" -eq 0 ]; then
    echo "Configured runtime pip packages: none"
    return 0
  fi

  echo "Installing configured runtime pip packages: ${packages[*]}"
  "$python_bin" -m pip install --no-cache-dir "${install_args[@]}" "${packages[@]}"
}

ci_install_uv_compatibility_shim() {
  local force_shim="${1:-false}"

  if [ "$force_shim" != "true" ] && command -v uv >/dev/null 2>&1; then
    uv --version
    return
  fi

  local python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
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
  local python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
  if [ "$python_bin" != "/usr/bin/python3" ] && \
     [ "$python_bin" != "/usr/local/bin/python3" ]; then
    ln -sf "$python_bin" /usr/local/bin/python3
  fi

  "$python_bin" -c "import torch; print('Torch:', torch.__version__)"
  ci_install_yq
  ci_install_envsubst
  ci_install_uv_compatibility_shim
  "$python_bin" -m pip install pybind11 --no-cache-dir
  ci_install_project "$@"
}

ci_install_local_tokenizer_dependencies() {
  # Transformers >= 4.47 validates mounted local tokenizer paths as Hub repo
  # ids. Keep the functional images on the version that accepts local paths.
  python3 -m pip install \
    "transformers<4.47.0" \
    "huggingface_hub<0.27.0" \
    --no-cache-dir --quiet
}

ci_validate_qwen_assets() {
  local data_root="${1:-/home/gitlab-runner/data}"
  local tokenizer_root="${2:-/home/gitlab-runner/tokenizers}"
  local data_prefix="$data_root/pile_wikipedia_demo/pile_wikipedia_demo"
  local tokenizer_path="$tokenizer_root/qwentokenizer"
  local -a required_paths=(
    "${data_prefix}.bin"
    "${data_prefix}.idx"
    "$tokenizer_path/tokenizer_config.json"
    "$tokenizer_path/tokenization_qwen.py"
    "$tokenizer_path/qwen.tiktoken"
  )
  local -a missing_paths=()
  local path

  for path in "${required_paths[@]}"; do
    if [ ! -f "$path" ]; then
      missing_paths+=("$path")
    fi
  done

  if [ "${#missing_paths[@]}" -ne 0 ]; then
    echo "::error::Functional assets are missing inside the container"
    for path in "${missing_paths[@]}"; do
      echo "  missing: $path"
    done
    return 1
  fi

  echo "Functional assets validated"
  echo "  dataset: $data_prefix"
  echo "  tokenizer: $tokenizer_path"
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
  "${CI_PYTHON_BIN:-$(command -v python3)}" -m pip install -e . --no-deps --no-build-isolation --no-cache-dir "$@"
}
