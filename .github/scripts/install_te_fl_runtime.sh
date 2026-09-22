#!/usr/bin/env bash

set -euo pipefail

: "${TE_FL_WHEEL_DIR:?TE_FL_WHEEL_DIR is required}"

python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
if [ ! -x "$python_bin" ]; then
  echo "::error::TE-FL Python executable not found: $python_bin" >&2
  exit 1
fi

if [ ! -d "$TE_FL_WHEEL_DIR" ]; then
  echo "::error::TE-FL wheel cache directory is missing: $TE_FL_WHEEL_DIR" >&2
  exit 1
fi

mapfile -t wheels < <(
  find "$TE_FL_WHEEL_DIR" -maxdepth 1 -type f -name 'transformer_engine*.whl' -print | sort
)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "::error::Expected exactly one TE-FL wheel, found ${#wheels[@]}" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

install_pip_args=()
install_pip_args_json="${TE_FL_INSTALL_PIP_ARGS_JSON:-[]}"
parsed_install_pip_args=''
if ! parsed_install_pip_args=$("$python_bin" - "$install_pip_args_json" <<'PY'
import json
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(
    isinstance(value, str)
    and value
    and not any(character in value for character in ("\n", "\r", "\t"))
    for value in values
):
    raise SystemExit("TE-FL install_pip_args must be a JSON string array")
for value in values:
    print(f"__CI_TE_FL_ARG__\t{value}")
PY
); then
  echo "::error::Invalid TE-FL pip install argument configuration" >&2
  exit 1
fi

while IFS=$'\t' read -r record_type arg; do
  case "$record_type" in
    __CI_TE_FL_ARG__)
      install_pip_args+=("$arg")
      ;;
    '')
      ;;
    *)
      # Vendor Python runtimes may print startup messages to stdout.
      ;;
  esac
done <<< "$parsed_install_pip_args"

# The CUDA image may already contain NVIDIA TE metadata and an extension. Remove
# it before installing the wheel so Python cannot resolve a mixed TE runtime.
"$python_bin" -m pip uninstall -y \
  transformer-engine transformer-engine-torch \
  transformer-engine-cu11 transformer-engine-cu12 transformer-engine-cu13 \
  >/dev/null 2>&1 || true

"$python_bin" -m pip install \
  --force-reinstall \
  --no-deps \
  --no-cache-dir \
  "${install_pip_args[@]}" \
  "${wheels[0]}"

"$python_bin" - <<'PY'
import sys
import transformer_engine

print(f"TE-FL Python: {sys.executable}")
print(f"TE-FL wheel import passed: {transformer_engine.__file__}")
PY
