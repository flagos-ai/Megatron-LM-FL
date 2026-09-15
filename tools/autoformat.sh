#!/bin/bash
set -euox pipefail

GIT_VERSION=$(git version | awk '{print $3}')
GIT_MAJOR=$(echo $GIT_VERSION | awk -F. '{print $1}')
GIT_MINOR=$(echo $GIT_VERSION | awk -F. '{print $2}')

if [[ $GIT_MAJOR -eq 2 && $GIT_MINOR -lt 31 ]]; then
    echo "Git version must be at least 2.31.0. Found $GIT_VERSION"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CHECK_ONLY=${CHECK_ONLY:-false}
SKIP_DOCS=${SKIP_DOCS:-false}

BASE_REF=${BASE_REF:-main}
git remote add autoformatter-remote "https://github.com/flagos-ai/Megatron-LM-FL.git" || true
git fetch autoformatter-remote ${BASE_REF}
CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base autoformatter-remote/${BASE_REF} megatron/core tests/ | grep '\.py$' || true)
MEGALENS_CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base autoformatter-remote/${BASE_REF} megatron/megalens/ | grep '\.py$' || true)
FORMAT_FILES="$CHANGED_FILES $MEGALENS_CHANGED_FILES"
ADDITIONAL_ARGS=""
ADDITIONAL_BLACK_ARGS=""
ADDITIONAL_PYLINT_ARGS=""
ADDITIONAL_RUFF_ARGS=""

if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_ARGS="--check"
    ADDITIONAL_BLACK_ARGS="--diff"
    ADDITIONAL_RUFF_ARGS="--no-fix"
else
    ADDITIONAL_RUFF_ARGS="--fix"
fi

if [[ $SKIP_DOCS == true ]]; then
    ADDITIONAL_PYLINT_ARGS="--disable=C0115,C0116"
fi

if [[ -n "$FORMAT_FILES" ]]; then
    black --skip-magic-trailing-comma --skip-string-normalization $ADDITIONAL_ARGS $ADDITIONAL_BLACK_ARGS --verbose $FORMAT_FILES
    isort $ADDITIONAL_ARGS $FORMAT_FILES
    ruff check $ADDITIONAL_RUFF_ARGS $FORMAT_FILES
    if [[ -n "$CHANGED_FILES" ]]; then
        pylint $ADDITIONAL_PYLINT_ARGS $CHANGED_FILES
        mypy --explicit-package-bases --follow-imports=skip $CHANGED_FILES || true
    fi
else
    echo Changeset is empty, all good.
fi
