"""Static contracts for the reproducible FlagOS work image."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "docker" / "Dockerfile.work"
_WHEEL_MANIFEST = _ROOT / "docker" / "work-wheels.sha256"
_FLAGSCALE_PATCH = _ROOT / "docker" / "patches" / "flagscale-megalens.patch"
_USERBUFFER_PATCH = (
    _ROOT
    / "docker"
    / "patches"
    / "transformer-engine-fl-v0.2.0-userbuffer-ops.patch"
)
_VERIFY_SCRIPT = _ROOT / "docker" / "verify_transformer_engine_userbuffer.py"

_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_transformer_engine_userbuffer", _VERIFY_SCRIPT
)
assert _VERIFY_SPEC is not None and _VERIFY_SPEC.loader is not None
_VERIFY_MODULE = importlib.util.module_from_spec(_VERIFY_SPEC)
_VERIFY_SPEC.loader.exec_module(_VERIFY_MODULE)


def _docker_arg(text: str, name: str) -> str:
    match = re.search(rf"^ARG {re.escape(name)}=([0-9a-f]+)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1)


def test_userbuffer_patch_is_pinned_by_the_work_image() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    patch_bytes = _USERBUFFER_PATCH.read_bytes()
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()

    assert _docker_arg(dockerfile, "TE_FL_USERBUFFER_PATCH_SHA256") == patch_sha256
    assert (
        '"${TE_FL_USERBUFFER_PATCH_SHA256}" '
        '"${PROJECT_ROOT}/docker/patches/'
        'transformer-engine-fl-v0.2.0-userbuffer-ops.patch"'
        in dockerfile
    )
    assert (
        'org.flagos.transformer-engine-fl.userbuffer.patch.sha256='
        '"${TE_FL_USERBUFFER_PATCH_SHA256}"'
        in dockerfile
    )
    assert '"transformer_engine_fl_userbuffer_patch_sha256": "%s"' in dockerfile
    assert 'docker/verify_transformer_engine_userbuffer.py' in dockerfile


def test_flagscale_patch_preserves_graph_compatibility_and_is_pinned() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    patch_bytes = _FLAGSCALE_PATCH.read_bytes()
    patch = patch_bytes.decode()
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()

    assert dockerfile.count(f'"{patch_sha256}"') == 2
    assert "requires --trace-cupti-kernels=off until the GPU gate" not in patch
    assert "Python scopes inside managed capture/replay are" in patch
    assert (
        "broadcast_packed_sequence_metadata = "
        "args.hybrid_context_parallel or args.sft"
    ) in patch
    assert patch.count("if broadcast_packed_sequence_metadata:") == 5
    assert (
        '"f4a31581bd93b6e53500e520a72312adf2f069e23634d176c97d707a0558c45c" '
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/utils.py"'
    ) in dockerfile
    assert (
        '"e5e6166aca7a7ac7db62e261217cc5e8eb6bdc829d5637c4f5f8cf2094968cf9" '
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/utils.py"'
    ) in dockerfile
    assert dockerfile.count(
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/utils.py"'
    ) == 3


def test_flagscale_range_skip_traces_only_the_actual_training_iteration() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    patch = _FLAGSCALE_PATCH.read_text(encoding="utf-8")

    ordered_snippets = (
        "if args.trace and args.skip_samples_range and args.rampup_batch_size is not None:",
        "if args.skip_samples_range:",
        "args.consumed_train_samples + current_global_batch_size",
        "elif args.skip_iters_range:",
        "args.skip_iters_range[0] <= iteration < args.skip_iters_range[1]",
        "if trace_iteration and not range_skip_active:",
        "if args.use_pytorch_profiler:",
        "while iteration >= start_skip_iteration and iteration < end_skip_iteration:",
        "if trace_iteration and range_skip_active:",
        ") = train_step(",
    )

    added_lines = "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    cursor = 0
    for snippet in ordered_snippets:
        cursor = added_lines.index(snippet, cursor) + len(snippet)
    assert "MegaLens tracing does not support --skip-samples-range" in patch
    assert "if trace_iteration and not range_skip_enabled:" not in patch
    assert (
        '"c79538410f1ada613d5230c0f7e2aa2dc1be3a6d11edde0cb90aab9fbe0ec153" '
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/training.py"'
        in dockerfile
    )


def test_work_image_wheel_manifest_matches_the_docker_argument() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    wheel_sha256 = _docker_arg(dockerfile, "TE_FL_WHEEL_SHA256")
    manifest = dict(
        line.split(maxsplit=1)[::-1]
        for line in _WHEEL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    assert (
        manifest["transformer_engine-2.14.0-cp312-cp312-linux_x86_64.whl"]
        == wheel_sha256
    )


def test_installed_userbuffer_adapter_verifier_checks_every_required_snippet(
    tmp_path: Path,
) -> None:
    for relative_path, snippets in _VERIFY_MODULE._EXPECTED_SNIPPETS.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(snippets), encoding="utf-8")

    _VERIFY_MODULE.verify(tmp_path)

    cuda_path = tmp_path / "backends" / "vendor" / "cuda" / "cuda.py"
    cuda_path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CommOverlapType"):
        _VERIFY_MODULE.verify(tmp_path)
