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
    assert "trace mode 0 requires --trace-cupti-kernels=off" not in patch
    assert "and args.continuous_trace_iterations != 1" not in patch
    assert "args.trace_mode != 0 and args.trace_cupti_kernels != 'off'" in patch
    assert "Python scopes inside managed capture/replay are" in patch
    assert (
        "broadcast_packed_sequence_metadata = "
        "args.hybrid_context_parallel or args.sft"
    ) in patch
    assert patch.count("if broadcast_packed_sequence_metadata:") == 5
    assert (
        '"8ae41c3a9397d48b056c7dd4373561c64e82ef7e43ba92a70377dd61d39a63b9" '
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/arguments.py"'
    ) in dockerfile
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


def test_flagscale_archive_source_contract_matches_enforced_checks() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    assert '#   CONTEXT="$(mktemp -d /tmp/flagos-work-context.XXXXXXXX)"' in dockerfile
    assert "#   DOCKER_BUILDKIT=0 docker build --network=none" in dockerfile
    assert '#   docker image inspect "${BASE_IMAGE}" >/dev/null' in dockerfile
    assert "EMERGING_OPTIMIZERS_ROOT" not in dockerfile
    assert "EMERGING_OPTIMIZERS_SOURCE_DATE_EPOCH" not in dockerfile
    assert "uv build" not in dockerfile
    assert (
        "#   EMERGING_OPTIMIZERS_REVISION="
        "d5363b4a418128cd8111983b191c4b8869a9766b"
        in dockerfile
    )
    assert (
        "#   EMERGING_OPTIMIZERS_WHEEL_SHA256="
        "03c111935f484bb89ca6e1954448130d926b0d3568832fd7181547680837640a"
        in dockerfile
    )
    assert '#   cp "${EMERGING_OPTIMIZERS_WHEEL}" "${CONTEXT}/wheels/"' in dockerfile
    assert (
        "#   FLAGSCALE_REVISION=6d775cd01d5c822f9413b9952a81652c917d273e"
        in dockerfile
    )
    assert (
        "#   FLAGSCALE_TREE=198de8867fee8bce6e140d100910c7899ab84690"
        in dockerfile
    )
    assert 'FLAGSCALE_REVISION="$(git -C "${FLAGSCALE_ROOT}" rev-parse HEAD)"' not in dockerfile
    flagscale_run = dockerfile.split("# FlagScale's runner", 1)[1].split(
        "# Build stages do not receive GPUs", 1
    )[0]

    git_guard = re.search(
        r"if git -C .*?; then \\\n(?P<body>.*?)\n    fi \\",
        flagscale_run,
        re.DOTALL,
    )
    assert git_guard is not None
    assert '"${FLAGSCALE_REVISION}"' in git_guard.group("body")
    assert '"${FLAGSCALE_TREE}"' in git_guard.group("body")

    before_patch, after_patch = flagscale_run.split(
        'git -C "${FLAGSCALE_ROOT}" apply /tmp/flagscale-megalens.patch', 1
    )
    file_pattern = (
        r'"[0-9a-f]{64}" "\$\{FLAGSCALE_ROOT\}/flagscale/train/'
        r'megatron/training/([^"/]+)"'
    )
    expected_files = {"arguments.py", "global_vars.py", "training.py", "utils.py"}
    assert set(re.findall(file_pattern, before_patch)) == expected_files
    assert set(re.findall(file_pattern, after_patch)) == expected_files
    assert "| sha256sum -c -" in before_patch
    assert "| sha256sum -c -" in after_patch


def test_flagscale_range_skip_traces_only_the_actual_training_iteration() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    patch = _FLAGSCALE_PATCH.read_text(encoding="utf-8")

    ordered_snippets = (
        "if args.trace and args.skip_samples_range and args.rampup_batch_size is not None:",
        "if args.skip_samples_range:",
        "args.consumed_train_samples + current_global_batch_size",
        "elif args.skip_iters_range:",
        "args.skip_iters_range[0] <= iteration < args.skip_iters_range[1]",
        "get_megalens_runtime() if getattr(args, 'trace', False) else None",
        "if megalens_runtime is not None and (skip_iteration or range_skip_active):",
        "megalens_runtime.tracer.close_trace_window()",
        "with ExitStack() as iteration_stack:",
        "if trace_iteration and not range_skip_active:",
        "if args.use_pytorch_profiler:",
        "if skip_iteration:",
        "dummy_train_step(train_data_iterator)",
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
        '"23ab0aa9c33b6c56972bc7299c51634fc3826d499bf4cd66695404d962bec883" '
        '"${FLAGSCALE_ROOT}/flagscale/train/megatron/training/training.py"'
        in dockerfile
    )


def test_flagscale_controlled_exit_flushes_megalens_before_system_exit() -> None:
    patch = _FLAGSCALE_PATCH.read_text(encoding="utf-8")
    assert (
        "+        shutdown_megalens_runtime(graceful=True)\n"
        "         sys.exit(exit_code)"
        in patch
    )


def test_work_image_artifact_manifest_matches_the_docker_arguments() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    manifest = dict(
        line.split(maxsplit=1)[::-1]
        for line in _WHEEL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    expected = {
        "transformer_engine-2.14.0-cp312-cp312-linux_x86_64.whl": (
            "TE_FL_WHEEL_SHA256"
        ),
        "flag_gems-5.0.2-py3-none-any.whl": "FLAG_GEMS_WHEEL_SHA256",
        "flagcx-0.13.0-cp312-cp312-linux_x86_64.whl": "FLAGCX_WHEEL_SHA256",
        "libflagcx.so": "FLAGCX_LIBRARY_SHA256",
    }
    assert {name: manifest[name] for name in expected} == {
        name: _docker_arg(dockerfile, argument) for name, argument in expected.items()
    }
    pip_installs = dockerfile.split("python -m pip install")[1:]
    assert len(pip_installs) == 6
    assert "ENV PIP_NO_INDEX=1" in dockerfile
    assert all("--no-index" in install.split("&&", 1)[0] for install in pip_installs)
    assert dockerfile.index("sha256sum -c /tmp/flagos-work-wheels.sha256") < (
        dockerfile.index("python -m pip install")
    )

    assert "ARG FLAGCX_SOURCE_TAG=v0.13.0" in dockerfile
    assert (
        _docker_arg(dockerfile, "FLAGCX_SOURCE_REVISION")
        == "0beba7aa7e76cc0885a9929f1975857bcd310d5e"
    )
    assert (
        _docker_arg(dockerfile, "FLAGCX_SOURCE_ARCHIVE_SHA256")
        == "9186f5fcfcbaceac834812e35423072ea45a8579c362c7e0cb51c3ae645000aa"
    )
    assert '"${FLAGCX_SOURCE_ARCHIVE}" | sha256sum -c -' in dockerfile
    assert (
        'install -m 0755 /tmp/flagos-wheels/libflagcx.so '
        '"${FLAGOS_ENV}/lib/libflagcx.so"'
        in dockerfile
    )
    assert (
        "libnccl.so.2 => ${FLAGOS_ENV}/lib/python3.12/site-packages/"
        "nvidia/nccl/lib/libnccl.so.2"
        in dockerfile
    )
    assert (
        'expected = {"flag-gems": "5.0.2", "flagcx": "0.13.0", '
        '"greenlet": "3.5.4", "grpcio": "1.83.0", "Markdown": "3.10.3", '
        '"packaging": "26.0", "PyYAML": "6.0.3", "SQLAlchemy": "2.0.48", '
        '"tensorboard": "2.20.0", "tensorboard-data-server": "0.7.2", '
        '"nvidia-nccl-cu12": "2.27.5"}'
        in dockerfile
    )
    for dependency in (
        "absl_py-2.4.0-py3-none-any.whl",
        "greenlet-3.5.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "grpcio-1.83.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "markdown-3.10.3-py3-none-any.whl",
        "nvidia_ml_py-13.595.45-py3-none-any.whl",
        "packaging-26.0-py3-none-any.whl",
        "pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "sqlalchemy-2.0.48-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "tensorboard-2.20.0-py3-none-any.whl",
        "tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl",
    ):
        assert dependency in manifest
        assert f"/tmp/flagos-wheels/{dependency}" in dockerfile
    for label in (
        "org.flagos.flag-gems.wheel.sha256",
        "org.flagos.flagcx.source.tag",
        "org.flagos.flagcx.source.revision",
        "org.flagos.flagcx.source.archive.sha256",
        "org.flagos.flagcx.wheel.sha256",
        "org.flagos.flagcx.library.sha256",
    ):
        assert label in dockerfile
    for field in (
        "flag_gems_wheel_sha256",
        "flagcx_source_tag",
        "flagcx_source_revision",
        "flagcx_source_archive_sha256",
        "flagcx_wheel_sha256",
        "flagcx_library_sha256",
    ):
        assert f'"{field}": "%s"' in dockerfile

    final_check = dockerfile.split(
        "# Build stages do not receive GPUs", 1
    )[1]
    assert dockerfile.count("export TORCH_DEVICE_BACKEND_AUTOLOAD=0") == 1
    assert final_check.index(
        "export TORCH_DEVICE_BACKEND_AUTOLOAD=0"
    ) < final_check.index("python -c")
    assert "ENV TORCH_DEVICE_BACKEND_AUTOLOAD" not in dockerfile


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
