"""Build a standalone wheel from the reviewed, committed FL plugin sources.

No source rewriting and no files under the installed `megatron` namespace.
Internal imports in the canonical plugin must already be relative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


FILES = (
    "NOTICE", "__init__.py", "checkpoint.py", "compat.py", "config.py", "distributed.py",
    "engine.py", "gpu_gradient_pipeline.py", "gpu_reduce.py", "kernels.py", "layout.py",
    "legacy_lce.py", "layer_update_pipeline.py", "session.py", "transport.py", "shard_h2d.py", "sharded_cpu_optimizer.py",
    "layer_adam/__init__.py", "layer_adam/builder.py", "layer_adam/layer_adam.py",
    "layer_adam/csrc/cpu_adam.cpp", "layer_adam/csrc/cpu_adam.h",
    "layer_adam/csrc/cpu_adam_impl.cpp", "layer_adam/csrc/simd.h",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path, help="New build directory")
    args = parser.parse_args()

    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(args.source), *cmd])

    if git("status", "--porcelain").strip():
        raise RuntimeError("Commit/preserve source changes before packaging")
    revision = git("rev-parse", "HEAD").decode().strip()
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output / "source"
    package = stage / "src" / "slideformer_megatron"
    package.mkdir(parents=True)
    manifest = {"source_commit": revision, "files": {}, "namespace": "slideformer_megatron"}
    prefix = "megatron/plugin/slideformer/"
    for name in FILES:
        content = git("show", f"{revision}:{prefix}{name}")
        if name.endswith(".py") and b"megatron.plugin" in content:
            raise RuntimeError(f"FL namespace dependency in {name}")
        for forbidden in (b"oom_optimizer", b"from osmoe", b"import osmoe", b"expert_scheduler", b"policy_search"):
            if forbidden in content.lower():
                raise RuntimeError(f"Private source boundary requires review: {name}")
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][name] = hashlib.sha256(content).hexdigest()
    (stage / "LICENSE").write_bytes(git("show", f"{revision}:LICENSE"))
    (stage / "NOTICE").write_bytes((package / "NOTICE").read_bytes())
    (package / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "pyproject.toml").write_text(f'''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "slideformer-megatron"
version = "0.1.0.dev0+{revision[:12]}"
description = "SlideFormer full fine-tuning offload for native Megatron Core models"
requires-python = ">=3.10"
dependencies = ["torch>=2.6", "packaging", "ninja"]
license = {{file = "LICENSE"}}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
slideformer_megatron = ["NOTICE", "source_manifest.json", "layer_adam/csrc/*.cpp", "layer_adam/csrc/*.h"]
''')
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(args.output.resolve() / "wheel"), str(stage.resolve())], check=True)
    manifest["wheels"] = {
        wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()
        for wheel in (args.output / "wheel").glob("*.whl")
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "commit": revision, "wheels": manifest["wheels"]}))


if __name__ == "__main__":
    main()
