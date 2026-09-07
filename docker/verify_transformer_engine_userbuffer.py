"""Verify that the installed TE-FL wheel contains the UserBuffer adapter fix."""

from __future__ import annotations

import sysconfig
from pathlib import Path


_EXPECTED_SNIPPETS = {
    "ops.py": (
        "def device_supports_multicast(self, device_id: int = -1) -> bool:",
        "def ubuf_built_with_mpi(self) -> bool:",
        "def get_stream_priority_range(self, device_id: int = -1) -> Tuple[int, int]:",
    ),
    "backends/vendor/cuda/cuda.py": (
        "return tex.device_supports_multicast(device_id)",
        "return tex.ubuf_built_with_mpi()",
        "return tex.get_stream_priority_range(device_id)",
        "comm_type = tex.CommOverlapType(int(comm_type))",
    ),
    "backends/vendor/cuda/register_ops.py": (
        'op_name="device_supports_multicast"',
        'op_name="ubuf_built_with_mpi"',
        'op_name="get_stream_priority_range"',
    ),
}


def verify(root: Path) -> None:
    missing: list[str] = []
    for relative_path, snippets in _EXPECTED_SNIPPETS.items():
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        missing.extend(
            f"{relative_path}: {snippet}" for snippet in snippets if snippet not in text
        )
    if missing:
        raise RuntimeError("TE-FL UserBuffer adapter check failed:\n" + "\n".join(missing))


if __name__ == "__main__":
    site_packages = Path(sysconfig.get_path("purelib"))
    verify(site_packages / "transformer_engine" / "plugin" / "core")

