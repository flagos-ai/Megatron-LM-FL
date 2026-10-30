# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check whether the selected PyTorch build provides ring exchange."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def describe_capability(torch_module: Any = torch) -> tuple[bool, str]:
    distributed = getattr(torch_module, "distributed", None)
    available = callable(getattr(distributed, "ring_exchange", None))
    module_path = Path(torch_module.__file__).resolve()
    status = "available" if available else "unavailable"
    detail = (
        f"ring_exchange={status} "
        f"torch={torch_module.__version__} "
        f"torch_path={module_path}"
    )
    return available, detail


def main() -> int:
    available, detail = describe_capability()
    print(detail, file=sys.stdout if available else sys.stderr)
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
