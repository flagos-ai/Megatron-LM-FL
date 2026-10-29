# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import subprocess
import sys
import textwrap


def test_core_imports_without_triton_and_paged_stash_fails_before_allocation():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from unittest.mock import patch

        class BlockTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    raise ModuleNotFoundError("Triton intentionally unavailable", name=fullname)
                return None

        sys.meta_path.insert(0, BlockTriton())
        import megatron.core.pipeline_parallel.schedules
        from megatron.core.transformer.moe import paged_stash

        assert not paged_stash.HAVE_TRITON
        with patch.object(paged_stash.torch, "empty", side_effect=AssertionError("allocated buffer")):
            try:
                paged_stash.PagedStashBuffer(
                    num_tokens=4, hidden_size=8, page_size=2, device="cpu",
                    overflow=None, host_spill=None, dtype=None,
                )
            except ImportError as error:
                assert str(error) == "MoE paged stash requires Triton"
            else:
                raise AssertionError("missing Triton must reject paged stash")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
