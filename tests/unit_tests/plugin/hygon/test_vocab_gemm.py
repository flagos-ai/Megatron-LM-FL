# Copyright (c) 2026, BAAI. All rights reserved.

"""Portable policy/math checks; the core TP/SP autograd protocol is not exercised."""

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch


def load_plugin():
    """Load the plugin with only core decorators stubbed, without GPU initialization."""
    path = Path(__file__).resolve().parents[4] / "megatron/plugin/hygon/tensor_parallel/layers.py"
    spec = importlib.util.spec_from_file_location("hygon_vocab_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    core = ModuleType("megatron.core.tensor_parallel")
    core.layers = SimpleNamespace(custom_fwd=lambda fn: fn, custom_bwd=lambda fn: fn)
    with mock.patch.dict(sys.modules, {"megatron.core.tensor_parallel": core}):
        spec.loader.exec_module(module)
    return module


plugin = load_plugin()


class TestVocabPolicy(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(torch.version, "hip", "test"))
        self.stack.enter_context(
            mock.patch.object(torch.compiler, "is_compiling", return_value=False)
        )
        self.stack.enter_context(
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        self.device_check = self.stack.enter_context(
            mock.patch.object(plugin, "_is_bw1100", return_value=True)
        )
        self.stack.enter_context(mock.patch.dict("os.environ", {"MEGATRON_HYGON_VOCAB_GEMM": "1"}))

    @staticmethod
    def tensors(hidden, rows):
        common = dict(
            dtype=torch.bfloat16, device=torch.device("cuda", 0), is_contiguous=lambda: True
        )
        x = SimpleNamespace(
            **common, shape=(rows, hidden), ndim=2, is_cuda=True, numel=lambda: rows * hidden
        )
        w = SimpleNamespace(**common, shape=(124160, hidden))
        return x, w

    def test_measured_shape_pairs_with_and_without_sp(self):
        for hidden, rows in ((5120, 8192), (5120, 16384), (2048, 32768), (2048, 65536)):
            for sp in (False, True):
                with self.subTest(hidden=hidden, rows=rows, sp=sp):
                    x, w = self.tensors(hidden, rows // 2 if sp else rows)
                    group = SimpleNamespace(size=lambda: 2) if sp else None
                    self.assertTrue(plugin._supports_vocab_gemm(x, w, sp, group))

    def test_unmeasured_cross_product_is_rejected(self):
        for hidden, rows in (
            (5120, 32768),
            (5120, 65536),
            (2048, 8192),
            (2048, 16384),
            (4096, 32768),
        ):
            with self.subTest(hidden=hidden, rows=rows):
                x, w = self.tensors(hidden, rows)
                self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))

    def test_fallback_guards(self):
        x, w = self.tensors(2048, 32768)
        self.assertFalse(plugin._supports_vocab_gemm(x, w, True, None))
        for changes in (
            {"dtype": torch.float32},
            {"is_cuda": False},
            {"is_contiguous": lambda: False},
            {"ndim": 4},
        ):
            with self.subTest(changes=changes):
                changed = SimpleNamespace(**{**vars(x), **changes})
                self.assertFalse(plugin._supports_vocab_gemm(changed, w, False, None))
        w.__fsdp_param__ = object()
        self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))

    def test_disable_compile_capture_and_other_device(self):
        x, w = self.tensors(2048, 32768)
        with mock.patch.dict("os.environ", {"MEGATRON_HYGON_VOCAB_GEMM": "0"}):
            self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))
        self.device_check.return_value = False
        self.assertFalse(plugin._supports_vocab_gemm(x, w, False, None))

    def test_helpers_preserve_tail_leading_dimensions_and_dgrad(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(3, 2, 8, dtype=torch.float64, generator=generator)
        w = torch.randn(16389, 8, dtype=torch.float64, generator=generator)
        dy = torch.randn(3, 16389, 2, dtype=torch.float64, generator=generator).transpose(1, 2)
        with torch.no_grad():
            torch.testing.assert_close(plugin._vocab_forward(x, w), x @ w.T)
            torch.testing.assert_close(plugin._vocab_dgrad(dy, w), dy @ w)


if __name__ == "__main__":
    unittest.main()
