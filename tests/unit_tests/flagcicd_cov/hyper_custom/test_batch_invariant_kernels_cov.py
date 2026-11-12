# Copyright (c) 2026, BAAI. All rights reserved.
"""Batch-invariant Triton kernels and aten overrides: matmul, log-softmax, mean, mode nesting."""

import math
import types

import pytest
import torch

import megatron.core.transformer.custom_layers.batch_invariant_kernels as bik


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    # The flagcicd_cov files share one process; never inherit or leak the aten overrides.
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()
    yield
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()


@pytest.mark.parametrize(
    "args,bytes_per_elem",
    [
        pytest.param(
            {"M": 64, "N": 128, "K": 32, "c_ptr": torch.empty(3, dtype=torch.float16)},
            2,
            id="c_ptr_fp16",
        ),
        pytest.param(
            {"M": 64, "N": 128, "K": 32, "c_ptr": torch.empty(3, dtype=torch.float32)},
            4,
            id="c_ptr_fp32",
        ),
        pytest.param({"M": 16, "N": 8, "K": 4, "FP8_OUTPUT": True}, 1, id="fp8_output"),
        pytest.param({"M": 16, "N": 8, "K": 4, "FP8_OUTPUT": False}, 2, id="fp16_output"),
    ],
)
def test_matmul_launch_metadata_reports_name_flops_and_bytes(args, bytes_per_elem):
    kernel = types.SimpleNamespace(name="matmul_kernel_persistent")
    metadata = bik._matmul_launch_metadata(lambda meta: None, kernel, args)
    m, n, k = args["M"], args["N"], args["K"]
    assert metadata["name"] == f"matmul_kernel_persistent [M={m}, N={n}, K={k}]"
    assert metadata[f"flops{bytes_per_elem * 8}"] == 2.0 * m * n * k
    assert metadata["bytes"] == bytes_per_elem * (m * k + n * k + m * n)
    assert set(metadata) == {"name", f"flops{bytes_per_elem * 8}", "bytes"}


def test_matmul_launch_metadata_appends_tiles_per_update_when_present():
    args = {
        "M": 64,
        "N": 128,
        "K": 32,
        "c_ptr": torch.empty(3, dtype=torch.float16),
        "tiles_per_update": 5,
    }
    kernel = types.SimpleNamespace(name="matmul_kernel_persistent")
    metadata = bik._matmul_launch_metadata(lambda meta: None, kernel, args)
    assert metadata["name"] == "matmul_kernel_persistent [M=64, N=128, K=32, tiles_per_update=05]"
    assert metadata["flops16"] == 2.0 * 64 * 128 * 32
    assert metadata["bytes"] == 2 * (64 * 32 + 128 * 32 + 64 * 128)


def test_get_batch_invariant_attention_block_size_returns_16_by_16_tiling():
    block = bik.get_batch_invariant_attention_block_size()
    assert block.block_m == 16
    assert block.block_n == 16


def test_import_module_if_available_returns_none_for_missing_module():
    assert bik._import_module_if_available("megatron.core.not_a_real_module_cov") is None
    assert bik._import_module_if_available("math") is math


def test_log_softmax_rejects_reduction_over_non_last_dimension():
    with pytest.raises(ValueError, match="only supports log_softmax along the last dimension"):
        bik.log_softmax(torch.randn(2, 3, 5), dim=0)


def test_addmm_batch_invariant_matches_torch_reference_on_cuda():
    torch.manual_seed(1234)
    a = torch.randn(9, 6, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(6, 7, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(7, device="cuda", dtype=torch.bfloat16)

    with bik.set_batch_invariant_mode(True):
        out = torch.addmm(bias, a, b)

    torch.testing.assert_close(
        out.cpu().float(),
        bias.cpu().float() + a.cpu().float() @ b.cpu().float(),
        rtol=1.6e-2,
        atol=1e-2,
    )


def test_aten_log_softmax_override_matches_torch_when_mode_enabled():
    torch.manual_seed(1234)
    x = torch.randn(6, 1300, device="cuda", dtype=torch.float32)

    with bik.set_batch_invariant_mode(True):
        out = torch.ops.aten._log_softmax(x, -1, False)

    torch.testing.assert_close(out.cpu(), torch.log_softmax(x.cpu(), dim=-1), rtol=1e-4, atol=1e-6)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "dim,keepdim",
    [
        pytest.param([1], False, id="single_dim"),
        pytest.param([1], True, id="single_dim_keepdim"),
        pytest.param([0, 2], False, id="multi_dim"),
        pytest.param([0, 2], True, id="multi_dim_keepdim"),
    ],
)
def test_aten_mean_dim_overrides_match_torch_when_mode_enabled(dim, keepdim):
    torch.manual_seed(1234)
    x = torch.randn(2, 5, 3, device="cuda", dtype=torch.float32)

    with bik.set_batch_invariant_mode(True):
        out = torch.mean(x, dim=dim, keepdim=keepdim)

    reference = x.cpu().mean(dim=dim, keepdim=keepdim)
    assert out.shape == reference.shape
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-5, atol=1e-6)


def test_set_batch_invariant_mode_nests_and_restores_previous_state():
    torch.manual_seed(1234)
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)
    reference = x.float().mean(dim=[0, 1])

    plain = torch.mean(x, dim=[0, 1])
    assert plain.dtype is torch.bfloat16
    torch.testing.assert_close(plain, reference.to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)

    with bik.set_batch_invariant_mode(True):
        overridden = torch.mean(x, dim=[0, 1])
        assert overridden.dtype is torch.float32
        torch.testing.assert_close(overridden, reference, rtol=1e-5, atol=1e-6)
        with bik.set_batch_invariant_mode(False):
            inner = torch.mean(x, dim=[0, 1])
            assert inner.dtype is torch.bfloat16
            torch.testing.assert_close(inner, reference.to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)
        restored = torch.mean(x, dim=[0, 1])
        assert restored.dtype is torch.float32
        torch.testing.assert_close(restored, reference, rtol=1e-5, atol=1e-6)

    after = torch.mean(x, dim=[0, 1])
    assert after.dtype is torch.bfloat16
    torch.testing.assert_close(after, reference.to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)


def test_enable_batch_invariant_mode_returns_early_when_already_enabled():
    torch.manual_seed(1234)
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)
    reference = x.float().mean(dim=[0, 1])

    with bik.set_batch_invariant_mode(True):
        lib_at_enable = bik._batch_invariant_LIB
        bik.enable_batch_invariant_mode()
        assert bik._batch_invariant_LIB is lib_at_enable
        overridden = torch.mean(x, dim=[0, 1])
        assert overridden.dtype is torch.float32
        torch.testing.assert_close(overridden, reference, rtol=1e-5, atol=1e-6)

    after = torch.mean(x, dim=[0, 1])
    assert after.dtype is torch.bfloat16
    torch.testing.assert_close(after, reference.to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)


def test_te_unpatch_leaves_te_symbols_untouched_when_already_restored():
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.cpp_extensions as te_cpp

    gemm_before = te_cpp.general_gemm
    rmsnorm_forward_before = te.RMSNorm.forward
    bik._te_unpatch_for_batch_invariant()
    assert te_cpp.general_gemm is gemm_before
    assert te.RMSNorm.forward is rmsnorm_forward_before
    assert bik._TE_GEMM_FUNC_ORIGS == {}
    assert bik._TE_RMSNORM_FUNC_ORIGS == {}
