"""Triton DSA kernels: drop-in local-computation kernels for the fused CSA path."""

from megatron.plugin.dsa_kernel.triton_dsa_kernels import (
    build_triton_dsa_namespace,
    triton_csa_fwd_flash_mla,
    triton_csa_sparse_attn_backward,
)

__all__ = [
    "build_triton_dsa_namespace",
    "triton_csa_fwd_flash_mla",
    "triton_csa_sparse_attn_backward",
]
