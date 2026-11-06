# Ascend GatedDeltaNet overrides

The Ascend adapter registers NPU implementations for `GatedDeltaNet._normalize_qk`,
`GatedDeltaNet._gated_delta_rule`, and `GatedDeltaNet._causal_conv1d` through the
Megatron plugin override registry. Set `MG_FL_PREFER=npu` before importing Megatron.
`TE_FL_PREFER` selects TransformerEngine implementations independently.

The dense GDN fast path accepts NPU FP16/BF16 Q/K/V with key and value head
dimension 128, FP32 gates, matching beta dtype, BSHD layout, fixed sequence lengths,
and no initial/final recurrent state. It is disabled for deterministic GDN mode.
Unsupported inputs and unavailable optional kernels delegate to upstream FLA;
kernel execution errors propagate. Dense NPU causal convolution uses PyTorch
depthwise conv1d; stateful, variable-length, or additional metadata options delegate
to FLA. The adapter preserves keyword metadata introduced by FLA 0.5.2.

Install the appropriate Ascend PyTorch/CANN runtime, FLA, and the optional
FLA-NPU AscendC/Triton operator package. The adapter does not install kernel
binaries. The validation environment used torch 2.7.1+cpu, torch-npu 2.7.1.post8,
FLA 0.5.2, and triton-ascend 3.2.2. The FLA-NPU package in that environment had
no distribution/version metadata; an installed-content fingerprint is recorded
with the validation artifacts rather than an inferred upstream commit.

Three test modules under `tests/unit_tests/ssm` cover adapter contracts, convolution
keyword compatibility, and override dispatch. Numerical cases require an Ascend
device and all optional kernels. They compare output and five input gradients
against an independent FP32 recurrence for FP16/BF16 and lengths 65/128; the test
fails on unintended fallback. Those cases exclude fused Q/K normalization and
are not a full-model NVIDIA accuracy comparison.

This implementation and the initial contract tests are adapted from
[flagos-ai/Megatron-LM-FL PR #153](https://github.com/flagos-ai/Megatron-LM-FL/pull/153),
head `9d4788a41d91fe8a13a2e1563eb39ea43184b674`. This submission rebases the adapter
onto official main `b471409ac414d760c42a4b159725c2acd2676166`, updates the tests for
the installed FLA keyword API, and adds independent NPU numerical validation.
It retains the original source copyright notices.
