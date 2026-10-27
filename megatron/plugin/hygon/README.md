# Hygon vocabulary projection

Enable the Hygon plugin before importing Megatron or constructing the model:

```bash
export MG_FL_PREFER=hygon
```

This uses the existing `@overridable` class and lazy plugin registry. It does
not replace global PyTorch functions or modify `megatron/core`. `MG_FL_PREFER`
selects Megatron plugins independently of `TE_FL_PREFER`; the measured training
configuration keeps `TE_FL_PREFER=vendor`.

The optimized path is limited to BW1100-family devices reporting `BW11*` and
`gfx938`, contiguous BF16 inputs and weights, and the measured vocabulary
projection shapes: weight `[124160, 5120]` and 8192 or 16384 input rows after
sequence-parallel all-gather. Inputs can be `[S, B, K]` or `[M, K]`; sequence
and batch dimensions are restored after the GEMM.

- Forward splits output channels into blocks of 16384, including the final
  partial block, and writes each `torch.mm` result directly into the output.
- Input gradients use `(weight.T @ grad_output_2d.T).T.contiguous()`.
- Bias handling, weight gradients, gradient accumulation/deferral, all-gather,
  all-reduce and reduce-scatter follow the existing core linear protocol.

The identities are mathematically equivalent; different GEMM execution paths
can produce different floating-point rounding. The optimization does not change
the reduction dimension or select a process-wide BLAS backend.

16384 is the maximum block width, not a required output size or multiple.
The loop clips each block to the remaining channels, so the algorithm also
handles smaller outputs and arbitrary tails without padding. The measured
local vocabulary size, 124160, already uses seven full blocks plus a final
9472-channel block. The helper is shape-generic, but 16384 is a measured
performance choice for the admitted shapes, not a universal optimum. Other
model shapes deliberately retain the core path until their performance is
measured; passing a numerical boundary check alone does not expand this policy.

Other shapes, dtypes, layouts and devices use the original core implementation.
Megatron-FSDP parameters, `torch.compile` and CUDA graph capture also fall back.
Set `MEGATRON_HYGON_VOCAB_GEMM=0` to disable both optimized GEMMs while retaining
the plugin selection. This plugin makes no changes to cross-entropy fusion or
its validation guards.
