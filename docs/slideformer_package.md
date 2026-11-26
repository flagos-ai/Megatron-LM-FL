# SlideFormer installable package for Megatron-LM-FL

This integration focuses on full fine-tuning: layer residency, activation
checkpointing/offload, CPU Adam, gradient reduction and same-topology recovery.
SSDP LoRA and its newer storage backends are separate work; they are not required
dependencies of this plugin. Native Megatron models, routing and TP/CP/EP groups
remain owned by MCore. No OsMoE expert scheduler or private runtime is included.

## Build and select

From a clean committed checkout in an existing compatible PyTorch/TE environment:

```bash
python tools/build_slideformer_package.py --output /path/to/new-build
python -m pip install --no-deps /path/to/new-build/wheel/*.whl
export MEGATRON_SLIDEFORMER_IMPLEMENTATION=package
export MEGATRON_SLIDEFORMER_ENABLE=1
export MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD=1
# Preserve native attention, MLP, norm, RoPE and cross-entropy:
export MEGATRON_SLIDEFORMER_KERNEL_POLICY=off
# Run the existing FL training entry with --clip-grad 0 and other normal arguments.
```

`in_tree` is the default implementation, retaining the prior training path.
`package` requires the installed `slideformer_megatron` namespace and adapter
API version 1. Missing/incompatible packages fail explicitly, without fallback.
`MEGATRON_SLIDEFORMER_ENABLE=0` disables the plugin even if the legacy FlagScale
enable variable remains set; an optional wheel is not required in that mode.
Choose the implementation before importing `megatron.training`.

The same selected implementation supplies the engine, kernel policy, GPU gradient
pipeline and sharded CPU optimizer. Startup emits `SLIDEFORMER_IMPLEMENTATION`
with the module path, API version and packaged source commit. Check this record
when comparing runs. A wheel installs no files under the `megatron` namespace.

The source is not duplicated: the builder extracts a fixed public whitelist from
`megatron/plugin/slideformer` at the recorded Git commit. It includes CPU Adam
C++ sources, LICENSE/NOTICE and a per-file hash manifest; it excludes internal
optimization notes and archival history. The extension compiles against the
user's active PyTorch ABI. The builder never installs or upgrades MCore/TE/CUDA.
`manifest.json` next to the build records the source revision and wheel SHA-256.

## API and native parallel composition

Existing FL training callbacks still own model construction, schedule, loss,
parameter-scheduler integration and checkpoint/restart. Their source selection
is isolated in `megatron/plugin/slideformer_loader.py`. The packaged engine has
relative imports and a small `compat.py` for version-dependent MCore helpers.
Kernel policy `off` preserves normal native forward operators. The qualified
terminal-projection recompute optimization described below is independent of
kernel policy and can be disabled separately. Existing `auto`/explicit policies
remain available and must be qualified separately.

The package also exposes `FullFineTuneSession` for MCore trainers that already
construct their own native model and `ProcessGroupCollection`. Its explicit
lifecycle is `begin_step -> forward/backward -> finalize_gradients -> step`;
the initial adapter is PP=1 and requires full fine-tuning. It does not patch the
upstream CLI. Portability to NVIDIA upstream is a separate qualification item;
the current priority is this FL package path.

DP/TP/CP/EP combinations require native gradient/update comparisons, not merely
successful model construction. Dense replicas reduce over DP×CP; expert replicas
reduce over expert-DP; TP-special gradients use MCore's native semantics. Native
router aux loss and expert updates must be checked with every MoE qualification.
Do not infer these process groups from global world size.

Current FL limitations still apply: zero global gradient clipping, no interleaved
VPP, same-topology checkpoint restore, no co-owned native optimizer/DDP, and no
CPU-sharded optimizer with PP/NVMe. Earlier installed-package runs covered small-model TP2, CP2, MoE DP2,
EP2 and CP2+EP2 through the general path. The Dense DP preset has its own
eligibility checks. See the [user guide](user-guide/slideformer_megatron_fl.md)
for scope and reproducible commands.

## Validation protocol

1. Run `tests/unit_tests/slideformer` with CUDA hidden for CPU regression.
2. Compare the installed package with native DDP for loss, all trainable gradients
   and master-weight updates. Use the same CPU Adam kernel in the reference to
   isolate communication/lifecycle correctness from optimizer arithmetic.
3. Through the FL CLI, train 15 steps and separately resume the step-10 checkpoint
   for 5 steps. Compare model, active moments, update counters, scheduler and RNG.
4. Disable the plugin, remove the wheel from the test import path and verify the
   native CLI still trains. Record the actual loaded implementation for enabled runs.
5. Qualify native TP2, CP2 and EP2 individually before larger combinations.

The SSDP delivery workspace records commands, pinned sources and raw logs under
`integrations/megatron-package`. It contains a guarded launcher for shared H100s;
never copy fixed GPU IDs from historical examples without an ownership check.
No formal third-party performance result is implied by these engineering tests.

## Single-batch overlap and terminal recomputation

GPU gradient storage defaults to two maximum-layer buffers. D2H completion
returns the GPU lease; the CPU gradient lease remains owned until CPU Adam has
finished. This bounds GPU memory independently from the CPU update queue and
keeps the next layer's backward from waiting for the preceding Adam update.
Released GPU slots rotate in FIFO order. In the DP path, release happens before
DMA completes; a LIFO pool would repeatedly wait on the newest slot while an
older slot remained free, defeating the second buffer. Reuse still waits for
the selected slot's completion event.
Set `MEGATRON_SLIDEFORMER_GPU_GRAD_BUFFER_COUNT=1` for the smaller memory budget.

With `slideformer-slot` activation checkpointing, terminal FC2 recomputation
can omit its forward GEMM: native dense TransformerLayer/MLP, native residual
add, bias-free TE FC2, zero hidden dropout, TP=CP=EP=1, and no quantization,
CUDA graphs or native recomputation/offload. Normal forward remains native TE;
this terminal backward uses PyTorch GEMMs, including the leased main_grad
accumulation target. Other structures keep their original implementation.
The counter `terminal_projection_recompute_skips` records actual use. This is
separate from kernel policy: set
`MEGATRON_SLIDEFORMER_TERMINAL_PROJECTION_RECOMPUTE=0` (or engine config
`terminal_projection_recompute=False`) to retain the native terminal TE backward.
Closing the engine restores the original FC2 method.


## Dense DP parameter transfer

The explicit `dense-dp` preset and `DataParallelTransport()` default to sharded
H2D plus native GPU reduce-scatter. Each replica transfers its own BF16 parameter
slice from CPU, then reconstructs the layer using MCore's parameter all-gather
for equal slices or native broadcasts for unequal slices. GPU reduction, CPU
Adam ownership and update dependencies remain unchanged.

Set `MEGATRON_SLIDEFORMER_SHARD_H2D=0` with the preset, or use
`DataParallelTransport(parameter_transfer="full")`, for the full-H2D comparison.
The `hsr-dp` preset keeps full H2D unless explicitly overridden. Callers without
an explicit preset/transport retain their prior behavior; the pure Dense DP
eligibility checks still reject unsupported TP/CP/EP/MoE combinations.

## Dense DP gradient precision

Native GPU reduce-scatter now defaults to BF16 reduction and BF16 D2H, matching
SlideDP's BF16 path. CPU master weights, Adam moments and update arithmetic stay
FP32. The optimizer accepts BF16 gradient buffers and converts while consuming;
there is no full-model FP32 gradient allocation for a single batch. More than one
microbatch uses FP32 CPU accumulation after BF16 D2H. GPU/host buffer leases,
completion events, prefetch and update ordering are unchanged.

Use `MegatronSlideFormerEngineConfig(gpu_reduce_dtype="fp32")` or
`MEGATRON_SLIDEFORMER_GPU_REDUCE_DTYPE=fp32` for the previous precision. CLI and
installed Session use the same field. HSR retains its existing BF16 D2H and
FP32 host reduction. The independent native MCore FP32-accumulation option keeps
BF16 wire traffic, performs the sum in FP32, then casts to the selected dtype on
GPU before scaling and D2H. Bucket element counts are unchanged between modes.

Checkpoints record the active GPU-reduce precision and reject mismatched resumes
before loading weights. Old GPU-reduce checkpoints without this metadata require
explicit FP32 mode. Printed `GPU_REDUCE_PRECISION` records wire/reduced/D2H and
optimizer dtypes; `gpu_reduce_d2h_bytes` counts actual submitted gradient bytes.
This policy is not a claim of numerical identity with FP32 reduction or of
convergence for arbitrary models or DP sizes.

## Layer communication and buffer dependencies

For contiguous equal-sized shards with matching input/reduction dtypes, the GPU
reducer follows MCore's distributed-optimizer layout: reduce-scatter writes into
the rank-owned view of the existing flat layer gradient. This issues one native
collective per layer and allocates neither packed input nor output scratch.
The rank-owned result is copied to a bounded pinned host slot in chunks of
`chunk_size_mb` (32 MiB by default). Uneven/empty shards, dtype conversion and
the separate FP32-accumulation algorithm retain the preceding bucketed path.
`MEGATRON_SLIDEFORMER_GPU_REDUCE_LAYOUT=bucketed` explicitly selects that path;
the default `auto` selects in-place reduction only when eligible. The existing
`GPU_REDUCE_BUCKET_MB` budget controls fallback scratch, not the native aliasing path.

The dependencies are separate and remain bounded:

- The D2H/reduction stream waits for the current compute stream before reading
  gradients. The reused GPU gradient slot remains leased until its last D2H.
- For an active GPU reducer, the next compute stream waits on that slot's event
  before zeroing or writing it. Autograd can submit other work without a host
  event synchronization. `MEGATRON_SLIDEFORMER_GPU_GRAD_REUSE=host` restores the
  preceding host wait for controlled comparison; inactive/host paths retain it.
- CPU Adam waits for the complete reduced host shard. The pinned slot is not
  overwritten until its CPU future completes; GPU completion alone cannot
  release this host lease. Worker failures still propagate via the future.
- Parameter H2D/gather still uses the existing parameter-pool reuse event and
  owner-specific compute-use event. Update draining and step finalization remain
  mandatory before the next step's parameter prefetch or checkpoint publication.

`gpu_reduce_inplace_layers`, `gpu_reduce_bucketed_layers`, and
`gpu_reduce_d2h_bytes` report the actual paths and traffic. These choices do not
change parameter ownership, communicated precision, optimizer state, or model
operators. Qualification and measured performance are recorded separately.
