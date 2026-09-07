<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# SlideFormer for Megatron-LM-FL

## Summary

This change ports the single-GPU training design from the SlideFormer paper
and reference implementation from a PyTorch + Hugging Face runtime to
Megatron-LM-FL + Megatron Core + Transformer Engine (TE).

The implementation keeps the SlideFormer invariants:

- persistent parameters, FP32 master weights, gradients, and Adam states live
  in CPU memory;
- only a bounded layer window is materialized on the GPU;
- parameter H2D, gradient D2H, activation movement, and CPU Adam are scheduled
  asynchronously;
- activation checkpointing stores layer boundaries on CPU and reconstructs the
  layer graph during backward;
- fused attention, MLP, normalization, RoPE, and linear cross-entropy kernels
  remain enabled by default.

It does not copy the Hugging Face module wrappers verbatim. Megatron owns the
model graph, checkpoint layout, tensor-parallel parameter layout, and TE
modules, so the port implements the same scheduling design at Megatron's
stable layer, optimizer, and training boundaries.

Current acceptance status: **ready for draft review for single-GPU training**.
The validated scope is dense decoder-only GPT models with
`TP=PP=DP=EP=1` and one microbatch. Multi-GPU training is intentionally out
of scope for this PR.

### Upstream references

- Paper: [SlideFormer (arXiv:2603.16428)](https://arxiv.org/abs/2603.16428)
- Reference implementation:
  [RegiaYoung/SlideFormer](https://github.com/RegiaYoung/SlideFormer)

The paper defines the layer-sliding training design. The repository is the
behavioral and implementation reference for this port; the exact baseline
revision used by the acceptance runs is recorded in the performance protocol
below and in `megatron/plugin/slideformer/NOTICE`.

## Repository boundary

Megatron-LM-FL is the owner of the SlideFormer engine:

- CPU/GPU parameter and gradient ownership;
- activation sliding checkpointing;
- layer-wise CPU Adam;
- SlideFormer checkpoint save/load;
- automatic fused-kernel selection and runtime verification.

FlagScale remains the launcher and configuration frontend. It enables the
plugin through `MEGATRON_SLIDEFORMER_*` environment variables and may add
end-to-end launch tests, but it does not carry a second engine implementation.

## Design mapping

| SlideFormer PyTorch/HF design | Megatron-LM-FL implementation | Reason for the adaptation |
| --- | --- | --- |
| Wrap HF embedding, decoder layers, norm, and LM head | Resolve Megatron GPTModel owners and attach hooks at MCore module boundaries | Preserves Megatron model construction and checkpoint names |
| One/two reusable max-layer cache units | Two Transformer/final-norm sliding slots plus one dedicated tied embedding/output slot | Avoids sizing every slot to a large tied vocabulary weight while preserving bounded reuse |
| Layer hooks launch H2D/D2H and CPU Adam | Megatron forward/backward boundaries plus TE-safe CUDA events and leases | TE custom autograd may retain parameter storage beyond a Python module hook |
| Saved-tensor activation offload | Non-reentrant layer checkpoint with one CPU boundary slot per decoder layer | Stores only the reconstruction boundary instead of retaining all TE intermediates |
| Separate asynchronous executors/streams | Shared depth-three H2D scheduler for parameter and activation prefetch; independent D2H/optimizer pipeline | Bounds outstanding transfers and preserves copy/compute overlap |
| Per-layer CPU LayerAdam | SlideFormer-owned LayerAdam for decoder and static parameter groups | Megatron's native optimizer cannot own parameters that are intentionally CPU-resident |
| HF/Liger model monkey patches | MCore/TE-aware structural policy followed by a runtime kernel report | Avoids maintaining model-family-specific HF replacement tables |
| HF checkpoint/model loading | Native Megatron model construction plus explicit HF-to-MCore weight mapping in the benchmark path | Keeps the training backend independent of HF `AutoModel` |

### Parameter and gradient lifecycle

Each managed owner has pageable FP32 master parameters and Adam states on CPU.
A shared pinned BF16 staging buffer converts and submits layer parameters to a
GPU slot. The next owner is prefetched while the current owner computes.

TE fused linear weight-gradient GEMMs write directly into an event-protected
flat GPU gradient slot. One layer-flat D2H transfer is submitted when the
layer is complete; the CPU slot is not recycled until both D2H and the
LayerAdam update finish. Parameter slots likewise carry CUDA completion
events, preventing a later owner from rebinding storage still referenced by
TE.

Untied models use two max-owner GPU parameter slots. Tied models keep the
shared embedding/output weight in one dedicated resident slot and size the
two sliding slots only for Transformer/final-norm owners. This keeps the
general two-unit SlideFormer policy without multiplying a vocabulary-sized
allocation.

### Sliding activation checkpoint

The `slideformer-slot` backend stores one BF16 boundary hidden state per
Transformer layer in CPU memory. During backward it prefetches the next
boundary, reconstructs the layer under non-reentrant checkpointing, and
releases the slot after the layer backward boundary.

This has the same layer-sliding semantics as SlideFormer, but it is not a
byte-for-byte copy of `sliding_checkpoint.py`. The reference HF path saves
the tensors selected by Hugging Face autograd hooks. The Megatron path
deliberately recomputes TE attention/MLP intermediates because their layout
and lifetime are TE implementation details. For compatible bias-free SwiGLU
with zero hidden dropout, the recompute path builds the FC2 Jacobian without
rerunning an unused FC2 forward GEMM.

### Scheduling

Parameter and activation H2D requests share a scheduler with a default maximum
depth of three. Sharing is not a correctness requirement; it is a bounded
backpressure policy that prevents the two prefetch sources from independently
filling the copy queue. An isolated scheduler experiment did not improve
throughput, so the shared scheduler remains the default.

Gradient D2H and CPU Adam use a separate pipeline. All slot reuse is guarded
by CUDA events and optimizer completion rather than by queue position alone.

## Automatic fused-kernel policy

`MEGATRON_SLIDEFORMER_KERNEL_POLICY=auto` and strict verification are the
defaults. The policy first selects the MCore Transformer Engine layer spec,
then inspects the constructed model and reports the effective backend.
Compatible Qwen3 training does not require users to enumerate fused kernels.

| Operation | Default | Ownership |
| --- | --- | --- |
| Dense causal attention | TE auto dispatch; strict mode requires FlashAttention availability | TE + `flash-attn` |
| TP=1 attention layout | Keep FlashAttention output batch-major through output projection | SlideFormer Megatron adapter |
| RMSNorm, including Q/K norm | TE MCore RMSNorm | TE |
| Standard dense RoPE | TE fused RoPE | TE |
| Bias-free SwiGLU | TE/MCore fused path | TE |
| Large SwiGLU FC1 | Preserve concatenated Megatron weight/checkpoint; execute gate/up views separately | TE linear + Liger SiLU×Mul |
| LM head + cross entropy | SlideFormer Legacy fused linear cross entropy | Adapted Liger Triton kernel |

TE auto is not equivalent to a Hugging Face `AutoModel` monkey patch. It
dispatches kernels inside known TE modules. The SlideFormer policy is
responsible for selecting those modules, validating the model constraints,
installing Legacy LCE at the stable GPT output boundary, and adapting the
single-GPU attention layout.

The TP=1 attention adapter retains Megatron's combined `linear_qkv` weight
and checkpoint format. FlashAttention output remains
`[batch, sequence, heads, head_dim]` through the output projection, and a
sequence-major view is restored only at the attention module boundary. It
removes a full attention-output layout copy without introducing a
batch-major QKV staging copy.

For a bias-free FC1 whose output reaches the configurable 0.5 GiB threshold,
the plugin views the concatenated FC1 weight as gate/up halves, runs two GEMMs,
and applies Liger SiLU×Mul. This avoids multi-GiB temporary allocations at
large batch sizes while retaining Megatron checkpoint compatibility. Smaller
shapes stay on the faster TE fused path.

The startup report must show the selected attention, QKV layout, MLP, loss,
normalization, and RoPE backends. Strict mode fails early if a required
backend is unavailable or the model violates its constraints.

## Integration

When enabled, `megatron/training/training.py`:

1. validates the current single-GPU scope and rejects Megatron native
   recompute;
2. resolves structural kernel choices before model construction;
3. builds the native Megatron GPTModel;
4. applies and verifies post-build kernel adapters;
5. skips Megatron DDP and the native optimizer;
6. applies the SlideFormer engine to the model chunk;
7. lets LayerAdam own decoder and static parameter groups;
8. saves and restores SlideFormer-owned state.

The primary switch is:

```bash
export MEGATRON_SLIDEFORMER_ENABLE=1
```

The safe defaults are declared in
`megatron/plugin/slideformer/config.py`. Every automatic backend has an
explicit override for diagnostics. Important controls include:

```bash
export MEGATRON_SLIDEFORMER_KERNEL_POLICY=auto
export MEGATRON_SLIDEFORMER_ATTENTION_BACKEND=auto
export MEGATRON_SLIDEFORMER_QKV_LAYOUT_BACKEND=auto
export MEGATRON_SLIDEFORMER_MLP_BACKEND=auto
export MEGATRON_SLIDEFORMER_LOSS_BACKEND=auto
export MEGATRON_SLIDEFORMER_NORM_BACKEND=auto
export MEGATRON_SLIDEFORMER_ROPE_BACKEND=auto
export MEGATRON_SLIDEFORMER_STRICT_KERNELS=1
export MEGATRON_SLIDEFORMER_MAX_OUTSTANDING_H2D=3
export MEGATRON_SLIDEFORMER_SPLIT_SWIGLU_THRESHOLD_GIB=0.5
```

FlagScale's legacy `FLAGSCALE_SLIDEFORMER_*` names remain accepted where
equivalents existed.

### Checkpoint ownership

Megatron's native optimizer checkpoint path is bypassed while the plugin owns
CPU-resident parameters. The engine writes:

```text
${save}/slideformer/iter_XXXXXXX.pt
${save}/slideformer/latest.pt
```

The checkpoint contains CPU master parameters, LayerAdam state, scheduler
state, and iteration. Load validates the owner count, parameter shapes, and
format version before rebinding execution storage.

## Supported scope and fail-fast checks

Validated:

- one CUDA GPU;
- dense decoder-only Megatron GPTModel;
- `TP=PP=DP=EP=1`;
- one microbatch;
- BF16 execution with FP32 master parameters and Adam states;
- dense causal attention without arbitrary attention bias;
- Qwen3 4B, 8B, and 14B class checkpoints;
- tied and untied embedding/output weights;
- CPU optimizer state, with optional NVMe optimizer-state plumbing.

Not part of this PR:

- `TP>1`, `PP>1`, `DP>1`, or `EP>1`;
- MoE end-to-end training;
- Megatron distributed optimizer co-ownership;
- arbitrary attention masks/biases on the FlashAttention adapter;
- a claim of equivalence for every Hugging Face model family;
- multi-GPU scheduling described by future SlideFormer work.

Unsupported combinations fail during setup instead of silently falling back
to a slower or semantically different path.

## Dependencies and provenance

The `slideformer` optional dependency group adds:

```toml
flash-attn = ">=2.8,<3"
liger-kernel = ">=0.7,<0.9"
ninja = ">=1.11"
```

Transformer Engine is already part of the Megatron-LM-FL stack and remains the
primary model-kernel provider. The tested environment used:

- Python 3.12.13;
- PyTorch 2.11.0+cu128;
- Transformer Engine 2.17.0;
- FlashAttention 2.8.3;
- Liger Kernel 0.8.0;
- Triton 3.6.0;
- Ninja 1.13.0;
- NVIDIA driver 570.86.10.

Ninja and a C++/OpenMP toolchain are needed the first time the LayerAdam CPU
extension is built. `tensornvme` is optional and imported only when NVMe
optimizer-state offload is enabled.

`megatron/plugin/slideformer/NOTICE` records the SlideFormer source, the
DeepSpeed-derived CPU LayerAdam components, the BSD-2-Clause Liger Legacy LCE
derivative, and FlashAttention. Derived source files retain their original
copyright and SPDX headers; unused CUDA FusedAdam/Apex components are not
shipped.

## Validation

### Automated tests

The following acceptance checks pass on this branch. Performance measurements
use the functionally identical engine at benchmark commit `5983b8ae9`; later
changes only package the JIT sources, normalize headers, and add this report.

| Suite | Result | Coverage |
| --- | ---: | --- |
| `tests/unit_tests/slideformer/test_slideformer_plugin.py` | 28 passed | config validation, kernel selection, TE/Liger adapters, layout discovery, tied weights, CPU initialization, asynchronous storage safety, checkpoint round-trip |
| FlagScale reporting regression | 4 passed | benchmark defaults, metadata consistency, mixed-configuration rejection |
| Megatron two-iteration training smoke | passed | real training entrypoint, automatic kernel report, optimizer steps, SlideFormer save path |
| Wheel packaging smoke | passed | plugin NOTICE and all CPU LayerAdam JIT C++/header sources are present in the built wheel |

The plugin tests include CUDA numerical checks for FlashAttention scaling,
TP=1 batch-major attention, Legacy LCE, split SwiGLU, optimizer update
equivalence, and checkpoint restore.

The entrypoint smoke uses a two-layer BF16 Qwen-style MCore GPT with TE,
FlashAttention auto selection, Q/K RMSNorm, SwiGLU, RoPE, mock data, and two
optimizer iterations under `torchrun --nproc-per-node=1`. Its startup report
confirmed TE auto attention, TP=1 batch-major attention, TE SwiGLU/RMSNorm/RoPE,
and SlideFormer Legacy LCE; both iterations saved SlideFormer-owned state.

Paired Qwen3-4B/BS64 testing reported a maximum three-step loss difference of
`6.49e-5` after enabling the TP=1 attention adapter. Qwen3-8B measured losses
remained within approximately `2e-4` of the correctness-safe reference.
Allocator retry and OOM counts were zero in the final formal runs.

### Performance protocol

The four figures below use:

- one NVIDIA RTX 4090;
- BF16, sequence length 1024;
- the same local Qwen3 checkpoint per native/Megatron pair;
- batch sizes 1, 2, 4, 8, 16, 32, and 64;
- three warmup and ten measured optimizer steps;
- a fresh process for every point;
- FlashAttention 2 and the default fused-kernel policy;
- process peak RSS from `/usr/bin/time -v`;
- native reference commit `debb9c533e0` (the local tree only adds
  disabled-by-default NVTX capture hooks; capture was off for these runs);
- Megatron-LM-FL benchmark commit `5983b8ae9`.

The scaling harness constructs the native Megatron GPTModel, maps the paired
HF checkpoint into MCore parameters, and applies the same plugin engine and
kernel policy used by the production training entrypoint. The separate
two-iteration smoke test covers the full `pretrain_gpt.py` integration. Figure
generation rejects a point unless its metadata confirms the 0.5 GiB split-FC1
policy, TP=1 batch-major attention, and Legacy LCE were actually active.

The source data is checked in beside the figures as
`images/slideformer/qwen3_scaling_pr_head.csv`.

![Qwen3 single-GPU throughput scaling](images/slideformer/qwen3_scaling_throughput.png)

![Qwen3 single-GPU allocated-memory scaling](images/slideformer/qwen3_scaling_gpu_allocated.png)

![Qwen3 single-GPU reserved-memory scaling](images/slideformer/qwen3_scaling_gpu_reserved.png)

![Qwen3 single-GPU process-RSS scaling](images/slideformer/qwen3_scaling_cpu_rss.png)

<!-- QWEN3_SCALING_SUMMARY_START -->
BS64 capacity-point summary:

| Model | Native / Megatron tok/s | Megatron delta | Native / Megatron allocated | Native / Megatron reserved | Native / Megatron CPU RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 4545.9 / 4496.7 | -1.08% | 12.105 / 12.180 GiB | 12.115 / 13.309 GiB | 67.527 / 68.927 GiB |
| Qwen3-8B | 2570.4 / 2533.7 | -1.43% | 16.641 / 15.578 GiB | 16.646 / 16.805 GiB | 115.113 / 116.869 GiB |
| Qwen3-14B | 1405.3 / 1386.6 | -1.33% | 21.751 / 20.674 GiB | 21.764 / 22.859 GiB | 210.619 / 212.386 GiB |

At BS64, throughput is within 1.43% of the reference for all three sizes.
Megatron allocated memory is 1.06-1.08 GiB lower for 8B/14B and essentially
equal for 4B. Reserved memory is 0.16 GiB above native for 8B and about
1.10-1.19 GiB above native for 4B/14B. CPU RSS is within 1.77 GiB.

The full curve is not uniformly within two percent. The largest measured gaps
are Qwen3-8B/BS8 (-7.69%) and Qwen3-14B/BS8 (-8.17%). Prior Nsight comparison
showed that these medium-batch split-FC1 points have lower or comparable
kernel sum but more exposed scheduling gaps than the HF runtime. They remain a
follow-up opportunity rather than being hidden by the capacity-point result.
<!-- QWEN3_SCALING_SUMMARY_END -->

Process RSS includes checkpoint/model construction as well as steady-state
training. The Megatron engine's persistent CPU tensors are FP32 master weights,
two FP32 Adam states, bounded BF16 gradient/staging pools, and one BF16
activation boundary per decoder layer. The reference HF process additionally
includes HF checkpoint materialization and its saved-tensor allocator high
water. The CPU plot should therefore be read as an end-to-end process limit,
not as the size of sliding checkpoint slots alone.

GPU allocated and reserved are both shown because TE workspaces and cross-stream
slot lifetimes affect the CUDA caching allocator. A lower allocated curve with
a slightly higher reserved curve indicates cached segments/fragmentation, not
an additional persistent copy of the model. The accepted flat parameter and
gradient slots reduced the earlier reserve gap; `cudaMallocAsync` increased
reserved memory in this workload and was not enabled.

### Profile-guided changes

Nsight Systems was used to separate kernel time, copy-engine time, and exposed
GPU gaps. The accepted changes address measured bottlenecks:

- layer-flat TE main-gradient D2H replaced per-parameter transfers;
- bounded parameter slots reduced allocator segments and fragmentation;
- split FC1 removed large SwiGLU allocator retries;
- FC2 recompute early-stop removed a backward-only redundant GEMM;
- the tied-weight resident slot avoided vocabulary-sized sliding slots;
- TP=1 batch-major attention removed complete attention-output layout copies.

The following alternatives were measured and rejected:

- forcing FlashAttention instead of TE auto dispatch;
- `cudaMallocAsync` for this workload;
- physically staging sequence-major hidden states into batch-major QKV input;
- splitting all SwiGLU shapes at a lower threshold;
- independent parameter/activation H2D queues;
- deeper unsafe gradient pipelines without the required ownership guarantees.

These results are why the PR keeps TE as the primary kernel system, uses Liger
only where TE has no equivalent Legacy LCE or where split SwiGLU needs a
pointwise operation, and adapts scheduling to TE lifetimes instead of copying
HF hooks literally.

## Review guide

Suggested review order:

1. `megatron/plugin/slideformer/config.py` and `layout.py` for scope,
   ownership, and fail-fast behavior;
2. `checkpoint.py` for activation boundary semantics;
3. `engine.py` for slot leases, event ordering, D2H/CPU Adam overlap, and
   checkpoint ownership;
4. `kernels.py` and `legacy_lce.py` for automatic TE/Liger policy and
   numerical constraints;
5. `megatron/training/training.py` for the integration boundary;
6. `tests/unit_tests/slideformer` for invariants and reproduction.

The main maintainer feedback requested by this draft is:

- whether plugin ownership at `training.py` is the preferred long-term
  integration point;
- whether kernel-policy selection should become a general Megatron registry;
- whether the SlideFormer-owned checkpoint should remain separate until
  multi-GPU support exists;
- which TE storage-lifetime contract should be used before extending the
  two-slot engine to tensor or pipeline parallelism.

## Acceptance decision

The single-GPU port preserves SlideFormer's layer-sliding design and training
semantics while using Megatron-native model structure and TE kernels. Kernel
selection is automatic and verified, Qwen3 correctness checks pass, and the
current-head 4B/8B/14B batch-scaling results are included above.

The change is suitable for a **draft Megatron-LM-FL PR** to obtain maintainer
feedback. Multi-GPU support, MoE, and FlagScale launcher polish should follow
as separate work after the backend interface is agreed.
