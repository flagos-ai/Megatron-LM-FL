# MegaLens related merge requests

[Megatron-LM-FL #104](https://github.com/flagos-ai/Megatron-LM-FL/pull/104) is the
main integration and coordination MR. It owns the implicit probes, Core facade,
training observation hooks, Trace runtime/export/loading, and the five retained
analysis families. Independent training and dependency fixes are reviewed in
their owning repositories with their own regression tests.

The [Core integration review guide](megalens-core-integration.md) describes probe
ownership, the native control-flow boundaries and checks for upstream updates.
The [change-origin inventory](megalens-change-origins.md) accounts for the main
MR's added lines and identifies the MixedPara source modules and target additions.

## Independent submissions

All nine submissions are Draft MRs. Follow their links for current merge status.

| Repository / target | MR | Change | Commit | Runtime or merge dependency |
|---|---|---|---|---|
| flagos-ai/Megatron-LM-FL / `main` | [#195](https://github.com/flagos-ai/Megatron-LM-FL/pull/195) | Core import compatibility without Triton | `eae39b944` | Enables Core imports without Triton. #104 currently follows upstream and requires real Triton until this repair merges. |
| flagos-ai/Megatron-LM-FL / `main` | [#196](https://github.com/flagos-ai/Megatron-LM-FL/pull/196) | Hybrid-CP helper calls and subsample metadata | `f581bb9cf` | Required only when using Hybrid-CP; ordinary CP2/TE profiles do not depend on it. |
| flagos-ai/Megatron-LM-FL / `main` | [#197](https://github.com/flagos-ai/Megatron-LM-FL/pull/197) | DualPipeV combined backward without shared experts | `953487750` | Required only for this native DualPipeV configuration; observation extensions remain separate. |
| flagos-ai/FlagScale / `main` | [#1305](https://github.com/flagos-ai/FlagScale/pull/1305) | Dense/packed batch broadcast correction | `fc8e063e9` | Independent of MegaLens; preserves SFT/Hybrid-CP packed transport. |
| flagos-ai/FlagScale / `main` | [#1306](https://github.com/flagos-ai/FlagScale/pull/1306) | Opt-in MegaLens lifecycle integration | `ca5fa3a34` | Requires a compatible Core build containing #104; has no source dependency on the dense-batch PR. |
| flagos-ai/TransformerEngine-FL / `main` | [#133](https://github.com/flagos-ai/TransformerEngine-FL/pull/133) | CUDA normalization quantizer dtype forwarding | `9357d5668` | Relevant to FP8 normalization using this CUDA adapter; native bindings and hardware validation pending. |
| deepseek-ai/DeepEP / `hybrid-ep` | [#775](https://github.com/deepseek-ai/DeepEP/pull/775) | Toolkit NVTX/CCCL build discovery | `7e2b06df3` | Targets hybrid-ep, not main; full HybridEP JIT/device validation pending. |
| flagos-ai/Megatron-LM-FL / `main` | [#209](https://github.com/flagos-ai/Megatron-LM-FL/pull/209) | ModelOpt checkpoint helper arguments, result and override path | `35d9631c6` | Independent native checkpoint repair; no runtime dependency on MegaLens. |
| flagos-ai/Megatron-LM-FL / `main` | [#210](https://github.com/flagos-ai/Megatron-LM-FL/pull/210) | Biencoder duplicate checkpoint-name argument | `ae80dda50` | Independent native checkpoint repair; no runtime dependency on MegaLens. |

The five native Megatron fixes add no MegaLens import, event or trace interface.
The DualPipeV fix does not include dedicated observation; the existing 44 generic
P2P wait observations remain in #104. The FlagScale integration changes three
training files and does not include the separate batch broadcast correction.

## Target revisions and adaptations

| Repository / target | Reviewed base |
|---|---|
| Megatron-LM-FL main | `bbc208e00f2a201783938ef549f5058d6047e4c0` (v0.18.2) |
| FlagScale main | `eb9dc045a8aa343c127de4e2b9dd4ad113e6d1e0` |
| TransformerEngine-FL main | `6df49429794d18109f0d04d924a14e032cdc220b` |
| DeepEP hybrid-ep | `4e890fd19a3a52f23cc7bd38a0d9b0b1276f70d6` |

- FlagScale's batch helper now lives in `training/utils/common_utils.py`.
  Its current training flow uses `step_batch_size_schedule`, which replaces the
  old overlay's dynamic-batch argument in the sample-range tracing restriction.
  Existing checkpoint, fault injection, RL and performance-monitor calls remain;
  the latest asynchronous checkpoint callbacks retain both new finalization arguments.
- TE-FL normalization still lacks native quantizer dtype normalization.
  The revised forwarding fix catches only conversion compatibility errors and
  preserves the object, argument and result contracts.
- DeepEP's active `hybrid-ep` setup still hardcodes NVTX interop and omits the
  toolkit CCCL include. The revised patch selects available libraries/headers
  from `CUDA_HOME`; it does not hardcode `/usr/local/cuda`, replace kernels,
  change JIT dependencies, or claim complete CUDA 12.8 support.

Do not create duplicate MRs for the following:

- Megatron main already contains the dense metadata condition and NVML
  empty-handle protection. #104 uses those upstream implementations.
- TE-FL main covers the key UserBuffer operations and enum conversion from the
  old v0.2.0 patch. Related merged work includes
  [#95](https://github.com/flagos-ai/TransformerEngine-FL/pull/95) and
  [#116](https://github.com/flagos-ai/TransformerEngine-FL/pull/116).
- DeepEP main has a different build layout and already includes a CCCL path;
  the old hybrid-ep patch is not a main patch.
- FlagGems `mul` exclusion and NCCL NVLS disabling are configuration workarounds,
  not completed kernel or communication-library fixes. FlagScale's default
  `$cmd; sync` exit-status propagation is a separate remaining runner issue.

## Validation and limits

| Submission | Checks passed | Evidence boundary |
|---|---|---|
| Megatron #195 | 1 | Fresh-process Core/schedule import with Triton blocked; paged-stash rejection before allocation |
| Megatron #196 | 1 | All three forward/backward sites and both TP metadata paths, using CPU transport stubs |
| Megatron #197 | 2 | CPU/AST guards in both combined backward paths |
| FlagScale #1305 | 12 | Dense single/first/last/MTP branches; SFT/Hybrid-CP matching sender/receiver protocol |
| FlagScale #1306 | 24 | Argument validation, lazy ownership, skip/exception boundaries and mode-1 alignment |
| TE-FL #133 | 22 | 14 normalization forwarding contracts and 8 existing UserBuffer contracts |
| DeepEP #775 | 13 | 12 toolkit/extension-configuration cases and the existing C++ standard check |
| Megatron #209 | 13 | Native helper signature/four-field results, restoration, path override, failures and sharded delegation |
| Megatron #210 | 12 | Real CPU checkpoint files, both optimizer settings, default/custom paths and full/query/context loads |

These seven refreshed submissions pass 75 CPU/contract checks against the
reviewed targets above. The two additional checkpoint MRs pass 13 and 12 cases.
Their unmodified main sources reproduce 9 and 12 failures respectively. The earlier
six-fix differential (19 failures) remains tied to its originally recorded bases.
The FlagScale integration also preserves all 120 original function ASTs after
removing its added observation calls/wrappers, with 20 paired observation markers.
Vendored globals and arguments imported with the current MegaLens Core.

Validation used Python 3.12 and PyTorch 2.8.0+cpu. Native Hybrid-CP/DualPipeV
standalone checks had Triton 3.4.0 available; the optional-import regression
explicitly blocked it. TE CPU tests used a native-extension substitute and a
package-version metadata stub. DeepEP checks inspected actual extension
configuration with a constructor substitute. These checks do not prove compiled
bindings, NVCC/JIT execution, FP8 numerics, GPU/NCCL ordering, CUDA Graph replay,
full distributed CI, training outcomes or runtime overhead.

New checkpoint tests pass Black/isort. Native formatting and import layouts are
preserved when unrelated to the repair, including Hybrid-CP's first multiline
backward call and DeepEP's existing imports. Refreshed scoped diffs pass whitespace
checks; inherited upstream formatting is not rewritten.

The main integration passes the source/target scanner and 715 CPU checks after
removing the relocated Triton-only assertion (previously 716). Its unchanged
shared DualPipeV wait-routing regression passes independently. It uses a real
Triton installation; the optional-import repair is not hidden inside Probe.
Packaging and GPU records retain their original source revisions.

## Merge and patch retirement

#104 contains no independent native training repair implementation or dedicated
repair tests. Hybrid-CP, paged stash and DualPipeV fwdbwd match the reviewed main
byte for byte. The upstream-covered dense/NVML-only regressions also leave #104.
The 44 generic DualPipeV P2P waits and their observation regression remain.

The active FlagScale overlay has only the three observation files. TE and DeepEP
repair patch payloads are removed. Docker still installs the immutable prebuilt
wheels with their original hashes and records the historical patch hashes as
artifact provenance; it does not read the removed patch files. The installed
UserBuffer verifier remains useful for that older wheel. This is not a dependency
upgrade or a claim that the old Graph training path works without its batch repair.

1. Merge independent repairs into their owning target, then merge main into
   `feature/megalens-cuda` normally. No history rewrite is required.
2. Until then, configurations needing Hybrid-CP, DualPipeV without shared experts,
   optional Triton, or FlagScale dense/Graph repairs must use the owning fix build.
   Ordinary CP2/TE profiles do not use the Hybrid-CP scheduler.
3. Provide a compatible Core build containing #104 before activating FlagScale
   #1306. Its dense broadcast fix can merge independently.
4. Validate the installed versions, module origins and affected training paths
   before changing pinned dependency wheels or claiming new GPU results.

Frozen archives and historical reproduction packages retain their original
patches, versions and evidence. Autoresume FLOPs propagation and DualPipeV's
nondefault activation-checkpoint call require separate semantic work; they are
not repaired by passing zero or dropping an argument in these submissions.
