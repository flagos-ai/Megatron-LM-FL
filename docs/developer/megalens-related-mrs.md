# MegaLens related merge requests

[Megatron-LM-FL #104](https://github.com/flagos-ai/Megatron-LM-FL/pull/104) is the
main integration and coordination MR. It owns the implicit probes, Core facade,
training observation hooks, Trace runtime/export/loading, and the five retained
analysis families. Independent training and dependency fixes are reviewed in
their owning repositories with their own regression tests.

## Independent submissions

All seven submissions are Draft MRs. Follow their links for current merge status.

| Repository / target | MR | Change | Commit | Runtime or merge dependency |
|---|---|---|---|---|
| flagos-ai/Megatron-LM-FL / `main` | [#195](https://github.com/flagos-ai/Megatron-LM-FL/pull/195) | Core import compatibility without Triton | `39aaba7bc` | Required for the existing CPU/no-Triton import guarantee; merge independently into main. |
| flagos-ai/Megatron-LM-FL / `main` | [#196](https://github.com/flagos-ai/Megatron-LM-FL/pull/196) | Hybrid-CP helper calls and subsample metadata | `b1f7716dd` | Required only when using Hybrid-CP; ordinary CP2/TE profiles do not depend on it. |
| flagos-ai/Megatron-LM-FL / `main` | [#197](https://github.com/flagos-ai/Megatron-LM-FL/pull/197) | DualPipeV combined backward without shared experts | `e0bfd2bdd` | Required only for this native DualPipeV configuration; observation extensions remain separate. |
| flagos-ai/FlagScale / `main` | [#1305](https://github.com/flagos-ai/FlagScale/pull/1305) | Dense/packed batch broadcast correction | `7a27af2e1` | Independent of MegaLens; preserves SFT/Hybrid-CP packed transport. |
| flagos-ai/FlagScale / `main` | [#1306](https://github.com/flagos-ai/FlagScale/pull/1306) | Opt-in MegaLens lifecycle integration | `415429a0f` | Requires a compatible Core build containing #104; has no source dependency on the dense-batch PR. |
| flagos-ai/TransformerEngine-FL / `main` | [#133](https://github.com/flagos-ai/TransformerEngine-FL/pull/133) | CUDA normalization quantizer dtype forwarding | `65e0c5a1c` | Relevant to FP8 normalization using this CUDA adapter; native bindings and hardware validation pending. |
| deepseek-ai/DeepEP / `hybrid-ep` | [#775](https://github.com/deepseek-ai/DeepEP/pull/775) | Toolkit NVTX/CCCL build discovery | `d1c05b412` | Targets hybrid-ep, not main; full HybridEP JIT/device validation pending. |

The three native Megatron fixes add no MegaLens import, event or trace interface.
The DualPipeV fix does not include dedicated observation; the existing 44 generic
P2P wait observations remain in #104. The FlagScale integration changes three
training files and does not include the separate batch broadcast correction.

## Target revisions and adaptations

| Repository / target | Reviewed base |
|---|---|
| Megatron-LM-FL main | `b471409ac414d760c42a4b159725c2acd2676166` (v0.18.2) |
| FlagScale main | `f8e9a62c499d526a6ffa47f533bcf342330b7326` |
| TransformerEngine-FL main | `2457a3928291bb9944b7123ad84243d0964a114b` |
| DeepEP hybrid-ep | `10d4dd7377d5bce900fbb4b80cce863764892b95` |
| DeepEP main, inspected only | `93eb6eb238127e96c6d7a4a625a6dad158348509` |

- FlagScale's batch helper now lives in `training/utils/common_utils.py`.
  Its current training flow uses `step_batch_size_schedule`, which replaces the
  old overlay's dynamic-batch argument in the sample-range tracing restriction.
  Existing checkpoint, fault injection, RL and performance-monitor calls remain.
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

These are 75 CPU/contract checks. Restoring the reviewed upstream source for the
six fixes produces 19 expected failures in the corresponding regression suites.
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

Black/isort checks passed for the affected native/TE files and relevant tests;
FlagScale's new tests passed Ruff and formatting; all staged diffs passed
whitespace and scope checks. The 781 main-integration CPU checks and packaging
results remain evidence for source revision `690c9d9f096da6bfe4c578361986e7d62442274b`.
This coordination change leaves those implementation bytes unchanged.

## Merge and patch retirement

#104 temporarily retains its working native fixes and patches used by the pinned
older work image. Opening these MRs does not establish that a dependency is merged
or that a replacement build has been validated.

1. Merge the independent native fixes into Megatron main, then merge updated main
   into `feature/megalens-cuda` normally. Reconcile duplicate regressions at that
   point; no forced push or historical rewrite is required.
2. Make the MegaLens-enabled Core build available before merging/activating the
   FlagScale lifecycle integration. The dense broadcast fix can merge independently.
3. For external fixes, pin the accepted source/build version and validate the
   installed package, overlay import source and affected training profile before
   removing the corresponding old patch from the work image.
4. Recheck trace-off/on on the affected PP/DP/TP/EP profiles, checkpoints and trace
   lifecycle. FP8, Hybrid-CP, DualPipeV and HybridEP require their own device paths.

Frozen archives and historical reproduction packages retain their original
patches, versions and evidence.
