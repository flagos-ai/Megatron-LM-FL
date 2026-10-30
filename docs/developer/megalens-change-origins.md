# MegaLens MR change origins

This inventory describes the reviewed implementation snapshot
`b1fe22a2b43cc73c34ae05cfb9c38810e42a2b01` relative to main
`bbc208e00f2a201783938ef549f5058d6047e4c0`. MixedPara is fixed at
`12fb7169ce30fdb62b50f86b41afa09336a523ea`; its configured source repository is
[Synlvejo/MixedPara](https://github.com/Synlvejo/MixedPara/tree/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens).
The inventory and P2P TODO are documentation added after this snapshot; their
own lines are outside the frozen totals below.

The snapshot has **52,214 additions / 730 deletions across 133 files**:
101 new files and 32 modified files. Counts are Git diff lines, including
comments, blank lines and reindented statements. A file's origin is not a claim
that every line was copied unchanged or that every changed line adds training work.

## Complete line budget

| File-level origin / role | Files | Added | Deleted | Share of additions | Origin and purpose |
|---|---:|---:|---:|---:|---|
| MixedPara-derived modules | 12 | 13,558 | 0 | 25.97% | Retained Trace, loading, five analyses, plotting style and shared utilities, with migration revisions. |
| Target Probe infrastructure | 13 | 2,956 | 0 | 5.66% | New lightweight Core facade, implicit probes and runtime adapters around source producer behavior. |
| Target event/nesting support | 2 | 608 | 0 | 1.16% | New event catalog and nested-span accounting for retained consumers; no PIG pipeline. |
| Native integration changes | 23 | 1,991 | 714 | 3.81% | Observation hooks and API/backend/lifecycle adaptation in existing FL Core, training and shared DualPipeV waits. |
| Regression / contract tests | 40 | 21,810 | 1 | 41.77% | Migration, boundary, failure and consumer contracts; 39 new suites and one existing native suite delta. |
| Test fixtures / schemas | 12 | 1,473 | 0 | 2.82% | Ten training YAML files plus run-manifest schema and source/target scanner fixture. |
| Validation runners / scanner | 13 | 6,085 | 0 | 11.65% | Training/Trace/checkpoint contracts, gate/FlagScale/NCCL runners, scanner and wheel import checker. |
| Pinned work environment | 6 | 1,459 | 0 | 2.79% | Docker definition, observation-only FlagScale patch, environment script, wheel hashes and installed-adapter verifier. |
| Documentation / evidence | 7 | 1,898 | 0 | 3.64% | Quickstart, Core review guide, related-MR guide, machine-readable validation and navigation. |
| Packaging / CI / lock metadata | 5 | 376 | 15 | 0.72% | Package/extras registration, test group and formatter scope; uv.lock contains generated dependency metadata. |
| **Total** | **133** | **52,214** | **730** | **100%** | Each changed path belongs to exactly one row. |

Tests, fixtures and validation tools account for **29,368 additions (56.25%)**.
The 12 MixedPara-derived modules account for **13,558 additions (25.97%)**.
The production groups together account for 19,113 additions (36.61%): derived
modules, Probe infrastructure, event/nesting support and native integration.
The remaining 3,733 additions are work-environment, documentation and build data.
Displayed percentages are rounded independently.

## MixedPara source modules

These 12 target files have the same paths in the fixed MixedPara baseline. Their
source files total 11,973 lines; their current target files total 13,558 lines.
The target total includes compatibility changes, corrections, formatting and
previously reviewed target capabilities, rather than 13,558 unchanged copied lines.
None is byte-identical; `paper_style.py` and `utils.py` are AST-identical. The other
ten have AST differences and must be reviewed as adapted source.

Paths below are relative to `megatron/megalens/` on both sides.

| Source / target module | Source lines | Target additions | Migration / revision scope |
|---|---:|---:|---|
| [analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/analyzer.py) | 350 | 364 | Retained analysis CLI and aggregation imports; explicit timeline-alignment option. |
| [data_loader.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/data_loader.py) | 811 | 861 | Trace pairing, normalized event records and query contracts; field/missing-evidence compatibility. |
| [dp_analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/dp_analyzer.py) | 1,201 | 1,216 | Retained five-result DP analysis/report boundary; consumer compatibility. |
| [ep_analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/ep_analyzer.py) | 2,613 | 2,733 | Retained EP analysis and reports; adapted event fields and evidence handling. |
| [hardware_monitor.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/hardware_monitor.py) | 299 | 341 | CPU/memory/GPU sampling with optional-provider and NVML capability handling. |
| [hybrid_analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/hybrid_analyzer.py) | 1,403 | 1,423 | Retained combined analysis; actual PP/TP fields, missing evidence and PP-rank aggregation fixes. |
| [paper_style.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/paper_style.py) | 83 | 85 | Formatting only in the compared file; complete AST matches the source. |
| [pp_analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/pp_analyzer.py) | 1,711 | 1,784 | Retained PP formulas/reporting; incomplete-workload samples handled explicitly. |
| [tp_analyzer.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/tp_analyzer.py) | 1,596 | 2,122 | Retained TP analysis plus target catalog/nested-span accounting and missing-evidence handling. |
| [trace.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/trace.py) | 1,221 | 1,424 | Source Tracer adapted for rank-local writers, sink/runtime lifecycle, cleanup and continuous CUPTI windows. |
| [trace_aggregate.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/trace_aggregate.py) | 453 | 969 | Source aggregation plus rank-local causal ordering, explicit timeline alignment and kernel viewer coordinates. |
| [utils.py](https://github.com/Synlvejo/MixedPara/blob/12fb7169ce30fdb62b50f86b41afa09336a523ea/megatron/megalens/utils.py) | 232 | 236 | Formatting only in the compared file; complete AST matches the source. |

The baseline also contains PIG, mitigation, ProgressiveDecoupler and online
components. Those components are outside this MR's retained module set.

## Target additions and integration

The Probe-infrastructure group has no same-path files in the source baseline.
It contains three Core facade/compatibility modules (`core/observability.py`,
`core/tensor_parallel/observability.py`, `core/transformer/moe/observability.py`),
seven `megalens/probes/` files, `megalens/core_adapter.py`, `runtime.py` and package
`__init__.py`. They organize and adapt source observation behavior to FL APIs,
raw Work objects, backend paths and lazy trace-off gates. Their current form
includes the later implementation-consolidation refactors; it is not a second
copy of the training implementation.

`event_catalog.py` and `nested_aggregation.py` are separate target additions:
explicit event roles, missing-evidence states and nested physical/logical span
accounting for the retained analysis consumers. They are not baseline files or
PIG components.

The 23 native-integration paths comprise 19 Core files, three training files and
the shared DualPipeV schedule wait consumer. Their added/deleted counts include
scope indentation and normal call changes. MixedPara supplies the model and
PP/TP/DP/EP producer behavior; FL supplies the native APIs, control flow and
backend differences. Combined/fine-grained scopes, shared-expert overlap and
backend-specific completion/metadata adaptations are target revisions. The
[Core review guide](megalens-core-integration.md) records their boundaries.
The remaining `_p2p_groups` extraction is an
[open TODO](megalens-core-integration.md#open-review-todos), not a completed cleanup.

The 40 test paths contain 39 newly added `tests/megalens/test_*.py` suites and the
observation assertion added to the existing native DualPipeV schedule suite.
The 13 validation-tool paths comprise 11 `tests/test_utils/runners/` files,
`tools/probe_contract_scan.py` and `tests/megalens/wheel_import_check.py`.
The source baseline has no matching paths for these tests, fixtures or tools;
they were built for this migration's source/target and FL integration contracts.
CPU/stub contracts and device runners have distinct evidence limits; their
presence does not mean every device profile has run on the current snapshot.

The 940-line `flagscale-megalens.patch` is diff text for the three observation
entry files at fixed FlagScale `6d775cd01d5c822f9413b9952a81652c917d273e`.
Its line count includes context, removed lines and reindented native statements.
The latest-main port is independently reviewed in FlagScale #1306. The work-image
files pin existing wheels and source provenance; TE/DeepEP repair payloads and
FlagScale's independent dense-batch fix have already left this MR.

The documentation group includes a 1,124-line validation JSON; the build group
includes `uv.lock` (+347/-9), which records resolved package metadata. These are
review/evidence and generated dependency records, respectively.
Independent repairs #195/#196/#197/#209/#210, FlagScale #1305 and TE/DeepEP fixes
remain in their [owning MRs](megalens-related-mrs.md), outside the main repair diff.

## Reproduce the inventory

```bash
rtk proxy git diff --numstat bbc208e00f2a201783938ef549f5058d6047e4c0...b1fe22a2b43cc73c34ae05cfb9c38810e42a2b01
rtk proxy git diff --name-status bbc208e00f2a201783938ef549f5058d6047e4c0...b1fe22a2b43cc73c34ae05cfb9c38810e42a2b01
```

For the source comparison, read each listed module at MixedPara
`12fb7169ce30fdb62b50f86b41afa09336a523ea` and compare it with the target snapshot.
AST equality ignores locations; byte equality uses the unmodified file content.
This documentation change does not rerun or upgrade any runtime/GPU evidence.

<details>
<summary>All 133 paths and their file-level origin assignment</summary>

| Changed path | Added | Deleted | Origin / role |
|---|---:|---:|---|
| `.dockerignore` | 18 | 0 | Pinned work environment |
| `.github/configs/cuda.yml` | 3 | 0 | Packaging / CI / lock metadata |
| `MANIFEST.in` | 1 | 0 | Packaging / CI / lock metadata |
| `README.md` | 1 | 0 | Documentation / evidence |
| `docker/Dockerfile.work` | 433 | 0 | Pinned work environment |
| `docker/common/hybridep-cu128-env.sh` | 6 | 0 | Pinned work environment |
| `docker/patches/flagscale-megalens.patch` | 940 | 0 | Pinned work environment |
| `docker/verify_transformer_engine_userbuffer.py` | 44 | 0 | Pinned work environment |
| `docker/work-wheels.sha256` | 18 | 0 | Pinned work environment |
| `docs/developer/megalens-core-integration.md` | 166 | 0 | Documentation / evidence |
| `docs/developer/megalens-core-validation.json` | 1,124 | 0 | Documentation / evidence |
| `docs/developer/megalens-related-mrs.md` | 137 | 0 | Documentation / evidence |
| `docs/index.md` | 1 | 0 | Documentation / evidence |
| `docs/user-guide/features/index.md` | 1 | 0 | Documentation / evidence |
| `docs/user-guide/features/megalens.md` | 468 | 0 | Documentation / evidence |
| `megatron/core/distributed/finalize_model_grads.py` | 84 | 15 | Native integration changes |
| `megatron/core/distributed/param_and_grad_buffer.py` | 122 | 56 | Native integration changes |
| `megatron/core/models/bert/bert_model.py` | 15 | 6 | Native integration changes |
| `megatron/core/models/gpt/fine_grained_callables.py` | 28 | 4 | Native integration changes |
| `megatron/core/models/gpt/gpt_model.py` | 57 | 40 | Native integration changes |
| `megatron/core/observability.py` | 287 | 0 | Target Probe infrastructure |
| `megatron/core/optimizer/optimizer.py` | 14 | 3 | Native integration changes |
| `megatron/core/pipeline_parallel/combined_1f1b.py` | 31 | 0 | Native integration changes |
| `megatron/core/pipeline_parallel/p2p_communication.py` | 73 | 17 | Native integration changes |
| `megatron/core/pipeline_parallel/schedules.py` | 434 | 164 | Native integration changes |
| `megatron/core/tensor_parallel/layers.py` | 81 | 11 | Native integration changes |
| `megatron/core/tensor_parallel/mappings.py` | 45 | 11 | Native integration changes |
| `megatron/core/tensor_parallel/observability.py` | 16 | 0 | Target Probe infrastructure |
| `megatron/core/transformer/attention.py` | 8 | 0 | Native integration changes |
| `megatron/core/transformer/mlp.py` | 8 | 0 | Native integration changes |
| `megatron/core/transformer/moe/moe_layer.py` | 178 | 47 | Native integration changes |
| `megatron/core/transformer/moe/observability.py` | 52 | 0 | Target Probe infrastructure |
| `megatron/core/transformer/moe/router.py` | 83 | 1 | Native integration changes |
| `megatron/core/transformer/moe/shared_experts.py` | 39 | 5 | Native integration changes |
| `megatron/core/transformer/moe/token_dispatcher.py` | 135 | 53 | Native integration changes |
| `megatron/core/transformer/transformer_layer.py` | 27 | 7 | Native integration changes |
| `megatron/core/utils.py` | 53 | 3 | Native integration changes |
| `megatron/megalens/__init__.py` | 8 | 0 | Target Probe infrastructure |
| `megatron/megalens/analyzer.py` | 364 | 0 | MixedPara-derived modules |
| `megatron/megalens/core_adapter.py` | 78 | 0 | Target Probe infrastructure |
| `megatron/megalens/data_loader.py` | 861 | 0 | MixedPara-derived modules |
| `megatron/megalens/dp_analyzer.py` | 1,216 | 0 | MixedPara-derived modules |
| `megatron/megalens/ep_analyzer.py` | 2,733 | 0 | MixedPara-derived modules |
| `megatron/megalens/event_catalog.py` | 238 | 0 | Target event/nesting support |
| `megatron/megalens/hardware_monitor.py` | 341 | 0 | MixedPara-derived modules |
| `megatron/megalens/hybrid_analyzer.py` | 1,423 | 0 | MixedPara-derived modules |
| `megatron/megalens/nested_aggregation.py` | 370 | 0 | Target event/nesting support |
| `megatron/megalens/paper_style.py` | 85 | 0 | MixedPara-derived modules |
| `megatron/megalens/pp_analyzer.py` | 1,784 | 0 | MixedPara-derived modules |
| `megatron/megalens/probes/__init__.py` | 9 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/dp.py` | 499 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/moe.py` | 522 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/p2p.py` | 747 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/pipeline.py` | 184 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/tp.py` | 339 | 0 | Target Probe infrastructure |
| `megatron/megalens/probes/training.py` | 124 | 0 | Target Probe infrastructure |
| `megatron/megalens/runtime.py` | 91 | 0 | Target Probe infrastructure |
| `megatron/megalens/tp_analyzer.py` | 2,122 | 0 | MixedPara-derived modules |
| `megatron/megalens/trace.py` | 1,424 | 0 | MixedPara-derived modules |
| `megatron/megalens/trace_aggregate.py` | 969 | 0 | MixedPara-derived modules |
| `megatron/megalens/utils.py` | 236 | 0 | MixedPara-derived modules |
| `megatron/plugin/dualpipev/dualpipev_schedules.py` | 137 | 45 | Native integration changes |
| `megatron/training/arguments.py` | 12 | 0 | Native integration changes |
| `megatron/training/global_vars.py` | 62 | 0 | Native integration changes |
| `megatron/training/training.py` | 265 | 226 | Native integration changes |
| `pyproject.toml` | 15 | 0 | Packaging / CI / lock metadata |
| `tests/megalens/fixtures/c5_2_run_manifest.schema.json` | 222 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_cp2_te_smoke.yaml` | 97 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_dp2_distopt_overlap_smoke.yaml` | 97 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_dp2_standard_overlap_smoke.yaml` | 96 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_ep2_smoke.yaml` | 102 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_gpt_eager_continuous_cupti_smoke.yaml` | 95 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_pp2_smoke.yaml` | 94 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_smoke.yaml` | 94 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_te_cuda_graph_full_cupti_smoke.yaml` | 99 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml` | 97 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/flagscale_single_node_tp2_sp_local_smoke.yaml` | 97 | 0 | Test fixtures / schemas |
| `tests/megalens/fixtures/probe_scan_gate.json` | 283 | 0 | Test fixtures / schemas |
| `tests/megalens/test_all_grads_sync_observability.py` | 475 | 0 | Regression / contract tests |
| `tests/megalens/test_analyzer.py` | 384 | 0 | Regression / contract tests |
| `tests/megalens/test_bert_encoder_observability.py` | 176 | 0 | Regression / contract tests |
| `tests/megalens/test_core_adapter.py` | 123 | 0 | Regression / contract tests |
| `tests/megalens/test_cp_probe_contract.py` | 368 | 0 | Regression / contract tests |
| `tests/megalens/test_dp_analyzer.py` | 281 | 0 | Regression / contract tests |
| `tests/megalens/test_dp_grad_sync_observability.py` | 1,096 | 0 | Regression / contract tests |
| `tests/megalens/test_dp_probe_contract.py` | 243 | 0 | Regression / contract tests |
| `tests/megalens/test_embedding_grads_allreduce_observability.py` | 736 | 0 | Regression / contract tests |
| `tests/megalens/test_ep_analyzer.py` | 1,051 | 0 | Regression / contract tests |
| `tests/megalens/test_ep_primitive_observability.py` | 994 | 0 | Regression / contract tests |
| `tests/megalens/test_event_catalog.py` | 246 | 0 | Regression / contract tests |
| `tests/megalens/test_flagscale_run_gate.py` | 968 | 0 | Regression / contract tests |
| `tests/megalens/test_gate_runner.py` | 168 | 0 | Regression / contract tests |
| `tests/megalens/test_gpt_loss_observability.py` | 204 | 0 | Regression / contract tests |
| `tests/megalens/test_gpt_model_phase_observability.py` | 240 | 0 | Regression / contract tests |
| `tests/megalens/test_hybrid_analyzer.py` | 577 | 0 | Regression / contract tests |
| `tests/megalens/test_moe_phase_observability.py` | 2,002 | 0 | Regression / contract tests |
| `tests/megalens/test_nested_aggregation.py` | 498 | 0 | Regression / contract tests |
| `tests/megalens/test_observability.py` | 382 | 0 | Regression / contract tests |
| `tests/megalens/test_offline_contracts.py` | 675 | 0 | Regression / contract tests |
| `tests/megalens/test_p2p_observability.py` | 1,462 | 0 | Regression / contract tests |
| `tests/megalens/test_p2p_order_nccl_runner.py` | 31 | 0 | Regression / contract tests |
| `tests/megalens/test_pipeline_schedule_observability.py` | 1,070 | 0 | Regression / contract tests |
| `tests/megalens/test_pp_analyzer.py` | 544 | 0 | Regression / contract tests |
| `tests/megalens/test_probe_contract_scan.py` | 330 | 0 | Regression / contract tests |
| `tests/megalens/test_run_manifest.py` | 278 | 0 | Regression / contract tests |
| `tests/megalens/test_runtime_lifecycle.py` | 997 | 0 | Regression / contract tests |
| `tests/megalens/test_shared_expert_overlap_observability.py` | 344 | 0 | Regression / contract tests |
| `tests/megalens/test_sp_layernorm_allreduce_observability.py` | 499 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_allgather_observability.py` | 440 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_allreduce_observability.py` | 234 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_analyzer.py` | 895 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_linear_observability.py` | 796 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_probe_contract.py` | 509 | 0 | Regression / contract tests |
| `tests/megalens/test_tp_reduce_scatter_observability.py` | 620 | 0 | Regression / contract tests |
| `tests/megalens/test_training_optimizer_observability.py` | 257 | 0 | Regression / contract tests |
| `tests/megalens/test_transformer_layer_phase_observability.py` | 286 | 0 | Regression / contract tests |
| `tests/megalens/test_work_image_contract.py` | 293 | 0 | Regression / contract tests |
| `tests/megalens/wheel_import_check.py` | 187 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/check_ring_exchange.py` | 33 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/cp_probe_contract.py` | 374 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/dp_probe_contract.py` | 380 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/gpt_probe_contract.py` | 815 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/megalens_run_manifest.py` | 424 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/p2p_probe_contract.py` | 584 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/run_flagscale_megalens.py` | 637 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/run_megalens_gate.py` | 208 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/run_p2p_order_nccl.py` | 220 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/tp_probe_contract.py` | 1,220 | 0 | Validation runners / scanner |
| `tests/test_utils/runners/training_run_contract.py` | 442 | 0 | Validation runners / scanner |
| `tests/unit_tests/a2a_overlap/test_schedule_dualpipev.py` | 38 | 1 | Regression / contract tests |
| `tools/autoformat.sh` | 10 | 6 | Packaging / CI / lock metadata |
| `tools/probe_contract_scan.py` | 561 | 0 | Validation runners / scanner |
| `uv.lock` | 347 | 9 | Packaging / CI / lock metadata |

</details>
