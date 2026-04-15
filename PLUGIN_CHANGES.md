# Plugin Changes (base → main)

Generated: 2026-04-15
Base: `0112156741214c833a47d9f92911400ae96d724d` (upstream core_v0.15.0rc7)
Main: `de0f47360450f1a179a4dd792ea02324856abdfa` (fork HEAD)

## Summary

- **182 files changed**, +15,689 / -713 lines
- **35 new plugin files** in `megatron/plugin/`
- **90+ modified core files** in `megatron/core/` (cur_platform replacements, @overridable decorators, hetero/dualpipev imports)
- **489 cur_platform call sites** across 82 core files
- **8 @overridable decorators** in core, **8 @override implementations** in plugin
- **27 CI/CD files** added or modified

---

## New Files (added by fork)

### Plugin System (`megatron/plugin/`)
```
megatron/plugin/__init__.py
megatron/plugin/decorators.py                          ← @overridable/@override decorator system
megatron/plugin/distributed/__init__.py
megatron/plugin/distributed/finalize_model_grads.py     ← @override for finalize_model_grads
megatron/plugin/dualpipev/dualpipev_schedules.py        ← DualPipeV pipeline schedules
megatron/plugin/dualpipev/fb_overlap/gpt_model.py
megatron/plugin/dualpipev/fb_overlap/modules/attention.py
megatron/plugin/dualpipev/fb_overlap/modules/token_dispatcher.py
megatron/plugin/dualpipev/fb_overlap/modules/utils.py
megatron/plugin/dualpipev/fb_overlap/overlap_funcs/bwd.py
megatron/plugin/dualpipev/fb_overlap/overlap_funcs/fwd.py
megatron/plugin/dualpipev/fb_overlap/overlap_funcs/fwdbwd.py
megatron/plugin/dualpipev/fb_overlap/transformer_block.py
megatron/plugin/dualpipev/fb_overlap/transformer_layer.py
megatron/plugin/hetero/__init__.py
megatron/plugin/hetero/p2p_communication.py             ← Heterogeneous p2p communication
megatron/plugin/hetero/parallel_context.py              ← Heterogeneous parallel context
megatron/plugin/models/__init__.py
megatron/plugin/models/common/__init__.py
megatron/plugin/models/common/language_module/__init__.py
megatron/plugin/models/common/language_module/language_module.py  ← @override for LanguageModule
megatron/plugin/optimizer/__init__.py
megatron/plugin/optimizer/clip_grads.py                 ← @override for clip_grads
megatron/plugin/optimizer/optimizer.py                  ← @override for optimizer methods
megatron/plugin/optimizer_param_scheduler.py            ← @override for OptimizerParamScheduler
megatron/plugin/platform/__init__.py
megatron/plugin/platform/platform_base.py               ← Base platform abstraction
megatron/plugin/platform/platform_cpu.py
megatron/plugin/platform/platform_cuda.py               ← NVIDIA CUDA platform
megatron/plugin/platform/platform_manager.py
megatron/plugin/platform/platform_musa.py               ← MetaX MUSA platform
megatron/plugin/platform/platform_register.py
megatron/plugin/platform/platform_txda.py               ← Hygon TXDA platform
megatron/plugin/tests/test_override_manager.py
megatron/plugin/utils.py
```

### CI/CD & Infrastructure
```
.github/configs/cuda.yml
.github/configs/metax.yml
.github/configs/template.yml
.github/workflows/all_tests_common.yml
.github/workflows/all_tests_cuda.yml
.github/workflows/all_tests_metax.yml
.github/workflows/flagscale-integration-tests.yml
.github/workflows/functional_tests_common.yml
.github/workflows/unit_tests_common.yml
docker/Dockerfile.fl.ci
```

### Tests & Other
```
megatron/__init__.py
megatron/core/optimizer/muon.py
tests/UNIT_TEST_GUIDE.md
tests/conftest.py
tests/functional_tests/run_functional_test_demo.sh
tests/test_utils/runners/check_results.py
tests/test_utils/runners/helpers.py
tests/test_utils/runners/parse_config.py
tests/test_utils/runners/run_functional_tests.sh
tests/test_utils/runners/run_tests.sh
tests/test_utils/runners/run_unit_tests.sh
tests/test_utils/runners/utils.sh
```

---

## @overridable Decorators in `megatron/core/` (8 total)

| File | Function | Line |
|------|----------|------|
| `megatron/core/distributed/finalize_model_grads.py` | `_allreduce_embedding_grad` | 206 |
| `megatron/core/optimizer/clip_grads.py` | `get_grad_norm_fp32` | 53 |
| `megatron/core/optimizer/clip_grads.py` | `count_zeros_fp32` | 186 |
| `megatron/core/optimizer/optimizer.py` | `MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan` | 498 |
| `megatron/core/optimizer/optimizer.py` | `ChainedOptimizer.load_state_dict` | 1268 |
| `megatron/core/models/common/language_module/language_module.py` | `LanguageModule._is_in_embd_group` | 62 |
| `megatron/core/models/common/language_module/language_module.py` | `LanguageModule.setup_embeddings_and_output_layer` | 169 |
| `megatron/core/optimizer_param_scheduler.py` | `OptimizerParamScheduler.get_lr` | 144 |

## @override Implementations in `megatron/plugin/` (8 total)

| File | Target | Method |
|------|--------|--------|
| `megatron/plugin/distributed/finalize_model_grads.py` | `finalize_model_grads` | `_allreduce_embedding_grad` |
| `megatron/plugin/optimizer/clip_grads.py` | `clip_grads` | `get_grad_norm_fp32` |
| `megatron/plugin/optimizer/clip_grads.py` | `clip_grads` | `count_zeros_fp32` |
| `megatron/plugin/optimizer/optimizer.py` | `MixedPrecisionOptimizer` | `_unscale_main_grads_and_check_for_nan` |
| `megatron/plugin/optimizer/optimizer.py` | `ChainedOptimizer` | `load_state_dict` |
| `megatron/plugin/models/common/language_module/language_module.py` | `LanguageModule` | `_is_in_embd_group` |
| `megatron/plugin/models/common/language_module/language_module.py` | `LanguageModule` | `setup_embeddings_and_output_layer` |
| `megatron/plugin/optimizer_param_scheduler.py` | `OptimizerParamScheduler` | `get_lr` |

---

## Modified Files in `megatron/core/` (90 files)

All modifications fall into these categories:
1. **cur_platform replacement** — `torch.cuda.*` → `cur_platform.*` (489 call sites across 82 files)
2. **@overridable decoration** — 8 functions decorated in 5 files
3. **Plugin imports** — `from megatron.plugin.platform import get_platform` (67 files), `from megatron.plugin.decorators import overridable` (5 files), hetero/dualpipev imports (4 files)
4. **Hetero integration** — `parallel_state.py`, `p2p_communication.py`
5. **DualPipeV integration** — `schedules.py`, `transformer/module.py`

### Import pattern
The fork uses `from megatron.plugin.platform import get_platform` then `cur_platform = get_platform()` at module level.

### Files with cur_platform replacements (82 files)
```
megatron/core/datasets/blended_megatron_dataset_builder.py
megatron/core/datasets/retro/utils.py
megatron/core/dist_checkpointing/exchange_utils.py
megatron/core/dist_checkpointing/strategies/async_utils.py
megatron/core/dist_checkpointing/strategies/filesystem_async.py
megatron/core/dist_checkpointing/strategies/fully_parallel.py
megatron/core/dist_checkpointing/strategies/state_dict_saver.py
megatron/core/dist_checkpointing/strategies/torch.py
megatron/core/dist_checkpointing/strategies/two_stage.py
megatron/core/dist_checkpointing/strategies/zarr.py
megatron/core/dist_checkpointing/tensor_aware_state_dict.py
megatron/core/distributed/distributed_data_parallel.py
megatron/core/distributed/fsdp/mcore_fsdp_adapter.py
megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py
megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py
megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py
megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py
megatron/core/distributed/param_and_grad_buffer.py
megatron/core/energy_monitor.py
megatron/core/export/trtllm/trtllm_weights_converter/distributed_trtllm_model_weights_converter.py
megatron/core/extensions/kitchen.py
megatron/core/extensions/transformer_engine.py
megatron/core/fp8_utils.py
megatron/core/full_cuda_graph.py
megatron/core/fusions/fused_indices_converter.py
megatron/core/fusions/fused_mla_yarn_rope_apply.py
megatron/core/fusions/fused_pad_routing_map.py
megatron/core/inference/communication_utils.py
megatron/core/inference/contexts/attention_context/mamba_metadata.py
megatron/core/inference/contexts/attention_context/mha_metadata.py
megatron/core/inference/contexts/dynamic_block_allocator.py
megatron/core/inference/contexts/dynamic_context.py
megatron/core/inference/engines/dynamic_engine.py
megatron/core/inference/engines/static_engine.py
megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py
megatron/core/inference/model_inference_wrappers/t5/t5_inference_wrapper.py
megatron/core/inference/unified_memory.py
megatron/core/inference/utils.py
megatron/core/models/bert/bert_model.py
megatron/core/models/common/embeddings/rope_utils.py
megatron/core/models/common/embeddings/rotary_pos_embedding.py
megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
megatron/core/models/common/language_module/language_module.py
megatron/core/models/common/model_chunk_schedule_plan.py
megatron/core/models/gpt/fine_grained_callables.py
megatron/core/models/gpt/gpt_model.py
megatron/core/models/vision/clip_vit_model.py
megatron/core/nccl_allocator.py
megatron/core/optimizer/clip_grads.py
megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
megatron/core/optimizer/distrib_optimizer.py
megatron/core/optimizer/grad_scaler.py
megatron/core/optimizer/muon.py
megatron/core/optimizer/optimizer.py
megatron/core/parallel_state.py
megatron/core/pipeline_parallel/bridge_communicator.py
megatron/core/pipeline_parallel/combined_1f1b.py
megatron/core/pipeline_parallel/p2p_communication.py
megatron/core/pipeline_parallel/schedules.py
megatron/core/pipeline_parallel/utils.py
megatron/core/rerun_state_machine.py
megatron/core/ssm/mamba_mixer.py
megatron/core/tensor_parallel/data.py
megatron/core/tensor_parallel/layers.py
megatron/core/tensor_parallel/mappings.py
megatron/core/tensor_parallel/random.py
megatron/core/tensor_parallel/utils.py
megatron/core/timers.py
megatron/core/transformer/attention.py
megatron/core/transformer/cuda_graphs.py
megatron/core/transformer/dot_product_attention.py
megatron/core/transformer/mlp.py
megatron/core/transformer/module.py
megatron/core/transformer/moe/experts.py
megatron/core/transformer/moe/moe_utils.py
megatron/core/transformer/moe/router.py
megatron/core/transformer/moe/shared_experts.py
megatron/core/transformer/moe/token_dispatcher.py
megatron/core/transformer/multi_token_prediction.py
megatron/core/transformer/transformer_layer.py
megatron/core/transformer/utils.py
megatron/core/utils.py
```

---

## CI/CD & Build Changes

### New files
- `.github/configs/cuda.yml` — CUDA test configuration
- `.github/configs/metax.yml` — MetaX test configuration
- `.github/configs/template.yml` — Config template
- `.github/workflows/all_tests_common.yml` — Common test workflow
- `.github/workflows/all_tests_cuda.yml` — CUDA test workflow
- `.github/workflows/all_tests_metax.yml` — MetaX test workflow
- `.github/workflows/flagscale-integration-tests.yml` — FlagScale integration tests
- `.github/workflows/functional_tests_common.yml` — Functional test workflow
- `.github/workflows/unit_tests_common.yml` — Unit test workflow
- `docker/Dockerfile.fl.ci` — Fork CI Docker image

### Modified files
- `pyproject.toml` — Fork-specific metadata and build config
- `.github/workflows/` — 17 existing workflows modified for fork

---

## Conflict Priority Classification

### P0 — Sacred (never lose fork version)
- `megatron/plugin/**` (35 files)
- Core files with `@overridable`: `finalize_model_grads.py`, `clip_grads.py`, `optimizer.py`, `language_module.py`, `optimizer_param_scheduler.py`

### P1 — Careful (merge both sides)
- 82 core files with `cur_platform` replacements
- `pyproject.toml`
- Core files with hetero/dualpipev imports: `parallel_state.py`, `p2p_communication.py`, `schedules.py`, `module.py`, `moe_utils.py`

### P2 — Upstream-preferred (accept upstream)
- All other files not listed above
