# Megatron-LM-FL SlideFormer Plugin

## 目标

这个补丁把 SlideFormer 的显存卸载逻辑迁入 `flagos-ai/Megatron-LM-FL`，让 Megatron-LM-FL 作为真正的训练后端承载 offload 能力。FlagScale 后续只负责前端启动训练，并通过环境变量把开关传给 Megatron-LM-FL。

当前实现不是新增一个独立后端，而是在 Megatron-LM-FL 内部以 plugin 形式接入：

- `megatron/plugin/slideformer/config.py`
- `megatron/plugin/slideformer/layout.py`
- `megatron/plugin/slideformer/checkpoint.py`
- `megatron/plugin/slideformer/engine.py`
- `megatron/plugin/slideformer/kernels.py`
- `megatron/training/training.py`

## 2026-07-31 单卡验收状态

单卡功能、正确性、CPU/GPU memory 和 BS64 吞吐验收均已通过。正式训练入口会在建模前选择必须影响
`TransformerConfig` 的开关，在建模后扫描实际模块结构并应用兼容 kernel。
默认 `kernel_policy=auto`，并且 `strict_kernels=true`；兼容模型不会依赖用户逐项
手动配置。

默认优先级如下：

| 模块 | 默认实现 | 不兼容或缺依赖时 |
| --- | --- | --- |
| dense causal attention | TE auto（Qwen3 实测选择 FlashAttention 2.8.3） | 显式选择 local backend |
| dense SwiGLU | TE MCore fused SwiGLU；超大 FC1 自动拆分 gate/up GEMM | 有 bias 时保留 Megatron bias-activation fusion |
| LM head + CE | SlideFormer Legacy LCE | 显式选择现代 `liger` 或 `megatron` |
| RMSNorm（含 Q/K norm） | TE MCore RMSNorm | 显式选择 local backend |
| standard dense RoPE | TE fused RoPE | 显式选择 local backend |

这里不使用 Liger 的 Hugging Face `AutoModel` monkey patch。MCore/TE 负责模型结构
kernel；Liger 提供 Legacy LCE 的 Triton CE kernel，以及超大 FC1 拆分路径中的
SiLU×Mul operation。这样无需维护一份 Qwen 专用自动替换表，也不会把 HF 模型结构
假设带入 Megatron。

对于单次 FC1 输出达到 4 GiB 的 bias-free SwiGLU，默认策略会保留 TE/Megatron
拼接后的 FC1 权重和 checkpoint 格式，但在 forward 中把它作为 gate/up 两个 view
执行两次 GEMM，再调用 Liger SiLU×Mul。这样兼容原有 checkpoint，同时避免 14B、
BS64 时单个约 4.25 GiB 临时张量造成的 CUDA allocator cache retry。阈值可通过
`MEGATRON_SLIDEFORMER_SPLIT_SWIGLU_THRESHOLD_GIB` 调整，设为 `0` 可关闭。

Qwen3-14B、BF16、seq=1024、BS64、3 warmup + 10 measured 的正式复测中，自动拆分
把 Megatron-LM-FL 从 54.8709 s/step 提升至 50.4093 s/step（1300.1 tokens/s），
peak allocated 从 22.041 GiB 降至 19.924 GiB，peak reserved 从 22.896 GiB 降至
22.273 GiB，allocator retry 从 529 降至 0。相对同协议 torch native 的
1405.3 tokens/s 仍慢约 7.5%，因此这项修正解决的是最明显的 FC1 分配问题，不能宣称
14B 已达到完全性能一致。

早期同一台 RTX 4090、同一 Qwen3-8B checkpoint、BF16、seq=1024、BS64、
3 warmup + 10 measured 的单点结果为：

```text
native SlideFormer:       27.3295 s/step, 2398.0 tokens/s, 15.645 GiB allocated
Megatron-LM-FL TE+Legacy: 27.7702 s/step, 2359.9 tokens/s, 14.428 GiB allocated
throughput delta:         -1.59%（早期单点）
allocated-memory delta:   -7.78%
```

后续同协议 BS1-64 全曲线复测中，BS64 native 为 2431.8 tokens/s，
Megatron-LM-FL 为 2338.8 tokens/s，差距为 -3.83%；因此当前不能宣称已经消除
约 4% 的高 batch 差距。TE debug 在 BS64/S1024/Qwen3 形状下
确认 `auto` 选择 FlashAttention 2.8.3。强制只允许 FlashAttention 的同协议复测
为 28.1621 s/step，并未带来收益，因此默认保留 TE 的兼容性选择逻辑。

最终实现进一步完成了三项生命周期修正：

- TE fused linear 的 wgrad GEMM 直接写入有 event/lease 保护的整层 GPU gradient slot，
  每个 TransformerLayer 只提交一次整层 D2H。
- checkpoint 外层 backward boundary 负责在 TE 自定义 autograd 完成后释放层；
  不在 backward 中把 TE 保存的 `Parameter.data` 替换成 CPU storage。
- GPU 参数使用 embedding/output 单大槽和 Transformer/final-norm 双滑动槽；
  activation prefetch 与参数 prefetch 共用深度 3 的 H2D 调度器。

同一 RTX 4090、Qwen3-8B checkpoint、BF16、seq=1024、BS64、
3 warmup + 10 measured 的最终正式结果：

| 实现 | step time | tokens/s | peak allocated | peak reserved | CPU RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SlideFormer torch native | 27.3295 s | 2398.0 | 15.645 GiB | 17.484 GiB | 约 132.955 GiB |
| Megatron-LM-FL + SlideFormer | 27.5889 s | 2375.4 | 16.005 GiB | 18.895 GiB | 117.003 GiB |

Megatron-LM-FL 吞吐差距为 -0.94%，allocated 多 0.360 GiB，reserved 多
1.410 GiB；10 个 measured loss 与同步安全参考的最大差约 `2e-4`，
allocator retry 和 OOM 均为 0。相比修正后但仍逐参数 D2H 的安全版本
30.7945 s / 22.955 GiB reserved，最终版本快 10.4%，并减少 4.061 GiB reserved。

CPU RSS 差异主要不是 sliding checkpoint 语义不同。两者都为 36 层保留约
18 GiB 的 BF16 boundary activation slots；Megatron 的显式统计为
30.513 GiB FP32 master、61.026 GiB Adam 状态、2.318 GiB shared gradients、
1.159 GiB BF16 staging 和 18 GiB activation slots，共 113.016 GiB unique。
torch native 的更高 RSS 主要来自 HF checkpoint/model materialization、运行时临时副本
和 allocator 高水位。Megatron reserved 原先较高则来自逐 parameter CUDA allocation
和碎片；共享平坦参数槽已把它从 22.955 GiB 降至 18.895 GiB。

## 当前支持范围

已支持：

- 单卡 Megatron GPTModel 类结构。
- `TP=1`、`PP=1`。
- decoder layer 参数 CPU master 常驻。
- layer 参数 forward/backward 分段 load/offload。
- decoder layer 梯度拷回 CPU。
- 默认复用有 lease/event 保护的最大层 CPU gradient 和 BF16 参数 staging
  缓冲，而不是为每层永久分配两份 execution-size CPU tensors。
- per-layer CPU Adam 更新，默认使用从 SlideFormer 主线迁入的 LayerAdam C++ fast path。
- embedding、final norm、output layer 等非 decoder-layer 参数也由 SlideFormer LayerAdam static groups 管理。
- activation offload 使用 saved tensor hooks + non-reentrant checkpoint。
- chunked H2D/D2H copy，默认 32MiB chunk。
- SlideFormer-owned state checkpoint 保存/恢复，路径为 `${save}/slideformer/latest.pt`。
- 可配置 LayerAdam NVMe optimizer-state offload 通路，默认关闭。
- gradient D2H 与 CPU Adam 调度重叠默认开启，可通过
  `MEGATRON_SLIDEFORMER_OVERLAP_GRAD_D2H_CPU_ADAM=0` 显式关闭。
- FlagScale 可通过环境变量启用 Megatron-LM-FL 插件。

暂不完整支持：

- `DP>1`、`TP>1`、`PP>1`、`EP>1`。
- Megatron distributed optimizer 与 SlideFormer-managed decoder 参数共同管理。
- latest OOM wrapper 等价的 full transfer-compute overlap。当前已有可选 GPU double buffer pool，但默认关闭，因为当前 24L 测试中会变慢。
- MoE end-to-end validation：当前任务范围明确不跑。

运行时依赖：

- 与当前 PyTorch/CUDA ABI 匹配的 `transformer-engine[pytorch]`。
- `flash-attn`，用于默认的 TE FlashAttention dispatch。
- `liger-kernel`，仅用于 Legacy LCE 的 Triton CE kernel。
- `ninja` 和可用的 C++ 编译器，用于首次构建 LayerAdam CPU 扩展。
- `tensornvme`，仅在启用 NVMe optimizer-state offload 时需要。

## 启用方式

直接运行 Megatron-LM-FL：

```bash
export MEGATRON_SLIDEFORMER_ENABLE=1
export MEGATRON_SLIDEFORMER_MODE=true_slideformer
export MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD=1
export MEGATRON_SLIDEFORMER_PARAM_PREFETCH=1
export MEGATRON_SLIDEFORMER_CHUNK_SIZE_MB=32
export MEGATRON_SLIDEFORMER_NVME_OFFLOAD_FRACTION=0.0
export MEGATRON_SLIDEFORMER_OFFLOAD_DIR=/path/to/offload_dir
export MEGATRON_SLIDEFORMER_SHARED_CPU_BUFFERS=1
# 0 表示跟随 optimizer pipeline depth；当前安全默认深度为 2
export MEGATRON_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT=0
export MEGATRON_SLIDEFORMER_CPU_PARAM_STAGING_BUFFER_COUNT=1
export MEGATRON_SLIDEFORMER_TE_FUSED_MAIN_GRAD=1
export MEGATRON_SLIDEFORMER_ACTIVATION_SLOT_PREFETCH=1
export MEGATRON_SLIDEFORMER_MAX_OUTSTANDING_H2D=3
export MEGATRON_SLIDEFORMER_KERNEL_POLICY=auto
export MEGATRON_SLIDEFORMER_ATTENTION_BACKEND=auto
export MEGATRON_SLIDEFORMER_MLP_BACKEND=auto
# 自动拆分 >=4 GiB 的 SwiGLU FC1 输出；设为 0 可禁用
export MEGATRON_SLIDEFORMER_SPLIT_SWIGLU_THRESHOLD_GIB=4.0
export MEGATRON_SLIDEFORMER_LOSS_BACKEND=auto
export MEGATRON_SLIDEFORMER_NORM_BACKEND=auto
export MEGATRON_SLIDEFORMER_ROPE_BACKEND=auto
export MEGATRON_SLIDEFORMER_STRICT_KERNELS=1
```

兼容旧的 FlagScale 环境变量：

```bash
export FLAGSCALE_SLIDEFORMER_MEGATRON=1
export FLAGSCALE_SLIDEFORMER_MODE=true_slideformer
export FLAGSCALE_SLIDEFORMER_ACTIVATION_OFFLOAD=1
```

## 训练入口行为

启用后，`megatron/training/training.py` 会：

1. 校验 `DP=TP=PP=EP=1`、`num_microbatches=1`，并拒绝 Megatron native recompute。
2. 建模前将全 `auto` 的结构 kernel 解析为 MCore Transformer Engine spec；
   strict 模式要求 FlashAttention 可用，实际 dispatch 仍由 TE 按输入选择。
3. 构建原生 Megatron GPTModel。
4. 建模后确认 TE attention/MLP/RMSNorm/RoPE 与 Legacy LCE 的实际生效状态，
   并打印结构化报告；显式 local backend 仍保留旧 adapter 兼容路径。
5. 不包 Megatron DDP。
6. 不构建 Megatron 原生 optimizer。
7. 对 model chunk 应用 `apply_true_megatron_slideformer()`。
8. decoder layer 参数、梯度、Adam 状态由 SlideFormer plugin 管理。
9. embedding、final norm、output layer 等 static groups 由 SlideFormer LayerAdam 管理。

当前如果训练配置触发 `--save`，Megatron 原生 checkpoint 会被跳过，避免 `optimizer=None` 和 CPU-owned decoder state 进入原生保存路径导致崩溃。SlideFormer engine 会保存自己的状态：

```text
${save}/slideformer/iter_XXXXXXX.pt
${save}/slideformer/latest.pt
```

如果 `--load` 目录下存在 `slideformer/latest.pt`，SlideFormer engine 会恢复 CPU master 参数、LayerAdam 状态和 iteration，并跳过 Megatron 原生 checkpoint load。

## 已验证命令

### 单元测试

```bash
PYTHONPATH=$PWD \
CUDA_VISIBLE_DEVICES=0 \
pytest -q tests/unit_tests/slideformer/test_slideformer_plugin.py
```

结果：

```text
22 passed
```

### Megatron-LM-FL 训练 smoke

```bash
PYTHONPATH=$PWD \
CUDA_VISIBLE_DEVICES=0 \
MEGATRON_SLIDEFORMER_ENABLE=1 \
MEGATRON_SLIDEFORMER_MODE=true_slideformer \
MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  pretrain_gpt.py \
  --num-layers 2 \
  --hidden-size 128 \
  --num-attention-heads 4 \
  --seq-length 64 \
  --max-position-embeddings 64 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --train-iters 2 \
  --lr 0.0002 \
  --min-lr 0.0002 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --tokenizer-type NullTokenizer \
  --vocab-size 512 \
  --mock-data \
  --bf16 \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --no-save-optim \
  --no-save-rng \
  --save /tmp/megatron_lm_fl_slideformer_smoke_ckpt \
  --save-interval 2 \
  --eval-interval 1000 \
  --log-interval 1
```

结果：

```text
Megatron-LM-FL SlideFormer true engine enabled; kernel report:
  attention=transformer_engine_auto_attention
  mlp=transformer_engine_swiglu
  loss=slideformer_legacy_linear_cross_entropy
  norm=transformer_engine_rmsnorm
  rope=transformer_engine_fused_rope
iteration 1/2 lm loss: 6.506352E+00
iteration 2/2 lm loss: 6.497377E+00
exit code: 0
```

## benchmark 结果

benchmark 脚本：

```text
tests/unit_tests/slideformer/benchmark_megatron_slideformer_alignment.py
```

### 2L / H128 / S64 / B1 / BF16

```text
baseline step: 0.006824s
SlideFormer step: 0.018420s
speed ratio: 0.3705x
baseline peak: 0.0203 GB
SlideFormer peak: 0.0179 GB
memory ratio: 0.8795x
max loss diff: 2.91e-4
```

### 12L / H768 / S512 / B1 / BF16

```text
baseline step: 0.032502s
SlideFormer step: 0.110543s
speed ratio vs Megatron baseline: 0.2940x
baseline peak: 0.9542 GB
SlideFormer peak: 0.1618 GB
memory ratio: 0.1696x
max loss diff: 2.52e-5
```

### 24L / H1024 / S1024 / B1 / BF16

```text
baseline step: 0.078795s
SlideFormer step: 0.331776s
speed ratio vs Megatron baseline: 0.2377x
baseline peak: 4.9546 GB
SlideFormer peak: 0.4466 GB
memory ratio: 0.0901x
max loss diff over 3 measured steps: 1.35e-3
one-update max loss diff: 7.78e-6
```

## 当前结论

当前 Megatron-LM-FL 插件已经完成了从 FlagScale prototype 到 Megatron-LM-FL 后端的主体迁移：

- loss 对齐通过。
- 单卡训练入口可跑通。
- 显存峰值显著下降，24L/H1024/S1024 下约为 Megatron baseline 的 9.0%。
- 默认 LayerAdam fast path 后，24L/H1024/S1024 step time 为 0.332s，快于此前同配置附近记录的 SlideFormer 主线约 0.461s。相对 Megatron baseline 仍慢，因为 offload 本身引入 CPU/GPU 传输和 CPU optimizer 开销。

后续优先级是整理单卡 PR、增加长时间训练与 checkpoint 恢复 CI；DP/TP/PP/EP
仍按当前任务边界留到后续工作。
