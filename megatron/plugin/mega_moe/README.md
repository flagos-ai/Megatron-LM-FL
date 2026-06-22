# SM90 MegaMoE Plugin

基于 DeepGEMM 的 SM90 (Hopper) FP8 融合 MoE 前向和反向 kernel，适用于 Megatron-LM。

## 架构概览

MegaMoE 将整个 MoE 层（dispatch → L1 GEMM → activation → L2 GEMM → combine）融合为单个 kernel，通过 symmetric memory 实现跨 rank 通信，无需额外的 all-to-all collective。

### 代码结构

```
mega_moe/
├── __init__.py              # Python API 入口
├── _check.py                # 依赖检查
├── _install_kernels.py      # 运行时将 .cuh 注入 deep_gemm include 路径
├── utils.py                 # 工具函数 (align, uneven_all_gather)
├── setup_ext.py             # C++ extension 编译脚本
│
├── csrc/                    # C++ host runtime (编译进 _mega_moe_C.so)
│   ├── python_api.cpp       # pybind11 模块入口
│   ├── apis/
│   │   ├── sm90_mega.hpp            # 前向：TMA描述符构建、启发式配置、kernel启动
│   │   └── sm90_mega_backward.hpp   # 反向：pool布局、重计算、cuBLASLt weight-grad
│   └── jit_kernels/
│       ├── heuristics/              # 启发式 tile/config 选择
│       │   ├── sm90_mega_moe.hpp
│       │   └── sm90_mega_moe_backward.hpp
│       └── impls/                   # kernel 生成和启动逻辑
│           ├── sm90_fp8_mega_moe.hpp
│           ├── sm90_bf16_mega_moe_backward.hpp
│           └── smxx_cublaslt.hpp
│
├── include/                 # .cuh CUDA kernel 源码 (运行时 JIT 编译)
│   └── deep_gemm/
│       ├── impls/
│       │   ├── sm90_fp8_mega_moe.cuh         # 前向 kernel (FP8 WGMMA + SwiGLU)
│       │   └── sm90_bf16_mega_moe_backward.cuh  # 反向 kernel (BF16)
│       ├── layout/
│       │   ├── mega_moe.cuh                  # expert pool 内存布局
│       │   └── sym_buffer.cuh                # symmetric buffer 数据结构
│       └── scheduler/
│           └── mega_moe.cuh                  # expert wave 调度器
│
└── tests/                   # 参考测试
    ├── test_mega_moe_hopper.py
    └── test_mega_moe_backward.py
```

### 两层编译模型

| 层级 | 文件类型 | 编译时机 | 产物 |
|------|----------|----------|------|
| Host Runtime | `.hpp` → `.cpp` | 用户运行 `setup_ext.py` | `_mega_moe_C.so` |
| CUDA Kernel | `.cuh` | 首次 kernel 调用时 (JIT) | `~/.deep_gemm/` 下的 `.cubin` |

- **Host Runtime** (`csrc/`): 负责 TMA descriptor 构建、启发式配置选择（block size, pipeline stages 等）、kernel launch。编译时依赖 DeepGEMM 的 JIT 基础设施头文件。
- **CUDA Kernel** (`include/`): 实际在 GPU 上执行的 kernel 实现。由 deep_gemm 的 NVRTC/NVCC JIT 编译器在运行时按需编译，支持 PCH 和 cubin 缓存。

---

## 安装步骤

### 前置条件

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| CUDA Toolkit | >= 12.3 (推荐 12.9) | 需要 nvcc 和 nvrtc |
| GPU | SM90 (H100/H800) | Hopper 架构 |
| PyTorch | >= 2.4 | 需 CUDA 支持和 symmetric memory |
| deep_gemm | >= 2.5.0 | JIT 编译基础设施 |
| DeepGEMM 源码 | 与 deep_gemm 版本匹配 | 编译时需要 csrc/ 和 third-party/ |

### 步骤 1: 安装 deep_gemm

```bash
# 方式 A: pip 安装（如有发布）
pip install deep_gemm>=2.5.0

# 方式 B: 从源码安装
git clone https://github.com/DeepSeek-AI/DeepGEMM.git
cd DeepGEMM
pip install -e .
```

### 步骤 2: 编译 _mega_moe_C extension

```bash
cd megatron/plugin/mega_moe/

# 设置 DeepGEMM 源码路径（需要 csrc/ 和 third-party/ 中的头文件）
export DEEP_GEMM_ROOT=/path/to/DeepGEMM

# 确保 CUDA_HOME 已设置
export CUDA_HOME=/usr/local/cuda

# 编译
python setup_ext.py build_ext --inplace
```

编译成功后会在当前目录生成 `_mega_moe_C.*.so`。

### 步骤 3: 验证安装

```python
# 快速验证
python -c "from megatron.plugin.mega_moe import SymmBuffer; print('MegaMoE OK')"
```

如果报错 `ImportError`，会给出具体缺失什么以及如何修复。

### 步骤 4: JIT kernel 首次编译（自动）

首次调用 kernel 时，deep_gemm 的 JIT 系统会自动编译 `.cuh` 源码为 `.cubin`。这个过程：
- 首次需要 30-60 秒
- 编译结果缓存在 `~/.deep_gemm/`（或 `$DG_JIT_CACHE_DIR`）
- 后续调用直接加载 cubin，无额外开销

---

## 使用方法

### 基本前向推理

```python
import torch
import torch.distributed as dist
from megatron.plugin.mega_moe import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe_sm90,
    fp8_mega_moe,
)

# 初始化 symmetric buffer
group = dist.group.WORLD
sym_buffer = get_symm_buffer_for_mega_moe(
    group=group,
    num_experts=64,
    num_max_tokens_per_rank=4096,
    num_topk=8,
    hidden=7168,
    intermediate_hidden=2048,
)

# 准备权重 (只需做一次)
l1_weights, l2_weights = transform_weights_for_mega_moe_sm90(
    l1_weights=(l1_fp8, l1_sf),
    l2_weights=(l2_fp8, l2_sf),
)

# 执行前向
y = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer)
```

### 前向+反向（训练）

```python
from megatron.plugin.mega_moe import MegaMoEFunction

# 使用 autograd Function 自动处理前向+反向
y = MegaMoEFunction.apply(
    x, x_sf, topk_idx, topk_weights,
    l1_w_fp8, l1_w_sf, l1_w_bf16,
    l2_w_fp8, l2_w_sf, l2_w_bf16,
    sym_buffer, None,   # cumulative_local_expert_recv_stats
    (128, 128, 128),    # recipe
    'swiglu',           # activation
    None,               # activation_clamp
    True,               # fast_math
)
loss = y.sum()
loss.backward()  # dx, dW1, dW2 自动计算
```

---

## 实现细节

### 前向 Kernel (SM90 FP8)

- **计算精度**: FP8 (e4m3) activations × FP8 weights → FP32 accumulator → BF16 output
- **Scale Factor**: per-128-K block (float32), WGMMA promote 加载
- **激活函数**: SwiGLU，在 L1 epilogue 中融合执行
- **TMA**: 2D descriptor 用于 activations/weights，multicast 到 cluster 内的 CTAs
- **Pipeline**: 多 stage software pipeline (2-4 stages)，TMA load 与 WGMMA 重叠
- **通信**: Symmetric memory barrier 驱动 expert dispatch/combine，zero-copy 跨 rank

### 反向 Kernel (SM90 BF16)

- **Phase 1**: 重计算前向激活值 (recompute checkpoint)
- **Phase 2**: L2 反向 GEMM + 融合 SwiGLU backward epilogue (通过 perm_buf 在 kernel 内完成 fwd→bwd pool 重排序)；L1 反向 GEMM + dx combine
- **Phase 3**: 计算 dW1/dW2 (grouped cuBLASLt BF16 GEMM，相同 K 的 expert 共享 desc/heuristic)

### 启发式配置

启发式系统根据以下参数自动选择最优 tile size 和 pipeline 配置：
- `num_tokens`: 影响 block_m (64 vs 128)
- `hidden / intermediate_hidden`: 影响 block_n, split-N 策略
- `num_ranks`: 影响 expert-per-wave, dispatch thread 数
- `num_experts`: 影响 pool 容量和调度

关键配置参数：
- `block_m`: 64 (decode) 或 128 (prefill)
- `block_n`: 128 / 256 / 512
- `block_k`: 64 / 128
- `num_stages`: 2-4
- `num_epilogue_threads`: 256 / 512

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEP_GEMM_ROOT` | (auto-detect) | DeepGEMM 源码路径（编译时） |
| `DG_JIT_CACHE_DIR` | `~/.deep_gemm` | JIT cubin 缓存目录 |
| `DG_JIT_DEBUG` | 0 | 打印 JIT 编译命令和生成的 kernel 代码 |
| `DG_JIT_PTXAS_VERBOSE` | 0 | 打印 ptxas 寄存器使用情况 |
| `DG_PRINT_CONFIGS` | 0 | 打印选中的 tile 配置 |

---

## 常见问题

### Q: 编译报 "Cannot find DeepGEMM source tree"

设置 `DEEP_GEMM_ROOT` 环境变量指向 DeepGEMM 仓库根目录（包含 `csrc/` 和 `third-party/`）。

### Q: 运行时报 "JIT compilation failed"

确保 `CUDA_HOME` 指向 CUDA >= 12.3，且 `nvcc --version` 可用。NVRTC >= 12.8 支持 PCH 加速编译。

### Q: 如何清除 JIT 缓存？

```bash
rm -rf ~/.deep_gemm
```

下次调用时会重新编译。

### Q: 能否在 SM100 (Blackwell) 上运行？

不能。本 plugin 仅支持 SM90。SM100 的 MegaMoE 使用 TMEM + UTCCP，需要不同的 kernel 实现，请使用 deep_gemm 原生的 SM100 路径。
