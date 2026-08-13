# FSA Headwise CP 单元测试

## 测试文件

- `test_fsa_headwise_cp.py` - FSA Headwise Context Parallel 单元测试

## 测试方法

### 核心测试：数值精度对比 (`test_cp_vs_no_cp_numerical_equivalence`)

**测试原理**：
- **Case 1 (baseline)**：不开启 CP（cp_size=1），在完整序列上批量运行 FSA（匹配真实训练场景）
- **Case 2 (test)**：开启 CP（cp_size>1），序列切分到多个 rank 并行计算
- **验证**：两种方式的输出应该数值完全一致（允许 bfloat16 精度误差）

**这是最关键的正确性测试**，确保 CP 并行化不改变计算结果。

## 测试场景覆盖

### 1. CP only (TP=1)

#### 1a. CP size < KV heads (all-to-all 通信)
- `tp=1, cp=2, q_heads=16, kv_heads=8` - 充足 KV heads
- `tp=1, cp=4, q_heads=32, kv_heads=16` - 充足 KV heads

#### 1b. CP size > KV heads (AllGather 通信)
- `tp=1, cp=4, q_heads=16, kv_heads=2` - 不足 KV heads
- `tp=1, cp=8, q_heads=32, kv_heads=4` - 不足 KV heads

### 2. TP + CP 组合

#### 2a. CP size <= KV heads per TP (all-to-all 通信)
- `tp=2, cp=2, q_heads=16, kv_heads=8` - 每个 TP rank 有 4 个 KV heads
- `tp=4, cp=2, q_heads=32, kv_heads=16` - 每个 TP rank 有 4 个 KV heads

#### 2b. CP size > KV heads per TP (AllGather 通信)
- `tp=2, cp=4, q_heads=16, kv_heads=4` - 每个 TP rank 有 2 个 KV heads
- `tp=4, cp=4, q_heads=32, kv_heads=8` - 每个 TP rank 有 2 个 KV heads

### 3. 显存占用对比测试
- 验证开启 CP 后，单卡 peak memory 低于不开 CP
- 使用长序列场景（4096/8192/16384）放大显存差异
- 输出详细的显存对比和节省比例

### 4. 错误场景测试
- Q heads 不能被 CP size 整除时抛出 ValueError

## 运行测试

### 使用测试脚本（推荐）

```bash
# 运行所有测试（2 GPUs - 只能测试 CP only, cp_size=2 的场景）
cd /share/project/lixianduo/codes/Megatron-LM-FL
./tests/unit_tests/extensions/run_fsa_tests.sh 2

# 运行所有测试（4 GPUs - 测试大部分场景）
./tests/unit_tests/extensions/run_fsa_tests.sh 4

# 运行所有测试（8 GPUs - 覆盖所有场景，包括 TP+CP 组合）
./tests/unit_tests/extensions/run_fsa_tests.sh 8

# 运行特定测试场景
./tests/unit_tests/extensions/run_fsa_tests.sh 4 "test_cp_vs_no_cp_numerical_equivalence[1-4-16-2"
```

### 手动运行

```bash
# 激活环境
source /share/project/lixianduo/envs/fsa-train/bin/activate

# 设置 GPU 数量
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4

# 运行测试
torchrun --nproc_per_node=$NUM_GPUS \
    -m pytest \
    -v \
    -s \
    tests/unit_tests/extensions/test_fsa_headwise_cp.py
```

## 测试输出示例

成功的测试会输出详细的数值对比信息：

```
============================================================
Numerical Equivalence Test
TP=1, CP=4
Q heads: 16 global, 16 per TP
KV heads: 2 global, 2 per TP
Communication mode: AllGather
------------------------------------------------------------
Max absolute diff: 2.345678e-04
Mean absolute diff: 1.234567e-05
Max relative diff: 3.456789e-04
Mean relative diff: 2.345678e-05
============================================================
PASSED
```

**关键指标**：
- `Max absolute diff < 5e-3` - 最大绝对误差（匹配 Megatron GDN 测试标准）
- `Max relative diff < 5e-3` - 最大相对误差
- `Communication mode` - 显示使用的通信模式（all-to-all 或 AllGather）

### 显存对比测试输出

```
============================================================
Memory Usage Comparison (cp_size=4)
  seq_len=8192, batch=1, q_heads=32, kv_heads=8, head_dim=128
============================================================
  No CP peak memory:   256.00 MB
  With CP peak memory: 72.00 MB
  Memory reduction:    71.9%
============================================================
PASSED
```

## 测试覆盖

✅ **数值精度验证**（最重要）
- CP 开启前后输出完全一致
- 覆盖 bfloat16 精度范围内的误差

✅ **混合通信模式**
- all-to-all: 当 `num_kv_heads_per_tp >= cp_size` 时
- AllGather: 当 `num_kv_heads_per_tp < cp_size` 时

✅ **并行策略组合**
- CP only (TP=1)
- TP + CP (TP>1)

✅ **形状验证**
- 输入：`[seq_local, batch, heads, head_dim]`
- 输出：`[seq_local, batch, heads, head_dim]`
- Gather 后与 baseline 对比

✅ **错误处理**
- Q heads 不能被 CP size 整除时的错误检查

✅ **显存优化验证**
- CP 开启后 peak memory 低于 no-CP
- 长序列场景（4K/8K/16K）下的显存对比
- 输出具体节省比例

## 注意事项

1. **GPU 要求**：
   - 最少 2 GPUs：测试 CP only (cp_size=2) 场景
   - 推荐 4 GPUs：测试大部分场景
   - 推荐 8 GPUs：覆盖所有场景，包括 TP+CP 组合
   - 测试会自动跳过 GPU 不足的场景

2. **依赖库**：需要安装 `flash_sparse_attn` 库
   ```bash
   pip install flash-sparse-attn
   ```

3. **分布式初始化**：
   - 使用 `Utils.initialize_model_parallel()` 初始化并行环境
   - 每个测试会初始化两次（no-CP baseline + with-CP test）
   - 每个测试后调用 `Utils.destroy_model_parallel()` 清理

4. **数值精度**：
   - 使用 bfloat16 精度
   - 允许的误差阈值：`atol=5e-3, rtol=5e-3`（与 Megatron GDN 测试标准一致）
   - 如果误差过大，测试会失败并显示详细的差异统计

5. **通信模式验证**：
   - 测试输出会显示使用的通信模式（all-to-all 或 AllGather）
   - 验证混合通信逻辑正确切换

## 故障排查

### 问题 1：ImportError: cannot import flash_sparse_attn
**解决方案**：安装 flash_sparse_attn 库
```bash
pip install flash-sparse-attn
```

### 问题 2：Numerical difference exceeds threshold
**可能原因**：
- FSA kernel 实现有误
- 通信逻辑错误（sequence gather 顺序、head 索引选择）
- 数值不稳定（softmax threshold 设置）

**调试方法**：
```bash
# 运行单个失败的测试，查看详细输出
./tests/unit_tests/extensions/run_fsa_tests.sh 4 "test_cp_vs_no_cp_numerical_equivalence[1-4-16-2"
```

### 问题 3：CUDA out of memory
**解决方案**：
- 减小测试参数（seq_len, batch_size）
- 使用更多 GPU 分摊内存
- 测试参数已经优化为较小值（seq_len=128-256）

### 问题 4：NCCL timeout / 通信超时
**解决方案**：
- 检查网络配置，确保所有 GPU 可以通信
- 增加超时时间：`export NCCL_TIMEOUT=300`
- 检查 GPU 互联拓扑：`nvidia-smi topo -m`

### 问题 5：测试被跳过 (SKIPPED)
**原因**：
- GPU 数量不足：需要更多 GPU
- flash_sparse_attn 库未安装
- 测试条件不满足

**检查**：查看跳过原因，测试会输出具体信息

## 性能分析

测试不仅验证正确性，还可以观察不同通信模式的性能：

- **all-to-all 模式**（`num_kv_heads_per_tp >= cp_size`）：
  - 通信量：每个 rank 只传输自己需要的 head
  - 更高效，推荐配置

- **AllGather 模式**（`num_kv_heads_per_tp < cp_size`）：
  - 通信量：每个 rank 接收所有 sequence chunks（约 1.33x 开销，CP=4 时）
  - 允许更大的 CP size

可以通过测试输出观察两种模式的实际运行时间差异。
