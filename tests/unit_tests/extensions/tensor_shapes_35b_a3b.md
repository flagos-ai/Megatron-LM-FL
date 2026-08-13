# Qwen3.6-35B-A3B FSA Attention Tensor Shapes

## Model Configuration (from `35b_a3b_llm_fsa_cp.yaml`)

```yaml
num_attention_heads: 16          # Global Q heads
num_query_groups: 2              # Global KV heads (GQA)
kv_channels: 256                 # head_dim
seq_length: 32768
micro_batch_size: 1
tensor_model_parallel_size: 4    # TP
context_parallel_size: 1         # CP (currently disabled)
sequence_parallel: True
```

## Derived Parameters

- **GQA ratio**: `num_q_per_kv = 16 / 2 = 8` (8 Q heads share 1 KV head)
- **TP split mode**: `num_query_groups (2) < tp_size (4)`
  - When `num_kv_heads < tp_size`, Megatron replicates KV heads across TP ranks
  - Each TP rank gets: **1 KV head (replicated)** + **4 Q heads**
  - TP rank 0: KV head 0, Q heads 0-3
  - TP rank 1: KV head 1, Q heads 4-7
  - TP rank 2: KV head 0 (replicated), Q heads 8-11
  - TP rank 3: KV head 1 (replicated), Q heads 12-15
- **Q heads per TP**: `16 / 4 = 4`
- **KV heads per TP**: `1` (replicated)
- **Sequence in attention**: `32768` (full, SP does not split attention sequence)
- **Sequence outside attention (LayerNorm/MLP with SP)**: `32768 / 4 = 8192`

> **Note**: Sequence Parallel (SP) splits sequence in LayerNorm/Dropout/MLP, but
> restores full sequence before attention via all-gather in ColumnParallelLinear.
> Attention always sees the full sequence length.

---

## Case 1: Current Configuration (CP=1, no Context Parallel)

### Per TP Rank Tensor Shapes (verified from training log)

```python
# Input to FSA kernel (each TP rank)
# Sequence Parallel restores full sequence before attention

# All TP ranks:
query:  [32768, 1, 4, 256]  # [seq_full, batch, 4 Q heads, head_dim]
key:    [32768, 1, 1, 256]  # [seq_full, batch, 1 KV head, head_dim]
value:  [32768, 1, 1, 256]
window_sizes: [1, 4]        # [1 KV head, 4]

# Output (each TP rank)
output: [32768, 1, 4, 256]  # [seq_full, batch, 4 Q heads, head_dim]
```

### GQA Mapping (per TP rank)
- TP rank 0: Q heads 0-3 → KV head 0
- TP rank 1: Q heads 4-7 → KV head 1
- TP rank 2: Q heads 8-11 → KV head 0 (replicated)
- TP rank 3: Q heads 12-15 → KV head 1 (replicated)

---

## Case 2: With Context Parallel (CP=4, hypothetical)

If `context_parallel_size: 4` were enabled:

### Global Parallelism
- **TP size**: 4, **CP size**: 4
- **Total ranks needed**: 16 (TP × CP)

### Per TP Rank (before CP split)
- `seq_len = 32768` (full, from SP all-gather)
- `num_q_heads_per_tp = 4`
- `num_kv_heads_per_tp = 1`

### Communication Mode Decision
- `num_kv_heads_per_tp (1) < cp_size (4)` → **AllGather mode**
- Each CP rank will handle `4 / 4 = 1` Q head
- All Q heads on a TP rank map to the same KV head (since only 1 KV head per TP rank)

### FSA Headwise CP Forward Pass

#### Input (local chunk per CP rank)
```python
# CP splits the full sequence into cp_size chunks
query_local:  [8192, 1, 4, 256]   # [seq/cp_size, batch, 4 Q heads, head_dim]
key_local:    [8192, 1, 1, 256]   # 1 KV head
value_local:  [8192, 1, 1, 256]
window_sizes: [1, 4]              # [1 KV head, 4]
```

#### Step 1: Q heads all-to-all (sequence → head)
```python
# Before: [8192, 1, 4, 256]
# Reshape to 3-d: [8192, 1, 4*256] = [8192, 1, 1024]
# all-to-all: [8192, 1, 1024] → [32768, 1, 256]
# Reshape to 4-d: [32768, 1, 1, 256]

q_full_seq: [32768, 1, 1, 256]  # Full sequence, 1 Q head per rank
```

**Global Q head index for each CP rank (on TP rank 0):**
- CP rank 0: Q head 0
- CP rank 1: Q head 1
- CP rank 2: Q head 2
- CP rank 3: Q head 3

#### Step 2: KV heads - AllGather mode
```python
# AllGather sequence dimension
# Before: [8192, 1, 1, 256]
# Reshape: [8192, 1, 256]
# AllGather: [8192, 1, 256] → [32768, 1, 256]
# Reshape: [32768, 1, 1, 256]

k_full_seq: [32768, 1, 1, 256]  # Full sequence, same 1 KV head
v_full_seq: [32768, 1, 1, 256]

# No need to select - all Q heads on this TP rank map to this 1 KV head
window_sizes_local: [1, 4]  # Single KV head's window config
```

#### Step 3: FSA kernel
```python
output = flash_sparse_attn_func(
    q_full_seq,      # [32768, 1, 1, 256]  (1 Q head)
    k_full_seq,      # [32768, 1, 1, 256]  (1 KV head)
    v_full_seq,      # [32768, 1, 1, 256]
    window_sizes=window_sizes_local,  # [1, 4]
    softmax_threshold=0.5,
    pack_gqa=False,
)
# output: [32768, 1, 1, 256]
```

#### Step 4: All-to-all backward (head → sequence)
```python
# Reshape: [32768, 1, 1*256] = [32768, 1, 256]
# all-to-all: [32768, 1, 256] → [8192, 1, 1024]
# Reshape: [8192, 1, 4, 256]

output_local: [8192, 1, 4, 256]  # Back to local sequence chunk, all 4 Q heads
```

---

## Memory Analysis

### No CP (current, verified)
Per TP rank peak activations during attention:
- Q/K/V: `(4×256 + 1×256 + 1×256) × 32768 × 1 × 2 bytes = 96 MB` (bfloat16)
- Attention scores (worst case, dense): `32768 × 4 × 32768 × 2 bytes = 8 GB`
- FSA sparse attention reduces this significantly via window pattern
- **Total (dense upper bound)**: ~8.1 GB per TP rank

### With CP=4
Per CP rank peak activations:
- Q/K/V local input: `(4×256 + 1×256 + 1×256) × 8192 × 1 × 2 bytes = 24 MB`
- After AllGather K/V: `1 × 256 × 32768 × 1 × 2 bytes = 16 MB`
- After Q all-to-all: `1 × 256 × 32768 × 1 × 2 bytes = 16 MB`
- Attention scores: `32768 × 1 × 32768 × 2 bytes = 2 GB` (1 Q head attends to 1 KV head)
- **Total (dense upper bound)**: ~2.1 GB per CP rank

**Memory reduction**: ~4× (from reducing Q heads per rank from 4 to 1)

---

## Test Scenario Mapping

The unit test scenario `(1, 4, 16, 2, 128, 2, 64)` maps to:
- `tp_size=1, cp_size=4`
- `num_q_heads=16, num_kv_heads=2` (same 8:1 ratio as 35B model)
- `seq_len=128` (scaled down from 32768)
- `head_dim=64` (scaled down from 256)

This tests the **AllGather mode** (`num_kv_heads_per_tp=2 < cp_size=4`).

### Key Difference from 35B Model
- **Test scenario**: `tp_size=1`, so `num_kv_heads_per_tp = 2` (normal split)
- **35B model**: `tp_size=4` with `num_kv_heads < tp_size`, so special split gives `num_kv_heads_per_tp = 1`
- Both trigger AllGather mode when CP is enabled (`num_kv_heads_per_tp < cp_size`)
- The test validates the AllGather communication path without the complexity of TP replication
