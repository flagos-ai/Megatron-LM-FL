#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
# Activate a compatible Python/TE/C++ environment before calling this script.
TP=${TP:-1}
CP=${CP:-1}
EP=${EP:-1}
GPUS=${SLIDEFORMER_GPU_IDS:-0}
IFS=, read -ra GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}
for gpu in "${GPU_LIST[@]}"; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU $gpu already has a compute context; aborting." >&2
    exit 3
  fi
done
export SLIDEFORMER_GPU_IDS="$GPUS"
export CUDA_VISIBLE_DEVICES=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MEGATRON_SLIDEFORMER_ENABLE=${MEGATRON_SLIDEFORMER_ENABLE:-1}
export MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD=1
export MEGATRON_SLIDEFORMER_KERNEL_POLICY=${MEGATRON_SLIDEFORMER_KERNEL_POLICY:-auto}
export MEGATRON_SLIDEFORMER_STRICT_KERNELS=1
EXTRA=()
if (( TP > 1 )); then EXTRA+=(--sequence-parallel); fi
if (( EP > 1 )); then
  EXTRA+=(--num-experts 4 --moe-router-topk 2 --moe-token-dispatcher-type alltoall --moe-aux-loss-coeff 0.01)
fi
if [[ -n "${PER_TOKEN:-}" ]]; then EXTRA+=(--calculate-per-token-loss); fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then EXTRA+=(--save "$CHECKPOINT_DIR" --save-interval "${SAVE_INTERVAL:-1}"); fi
if [[ -n "${LOAD_DIR:-}" ]]; then EXTRA+=(--load "$LOAD_DIR" --use-checkpoint-opt-param-scheduler); fi
python -m torch.distributed.run --standalone --nproc-per-node="$NPROC" tests/functional_tests/slideformer/rank_entry.py \
 --num-layers 2 --hidden-size 128 --ffn-hidden-size 384 \
 --num-attention-heads 4 --num-query-groups 4 --group-query-attention --kv-channels 32 \
 --seq-length 128 --max-position-embeddings 128 \
 --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH:-4}" --train-iters "${TRAIN_ITERS:-3}" \
 --tensor-model-parallel-size "$TP" --context-parallel-size "$CP" --expert-model-parallel-size "$EP" --pipeline-model-parallel-size ${PP:-1} \
 --transformer-impl transformer_engine --bf16 --normalization RMSNorm --qk-layernorm --swiglu --disable-bias-linear \
 --position-embedding-type rope --rotary-base 1000000 --attention-dropout 0 --hidden-dropout 0 \
 --mock-data --tokenizer-type NullTokenizer --vocab-size 1024 \
 --lr 0.0001 --min-lr 0.00001 --lr-decay-style cosine --lr-warmup-iters 0 \
 --weight-decay 0.01 --clip-grad 0.0 --seed 42 --log-interval 1 --eval-iters 0 --eval-interval 1000 \
 --no-gradient-accumulation-fusion --distributed-timeout-minutes 4 --num-workers 0 "${EXTRA[@]}" "$@"
