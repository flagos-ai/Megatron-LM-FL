#!/bin/bash
# FSA Headwise CP Unit Test Runner
#
# Usage:
#   # Run all FSA headwise CP tests with 2 GPUs
#   ./run_fsa_tests.sh 2
#
#   # Run all FSA headwise CP tests with 4 GPUs
#   ./run_fsa_tests.sh 4
#
#   # Run specific test scenario
#   ./run_fsa_tests.sh 4 "test_fsa_headwise_cp_forward[4-16-2-128-2-64]"

set -e

# Default to 2 GPUs if not specified
NUM_GPUS=${1:-2}
TEST_PATTERN=${2:-""}

echo "========================================"
echo "FSA Headwise CP Unit Tests"
echo "========================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Test pattern: ${TEST_PATTERN:-all tests}"
echo "========================================"

# Activate conda environment (skip if already activated)
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "fsa-train" ]; then
    eval "$(conda shell.bash hook)"
    conda activate /share/project/lixianduo/envs/fsa-train
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL communication settings
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# Run tests with pytest (single-node)
cd /share/project/lixianduo/codes/Megatron-LM-FL

if [ -z "$TEST_PATTERN" ]; then
    # Run all FSA headwise CP tests
    torchrun --rdzv_backend static \
        --nnodes 1 \
        --nproc_per_node=$NUM_GPUS \
        -m pytest \
        -v \
        -s \
        tests/unit_tests/extensions/test_fsa_headwise_cp.py
else
    # Run specific test
    torchrun --rdzv_backend static \
        --nnodes 1 \
        --nproc_per_node=$NUM_GPUS \
        -m pytest \
        -v \
        -s \
        -k "$TEST_PATTERN" \
        tests/unit_tests/extensions/test_fsa_headwise_cp.py
fi

echo "========================================"
echo "Tests completed!"
echo "========================================"
