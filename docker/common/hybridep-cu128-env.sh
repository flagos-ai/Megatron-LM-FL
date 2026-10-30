# Source this file only for HybridEP processes that JIT-compile CUDA kernels.
# The CUDA 12.8 toolkit remains the compiler and runtime; this prepend path
# supplies the complete CUDA 12.9.27 CCCL headers required by HybridEP.
_flagos_cccl_include="${FLAGOS_ENV:-/root/miniconda3/envs/flagscale-train}/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export NVCC_PREPEND_FLAGS="-I${_flagos_cccl_include}${NVCC_PREPEND_FLAGS:+ ${NVCC_PREPEND_FLAGS}}"
unset _flagos_cccl_include
