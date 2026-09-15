#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// ============================================================================
// Device helpers
// ============================================================================

template <bool kFastMath>
__forceinline__ __device__ float sm90_bf16_bw_sigmoid(float x) {
    const float e = kFastMath ? __expf(-x) : expf(-x);
    return kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
}

template <bool kFastMath>
__forceinline__ __device__ void sm90_bf16_swiglu_backward(
    float d_a, float gate, float up, float topk_w,
    float& d_gate, float& d_up) {
    // Forward applied topk_w inside L1 epilogue: a_scaled = swiglu * topk_w.
    // To backprop through the scalar multiply: d_swiglu = d_a_scaled * topk_w.
    // Guard: padding positions have topk_w=0 and d_a may be garbage/NaN from
    // uninitialized d_o_pool through the L2 GEMM. Early-out to avoid 0*NaN=NaN.
    if (topk_w == 0.0f) { d_gate = 0.0f; d_up = 0.0f; return; }
    const float d_swiglu = d_a * topk_w;
    const float sig = sm90_bf16_bw_sigmoid<kFastMath>(gate);
    const float silu_gate = gate * sig;
    d_up = d_swiglu * silu_gate;
    d_gate = d_swiglu * up * sig * (1.0f + gate * (1.0f - sig));
}

// ============================================================================
// Kernel 1: Backward L2
// Computes: d_a[M,IH] = d_o[M,H] @ W2[H,IH]   (activation gradient)
// Fused:    SwiGLU backward in epilogue → d_h[M,2*IH]
//
// The epilogue reads gate/up from recomp_h (forward pool order, via perm lookup)
// and writes d_h to d_h_buffer (backward pool order). This eliminates the
// host-side reorder + sync between L2 and L1, and removes SwiGLU from L1.
//
// Weight gradient dW2 is computed host-side after both kernels finish, using
// cuBLASLt per-expert GEMM:
//   dW2[e, H, IH] = d_o_pool[M_e, H]^T @ recomp_a[M_e, IH]
//
// Phase 2 implementation: uses cooperative BF16 WGMMA mainloop for the
// activation gradient GEMM + fused SwiGLU backward epilogue.
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kFastMath,
    bool kRecompute,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_bf16_mega_moe_backward_l2_impl(
    void* d_h_buffer,                        // [pool, 2*IH] BF16 — SwiGLU BW output (bwd order)
    void* d_o_pool,                          // [pool, H] BF16 — dispatch fills, GEMM reads
    float* dW2,
    const void* l2_weights_bf16,
    const void* recomp_h,                    // [pool, 2*IH] BF16 — gate/up (fwd order)
    const void* recomp_a,
    float* topk_weights_pool,                // [pool] FP32 — dispatch fills (for SwiGLU BW)
    uint32_t* perm_buf,                      // [pool] — dispatch warp fills bwd→fwd pos mapping
    const int32_t* fwd_lookup_table,         // [num_ranks * max_tokens * topk] key→fwd_pool_pos
    const uint32_t fwd_lookup_stride_r,      // = max_tokens_per_rank * topk
    const uint32_t fwd_lookup_stride_t,      // = topk
    const void* input_topk_idx,              // [T, topk] int64 — from forward sym_buffer
    const void* input_topk_weights,          // [T, topk] float — from forward sym_buffer
    const void* dy_source,                   // [T, H] BF16 — in sym_buffer, NVLink-accessible
    int* cumulative_local_expert_recv_stats,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dy,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_d_a_output) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    // Suppress unused-parameter warnings (dW2 computed host-side via cuBLASLt;
    // tensor_map_d_a_output no longer used — epilogue writes d_h directly)
    (void)dW2;
    (void)recomp_a;
    (void)tensor_map_d_a_output;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // WGMMA type for BF16: M=64, K=16
    // A operand (dy) is K-major in smem (K contiguous in gmem).
    // B operand (weights) is MN-major in smem (N contiguous in gmem).
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kNumEpilogueWarpgroups;
    using WGMMA = typename mma::sm90::BF16MMASelector<WG_BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::MN>::type;
    static_assert(WGMMA::K == 16, "BF16 WGMMA K must be 16");
    constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;

    constexpr uint32_t kBF16Bytes = sizeof(nv_bfloat16);
    constexpr uint32_t kSwizzleMode = BLOCK_K * kBF16Bytes;
    static_assert(kSwizzleMode <= 128, "Swizzle mode must be <= 128B");
    constexpr uint32_t kAlign = 1024;

    // Shared memory layout: [A stages][B stages][barriers]
    extern __shared__ __align__(kAlign) uint8_t smem_buffer[];
    constexpr uint32_t SMEM_A_PER_STAGE = BLOCK_M * BLOCK_K * kBF16Bytes;
    constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * kBF16Bytes;

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + i * SMEM_A_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer +
            kNumStages * SMEM_A_PER_STAGE + i * SMEM_B_PER_STAGE);
    });
    constexpr uint32_t BARRIER_OFFSET = kNumStages * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);
    auto* barrier_base = reinterpret_cast<Barrier*>(smem_buffer + BARRIER_OFFSET);
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_base + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages + i; });

    // Thread indexing
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = thread_idx % 32;
    const uint32_t sm_idx = blockIdx.x;

    // Barrier init
    if (warp_idx == 0) {
        if (lane_idx < kNumStages) {
            full_barriers[lane_idx]->init(1);
            empty_barriers[lane_idx]->init(kNumEpilogueWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfiguration: warpgroup-collective setmaxnreg requires all
    // 4 warps in a warpgroup to participate. Warpgroup 0 (warps 0-3) decreases
    // registers; warpgroups 1+ (warps 4-19, math) increase registers.
    // Budget: 128 threads × 24 + 512 threads × 120 = 3072 + 61440 = 64512 ≤ 65536
    if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Warps 0-3 (dispatch + TMA + idle): dealloc registers
        cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
        // Warps 4-19 (math warpgroups): alloc registers
        cutlass::arch::warpgroup_reg_alloc<120>();
    }

    // Block-level scheduling constants
    constexpr uint32_t kNumExPerRank = kNumExpertsPerRank;
    constexpr uint32_t kNumKBlocks = kHidden / BLOCK_K;
    constexpr uint32_t kNumNBlocks = kIntermediateHidden / BLOCK_N;

    // =====================================================================
    // ROLE: TMA LOAD WARP
    // =====================================================================
    if (warp_idx == kNumDispatchWarps) {

        // Wait for per-expert token counts to be available from the workspace.
        // The dispatch warps write expert_recv_count_sum via NVLink exchange;
        // each entry encodes (num_ranks_arrived << 32 | token_count).
        // Poll until all ranks have reported for each expert.
        const auto workspace = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

        // Compute total work items by scanning expert counts.
        // We avoid storing per-expert arrays (register pressure in 24-reg TMA warp).
        uint32_t total_pool_blocks = 0;
        for (uint32_t e = 0; e < kNumExPerRank; ++e) {
            const auto sum_ptr = workspace.get_expert_recv_count_sum_ptr(e);
            uint64_t sum_val;
            do {
                // ld_volatile + exact equality match the forward scheduler
                // (scheduler/mega_moe.cuh:fetch_expert_recv_count). The high
                // 32 bits hold kNumSMs per rank (one atomic_add per CTA in the
                // count phase), so the fully-published value is kNumSMs*kNumRanks.
                // Polling `< kNumRanks` exits after the FIRST rank publishes for
                // kNumRanks>=2 (132 >= 2), reading a partial count — which then
                // starves the arrival-count wait in other warps and deadlocks.
                sum_val = ptx::ld_volatile(sum_ptr);
            } while (static_cast<uint32_t>(sum_val >> 32) != kNumSMs * kNumRanks);
            const uint32_t count = static_cast<uint32_t>(sum_val & 0xffffffff);
            total_pool_blocks += math::ceil_div(count, BLOCK_M);
        }
        const uint32_t total_blocks = total_pool_blocks * kNumNBlocks;

        uint32_t stage = 0, ph = 0;
        auto adv = [&](uint32_t& k) { ++k; if (++stage >= kNumStages) { stage = 0; ph ^= 1; } };

        for (uint32_t block_id = sm_idx; block_id < total_blocks; block_id += kNumSMs) {
            // Map block_id → (expert_idx, local_m_block, n_block) by scanning experts
            uint32_t expert_idx = 0, pool_block_start = 0;
            uint32_t remaining = block_id;
            for (uint32_t e = 0; e < kNumExPerRank; ++e) {
                const uint32_t cnt = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(e) & 0xffffffff);
                const uint32_t e_m_blocks = math::ceil_div(cnt, BLOCK_M);
                const uint32_t expert_work = e_m_blocks * kNumNBlocks;
                if (remaining < expert_work) { expert_idx = e; break; }
                remaining -= expert_work;
                pool_block_start += e_m_blocks;
            }
            const uint32_t local_m_block = remaining / kNumNBlocks;
            const uint32_t n_block = remaining % kNumNBlocks;
            const uint32_t pool_block_idx = pool_block_start + local_m_block;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block * BLOCK_N;

            // Wait for the dispatch to have filled this pool block's d_o before
            // TMA-loading it. Must wait on EVERY (expert, m_block) — not just
            // n_block==0 — because the N-blocks of a pool block run on different
            // CTAs with no ordering between them. Gating on n_block==0 let
            // n_block>0 CTAs read d_o before the dispatch (which fills later
            // experts last) had written it, shrinking d_a more for later experts.
            {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const uint32_t expert_count = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(expert_idx) & 0xffffffff);
                const uint32_t expert_m_blks = math::ceil_div(expert_count, BLOCK_M);
                const uint32_t tokens_in_block = (local_m_block + 1 == expert_m_blks)
                    ? expert_count - local_m_block * BLOCK_M
                    : BLOCK_M;
                while (ptx::ld_acq(ptr) < tokens_in_block);
            }

            for (uint32_t k_block = 0; k_block < kNumKBlocks; adv(k_block)) {
                empty_barriers[stage]->wait(ph ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t k_idx = k_block * BLOCK_K;
                    // A: d_o tile [BLOCK_M rows starting at m_idx, BLOCK_K cols at k_idx]
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleMode, nv_bfloat16>(
                        &tensor_map_dy, full_barriers[stage],
                        smem_a[stage], k_idx, m_idx, 1);
                    // B: W2 tile [expert_idx, K=k_idx, N=n_idx]
                    tma::copy<BLOCK_N, BLOCK_K, kSwizzleMode, nv_bfloat16, true>(
                        &tensor_map_l2_weights, full_barriers[stage],
                        smem_b[stage], n_idx, k_idx, 1, expert_idx);
                    full_barriers[stage]->arrive_and_expect_tx(
                        SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);
                }
                __syncwarp();
            }
        }

    // =====================================================================
    // ROLE: MATH WARPGROUPS
    // =====================================================================
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {

        const uint32_t epilogue_warp_idx = warp_idx - kNumDispatchWarps - kNumMMANonEpilogueWarps;
        const uint32_t wg_idx = epilogue_warp_idx / 4;
        const uint32_t local_warp_in_wg = epilogue_warp_idx % 4;
        const uint32_t local_tid = local_warp_in_wg * 32 + lane_idx;

        // Descriptor base for stage 0
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleMode>(
            smem_a[0], 0u, 0u);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::MN, BLOCK_N, BLOCK_K, kSwizzleMode>(
            smem_b[0], wg_idx * WG_BLOCK_N, 0u);
        const uint32_t a_desc_base_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_base_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Wait for per-expert token counts (same mechanism as TMA warp)
        const auto workspace_math = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

        uint32_t expert_pool_offsets[kNumExPerRank];
        uint32_t expert_m_blocks[kNumExPerRank];
        uint32_t total_pool_blocks = 0;
        for (uint32_t e = 0; e < kNumExPerRank; ++e) {
            const auto sum_ptr = workspace_math.get_expert_recv_count_sum_ptr(e);
            uint64_t sum_val;
            do {
                // ld_volatile + exact equality match the forward scheduler
                // (scheduler/mega_moe.cuh:fetch_expert_recv_count). The high
                // 32 bits hold kNumSMs per rank (one atomic_add per CTA in the
                // count phase), so the fully-published value is kNumSMs*kNumRanks.
                // Polling `< kNumRanks` exits after the FIRST rank publishes for
                // kNumRanks>=2 (132 >= 2), reading a partial count — which then
                // starves the arrival-count wait in other warps and deadlocks.
                sum_val = ptx::ld_volatile(sum_ptr);
            } while (static_cast<uint32_t>(sum_val >> 32) != kNumSMs * kNumRanks);
            const uint32_t count = static_cast<uint32_t>(sum_val & 0xffffffff);
            expert_pool_offsets[e] = total_pool_blocks;
            expert_m_blocks[e] = math::ceil_div(count, BLOCK_M);
            total_pool_blocks += expert_m_blocks[e];
        }
        const uint32_t total_blocks = total_pool_blocks * kNumNBlocks;

        uint32_t stage = 0, ph = 0;
        auto adv = [&](uint32_t& k) { ++k; if (++stage >= kNumStages) { stage = 0; ph ^= 1; } };

        for (uint32_t block_id = sm_idx; block_id < total_blocks; block_id += kNumSMs) {
            // Map block_id → (expert_idx, local_m_block, n_block)
            uint32_t expert_idx = 0;
            uint32_t remaining = block_id;
            for (uint32_t e = 0; e < kNumExPerRank; ++e) {
                const uint32_t expert_work = expert_m_blocks[e] * kNumNBlocks;
                if (remaining < expert_work) { expert_idx = e; break; }
                remaining -= expert_work;
            }
            const uint32_t local_m_block = remaining / kNumNBlocks;
            const uint32_t n_block = remaining % kNumNBlocks;
            const uint32_t pool_block_idx = expert_pool_offsets[expert_idx] + local_m_block;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block * BLOCK_N;

            // Zero accumulators
            float accum[kAccumPerThread];
            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread; ++i) accum[i] = 0.0f;

            // WGMMA mainloop over K=kHidden
            for (uint32_t k_block = 0; k_block < kNumKBlocks; adv(k_block)) {
                full_barriers[stage]->wait(ph);

                const uint32_t a_lo = a_desc_base_lo + stage * (SMEM_A_PER_STAGE / 16);
                const uint32_t b_lo = b_desc_base_lo + stage * (SMEM_B_PER_STAGE / 16);

                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();

                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleMode, nv_bfloat16>(
                        a_lo, 0u, k * WGMMA::K, 0u);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::MN, BLOCK_N, BLOCK_K, kSwizzleMode, nv_bfloat16>(
                        b_lo, 0u, k * WGMMA::K, 0u);  // N.B.: wg N-offset is already baked into b_desc_base_lo via make_gmma_desc; passing it here too double-counts it (wg>=2 read OOB -> NaN).
                    WGMMA::wgmma(a_desc, b_desc, accum, true);
                }

                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }

            // Epilogue: fused SwiGLU backward
            // d_a (accumulator) is in registers. Read gate/up from recomp_h via
            // perm lookup (forward pool order → backward pool order), compute
            // d_gate/d_up, write d_h to d_h_buffer (backward order).
            constexpr uint32_t k2IH = 2 * kIntermediateHidden;
            auto* recomp_h_ptr = reinterpret_cast<const nv_bfloat16*>(recomp_h);
            auto* dh_out = reinterpret_cast<nv_bfloat16*>(d_h_buffer);

            const uint32_t warp_in_wg = local_warp_in_wg;
            const uint32_t base_row = warp_in_wg * 16;
            const uint32_t r0 = base_row + lane_idx / 4;
            const uint32_t r1 = r0 + 8;
            const uint32_t col_base = (lane_idx % 4) * 2;

            // Load perm and topk_weights for the two rows this thread touches.
            // perm_buf[bwd_row] → fwd_row (for reading gate/up from forward-order recomp_h).
            const uint32_t bwd_row0 = m_idx + r0;
            const uint32_t bwd_row1 = m_idx + r1;
            const uint32_t fwd_row0 = (bwd_row0 < kNumMaxPoolTokens) ? perm_buf[bwd_row0] : 0;
            const uint32_t fwd_row1 = (bwd_row1 < kNumMaxPoolTokens) ? perm_buf[bwd_row1] : 0;
            const float tw0 = (bwd_row0 < kNumMaxPoolTokens) ? topk_weights_pool[bwd_row0] : 0.0f;
            const float tw1 = (bwd_row1 < kNumMaxPoolTokens) ? topk_weights_pool[bwd_row1] : 0.0f;

            // Absolute column base within the IH dimension (warpgroup N-offset included)
            const uint32_t abs_col_base = n_idx + wg_idx * WG_BLOCK_N + col_base;

            // Cast output to uint32_t* for vectorized 4B (2×BF16) stores
            auto* dh_out_u32 = reinterpret_cast<uint32_t*>(dh_out);
            auto* recomp_h_u32 = reinterpret_cast<const uint32_t*>(recomp_h_ptr);

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
                const uint32_t col = abs_col_base + i * 8;
                // col is always 2-aligned (col_base = lane%4 * 2, i*8 is even)
                const uint32_t col_half = col / 2;  // uint32_t index (2 BF16 per u32)
                constexpr uint32_t ih_half = kIntermediateHidden / 2;
                constexpr uint32_t k2ih_half = 2 * kIntermediateHidden / 2;

                if (bwd_row0 < kNumMaxPoolTokens and col + 1 < kIntermediateHidden) {
                    float da0 = accum[i*4+0], da1 = accum[i*4+1];
                    // Vectorized 4B load: gate[col:col+2] and up[col:col+2]
                    uint32_t gate_packed0 = recomp_h_u32[fwd_row0 * k2ih_half + col_half];
                    uint32_t up_packed0   = recomp_h_u32[fwd_row0 * k2ih_half + ih_half + col_half];
                    float gate0 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate_packed0)));
                    float gate1 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate_packed0 >> 16)));
                    float up0   = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(up_packed0)));
                    float up1   = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(up_packed0 >> 16)));

                    float dg0, du0, dg1, du1;
                    sm90_bf16_swiglu_backward<kFastMath>(da0, gate0, up0, tw0, dg0, du0);
                    sm90_bf16_swiglu_backward<kFastMath>(da1, gate1, up1, tw0, dg1, du1);

                    // Vectorized 4B store: pack 2 BF16 into uint32_t
                    dh_out_u32[bwd_row0 * k2ih_half + col_half]            = math::cast_into_bf16_and_pack(dg0, dg1);
                    dh_out_u32[bwd_row0 * k2ih_half + ih_half + col_half]  = math::cast_into_bf16_and_pack(du0, du1);
                }
                if (bwd_row1 < kNumMaxPoolTokens and col + 1 < kIntermediateHidden) {
                    float da2 = accum[i*4+2], da3 = accum[i*4+3];
                    uint32_t gate_packed1 = recomp_h_u32[fwd_row1 * k2ih_half + col_half];
                    uint32_t up_packed1   = recomp_h_u32[fwd_row1 * k2ih_half + ih_half + col_half];
                    float gate2 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate_packed1)));
                    float gate3 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(gate_packed1 >> 16)));
                    float up2   = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(up_packed1)));
                    float up3   = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(up_packed1 >> 16)));

                    float dg2, du2, dg3, du3;
                    sm90_bf16_swiglu_backward<kFastMath>(da2, gate2, up2, tw1, dg2, du2);
                    sm90_bf16_swiglu_backward<kFastMath>(da3, gate3, up3, tw1, dg3, du3);

                    dh_out_u32[bwd_row1 * k2ih_half + col_half]            = math::cast_into_bf16_and_pack(dg2, dg3);
                    dh_out_u32[bwd_row1 * k2ih_half + ih_half + col_half]  = math::cast_into_bf16_and_pack(du2, du3);
                }
            }
        }
    }
    // =====================================================================
    // ROLE: DISPATCH WARPS — Combine Backward (NVLink pull dy → d_o_pool)
    // Mirrors the forward dispatch protocol: count tokens per expert,
    // exchange counts, pull BF16 dy tokens from source ranks into the
    // local d_o_pool, write TokenSrcMetadata for L1 scatter, and signal
    // arrival counts so TMA load warps can proceed.
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        // Workspace and buffer setup
        const auto workspace = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

        constexpr auto bf16_hidden_layout = layout::Data(kHidden * sizeof(nv_bfloat16));

        // Shared memory for dispatch: expert_count at the end of GEMM smem
        constexpr uint32_t kDispatchSmemOffset = BARRIER_OFFSET + 2 * kNumStages * sizeof(Barrier);
        auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer + kDispatchSmemOffset);
        constexpr uint32_t kSmemSendBufferOffset = kDispatchSmemOffset +
            math::constexpr_align(kNumExperts * static_cast<uint32_t>(sizeof(uint32_t)), 1024u);
        const auto smem_send_buffers = layout::Buffer(bf16_hidden_layout, kNumDispatchWarps, 1,
            smem_buffer + kSmemSendBufferOffset);

        // Barrier indices for dispatch
        constexpr uint32_t kDispatchBarrierIdx = 0;
        constexpr uint32_t kDispatchGridSyncIndex = 0;
        constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;

        // Clear expert counts
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
            smem_expert_count[i] = 0;
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Read topk_idx and count tokens per expert
        constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;

        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(reinterpret_cast<const int64_t*>(input_topk_idx) + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Publish recv counts to remote ranks
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Build cumulative expert recv stats for the scheduler
        // (needed by TMA load warps to know valid M per expert)
        auto sched_workspace = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

        // Token / BF16 pull loop — pull dy from source ranks into d_o_pool
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        auto* pull_mbarrier = reinterpret_cast<Barrier*>(
            smem_buffer + kSmemSendBufferOffset +
            math::align<uint32_t>(static_cast<uint32_t>(smem_send_buffers.get_num_bytes()), 128u) +
            warp_idx * sizeof(Barrier));

        if (lane_idx == 0)
            pull_mbarrier->init(1);
        __syncwarp();

        // Fetch expert recv counts for scheduling
        uint32_t expert_recv_counts[kNumExPerRank];
        for (uint32_t e = 0; e < kNumExPerRank; ++e) {
            const auto sum_ptr = workspace.get_expert_recv_count_sum_ptr(e);
            uint64_t sum_val;
            do {
                // ld_volatile + exact equality match the forward scheduler
                // (scheduler/mega_moe.cuh:fetch_expert_recv_count). The high
                // 32 bits hold kNumSMs per rank (one atomic_add per CTA in the
                // count phase), so the fully-published value is kNumSMs*kNumRanks.
                // Polling `< kNumRanks` exits after the FIRST rank publishes for
                // kNumRanks>=2 (132 >= 2), reading a partial count — which then
                // starves the arrival-count wait in other warps and deadlocks.
                sum_val = ptx::ld_volatile(sum_ptr);
            } while (static_cast<uint32_t>(sum_val >> 32) != kNumSMs * kNumRanks);
            expert_recv_counts[e] = static_cast<uint32_t>(sum_val & 0xffffffff);
        }

        // Write cumulative stats (SM 0 only, warp 0)
        if (sm_idx == 0 and warp_idx == 0 and lane_idx == 0) {
            int cum = 0;
            cumulative_local_expert_recv_stats[0] = 0;
            for (uint32_t e = 0; e < kNumExPerRank; ++e) {
                cum += expert_recv_counts[e];
                cumulative_local_expert_recv_stats[e + 1] = cum;
            }
        }

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= static_cast<int>(kNumExPerRank))
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += expert_recv_counts[current_expert_idx];
            }
            if (current_expert_idx >= static_cast<int>(kNumExPerRank))
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection (identical to forward)
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                if (num_active_ranks == 0) break;
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

            // Pull BF16 dy token from source rank via NVLink TMA
            constexpr uint32_t kTokenBytes = kHidden * sizeof(nv_bfloat16);
            if (cute::elect_one_sync()) {
                const auto* dy_token_ptr = reinterpret_cast<const uint8_t*>(dy_source) +
                    static_cast<uint64_t>(src_token_idx) * kTokenBytes;
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(dy_token_ptr,
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kTokenBytes);
            }
            __syncwarp();

            // Pull topk_weight for this slot
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    reinterpret_cast<const float*>(input_topk_weights) + src_token_topk_idx,
                    current_rank_in_expert_idx);
                topk_weights_pool[pool_token_idx] = weight;
            }
            __syncwarp();

            // Wait for TMA load, then TMA store into d_o_pool
            if (cute::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kTokenBytes);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                // Store to local d_o_pool
                auto* dst_ptr = reinterpret_cast<uint8_t*>(d_o_pool) +
                    static_cast<uint64_t>(pool_token_idx) * kTokenBytes;
                ptx::tma_store_1d(dst_ptr, pull_buffer.get_base_ptr(), kTokenBytes);

                // Write TokenSrcMetadata for L1 kernel scatter
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Write perm_buf: lookup forward pool position for this token
                // key = rank * stride_r + token * stride_t + topk_idx
                {
                    const uint64_t lookup_key =
                        static_cast<uint64_t>(current_rank_in_expert_idx) * fwd_lookup_stride_r +
                        static_cast<uint64_t>(src_token_idx) * fwd_lookup_stride_t +
                        static_cast<uint64_t>(src_topk_idx);
                    const int32_t fwd_pos = fwd_lookup_table[lookup_key];
                    perm_buf[pool_token_idx] = (fwd_pos >= 0)
                        ? static_cast<uint32_t>(fwd_pos) : pool_token_idx;
                }

                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();

                // Signal arrival for this pool block
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }
    }

#endif  // __CUDA_ARCH__
}

// ============================================================================
// Kernel 2: Backward L1
// Computes:
//   1. d_x[M,H] = d_h[M,2*IH] @ W1[2*IH,H]  (activation gradient via WGMMA)
//   2. Dispatch backward: scatter d_x to source ranks, combine reduction
//
// SwiGLU backward is already fused into L2's epilogue. d_h is ready in
// d_h_buffer (backward pool order) when this kernel launches.
//
// Weight gradient dW1 is computed host-side after both kernels finish, using
// cuBLASLt per-expert GEMM:
//   dW1[e, 2*IH, H] = d_h_pool[M_e, 2*IH]^T @ x_pool[M_e, H]
// This avoids atomicAdd contention and leverages cuBLASLt TF32 throughput.
//
// Kernel structure:
//   Dispatch warps: idle during GEMM, active for combine scatter afterward
//   TMA warp: load d_h + W1 tiles (no arrival polling — d_h ready from L2)
//   Math warpgroups: WGMMA → scatter epilogue (NVLink write to remote combine buf)
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kFastMath,
    bool kRecompute,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_bf16_mega_moe_backward_l1_impl(
    void* dx,                                    // [pool, H] BF16 — GEMM output (pool layout)
    void* dx_final,                              // [T, H] BF16 — final combined output
    void* dx_combine_buffer,                     // [topk, T, H] BF16 — scatter target for dispatch BW
    float* dW1,
    const void* l1_weights_bf16,
    const void* x_bf16,
    const void* input_topk_idx,                  // [T, topk] int64 — for combine reduction
    int* cumulative_local_expert_recv_stats,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_d_h,      // d_h_buffer TMA
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dx_output) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    // Suppress unused-parameter warnings (dW1 computed host-side via cuBLASLt;
    // dx pool buffer no longer needed — scatter writes directly to remote combine buffer;
    // tensor_map_dx_output no longer needed — scatter bypasses TMA store)
    (void)dW1;
    (void)x_bf16;
    (void)dx;
    (void)tensor_map_dx_output;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // L1 backward GEMM: d_x[M,H] = d_h[M,2*IH] @ W1[2*IH,H]
    // K = 2*IH, N = H
    // A (d_h): K-major (2*IH contiguous in gmem)
    // B (W1): MN-major (H=N contiguous in gmem, W1 stored as [E, 2*IH, H])
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kNumEpilogueWarpgroups;
    using WGMMA = typename mma::sm90::BF16MMASelector<WG_BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::MN>::type;
    static_assert(WGMMA::K == 16, "BF16 WGMMA K must be 16");
    constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;

    constexpr uint32_t kBF16Bytes = sizeof(nv_bfloat16);
    constexpr uint32_t kSwizzleMode = BLOCK_K * kBF16Bytes;
    static_assert(kSwizzleMode <= 128, "Swizzle mode must be <= 128B");
    constexpr uint32_t kAlign = 1024;
    constexpr uint32_t k2IH = 2 * kIntermediateHidden;

    // Shared memory: [A stages][B stages][barriers]
    extern __shared__ __align__(kAlign) uint8_t smem_buffer[];
    constexpr uint32_t SMEM_A_PER_STAGE = BLOCK_M * BLOCK_K * kBF16Bytes;
    constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * kBF16Bytes;

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + i * SMEM_A_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer +
            kNumStages * SMEM_A_PER_STAGE + i * SMEM_B_PER_STAGE);
    });
    constexpr uint32_t BARRIER_OFFSET = kNumStages * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);
    auto* barrier_base = reinterpret_cast<Barrier*>(smem_buffer + BARRIER_OFFSET);
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_base + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages + i; });

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = thread_idx % 32;
    const uint32_t sm_idx = blockIdx.x;

    // Barrier init
    if (warp_idx == 0) {
        if (lane_idx < kNumStages) {
            full_barriers[lane_idx]->init(1);
            empty_barriers[lane_idx]->init(kNumEpilogueWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Scheduling: K of the GEMM = 2*IH, N = H
    constexpr uint32_t kNumKBlocks = k2IH / BLOCK_K;
    constexpr uint32_t kNumNBlocks = kHidden / BLOCK_N;
    constexpr uint32_t kNumExPerRank = kNumExpertsPerRank;

    // =====================================================================
    // Phase B: TMA Load + WGMMA for d_x = d_h @ W1
    // SwiGLU backward is now fused per-block into dispatch warps, running
    // concurrently with TMA load + GEMM (mirrors forward's dispatch ∥ GEMM).
    // =====================================================================

    // Register reconfiguration: warpgroup-collective setmaxnreg requires all
    // 4 warps in a warpgroup to participate. Warpgroup 0 (warps 0-3) decreases
    // registers; warpgroups 1+ (warps 4-19, math) increase registers.
    // Budget: 128 threads × 24 + 512 threads × 120 = 3072 + 61440 = 64512 ≤ 65536
    if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<120>();
    }

    // ROLE: DISPATCH WARPS — idle during GEMM phase
    // SwiGLU backward is now fused into L2's epilogue. d_h is ready when L1 launches.
    // Dispatch warps will be used later for combine backward (scatter d_x to source
    // ranks) — that code follows after the math warpgroup section.
    if (warp_idx < kNumDispatchWarps) {
        // Nothing to do during GEMM — combine backward handled after GEMM completes.
        // (Combine code is at the end of this kernel, gated by a grid_sync.)

    // ROLE: TMA LOAD WARP
    } else if (warp_idx == kNumDispatchWarps) {

        // L1 launches after L2 completes → d_h_buffer is fully written, no polling needed.
        // Compute total work items by scanning cumulative stats.
        uint32_t total_pool_blocks = 0;
        for (uint32_t e = 0; e < kNumExPerRank; ++e) {
            const uint32_t count = static_cast<uint32_t>(
                cumulative_local_expert_recv_stats[e + 1] - cumulative_local_expert_recv_stats[e]);
            total_pool_blocks += math::ceil_div(count, BLOCK_M);
        }
        const uint32_t total_blocks = total_pool_blocks * kNumNBlocks;

        uint32_t stage = 0, ph = 0;
        auto adv = [&](uint32_t& k) { ++k; if (++stage >= kNumStages) { stage = 0; ph ^= 1; } };

        for (uint32_t block_id = sm_idx; block_id < total_blocks; block_id += kNumSMs) {
            // Map block_id → (expert_idx, local_m_block, n_block) by scanning experts
            uint32_t expert_idx = 0, pool_block_start = 0;
            uint32_t remaining = block_id;
            for (uint32_t e = 0; e < kNumExPerRank; ++e) {
                const uint32_t cnt = static_cast<uint32_t>(
                    cumulative_local_expert_recv_stats[e + 1] - cumulative_local_expert_recv_stats[e]);
                const uint32_t e_m_blocks = math::ceil_div(cnt, BLOCK_M);
                const uint32_t expert_work = e_m_blocks * kNumNBlocks;
                if (remaining < expert_work) { expert_idx = e; break; }
                remaining -= expert_work;
                pool_block_start += e_m_blocks;
            }
            const uint32_t local_m_block = remaining / kNumNBlocks;
            const uint32_t n_block = remaining % kNumNBlocks;
            const uint32_t pool_block_idx = pool_block_start + local_m_block;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block * BLOCK_N;

            for (uint32_t k_block = 0; k_block < kNumKBlocks; adv(k_block)) {
                empty_barriers[stage]->wait(ph ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t k_idx = k_block * BLOCK_K;
                    // A: d_h [BLOCK_M, BLOCK_K] from row m_idx, col k_idx
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleMode, nv_bfloat16>(
                        &tensor_map_d_h, full_barriers[stage],
                        smem_a[stage], k_idx, m_idx, 1);
                    // B: W1 [expert_idx, K=k_idx, N=n_idx]
                    tma::copy<BLOCK_N, BLOCK_K, kSwizzleMode, nv_bfloat16, true>(
                        &tensor_map_l1_weights, full_barriers[stage],
                        smem_b[stage], n_idx, k_idx, 1, expert_idx);
                    full_barriers[stage]->arrive_and_expect_tx(
                        SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);
                }
                __syncwarp();
            }
        }

    // ROLE: MATH WARPGROUPS
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {

        const uint32_t epilogue_warp_idx = warp_idx - kNumDispatchWarps - kNumMMANonEpilogueWarps;
        const uint32_t wg_idx = epilogue_warp_idx / 4;
        const uint32_t local_warp_in_wg = epilogue_warp_idx % 4;
        const uint32_t local_tid = local_warp_in_wg * 32 + lane_idx;

        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleMode>(
            smem_a[0], 0u, 0u);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::MN, BLOCK_N, BLOCK_K, kSwizzleMode>(
            smem_b[0], wg_idx * WG_BLOCK_N, 0u);
        const uint32_t a_desc_base_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_base_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Read per-expert pool block offsets (same as TMA warp)
        uint32_t expert_pool_offsets[kNumExPerRank];
        uint32_t expert_m_blocks[kNumExPerRank];
        uint32_t total_pool_blocks = 0;
        for (uint32_t e = 0; e < kNumExPerRank; ++e) {
            expert_pool_offsets[e] = total_pool_blocks;
            const uint32_t count = static_cast<uint32_t>(
                cumulative_local_expert_recv_stats[e + 1] - cumulative_local_expert_recv_stats[e]);
            expert_m_blocks[e] = math::ceil_div(count, BLOCK_M);
            total_pool_blocks += expert_m_blocks[e];
        }
        const uint32_t total_blocks = total_pool_blocks * kNumNBlocks;

        // Workspace and dx_combine_buffer layout for inline NVLink scatter
        const auto scatter_workspace = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);
        constexpr auto bf16_hidden_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
        const auto dx_combine_buf = layout::Buffer(bf16_hidden_layout, kNumTopk, kNumMaxTokensPerRank,
            dx_combine_buffer);

        uint32_t stage = 0, ph = 0;
        auto adv = [&](uint32_t& k) { ++k; if (++stage >= kNumStages) { stage = 0; ph ^= 1; } };

        for (uint32_t block_id = sm_idx; block_id < total_blocks; block_id += kNumSMs) {
            // Map block_id → (expert_idx, local_m_block, n_block)
            uint32_t expert_idx = 0;
            uint32_t remaining = block_id;
            for (uint32_t e = 0; e < kNumExPerRank; ++e) {
                const uint32_t expert_work = expert_m_blocks[e] * kNumNBlocks;
                if (remaining < expert_work) { expert_idx = e; break; }
                remaining -= expert_work;
            }
            const uint32_t local_m_block = remaining / kNumNBlocks;
            const uint32_t n_block = remaining % kNumNBlocks;
            const uint32_t pool_block_idx = expert_pool_offsets[expert_idx] + local_m_block;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block * BLOCK_N;

            // Zero accumulators
            float accum[kAccumPerThread];
            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread; ++i) accum[i] = 0.0f;

            // WGMMA mainloop: K = 2*IH
            for (uint32_t k_block = 0; k_block < kNumKBlocks; adv(k_block)) {
                full_barriers[stage]->wait(ph);

                const uint32_t a_lo = a_desc_base_lo + stage * (SMEM_A_PER_STAGE / 16);
                const uint32_t b_lo = b_desc_base_lo + stage * (SMEM_B_PER_STAGE / 16);

                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();

                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleMode, nv_bfloat16>(
                        a_lo, 0u, k * WGMMA::K, 0u);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::MN, BLOCK_N, BLOCK_K, kSwizzleMode, nv_bfloat16>(
                        b_lo, 0u, k * WGMMA::K, 0u);
                    WGMMA::wgmma(a_desc, b_desc, accum, true);
                }

                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }

            // ─── Epilogue: BF16 cast + inline NVLink scatter ───────────────
            // Mirrors forward L2 epilogue (sm90_fp8_mega_moe.cuh:1687-1765).
            // Each thread owns specific (row, col) accumulator elements in the
            // WGMMA output layout. We convert to BF16, then scatter directly to
            // the remote rank's dx_combine_buffer via sym_buffer.map().
            //
            // WGMMA BF16 M=64 output layout per thread (within a warpgroup):
            //   warp_in_wg (0-3) → 16-row slice: base_row = warp_in_wg * 16
            //   lane_idx / 4     → row offset within 8-row half (r0 = base + offset)
            //   lane_idx % 4     → column pair start: col_base = (lane%4) * 2
            //   Each accum chunk of 4 floats: (r0,col), (r0,col+1), (r1,col), (r1,col+1)
            //   where r1 = r0 + 8.

            const uint32_t base_row = local_warp_in_wg * 16;
            const uint32_t r0 = base_row + lane_idx / 4;
            const uint32_t r1 = r0 + 8;
            const uint32_t col_base = (lane_idx % 4) * 2;

            // Compute absolute column offset for this warpgroup's output slice
            const uint32_t wg_col_offset = n_idx + wg_idx * WG_BLOCK_N;

            // Load per-row token metadata once (same for all column iterations)
            const bool valid_r0 = (m_idx + r0 < kNumMaxPoolTokens);
            const bool valid_r1 = (m_idx + r1 < kNumMaxPoolTokens);

            layout::TokenSrcMetadata meta_r0, meta_r1;
            if (valid_r0) meta_r0 = *scatter_workspace.get_token_src_metadata_ptr(m_idx + r0);
            if (valid_r1) meta_r1 = *scatter_workspace.get_token_src_metadata_ptr(m_idx + r1);

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
                const uint32_t col = col_base + i * 8;

                if (valid_r0 and col < WG_BLOCK_N and meta_r0.rank_idx < kNumRanks) {
                    // Pack two BF16 values into a uint32 for a single 4-byte NVLink write
                    const uint32_t packed_r0 = math::cast_into_bf16_and_pack(accum[i*4+0], accum[i*4+1]);
                    // Destination: dx_combine_buffer[topk_slot][token_idx] + byte offset for this col
                    auto* dst_base_r0 = reinterpret_cast<uint8_t*>(
                        dx_combine_buf.get_rank_buffer(meta_r0.topk_idx)
                                      .get_data_buffer(meta_r0.token_idx).get_base_ptr());
                    auto* dst_r0 = reinterpret_cast<uint32_t*>(
                        dst_base_r0 + (wg_col_offset + col) * sizeof(nv_bfloat16));
                    *sym_buffer.map(dst_r0, meta_r0.rank_idx) = packed_r0;
                }

                if (valid_r1 and col < WG_BLOCK_N and meta_r1.rank_idx < kNumRanks) {
                    const uint32_t packed_r1 = math::cast_into_bf16_and_pack(accum[i*4+2], accum[i*4+3]);
                    auto* dst_base_r1 = reinterpret_cast<uint8_t*>(
                        dx_combine_buf.get_rank_buffer(meta_r1.topk_idx)
                                      .get_data_buffer(meta_r1.token_idx).get_base_ptr());
                    auto* dst_r1 = reinterpret_cast<uint32_t*>(
                        dst_base_r1 + (wg_col_offset + col) * sizeof(nv_bfloat16));
                    *sym_buffer.map(dst_r1, meta_r1.rank_idx) = packed_r1;
                }
            }

            // NOTE: Weight gradient dW1 computed host-side via cuBLASLt after
            // both kernels finish: dW1[e] = d_h_pool[e]^T @ x_pool[e]
        }

        // System-scope fence after all scatter writes — ensures NVLink writes
        // from this thread are globally visible before the nvlink_barrier signals
        // remote ranks. One fence at the end (not per-tile) is correct because
        // the remote rank's combine only begins after nvlink_barrier completes.
        __threadfence_system();
    }

    // Block-level barrier: all threads converge here — dispatch warps arrive
    // (idle during GEMM), TMA/math warps arrive after finishing all GEMM tiles
    // + scatter + threadfence_system.
    __syncthreads();

    // =====================================================================
    // ROLE: DISPATCH WARPS — NVLink barrier + Combine reduction
    // The NVLink scatter is fused into the math warpgroup epilogue above.
    // Dispatch warps: (1) cross-rank NVLink barrier ensuring all remote
    // scatters are visible, then (2) combine reduction: dx_final[t] = sum_k dx_combine[k][t].
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        const auto workspace = layout::Workspace(
            sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

        constexpr uint32_t kDispatchBarrierIdx = 0;
        constexpr uint32_t kDispatchGridSyncIndex = 0;
        constexpr uint32_t kScatterBarrierTag = 4;

        // NVLink barrier: ensure all ranks' scatter writes are globally visible.
        // sync_prologue = true: grid_sync first so that the nvlink signal is only
        // sent after ALL SMs on this rank have completed their scatter + fence.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kScatterBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, true);

        // ─── Combine reduction: dx[t] = sum_k dx_combine[k][t] ───
        // Each warp processes a strided subset of tokens
        constexpr uint32_t kTokenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kVecSize = sizeof(uint4);  // 16 bytes = 8 BF16
        constexpr uint32_t kElemsPerVec = kVecSize / sizeof(nv_bfloat16);
        constexpr uint32_t kVecsPerRow = kHidden / kElemsPerVec;
        constexpr uint32_t kNumCombineWarps = kNumDispatchWarps;
        constexpr uint32_t kNumGlobalCombineWarps = kNumSMs * kNumCombineWarps;

        for (uint32_t token_idx = sm_idx * kNumCombineWarps + warp_idx;
             token_idx < num_tokens;
             token_idx += kNumGlobalCombineWarps) {
            // For each vector position in the hidden dimension
            for (uint32_t vec_idx = lane_idx; vec_idx < kVecsPerRow; vec_idx += 32) {
                // Accumulate across topk slots
                float2 accum[kElemsPerVec / 2] = {};

                #pragma unroll
                for (uint32_t k = 0; k < kNumTopk; ++k) {
                    const int64_t expert_idx_for_slot = __ldg(
                        reinterpret_cast<const int64_t*>(input_topk_idx) + token_idx * kNumTopk + k);
                    if (expert_idx_for_slot < 0) continue;

                    const auto* slot_ptr = reinterpret_cast<const uint4*>(
                        reinterpret_cast<const uint8_t*>(dx_combine_buffer) +
                        (static_cast<uint64_t>(k) * kNumMaxTokensPerRank + token_idx) * kTokenBytes);
                    const uint4 data = slot_ptr[vec_idx];
                    const auto* bf16_data = reinterpret_cast<const nv_bfloat162*>(&data);

                    #pragma unroll
                    for (uint32_t l = 0; l < kElemsPerVec / 2; ++l) {
                        float2 vals = __bfloat1622float2(bf16_data[l]);
                        accum[l].x += vals.x;
                        accum[l].y += vals.y;
                    }
                }

                // Write final dx[token_idx] in BF16
                uint4 result;
                auto* result_bf16 = reinterpret_cast<nv_bfloat162*>(&result);
                #pragma unroll
                for (uint32_t l = 0; l < kElemsPerVec / 2; ++l)
                    result_bf16[l] = __float22bfloat162_rn(accum[l]);

                auto* out_ptr = reinterpret_cast<uint4*>(
                    reinterpret_cast<uint8_t*>(dx_final) +
                    static_cast<uint64_t>(token_idx) * kTokenBytes);
                out_ptr[vec_idx] = result;
            }
        }
    }

#endif  // __CUDA_ARCH__
}

} // namespace deep_gemm

#pragma clang diagnostic pop
