#pragma once

#include <torch/python.h>
#include <algorithm>
#include <cstdlib>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe_backward.hpp"
#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 BF16 MegaMoE Backward L2 Runtime
// ----------------------------------------------------------------------------
// JIT host-side runtime for the backward L2 kernel (Backward Combine + L2 BW).
// Mirrors SM90FP8MegaMoERuntime for the forward path.
// ============================================================================

class SM90BF16MegaMoEBackwardL2Runtime final : public LaunchRuntime<SM90BF16MegaMoEBackwardL2Runtime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        bool fast_math;
        bool recompute;
        MegaMoESM90BackwardConfig config;

        // Runtime arguments
        void* d_h_buffer;              // [pool, 2*IH] BF16 — SwiGLU BW output (bwd order)
        void* d_o_pool;                // [pool, H] BF16 — dispatch fills, GEMM reads
        float* dW2;                    // [E, H, IH] FP32 weight grad (host-side cuBLASLt)
        const void* l2_weights_bf16;   // [E, H, IH] BF16
        const void* recomp_h;          // [pool, 2*IH] BF16 — gate/up (fwd order)
        const void* recomp_a;          // [pool, IH] BF16 (unused, for dW2)
        float* topk_weights_pool;      // [pool] FP32 — dispatch fills, epilogue reads
        uint32_t* perm_buf;            // [pool] — dispatch warp fills bwd→fwd pos mapping
        const int32_t* fwd_lookup_table;  // [num_ranks * max_tokens * topk] — key→fwd_pool_pos
        uint32_t fwd_lookup_stride_r;  // = max_tokens_per_rank * topk
        uint32_t fwd_lookup_stride_t;  // = topk
        const void* input_topk_idx;    // [T, topk] int64 — from forward sym_buffer
        const void* input_topk_weights; // [T, topk] float — from forward sym_buffer
        const void* dy_source;         // [T, H] BF16 — in sym_buffer, NVLink-accessible
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // TMA descriptors
        CUtensorMap tensor_map_d_o;    // for GEMM A operand (d_o_pool)
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_d_a_output;  // kept for ABI compat (voided in kernel)

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
// L2 v7: dispatch warp builds perm in-kernel (no post-L2 sync)
#include <deep_gemm/impls/sm90_bf16_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_moe_backward_l2_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    args.fast_math ? "true" : "false",
    args.recompute ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.d_h_buffer,
            args.d_o_pool,
            args.dW2,
            args.l2_weights_bf16,
            args.recomp_h,
            args.recomp_a,
            args.topk_weights_pool,
            args.perm_buf,
            args.fwd_lookup_table,
            args.fwd_lookup_stride_r,
            args.fwd_lookup_stride_t,
            args.input_topk_idx,
            args.input_topk_weights,
            args.dy_source,
            args.cumulative_local_expert_recv_stats,
            static_cast<uint32_t>(args.num_tokens),
            args.sym_buffer_ptrs,
            args.tensor_map_d_o,
            args.tensor_map_l2_weights,
            args.tensor_map_d_a_output
        ));
    }
};

// ============================================================================
// SM90 BF16 MegaMoE Backward L1 Runtime
// ----------------------------------------------------------------------------
// JIT host-side runtime for the backward L1 kernel (Linear1 BW + Dispatch BW).
// SwiGLU BW is now fused into L2's epilogue — L1 is a pure GEMM + scatter.
// ============================================================================

class SM90BF16MegaMoEBackwardL1Runtime final : public LaunchRuntime<SM90BF16MegaMoEBackwardL1Runtime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        bool fast_math;
        bool recompute;
        MegaMoESM90BackwardConfig config;

        // Runtime arguments
        void* dx;                       // [pool, H] BF16 pool-layout GEMM output
        void* dx_final;                 // [T, H] BF16 final output (after combine)
        void* dx_combine_buffer;        // [topk, T, H] BF16 scatter target (in sym_buffer)
        float* dW1;                     // [E, 2*IH, H] FP32 weight grad
        const void* l1_weights_bf16;    // [E, 2*IH, H] BF16
        const void* x_bf16;            // [pool, H] BF16 (dequantized input for dW1)
        const void* input_topk_idx;     // [T, topk] int64 — for combine reduction
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // TMA descriptors
        CUtensorMap tensor_map_d_h;     // d_h_buffer [pool, 2*IH] TMA
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_dx_output;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
// L1 v7: remove redundant workspace zero (API handles it)
#include <deep_gemm/impls/sm90_bf16_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_moe_backward_l1_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    args.fast_math ? "true" : "false",
    args.recompute ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.dx,
            args.dx_final,
            args.dx_combine_buffer,
            args.dW1,
            args.l1_weights_bf16,
            args.x_bf16,
            reinterpret_cast<const void*>(args.input_topk_idx),
            args.cumulative_local_expert_recv_stats,
            static_cast<uint32_t>(args.num_tokens),
            args.sym_buffer_ptrs,
            args.tensor_map_d_h,
            args.tensor_map_l1_weights,
            args.tensor_map_dx_output
        ));
    }
};

// ============================================================================
// Host-side backward launch function
// Launches L2 and L1 fused GEMM kernels for activation gradients (dx).
// Weight gradients (dW1, dW2) are computed separately after both kernels
// finish — see Phase 3 in sm90_mega_backward.hpp (cuBLASLt per-expert GEMM).
// ============================================================================

static void sm90_bf16_mega_moe_backward(
    const torch::Tensor& dx_final,
    const torch::Tensor& dx_pool,
    const torch::Tensor& dW1, const torch::Tensor& dW2,
    const torch::Tensor& l1_weights_bf16, const torch::Tensor& l2_weights_bf16,
    const torch::Tensor& d_a_buffer, const torch::Tensor& d_o_buffer,
    const torch::Tensor& recomp_h_buffer, const torch::Tensor& recomp_a_buffer,
    const torch::Tensor& x_bf16_buffer,
    const torch::Tensor& topk_weights_pool,
    const torch::Tensor& dx_combine_buffer,
    const torch::Tensor& d_h_buffer,          // [pool, 2*IH] — L2 epilogue writes SwiGLU BW output
    const torch::Tensor& input_topk_idx,
    const torch::Tensor& input_topk_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const void* dy_source,                   // dy copied into sym_buffer (NVLink-accessible)
    const torch::Tensor& fwd_token_src_metadata,  // Forward dispatch metadata for perm correction
    const std::vector<int>& expert_counts,        // FULL per-local-expert counts (all ranks)
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& recompute, const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = 0;  // BF16 path has no SF

    // Get backward heuristic config
    const auto config = get_mega_moe_backward_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    const auto num_sms = device_runtime->get_num_sms();

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // ─── Build fwd_lookup_table: maps (rank, token, topk) → fwd_pool_pos ───
    // The dispatch warp will use this table to dynamically build perm_buf
    // inside the kernel, eliminating the need for post-L2 sync + correction.
    // Must match the forward kernel's block_m selection (sm90_mega_moe.hpp)
    const auto [fwd_block_m, fwd_epilogue_unused_] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int kFwdPoolBlockM = fwd_block_m;
    const int kBwdPoolBlockM = config.block_m;

    auto make_pool_offsets = [&](int block_m) {
        std::vector<int64_t> offs(num_experts_per_rank, 0);
        for (int e = 1; e < num_experts_per_rank; ++e)
            offs[e] = offs[e-1] +
                ((expert_counts[e-1] + block_m - 1) / block_m) * block_m;
        return offs;
    };
    const std::vector<int64_t> fwd_pool_offsets = make_pool_offsets(kFwdPoolBlockM);
    const std::vector<int64_t> bwd_pool_offsets = make_pool_offsets(kBwdPoolBlockM);

    // Build fwd_lookup_table on CPU: key = rank * stride_r + token * stride_t + topk_idx
    // Value = fwd_pool_pos. Key space is small: num_ranks * max_tokens * topk.
    const uint32_t fwd_lookup_stride_t = static_cast<uint32_t>(num_topk);
    const uint32_t fwd_lookup_stride_r = static_cast<uint32_t>(num_max_tokens_per_rank) * fwd_lookup_stride_t;
    const int64_t fwd_lookup_size = static_cast<int64_t>(num_ranks) * fwd_lookup_stride_r;

    auto fwd_lookup_cpu = torch::full({fwd_lookup_size}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* fl_ptr = fwd_lookup_cpu.data_ptr<int32_t>();

    // Parse forward TokenSrcMetadata to populate the lookup table
    auto fwd_md_cpu = fwd_token_src_metadata.to(torch::kCPU);
    auto* fwd_meta = reinterpret_cast<const layout::TokenSrcMetadata*>(fwd_md_cpu.data_ptr());
    for (int e = 0; e < num_experts_per_rank; ++e) {
        for (int i = 0; i < expert_counts[e]; ++i) {
            const auto& m = fwd_meta[fwd_pool_offsets[e] + i];
            if (m.rank_idx == 0xFFFFFFFF) continue;
            uint64_t key = static_cast<uint64_t>(m.rank_idx) * fwd_lookup_stride_r +
                           static_cast<uint64_t>(m.token_idx) * fwd_lookup_stride_t +
                           static_cast<uint64_t>(m.topk_idx);
            fl_ptr[key] = static_cast<int32_t>(fwd_pool_offsets[e] + i);
        }
    }
    auto fwd_lookup_gpu = fwd_lookup_cpu.to(recomp_h_buffer.device());

    // Allocate perm_buf on GPU — dispatch warp will fill it in-kernel
    auto perm_buf = torch::zeros({static_cast<int64_t>(config.num_max_pool_tokens)},
        torch::TensorOptions().dtype(torch::kInt32).device(recomp_h_buffer.device()));

    // NOTE: Workspace zeroing + TokenSrcMetadata sentinel (0xFF) + cross-rank sync
    // are handled by the caller (sm90_mega_backward.hpp) BEFORE this function is
    // called. Do NOT re-zero here — it would overwrite the 0xFF sentinel that
    // causes L1 scatter to skip padding positions.

    // ─── Kernel 1: Backward L2 (with fused SwiGLU BW epilogue) ──────────
    // L2 GEMM: d_a[M,IH] = d_o_pool[M,H] @ W2[H,IH]. K=H, N=IH.
    // Epilogue: SwiGLU BW → d_h[M,2*IH] written to d_h_buffer.
    // A (d_o_pool): [pool, H], H contiguous → K-major 2D TMA
    const auto tensor_map_dy = make_tma_2d_desc(d_o_buffer,
        hidden, config.num_max_pool_tokens,
        config.block_k, config.block_m,
        static_cast<int>(d_o_buffer.stride(0)),
        config.swizzle_acts_mode);
    // B (W2): [E_local, H, IH], IH contiguous → MN-major 3D TMA
    const auto tensor_map_l2_weights = make_tma_3d_desc(l2_weights_bf16,
        intermediate_hidden, hidden, num_experts_per_rank,
        config.block_n, config.block_k, 1,
        static_cast<int>(l2_weights_bf16.stride(-2)),
        static_cast<int>(l2_weights_bf16.stride(-3)),
        config.swizzle_weights_mode);
    // Output TMA descriptor (kept for ABI but voided in kernel — epilogue writes d_h directly)
    const auto tensor_map_d_a_output = make_tma_2d_desc(d_a_buffer,
        intermediate_hidden, config.num_max_pool_tokens,
        config.block_n, config.block_m,
        static_cast<int>(d_a_buffer.stride(0)),
        0);

    const SM90BF16MegaMoEBackwardL2Runtime::Args l2_args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .fast_math = fast_math,
        .recompute = recompute,
        .config = config,
        .d_h_buffer = d_h_buffer.data_ptr(),
        .d_o_pool = d_o_buffer.data_ptr(),
        .dW2 = dW2.data_ptr<float>(),
        .l2_weights_bf16 = l2_weights_bf16.data_ptr(),
        .recomp_h = recomp_h_buffer.data_ptr(),
        .recomp_a = recomp_a_buffer.data_ptr(),
        .topk_weights_pool = topk_weights_pool.data_ptr<float>(),
        .perm_buf = reinterpret_cast<uint32_t*>(perm_buf.data_ptr<int32_t>()),
        .fwd_lookup_table = fwd_lookup_gpu.data_ptr<int32_t>(),
        .fwd_lookup_stride_r = fwd_lookup_stride_r,
        .fwd_lookup_stride_t = fwd_lookup_stride_t,
        .input_topk_idx = input_topk_idx.data_ptr(),
        .input_topk_weights = input_topk_weights.data_ptr(),
        .dy_source = dy_source,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_d_o = tensor_map_dy,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_d_a_output = tensor_map_d_a_output,
        .launch_args = LaunchArgs(num_sms,
            config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
            config.smem_size, config.cluster_size)
    };
    const auto l2_code = SM90BF16MegaMoEBackwardL2Runtime::generate(l2_args);
    const auto l2_runtime = compiler->build("sm90_bf16_mega_moe_backward_l2", l2_code);
    SM90BF16MegaMoEBackwardL2Runtime::launch(l2_runtime, l2_args);

    // ─── Debug: check d_h_buffer and d_o_pool after L2 ──────────────────
    if (std::getenv("DG_BWD_DEBUG")) {
        DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));
        auto dh_f = d_h_buffer.to(torch::kFloat32);
        auto do_f = d_o_buffer.to(torch::kFloat32);
        auto tw_f = topk_weights_pool.to(torch::kFloat32);
        float dh_norm = dh_f.norm().item<float>();
        float do_norm = do_f.norm().item<float>();
        float tw_norm = tw_f.norm().item<float>();
        int dh_nan = torch::isnan(dh_f).any().item<int>();
        int do_nan = torch::isnan(do_f).any().item<int>();
        fprintf(stderr, "[C++ diag rank=%d] after L2: d_h_buffer norm=%.4f nan=%d, "
                "d_o_pool norm=%.4f nan=%d, topk_w_pool norm=%.4f\n",
                rank_idx, dh_norm, dh_nan, do_norm, do_nan, tw_norm);

        // Check perm_buf validity (now filled by dispatch warp in-kernel)
        auto perm_cpu = perm_buf.to(torch::kCPU);
        auto* pp = perm_cpu.data_ptr<int32_t>();
        int num_pool = config.num_max_pool_tokens;
        int bad_perm = 0;
        for (int i = 0; i < num_pool; ++i) {
            if (pp[i] < 0 || pp[i] >= num_pool) { bad_perm++; }
        }
        fprintf(stderr, "[C++ diag rank=%d] perm_buf: %d out-of-range (of %d)\n",
                rank_idx, bad_perm, num_pool);
    }

    // ─── Zero grid_sync + launch L1 (no host sync needed!) ────────────────
    // L2's dispatch warp built perm_buf in-kernel using fwd_lookup_table.
    // The epilogue used the correct perm to read gate/up from recomp_h,
    // producing correct d_h directly — no post-L2 correction needed.
    // Only zero grid_sync counters (stream-ordered after L2) so L1's nvlink_barrier works.
    {
        auto* ws_ptr = reinterpret_cast<void*>(sym_buffer_ptrs[rank_idx]);
        constexpr auto kGridSyncBytes = layout::Workspace::kNumMaxGridSyncCounters * sizeof(uint32_t);
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(ws_ptr, 0, kGridSyncBytes, c10::cuda::getCurrentCUDAStream()));
    }

    // ─── Kernel 2: Backward L1 (pure GEMM + scatter) ────────────────────
    // L1 GEMM: dx[M,H] = d_h[M,2*IH] @ W1[2*IH,H]. K=2*IH, N=H.
    // A (d_h): [pool, 2*IH], 2*IH contiguous → K-major 2D TMA
    const auto tensor_map_d_h = make_tma_2d_desc(d_h_buffer,
        2 * intermediate_hidden, config.num_max_pool_tokens,
        config.block_k, config.block_m,
        static_cast<int>(d_h_buffer.stride(0)),
        config.swizzle_acts_mode);
    // B (W1): [E_local, 2*IH, H], H contiguous → MN-major 3D TMA
    const auto tensor_map_l1_weights = make_tma_3d_desc(l1_weights_bf16,
        hidden, 2 * intermediate_hidden, num_experts_per_rank,
        config.block_n, config.block_k, 1,
        static_cast<int>(l1_weights_bf16.stride(-2)),
        static_cast<int>(l1_weights_bf16.stride(-3)),
        config.swizzle_weights_mode);
    // Output (dx): [pool, H], H contiguous → 2D TMA store (no swizzle)
    const auto tensor_map_dx_output = make_tma_2d_desc(dx_pool,
        hidden, config.num_max_pool_tokens,
        config.block_n, config.block_m,
        static_cast<int>(dx_pool.stride(0)),
        0);

    const SM90BF16MegaMoEBackwardL1Runtime::Args l1_args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .fast_math = fast_math,
        .recompute = recompute,
        .config = config,
        .dx = dx_pool.data_ptr(),
        .dx_final = dx_final.data_ptr(),
        .dx_combine_buffer = dx_combine_buffer.data_ptr(),
        .dW1 = dW1.data_ptr<float>(),
        .l1_weights_bf16 = l1_weights_bf16.data_ptr(),
        .x_bf16 = x_bf16_buffer.data_ptr(),
        .input_topk_idx = input_topk_idx.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_d_h = tensor_map_d_h,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_dx_output = tensor_map_dx_output,
        .launch_args = LaunchArgs(num_sms,
            config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
            config.smem_size, config.cluster_size)
    };
    const auto l1_code = SM90BF16MegaMoEBackwardL1Runtime::generate(l1_args);
    const auto l1_runtime = compiler->build("sm90_bf16_mega_moe_backward_l1", l1_code);
    SM90BF16MegaMoEBackwardL1Runtime::launch(l1_runtime, l1_args);

    // ─── Debug: check dx outputs after L1 ───────────────────────────────
    if (std::getenv("DG_BWD_DEBUG")) {
        DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));
        auto dxf_f = dx_final.to(torch::kFloat32);
        auto dxp_f = dx_pool.to(torch::kFloat32);
        auto dxc_f = dx_combine_buffer.to(torch::kFloat32);
        float dxf_norm = dxf_f.norm().item<float>();
        float dxp_norm = dxp_f.norm().item<float>();
        float dxc_norm = dxc_f.norm().item<float>();
        int dxf_nan = torch::isnan(dxf_f).any().item<int>();
        fprintf(stderr, "[C++ diag rank=%d] after L1: dx_final norm=%.4f nan=%d, "
                "dx_pool norm=%.4f, dx_combine norm=%.4f\n",
                rank_idx, dxf_norm, dxf_nan, dxp_norm, dxc_norm);
    }


    // ─── Post-kernel reorder for dW cuBLASLt ────────────────────────────
    // recomp_a and x_bf16 are still in forward pool order. They're needed for
    // dW2 and dW1 respectively. Remap to backward order using perm_buf
    // (correctly filled by L2 dispatch warp: perm[bwd_pos] = fwd_pos).
    {
        auto perm_gpu = perm_buf.to(torch::kInt64);
        recomp_a_buffer.copy_(recomp_a_buffer.clone().index_select(0, perm_gpu));
        x_bf16_buffer.copy_(x_bf16_buffer.clone().index_select(0, perm_gpu));
    }
}

} // namespace deep_gemm
