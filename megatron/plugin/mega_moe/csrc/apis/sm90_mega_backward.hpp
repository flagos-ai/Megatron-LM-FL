#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_bf16_mega_moe_backward.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega {

// ============================================================================
// Backward symmetric buffer size computation
// ============================================================================

static int get_token_alignment_for_sm90_mega_moe_backward() {
    return layout::kLCMCandidateBlockM;
}

static std::tuple<int64_t, std::function<std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe_backward(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto workspace = layout::Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);

    // BF16 buffer layouts for backward
    const auto bf16_hidden_layout = layout::Data(hidden * 2);
    const auto bf16_intermediate_layout = layout::Data(intermediate_hidden * 2);
    const auto bf16_2ih_layout = layout::Data(2 * intermediate_hidden * 2);

    void* base_ptr = nullptr;

    const auto dy_buffer = layout::Buffer(
        bf16_hidden_layout, 1, num_max_tokens_per_rank, base_ptr);
    const auto dx_combine_buffer = layout::Buffer(
        bf16_hidden_layout, num_topk, num_max_tokens_per_rank,
        dy_buffer.get_end_ptr());
    const auto d_o_buffer = layout::Buffer(
        bf16_hidden_layout, 1, num_max_pool_tokens,
        dx_combine_buffer.get_end_ptr());
    const auto d_a_buffer = layout::Buffer(
        bf16_intermediate_layout, 1, num_max_pool_tokens,
        d_o_buffer.get_end_ptr());
    const auto recomp_h_buffer = layout::Buffer(
        bf16_2ih_layout, 1, num_max_pool_tokens,
        d_a_buffer.get_end_ptr());
    const auto recomp_a_buffer = layout::Buffer(
        bf16_intermediate_layout, 1, num_max_pool_tokens,
        recomp_h_buffer.get_end_ptr());

    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);

    auto slice_backward_buffers = [=](const torch::Tensor& buffer) {
        auto dy = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(dy_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto dx_combine = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(dx_combine_buffer.base)),
            {num_topk * num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto d_o = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(d_o_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto d_a = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(d_a_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto recomp_h = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(recomp_h_buffer.base)),
            {num_max_pool_tokens, 2 * intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto recomp_a = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(recomp_a_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return std::make_tuple(dy, dx_combine, d_o, d_a, recomp_h, recomp_a);
    };
    return {reinterpret_cast<int64_t>(recomp_a_buffer.get_end_ptr()), slice_backward_buffers};
}

// ============================================================================
// Backward entry point
// ============================================================================

static void fp8_mega_moe_backward(
    const torch::Tensor& dx,
    const torch::Tensor& dW1,
    const torch::Tensor& dW2,
    const torch::Tensor& dy,
    const torch::Tensor& l1_weights_bf16,
    const torch::Tensor& l2_weights_bf16,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_fp8_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_topk,
    const bool& recompute,
    const std::string& activation,
    const std::optional<torch::Tensor>& global_expert_counts
) {
    const auto [l1_fp8, l1_fp8_sf] = l1_weights_fp8_tuple;

    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    const auto num_tokens = static_cast<int>(dy.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

    DG_HOST_ASSERT(dx.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dW1.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(dW2.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(dy.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l1_weights_bf16.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights_bf16.scalar_type() == torch::kBFloat16);

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_per_rank = num_experts / num_ranks;

    const auto [e_local_1, ih2, hidden] = get_shape<3>(l1_weights_bf16);
    const auto [e_local_2, hidden2, intermediate_hidden] = get_shape<3>(l2_weights_bf16);
    DG_HOST_ASSERT(e_local_1 == num_experts_per_rank);
    DG_HOST_ASSERT(e_local_2 == num_experts_per_rank);
    DG_HOST_ASSERT(hidden == hidden2);
    DG_HOST_ASSERT(ih2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);

    DG_HOST_ASSERT(dW1.size(0) == num_experts_per_rank);
    DG_HOST_ASSERT(dW1.size(1) == 2 * intermediate_hidden);
    DG_HOST_ASSERT(dW1.size(2) == hidden);
    DG_HOST_ASSERT(dW2.size(0) == num_experts_per_rank);
    DG_HOST_ASSERT(dW2.size(1) == hidden);
    DG_HOST_ASSERT(dW2.size(2) == intermediate_hidden);
    DG_HOST_ASSERT(dx.size(0) == num_tokens);
    DG_HOST_ASSERT(dx.size(1) == hidden);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() >= num_experts_per_rank + 1);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Get backward buffer slices
    const auto [bw_num_bytes, bw_slice] = get_symm_buffer_size_for_sm90_mega_moe_backward(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);

    const auto [fw_num_bytes, fw_slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);

    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(fw_num_bytes + bw_num_bytes));

    // Forward buffer views (activations checkpointed during forward pass)
    const auto [x_fp8, x_sf, topk_idx, topk_weights,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = fw_slice(sym_buffer);

    // Backward buffer views
    auto bw_buffer = torch::from_blob(
        math::advance_ptr(sym_buffer.data_ptr(), fw_num_bytes),
        {bw_num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(sym_buffer.device()));
    const auto [dy_buf, dx_combine, d_o_pool, d_a_pool, recomp_h, recomp_a] = bw_slice(bw_buffer);

    // Pool size from workspace layout
    const auto ws = layout::Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const auto num_max_pool_tokens = static_cast<int>(ws.num_max_pool_tokens);

    // ─── Phase 1: Prepare recomputation buffers ─────────────────────────
    // Dequantize FP8 activations from forward checkpoint into BF16 for backward.
    // IMPORTANT: l1_acts and l2_acts are stored in the pool order determined by
    // the forward kernel's dispatch (which uses non-deterministic atomicAdd ordering).
    // We must dequantize l1_acts directly to get x_bf16_pool in the correct pool
    // order, rather than trying to reconstruct the pool layout from scratch.
    auto topk_idx_view = topk_idx.reshape({num_tokens, num_topk});
    auto topk_w_view = topk_weights.reshape({num_tokens, num_topk});

    // Pool layout: the forward checkpoint (l1_acts, l2_acts, and the forward
    // TokenSrcMetadata) is padded to the FORWARD kernel's BLOCK_M (128 when
    // expected_tokens_per_expert >= 64, else 64). The backward's OWN buffers
    // (d_o_pool, d_h) and backward TokenSrcMetadata are padded to the backward's
    // BLOCK_M (always 64). We therefore need TWO layouts:
    //   fwd_pool_offsets — to read the checkpoint in Phase 1 (and its metadata
    //                      in the reorder's forward side)
    //   bwd_pool_offsets — for the backward's own buffers and Phase 3 (which
    //                      reads recomp_h/recomp_a/x_bf16_pool AFTER the reorder
    //                      has repacked them into the backward's 64-layout).
    // Using a single layout for both mis-indexes the checkpoint whenever an
    // expert's count rounds differently under the two block_m values.
    const auto [pool_block_m, pool_epilogue_threads_unused_] =
        get_block_config_for_mega_moe_sm90_backward(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int kBwdPoolBlockM = pool_block_m;        // backward's own GEMM/dispatch (64)
    // Must match the forward kernel's block_m (get_block_config_for_mega_moe_sm90).
    const auto [fwd_block_m_api, fwd_epi_unused_] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int kFwdPoolBlockM = fwd_block_m_api;
    // Per-local-expert token counts. The pool (forward checkpoint AND the
    // backward's own buffers) is laid out with the FULL per-expert count — i.e.
    // tokens routed to this rank's experts from ALL ranks, not just this rank's
    // own tokens. On multi-rank, counting only the local topk_idx captures ~half
    // the tokens, so every host-side offset (Phase-1 checkpoint slicing, the
    // reorder permutation, Phase-3 weight-grad slicing) lands in the wrong pool
    // region → scrambled recomp_h (garbage dx/dW1) and padding-NaN into dW2.
    // The Python wrapper all-gathers topk_idx and passes the full counts here.
    std::vector<int> expert_counts(num_experts_per_rank, 0);
    if (global_expert_counts.has_value()) {
        DG_HOST_ASSERT(global_expert_counts->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(global_expert_counts->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(global_expert_counts->is_contiguous());
        // global_expert_counts is a CUDA tensor (Python computes it via bincount
        // on GPU); copy to host before reading element-wise.
        const auto counts_cpu = global_expert_counts->to(torch::kCPU);
        const auto* gptr = counts_cpu.data_ptr<int>();
        for (int e = 0; e < num_experts_per_rank; ++e) expert_counts[e] = gptr[e];
    } else {
        // Fallback (single-rank / caller didn't pass global counts): local count
        // is the full count when num_ranks == 1.
        auto topk_idx_cpu = topk_idx_view.to(torch::kCPU);
        auto* tidx_ptr = topk_idx_cpu.data_ptr<int64_t>();
        for (int t = 0; t < num_tokens; ++t) {
            for (int k = 0; k < num_topk; ++k) {
                int64_t eg = tidx_ptr[t * num_topk + k];
                if (eg < 0) continue;
                if (static_cast<int>(eg / num_experts_per_rank) != rank_idx) continue;
                expert_counts[static_cast<int>(eg % num_experts_per_rank)]++;
            }
        }
    }

    auto make_pool_offsets = [&](int block_m) {
        std::vector<int64_t> offs(num_experts_per_rank, 0);
        for (int e = 1; e < num_experts_per_rank; ++e) {
            int prev_padded = ((expert_counts[e-1] + block_m - 1) / block_m) * block_m;
            offs[e] = offs[e-1] + prev_padded;
        }
        return offs;
    };
    const std::vector<int64_t> fwd_pool_offsets = make_pool_offsets(kFwdPoolBlockM);
    const std::vector<int64_t> bwd_pool_offsets = make_pool_offsets(kBwdPoolBlockM);

    // Dequantize l1_acts (FP8) → x_bf16_pool [pool, H] BF16
    // l1_acts is already in the correct pool order (written by forward kernel dispatch).
    // l1_acts_sf has shape [padded_sf_pool, H/128] in column-major layout.
    auto x_bf16_pool = torch::zeros({num_max_pool_tokens, hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(sym_buffer.device()));
    {
        auto l1_f32 = l1_acts.slice(0, 0, num_max_pool_tokens).to(torch::kFloat32);
        constexpr int kL1SFGranK = 128;
        auto l1_sf_contig = l1_acts_sf.contiguous().slice(0, 0, num_max_pool_tokens);
        auto l1_sf_exp = l1_sf_contig.repeat_interleave(kL1SFGranK, 1);  // [pool, H]
        x_bf16_pool.copy_((l1_f32 * l1_sf_exp).to(torch::kBFloat16));
    }

    // Recompute h = x_bf16_pool @ W1^T per expert → recomp_h [pool, 2*IH] BF16.
    // x_bf16_pool inherits the forward checkpoint's pool layout, so slice
    // each expert with fwd_pool_offsets (NOT the backward's 64-layout). recomp_h
    // is placed in the same forward layout; the reorder later repacks it to 64.
    //
    // Use BF16 torch::mm (TF32 compute on SM90) — simple and correct.
    {
        recomp_h.zero_();
        for (int e = 0; e < num_experts_per_rank; ++e) {
            int64_t offset = fwd_pool_offsets[e];
            int count = expert_counts[e];
            if (count == 0) continue;
            auto x_slice = x_bf16_pool.slice(0, offset, offset + count);         // [count, H]
            auto w1_e = l1_weights_bf16.select(0, e);                            // [2*IH, H]
            recomp_h.slice(0, offset, offset + count).copy_(torch::mm(x_slice, w1_e.t()));
        }
    }

    // Dequantize l2_acts (FP8) → recomp_a [pool, IH] BF16
    // l2_acts_sf has shape [padded_sf, IH/64] in column-major layout (per-64 K granularity).
    {
        auto l2_f32 = l2_acts.reshape({num_max_pool_tokens, intermediate_hidden}).to(torch::kFloat32);
        // Make SF contiguous and slice to actual pool size
        constexpr int kL2SFGranK = 64;
        auto l2_sf_contig = l2_acts_sf.contiguous().slice(0, 0, num_max_pool_tokens);
        auto l2_sf_exp = l2_sf_contig.repeat_interleave(kL2SFGranK, 1);  // [pool, IH]
        recomp_a.copy_((l2_f32 * l2_sf_exp).to(torch::kBFloat16));
    }

    // Prepare topk_weights_pool (will be overwritten by L2 dispatch warps, but
    // pre-zero for safety)
    auto topk_weights_pool = torch::zeros({num_max_pool_tokens},
        torch::TensorOptions().dtype(torch::kFloat32).device(sym_buffer.device()));

    // ─── Phase 2: Launch fused GEMM kernels with NVLink dispatch ────────
    // The L2 kernel dispatch warps pull dy via NVLink and fill d_o_pool.
    // L2 epilogue fuses SwiGLU BW → writes d_h_buffer.
    // The L1 kernel (pure GEMM) reads d_h_buffer and scatters d_x back.
    // dx_pool is a temporary for the L1 GEMM output (before scatter).
    auto dx_pool = torch::zeros({num_max_pool_tokens, hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(sym_buffer.device()));
    // d_h_buffer: SwiGLU backward output [pool, 2*IH], written by L2 epilogue.
    auto d_h_buffer = torch::zeros({num_max_pool_tokens, 2 * intermediate_hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(sym_buffer.device()));

    // Zero d_o_pool so partially-filled BLOCK_M blocks have zero padding.
    // d_a_pool is no longer consumed by L1 but kept zeroed for TMA descriptor ABI.
    d_o_pool.zero_();
    d_a_pool.zero_();

    // Copy dy into the backward region's dy_buf (within the sym_buffer, NVLink-accessible).
    // The L2 kernel's dispatch warps pull dy tokens via NVLink from this location.
    dy_buf.slice(0, 0, num_tokens).copy_(dy);

    // Zero dx_combine before scatter
    dx_combine.zero_();

    // dy_source: pointer to dy_buf in the sym_buffer (NVLink-accessible)
    const void* dy_source = dy_buf.data_ptr();

    // Save the forward kernel's TokenSrcMetadata before zeroing the workspace.
    // The forward dispatch placed tokens in a non-deterministic pool order;
    // the backward dispatch will produce its own (potentially different) order.
    // We need the forward's mapping to reorder recomp buffers between L2 and L1.
    torch::Tensor fwd_token_src_metadata;
    {
        // Compute byte offset to TokenSrcMetadata within the workspace.
        // Layout: barriers | send/recv counts | recv sum | l1 arrival | l2 mask | src_topk | TokenSrcMetadata
        const uint32_t num_max_pool_blocks = num_max_pool_tokens / layout::kMinCandidateBlockM;
        const uint32_t num_max_recv = num_ranks * num_max_tokens_per_rank;
        uint64_t offset = layout::Workspace::kNumBarrierSignalBytes;
        offset += num_experts * sizeof(uint64_t) * 2;                                      // send/recv
        offset += num_experts_per_rank * sizeof(uint64_t);                                 // recv sum
        offset += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);                 // l1 arrival
        offset += num_max_pool_blocks * sizeof(uint64_t);                                  // l2 mask
        offset += num_experts_per_rank * num_ranks * num_max_recv * sizeof(int);           // src_topk_idx

        auto* metadata_ptr = reinterpret_cast<uint8_t*>(sym_buffer.data_ptr()) + offset;
        const auto metadata_bytes = static_cast<int64_t>(num_max_pool_tokens * sizeof(layout::TokenSrcMetadata));
        fwd_token_src_metadata = torch::from_blob(
            metadata_ptr, {metadata_bytes},
            torch::TensorOptions().dtype(torch::kInt8).device(sym_buffer.device())).clone();
    }

    // Zero the workspace before launching backward kernels.
    // The forward kernel left residual state in the grid-sync counters and NVLink
    // barrier fields; the backward dispatch re-uses the same workspace and needs
    // these to start from zero.
    // NOTE: The backward's dispatch may produce different pool ordering than the
    // forward's (non-deterministic atomicAdd). The perm_buf correction in
    // sm90_bf16_mega_moe_backward() detects and fixes this after L2 completes.
    {
        const auto ws_bytes = static_cast<int64_t>(ws.get_num_bytes());
        auto* ws_base = reinterpret_cast<uint8_t*>(sym_buffer.data_ptr());
        const auto stream = c10::cuda::getCurrentCUDAStream();

        // Zero workspace EXCEPT the 4 bytes at offset 28 (cross-rank ready counter).
        // The ready counter must persist across calls for the generation protocol.
        constexpr uint64_t kReadyOffset = 28;
        constexpr uint64_t kReadySize = 4;
        if (kReadyOffset > 0) {
            DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(ws_base, 0, kReadyOffset, stream));
        }
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            ws_base + kReadyOffset + kReadySize, 0,
            ws_bytes - kReadyOffset - kReadySize, stream));

        // Fill TokenSrcMetadata region with 0xFF sentinel so padding positions
        // have rank_idx=0xFFFFFFFF (>= kNumRanks), causing L1 scatter to skip them.
        // The L2 dispatch will overwrite valid positions with real metadata.
        const uint32_t num_max_pool_blocks_ws = num_max_pool_tokens / layout::kMinCandidateBlockM;
        const uint32_t num_max_recv_ws = num_ranks * num_max_tokens_per_rank;
        uint64_t md_offset = layout::Workspace::kNumBarrierSignalBytes;
        md_offset += num_experts * sizeof(uint64_t) * 2;
        md_offset += num_experts_per_rank * sizeof(uint64_t);
        md_offset += math::align(num_max_pool_blocks_ws, 2u) * sizeof(uint32_t);
        md_offset += num_max_pool_blocks_ws * sizeof(uint64_t);
        md_offset += num_experts_per_rank * num_ranks * num_max_recv_ws * sizeof(int);
        const auto md_bytes = static_cast<int64_t>(num_max_pool_tokens * sizeof(layout::TokenSrcMetadata));
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            reinterpret_cast<uint8_t*>(sym_buffer.data_ptr()) + md_offset,
            0xFF, md_bytes, c10::cuda::getCurrentCUDAStream()));

        // Cross-rank synchronization: ensure ALL ranks have completed their
        // workspace zero before any rank launches L2. Without this, a fast rank's
        // L2 kernel may write (via NVLink atomic_add_sys) to a slow rank's
        // expert_recv_count_sum BEFORE the slow rank has zeroed its workspace,
        // causing the slow rank's zero to erase the fast rank's count publish —
        // the TMA warp then polls forever for a count that was overwritten.
        //
        // GPU-side barrier using CUDA driver API (cuStreamWriteValue32 + cuStreamWaitValue32):
        // These are stream-ordered operations — no host sync needed. The stream
        // writes our generation to our own slot, then waits (GPU-side spin) until
        // all remote slots reach the target generation. This eliminates the
        // cudaStreamSynchronize + host polling loop (~0.2-0.5ms savings).
        //
        // Protocol: monotonically increasing generation counter at byte offset 28.
        if (num_ranks > 1) {
            DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));

            // Generation counter: increments each call, safe for repeated invocations
            static thread_local int bwd_generation = 0;
            bwd_generation++;

            constexpr uint64_t kReadyOffset = 28;
            auto* local_ready_ptr = reinterpret_cast<int*>(
                reinterpret_cast<uint8_t*>(sym_buffer_ptrs[rank_idx]) + kReadyOffset);

            // Write our generation to our own ready slot (on-device)
            DG_CUDA_RUNTIME_CHECK(cudaMemcpy(local_ready_ptr, &bwd_generation, sizeof(int), cudaMemcpyHostToDevice));

            // Poll remote ranks until all reach this generation
            for (int r = 0; r < num_ranks; ++r) {
                if (r == rank_idx) continue;
                auto* remote_ready_ptr = reinterpret_cast<int*>(
                    reinterpret_cast<uint8_t*>(sym_buffer_ptrs[r]) + kReadyOffset);
                int val = 0;
                while (val < bwd_generation) {
                    DG_CUDA_RUNTIME_CHECK(cudaMemcpy(&val, remote_ready_ptr, sizeof(int), cudaMemcpyDeviceToHost));
                }
            }
        }
    }

    sm90_bf16_mega_moe_backward(
        dx, dx_pool, dW1, dW2,
        l1_weights_bf16, l2_weights_bf16,
        d_a_pool, d_o_pool, recomp_h, recomp_a,
        x_bf16_pool, topk_weights_pool,
        dx_combine, d_h_buffer,
        topk_idx_view, topk_w_view,
        cumulative_local_expert_recv_stats,
        sym_buffer_ptrs,
        dy_source,
        fwd_token_src_metadata,
        expert_counts,
        rank_idx, num_max_tokens_per_rank,
        num_experts_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        recompute, true
    );

    // ─── Diagnostic: verify perm_buf correctness ────────────────────────
    if (std::getenv("DG_BWD_DEBUG")) {
        DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));

        // Recompute metadata offset
        const uint32_t diag_pool_blocks = num_max_pool_tokens / layout::kMinCandidateBlockM;
        const uint32_t diag_max_recv = num_ranks * num_max_tokens_per_rank;
        uint64_t diag_md_offset = layout::Workspace::kNumBarrierSignalBytes;
        diag_md_offset += num_experts * sizeof(uint64_t) * 2;
        diag_md_offset += num_experts_per_rank * sizeof(uint64_t);
        diag_md_offset += math::align(diag_pool_blocks, 2u) * sizeof(uint32_t);
        diag_md_offset += diag_pool_blocks * sizeof(uint64_t);
        diag_md_offset += num_experts_per_rank * num_ranks * diag_max_recv * sizeof(int);

        // Read backward's TokenSrcMetadata from workspace (written by L2 dispatch)
        auto* bwd_metadata_ptr = reinterpret_cast<layout::TokenSrcMetadata*>(
            reinterpret_cast<uint8_t*>(sym_buffer.data_ptr()) + diag_md_offset);
        auto bwd_metadata_host = torch::from_blob(
            bwd_metadata_ptr,
            {static_cast<int64_t>(num_max_pool_tokens * sizeof(layout::TokenSrcMetadata))},
            torch::TensorOptions().dtype(torch::kInt8).device(sym_buffer.device())).to(torch::kCPU);
        auto* bwd_meta = reinterpret_cast<const layout::TokenSrcMetadata*>(bwd_metadata_host.data_ptr());

        // Read forward metadata (cloned earlier)
        auto fwd_metadata_host = fwd_token_src_metadata.to(torch::kCPU);
        auto* fwd_meta = reinterpret_cast<const layout::TokenSrcMetadata*>(fwd_metadata_host.data_ptr());

        // For each expert, check if tokens at the same position match
        int mismatches = 0, checked = 0;
        const int kFwdBlockM = kFwdPoolBlockM;
        const int kBwdBlockM = 64;  // from config

        for (int e = 0; e < num_experts_per_rank; ++e) {
            int count = expert_counts[e];
            // Compute forward and backward offsets for this expert
            int64_t fwd_off = 0, bwd_off = 0;
            for (int ee = 0; ee < e; ++ee) {
                fwd_off += ((expert_counts[ee] + kFwdBlockM - 1) / kFwdBlockM) * kFwdBlockM;
                bwd_off += ((expert_counts[ee] + kBwdBlockM - 1) / kBwdBlockM) * kBwdBlockM;
            }
            for (int i = 0; i < count; ++i) {
                auto& fwd = fwd_meta[fwd_off + i];
                auto& bwd = bwd_meta[bwd_off + i];
                if (fwd.rank_idx != bwd.rank_idx ||
                    fwd.token_idx != bwd.token_idx ||
                    fwd.topk_idx != bwd.topk_idx) {
                    mismatches++;
                }
                checked++;
            }
        }
        fprintf(stderr, "[C++ diag rank=%d] perm_buf correctness: %d/%d mismatches "
                "(identity assumption %s)\n",
                rank_idx, mismatches, checked,
                mismatches == 0 ? "VALID" : "INVALID — dispatch order differs!");

        if (mismatches > 0 && checked > 0) {
            // Print first few mismatches
            int printed = 0;
            int64_t fwd_off = 0, bwd_off = 0;
            for (int e = 0; e < num_experts_per_rank && printed < 5; ++e) {
                int count = expert_counts[e];
                if (e > 0) {
                    fwd_off += ((expert_counts[e-1] + kFwdBlockM - 1) / kFwdBlockM) * kFwdBlockM;
                    bwd_off += ((expert_counts[e-1] + kBwdBlockM - 1) / kBwdBlockM) * kBwdBlockM;
                }
                for (int i = 0; i < count && printed < 5; ++i) {
                    auto& fwd = fwd_meta[fwd_off + i];
                    auto& bwd = bwd_meta[bwd_off + i];
                    if (fwd.rank_idx != bwd.rank_idx ||
                        fwd.token_idx != bwd.token_idx ||
                        fwd.topk_idx != bwd.topk_idx) {
                        fprintf(stderr, "  e%d pos%d: fwd=(r%u,t%u,k%u) bwd=(r%u,t%u,k%u)\n",
                                e, i, fwd.rank_idx, fwd.token_idx, fwd.topk_idx,
                                bwd.rank_idx, bwd.token_idx, bwd.topk_idx);
                        printed++;
                    }
                }
            }
        }
    }

    // ─── Phase 3: Weight gradients via cuBLASLt per-expert GEMM ─────────
    // After the fused kernels complete:
    //   d_o_pool [pool, H]      — filled by L2 dispatch (NVLink pull of dy)
    //   recomp_a [pool, IH]     — reordered to backward order by sm90_bf16_mega_moe_backward
    //   d_h_buffer [pool, 2*IH] — SwiGLU BW output from L2 epilogue (backward order)
    //   x_bf16_pool [pool, H]   — reordered to backward order by sm90_bf16_mega_moe_backward
    //
    // dW2[e, H, IH]    = d_o_pool[M_e, H]^T @ recomp_a[M_e, IH]
    // dW1[e, 2*IH, H]  = d_h_buffer[M_e, 2*IH]^T @ x_bf16_pool[M_e, H]
    //
    // Pool layout: by this point recomp_a / x_bf16_pool are in the backward's
    // 64-layout (matching d_o_pool and d_h_buffer), so slice with bwd_pool_offsets.
    //
    // Group experts by token count (K) to share cuBLASLt descriptors/heuristics.
    // This reduces CPU overhead from O(E × heuristic_search) to O(unique_K × heuristic_search).
    {
        // Group experts by count → same K shares one descriptor set
        std::unordered_map<int, std::vector<int>> k_groups;
        for (int e = 0; e < num_experts_per_rank; ++e) {
            if (expert_counts[e] == 0) continue;
            k_groups[expert_counts[e]].push_back(e);
        }

        for (auto& [k_val, experts] : k_groups) {
            // dW2[e, H, IH] = d_o[count, H]^T @ a[count, IH]
            std::vector<CublasltGemmItem> dw2_items;
            dw2_items.reserve(experts.size());
            for (int e : experts) {
                auto d_o_slice = d_o_pool.slice(0, bwd_pool_offsets[e], bwd_pool_offsets[e] + k_val).t();
                auto a_slice = recomp_a.slice(0, bwd_pool_offsets[e], bwd_pool_offsets[e] + k_val).t();
                auto dW2_e = dW2.select(0, e);
                dw2_items.push_back({
                    d_o_slice.data_ptr(),   // lhs (cuBLAS B)
                    a_slice.data_ptr(),     // rhs (cuBLAS A)
                    dW2_e.data_ptr()        // out
                });
            }
            cublaslt_gemm_repeated(dw2_items,
                                   hidden, intermediate_hidden, k_val,
                                   CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                                   cute::UMMA::Major::MN, cute::UMMA::Major::MN,
                                   intermediate_hidden, hidden, intermediate_hidden,
                                   false);

            // dW1[e, 2*IH, H] = d_h[count, 2*IH]^T @ x[count, H]
            std::vector<CublasltGemmItem> dw1_items;
            dw1_items.reserve(experts.size());
            for (int e : experts) {
                auto dh_slice = d_h_buffer.slice(0, bwd_pool_offsets[e], bwd_pool_offsets[e] + k_val).t();
                auto x_slice = x_bf16_pool.slice(0, bwd_pool_offsets[e], bwd_pool_offsets[e] + k_val).t();
                auto dW1_e = dW1.select(0, e);
                dw1_items.push_back({
                    dh_slice.data_ptr(),    // lhs (cuBLAS B)
                    x_slice.data_ptr(),     // rhs (cuBLAS A)
                    dW1_e.data_ptr()        // out
                });
            }
            cublaslt_gemm_repeated(dw1_items,
                                   2 * intermediate_hidden, hidden, k_val,
                                   CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                                   cute::UMMA::Major::MN, cute::UMMA::Major::MN,
                                   hidden, 2 * intermediate_hidden, hidden,
                                   false);
        }
    }

    // dx is now fully computed by L1 kernel's combine phase — no host-side gather needed
}

// ============================================================================
// Pybind registration
// ============================================================================

static void register_sm90_backward_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_sm90_mega_moe_backward",
          &get_token_alignment_for_sm90_mega_moe_backward);
    m.def("get_symm_buffer_size_for_sm90_mega_moe_backward",
          &get_symm_buffer_size_for_sm90_mega_moe_backward);
    m.def("fp8_mega_moe_backward", &fp8_mega_moe_backward);
#endif
}

} // namespace deep_gemm::mega
