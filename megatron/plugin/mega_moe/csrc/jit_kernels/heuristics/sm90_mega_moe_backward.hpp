#pragma once

#include "mega_moe.hpp"
#include "sm90.hpp"

namespace deep_gemm {

struct MegaMoESM90BackwardConfig {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_experts_per_wave;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90BackwardConfig& config) {
        os << "MegaMoESM90BackwardConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

// Block size heuristic for backward.
// The backward GEMM uses 4 epilogue warpgroups that split only along N
// (each warpgroup computes WG_BLOCK_N = BLOCK_N/4 columns at WGMMA M=64 rows).
// With BLOCK_M=64, the 4 warpgroups together cover exactly 64 rows × 256 cols
// = BLOCK_M × BLOCK_N. Using BLOCK_M=128 would leave rows 64-127 uncomputed.
static std::tuple<int, int> get_block_config_for_mega_moe_sm90_backward(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    (void)num_ranks; (void)num_experts; (void)num_max_tokens_per_rank;
    (void)num_topk; (void)num_tokens;
    // BLOCK_M must equal WGMMA::M (64) so that 4 N-split warpgroups cover the full tile.
    // Keep 512 epilogue threads (4 warpgroups) for maximum N-parallelism.
    return {64, 512};
}

// Experts per wave (same as forward)
static int get_num_experts_per_wave_for_mega_moe_sm90_backward(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    // Reuse forward heuristic
    return get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

// Pipeline config for backward: BF16 tiles are 2× FP8, so stages drop from 4→2
static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90_backward(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;

    // Dispatch region: BF16 tokens (hidden * 2 bytes each, vs FP8 hidden bytes in forward)
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden * 2), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    // Pull mbarriers: one per dispatch warp (16 bytes each, aligned)
    const int smem_pull_mbarriers_size = align(
        num_dispatch_warps * 16, 128);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size + smem_pull_mbarriers_size;

    // Epilogue C/D region: BF16 output tiles
    // Backward L2: d_a output [block_m, intermediate_hidden] BF16
    // Backward L1: d_x output [block_m, hidden] BF16
    // Use max of both (hidden is typically larger)
    const int smem_cd = align(block_m * block_n * static_cast<int>(sizeof(nv_bfloat16)), kSmemAlignment);

    // Per-stage GEMM tiles: BF16 A/B matrices
    // With block_k=64 for BF16 (constrained by 128B max swizzle mode):
    // WGMMA K=16, so 64/16=4 WGMMA iterations per K-block
    const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_b_per_stage = block_n * block_k * static_cast<int>(sizeof(nv_bfloat16));
    // No scale factors in SMEM for BF16 (weights are BF16, no quantization in backward)
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    // Calculate num_stages: BF16 tiles are 2× FP8, so expect ~2 stages (vs 4 for FP8)
    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 1);  // Must fit at least 1 stage
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

// Main config selector
static MegaMoESM90BackwardConfig get_mega_moe_backward_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    // Block sizes
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90_backward(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);

    // BLOCK_N must be ≥ 4 * (swizzle_mode / sizeof(BF16)) = 4*64 = 256
    // to ensure each of the 4 epilogue warpgroups' WG_BLOCK_N aligns to
    // the swizzle atom (BLOCK_MN_ATOM = swizzle_mode / sizeof(BF16) = 64).
    // Using block_n=128 with swizzle=128 would give WG_BLOCK_N=32 < 64 → assertion failure.
    const int block_n = 256;
    const int block_k = 64;  // BF16: swizzle = block_k * 2 = 128B (max supported)
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90_backward(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    // With 512 epilogue threads (4 warpgroups), non-epilogue/dispatch each get 64 threads (2 warps)
    const int num_dispatch_threads = 64;
    const int num_non_epilogue_threads = 64;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    // Pipeline stages: BF16 → fewer stages than FP8
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90_backward(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoESM90BackwardConfig {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90BackwardConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
