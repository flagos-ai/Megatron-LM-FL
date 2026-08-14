"""KunLunXin Transformer Engine overrides."""

import importlib.util

from megatron.plugin.kunlunxin.debug import debug_patch


@debug_patch("extensions.transformer_engine._sharded_state_dict_grouped")
def _sharded_state_dict_grouped(self, tp_axis_map, prefix="", sharded_offsets=(), metadata=None):
    """Build TEGroupedLinear sharded state dict with XME large-weight handling."""
    if importlib.util.find_spec("megatron.training") is None:
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        return TEGroupedLinear._sharded_state_dict_grouped.__wrapped__(
            self, tp_axis_map, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
    from megatron.core.parallel_state import (
        get_data_parallel_rank,
        get_expert_data_parallel_rank,
        get_expert_model_parallel_rank,
        get_expert_model_parallel_world_size,
        get_expert_tensor_parallel_rank,
        get_expert_tensor_parallel_world_size,
        get_tensor_model_parallel_rank,
    )
    from megatron.core.transformer.utils import (
        _get_extra_state_offsets,
        make_sharded_tensors_for_checkpoint,
    )

    singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
    sharded_state_dict = {}
    full_state_dict = self.state_dict(prefix="", keep_vars=True)
    num_global_experts = get_expert_model_parallel_world_size() * self.num_gemms
    local_expert_indices_offset = get_expert_model_parallel_rank() * self.num_gemms
    ep_axis = len(sharded_offsets)

    if "large_weight" in full_state_dict:
        state_dict = {"large_weight": full_state_dict["large_weight"]}
        if "_extra_state" in full_state_dict:
            state_dict["_extra_state"] = full_state_dict["_extra_state"]
        if self.use_bias and "large_bias" in full_state_dict:
            state_dict["large_bias"] = full_state_dict["large_bias"]

        ep_rank = get_expert_model_parallel_rank()
        ep_world_size = get_expert_model_parallel_world_size()
        ep_offsets = (ep_axis, ep_rank, ep_world_size)

        new_tp_axis_map = {}
        if tp_axis_map["0.weight"] == 0:
            new_tp_axis_map["large_weight"] = 1
            new_tp_axis_map["large_bias"] = 1
        else:
            new_tp_axis_map["large_weight"] = 2

        for layer_name, tensor in state_dict.items():
            new_sharded_offsets = [*sharded_offsets, ep_offsets]
            if layer_name in new_tp_axis_map:
                tp_axis = new_tp_axis_map[layer_name]
                tp_rank = get_expert_tensor_parallel_rank()
                tp_size = get_expert_tensor_parallel_world_size()
                new_sharded_offsets.append((tp_axis, tp_rank, tp_size))

            dp_replica_id = get_data_parallel_rank(with_context_parallel=True)
            replica_id = (0, 0, dp_replica_id)
            layer_key = f"{prefix}{layer_name}"
            if layer_name.endswith('_extra_state'):
                replica_id = (
                    0,
                    get_tensor_model_parallel_rank(),
                    get_data_parallel_rank(with_context_parallel=True),
                )
                sharded_state_dict[layer_key] = ShardedObject(
                    layer_key,
                    tensor,
                    *_get_extra_state_offsets(new_sharded_offsets),
                    replica_id,
                )
            else:
                sharded_state_dict[layer_key] = ShardedTensor.from_rank_offsets(
                    layer_key,
                    tensor,
                    *new_sharded_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=0,
                )
    else:
        extra_states = None
        if "_extra_state" in full_state_dict:
            extra_states = self._split_extra_state(full_state_dict["_extra_state"])
        for gemm_idx in range(self.num_gemms):
            global_expert_idx = local_expert_indices_offset + gemm_idx
            state_dict = {f"{gemm_idx}.weight": full_state_dict[f"weight{gemm_idx}"]}
            if extra_states is not None:
                state_dict[f"{gemm_idx}._extra_state"] = extra_states[gemm_idx]
            if self.use_bias:
                state_dict[f"{gemm_idx}.bias"] = full_state_dict[f"bias{gemm_idx}"]
            if singleton_local_shards:
                expert_prefix = f"{global_expert_idx}.{prefix}"
                new_sharded_offsets = sharded_offsets
            else:
                expert_prefix = prefix
                new_sharded_offsets = (
                    *sharded_offsets,
                    (ep_axis, global_expert_idx, num_global_experts),
                )
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict, '', tp_axis_map, new_sharded_offsets
            )
            replace_prefix_for_sharding(sub_sd, f"{gemm_idx}.", expert_prefix)
            sharded_state_dict[f"{prefix}weight{gemm_idx}"] = sub_sd[f"{gemm_idx}.weight"]
            if extra_states is not None:
                sharded_state_dict[
                    f"{prefix}_extra_state{'' if gemm_idx == 0 else gemm_idx}"
                ] = sub_sd[f"{gemm_idx}._extra_state"]
            if self.use_bias:
                sharded_state_dict[f"{prefix}bias{gemm_idx}"] = sub_sd[f"{gemm_idx}.bias"]

    for sh_ten in sharded_state_dict.values():
        replica_id = sh_ten.replica_id
        assert len(replica_id) == 3, f"Expected replica_id in (PP, TP, DP) format, got: {replica_id}"
        if getattr(sh_ten, "is_data_parallel_fully_shard", False):
            edp_replica_id = 0
        else:
            edp_replica_id = get_expert_data_parallel_rank()
        sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
    return sharded_state_dict
