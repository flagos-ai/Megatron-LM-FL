"""Count indexed expert weights as tensor-parallel shards on NPU."""


def set_parallel_attributes(self):
    """Mark indexed expert weights as TP shards so they are not treated as duplicates.

    When explicit_expert_comm clears TE's parallel_mode, only the grouped weight
    receives tensor_model_parallel, so every indexed weight looks like a duplicate
    to param_is_not_tensor_parallel_duplicate(). Gradient norm, clipping and the
    zero count then keep the expert weights of tensor-parallel rank 0 only. The
    other ranks do not hold copies of them: they hold another shard of each expert
    when the expert TP size is above one, or other experts under expert parallelism.
    """
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    TEGroupedLinear._set_parallel_attributes.__wrapped__(self)
    if not self.explicit_expert_comm or self._original_parallel_mode not in ("column", "row"):
        return
    for i in range(self.num_gemms):
        weight = getattr(self, f"weight{i}", None)
        if weight is not None:
            weight.tensor_model_parallel = True
