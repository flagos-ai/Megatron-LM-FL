# Copyright (c) 2026, BAAI. All rights reserved.
"""Pooler: sequence-parallel gather equals full-input pooling."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.bert.pooler import Pooler
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_pooler_gathers_sequence_parallel_shards_before_pooling(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
    )
    pooler = Pooler(config.hidden_size, config.init_method, config, sequence_parallel=True).to(
        get_current_device()
    )

    device = get_current_device()
    full_hidden_states = torch.arange(32, dtype=torch.float32, device=device).reshape(4, 1, 8)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    local_shard = full_hidden_states[tp_rank * 2 : (tp_rank + 1) * 2].contiguous()

    pooled = pooler(local_shard)

    expected = torch.tanh(pooler.dense(full_hidden_states[0]))
    torch.testing.assert_close(pooled, expected)
