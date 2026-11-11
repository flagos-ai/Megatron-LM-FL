# Copyright (c) 2026, BAAI. All rights reserved.
"""FSDP DTensor checkpoints: expert key remapping, FP8 extra-state stripping, state-dict diff reporting, and DCP round-trip validation."""

import logging

import pytest
import torch
import torch.distributed.checkpoint
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import make_fsdp_dtensor
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import FSDPDistributedIndex
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    expert_param_local_key,
    get_global_unique_param_name,
    handle_experts_in_state_dict,
    handle_fp8_extra_state_case,
    print_diff_in_state_dicts,
    validate_loaded_state_dict,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def test_handle_fp8_extra_state_case_removes_only_extra_state_keys():
    state_dict = {
        'decoder.layers.0.self_attention.linear_qkv._extra_state': 'fp8-meta',
        'decoder.layers.0.mlp.linear_fc1._extra_state': 'fp8-meta',
        'decoder.layers.0.mlp.linear_fc1.weight': torch.ones(2),
    }
    handle_fp8_extra_state_case(state_dict)
    assert set(state_dict) == {'decoder.layers.0.mlp.linear_fc1.weight'}
    assert torch.equal(state_dict['decoder.layers.0.mlp.linear_fc1.weight'], torch.ones(2))


def test_print_diff_in_state_dicts_lists_keys_by_section(caplog):
    metadata = {'model': {'only_meta': torch.ones(2), 'both': torch.ones(2, 4), 'count': 3}}
    load = {'model': {'only_load': torch.ones(3), 'both': torch.ones(2, 5), 'count': 3}}
    with caplog.at_level(logging.INFO, logger='megatron.core.transformer.fsdp_dtensor_checkpoint'):
        print_diff_in_state_dicts(metadata, load, limit=10)

    assert 'Keys only in checkpoint metadata_state_dict' in caplog.text
    assert 'model.only_meta' in caplog.text
    assert 'model.only_load' in caplog.text
    assert 'Keys in both but with different shapes' in caplog.text
    assert 'model.both' in caplog.text
    assert 'model.count' not in caplog.text


class TestExpertKeyRewriting:
    def setup_method(self, method):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        self.offset = parallel_state.get_expert_model_parallel_rank() * 4

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_expert_keys_are_rewritten_to_global_indices_per_ep_rank(self):
        state_dict = {
            'decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight': 'sequential',
            'decoder.layers.0.mlp.experts.linear_fc1.weight2': 'grouped',
            'decoder.layers.0.mlp.experts.linear_fc1.weight2_w': 'grouped-swiglu',
            'decoder.layers.0.mlp.linear_fc1.weight': 'dense',
        }
        handled = handle_experts_in_state_dict(state_dict, num_experts=8)

        expected_keys = {
            f'decoder.layers.0.mlp.experts.local_experts.{1 + self.offset}.linear_fc1.weight',
            f'decoder.layers.0.mlp.experts.linear_fc1.weight{2 + self.offset}',
            f'decoder.layers.0.mlp.experts.linear_fc1.weight{2 + self.offset}_w',
            'decoder.layers.0.mlp.linear_fc1.weight',
        }
        assert set(handled) == expected_keys
        assert (
            handled[
                f'decoder.layers.0.mlp.experts.local_experts.{1 + self.offset}.linear_fc1.weight'
            ]
            == 'sequential'
        )
        assert (
            handled[f'decoder.layers.0.mlp.experts.linear_fc1.weight{2 + self.offset}'] == 'grouped'
        )
        assert (
            handled[f'decoder.layers.0.mlp.experts.linear_fc1.weight{2 + self.offset}_w']
            == 'grouped-swiglu'
        )
        assert handled['decoder.layers.0.mlp.linear_fc1.weight'] == 'dense'

    def test_expert_param_local_key_shifts_global_indices_to_local(self):
        sequential = expert_param_local_key(
            'decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight', num_experts=8
        )
        assert sequential == (
            f'decoder.layers.0.mlp.experts.local_experts.{5 - self.offset}.linear_fc1.weight'
        )
        grouped = expert_param_local_key(
            'decoder.layers.0.mlp.experts.linear_fc1.weight5', num_experts=8
        )
        assert grouped == f'decoder.layers.0.mlp.experts.linear_fc1.weight{5 - self.offset}'
        dense = expert_param_local_key('decoder.layers.0.mlp.linear_fc1.weight', num_experts=8)
        assert dense == 'decoder.layers.0.mlp.linear_fc1.weight'


class TestGetGlobalUniqueParamName:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True
        )
        spec = get_gpt_layer_local_spec()
        decoder_layer = TransformerLayer(config, spec.submodules, layer_number=2)
        mtp_layer = TransformerLayer(config, spec.submodules, layer_number=1)
        container = torch.nn.ModuleDict(
            {
                'decoder': torch.nn.ModuleDict({'layers': torch.nn.ModuleList([decoder_layer])}),
                'mtp': torch.nn.ModuleDict({'layers': torch.nn.ModuleList([mtp_layer])}),
            }
        )
        self.root = torch.nn.Module()
        self.root.module = container
        self.root.config = config

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_transformer_layer_param_is_renumbered_by_layer_number(self):
        param = self.root.module['decoder']['layers'][0].mlp.linear_fc1.weight
        name = get_global_unique_param_name([self.root], param)
        assert name == 'module.decoder.layers.1.mlp.linear_fc1.weight'

    def test_mtp_param_keeps_its_recorded_layer_index(self):
        param = self.root.module['mtp']['layers'][0].mlp.linear_fc1.weight
        name = get_global_unique_param_name([self.root], param)
        assert name == 'module.mtp.layers.0.mlp.linear_fc1.weight'

    def test_unknown_param_raises_value_error(self):
        with pytest.raises(ValueError, match='Parameter not found in model chunks'):
            get_global_unique_param_name([self.root], torch.nn.Parameter(torch.ones(2)))


class TestValidateLoadedStateDict:
    def test_saved_shards_validate_against_the_original_tensor(self, model_parallel, tmp_path):
        model_parallel()
        world = torch.distributed.get_world_size()
        mesh = init_device_mesh('cuda', (world, 1), mesh_dim_names=('dp_shard', 'tp'))
        dist_index = FSDPDistributedIndex(device_mesh=mesh, dp_shard_dim='dp_shard', tp_dim='tp')
        full = torch.arange(float(8 * 4)).reshape(8, 4)
        rows_per_rank = 8 // world
        dp_rank = mesh.get_local_rank(mesh_dim=0)
        piece = (
            full[dp_rank * rows_per_rank : (dp_rank + 1) * rows_per_rank].contiguous().to('cuda')
        )
        template = torch.empty(8, 4)
        dtensor = make_fsdp_dtensor(
            piece,
            template,
            dist_index,
            is_sharded_param=True,
            run_check=True,
            update_uneven_dtensor_chunk_meta=True,
        )
        assert isinstance(dtensor, DTensor)
        assert tuple(dtensor.shape) == (8, 4)

        state_dict = {'plain': torch.arange(6.0).reshape(2, 3), 'sharded': dtensor, 'iteration': 7}
        checkpoint_dir = tmp_path / 'fsdp-dtensor-ckpt'
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=str(checkpoint_dir))
        assert (checkpoint_dir / '.metadata').exists()

        validate_loaded_state_dict(state_dict, str(checkpoint_dir))
