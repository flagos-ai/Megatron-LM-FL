# Copyright (c) 2026, BAAI. All rights reserved.
"""FSDP DTensor checkpoint splitting: SWiGLU fc1 and GDN fused projections split into per-component DTensor shards."""

import torch
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import make_fsdp_dtensor
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import FSDPDistributedIndex
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    handle_gdn_in_state_dict,
    handle_swiglu_in_state_dict,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

TP_SIZE = 4
DP_SHARD_SIZE = 2
GDN_IN_PROJ_NAMES = ['query', 'key', 'value', 'z', 'beta', 'alpha']
GDN_CONV1D_NAMES = ['query', 'key', 'value']


class _GdnFusedProjections(torch.nn.Module):
    """Attribute contract of GatedDeltaNet's fused projections.

    GatedDeltaNet itself needs flash-linear-attention, which is not installed in the
    CUDA CI image; handle_gdn_in_state_dict reads only the qk_dim/v_dim/in_proj_dim
    attributes plus the in_proj and conv1d parameters, so this module mirrors that
    structure (a linear in_proj and a grouped nn.Conv1d, as in GatedDeltaNet).
    """

    def __init__(self, hidden_size, qk_dim, v_dim, num_value_heads, tp_size, conv_kernel):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.in_proj_dim = 2 * (qk_dim + v_dim + num_value_heads)
        self.num_value_heads = num_value_heads
        self.tp_size = tp_size
        self.in_proj = torch.nn.Linear(hidden_size, self.in_proj_dim, bias=False)
        conv_dim = 2 * qk_dim + v_dim
        self.conv1d = torch.nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=conv_kernel,
            groups=conv_dim,
            bias=True,
            padding=conv_kernel - 1,
        )


def _sharded_fsdp_tensor(full, dist_index, mesh):
    """Build this rank's FSDP shard of ``full`` as a DTensor plus its reference slices.

    Mirrors the megatron param_and_grad_buffer layout: each rank holds the
    dp_shard-th contiguous slice of its TP-local flat weight, recorded in
    ``megatron_fsdp_slice`` in TP-local flat coordinates.
    """
    rows = full.shape[0]
    rows_tp = rows // TP_SIZE
    dp_rank, tp_rank = mesh.get_local_rank(mesh_dim=0), mesh.get_local_rank(mesh_dim=1)
    tp_local_flat = full[tp_rank * rows_tp : (tp_rank + 1) * rows_tp].reshape(-1).to('cuda')
    shard_len = tp_local_flat.numel() // DP_SHARD_SIZE
    piece = tp_local_flat[dp_rank * shard_len : (dp_rank + 1) * shard_len]
    template = torch.empty(rows_tp, *full.shape[1:])
    set_tensor_model_parallel_attributes(template, True, 0, 1)
    dtensor = make_fsdp_dtensor(
        piece.view(-1, *full.shape[1:]),
        template,
        dist_index,
        is_sharded_param=True,
        run_check=True,
        update_uneven_dtensor_chunk_meta=True,
    )
    return dtensor, slice(dp_rank * shard_len, (dp_rank + 1) * shard_len), tp_local_flat


def _fsdp_parameter(dtensor, dist_index, fsdp_slice):
    param = torch.nn.Parameter(dtensor)
    param.megatron_fsdp_dist_index = dist_index
    param.megatron_fsdp_slice = fsdp_slice
    set_tensor_model_parallel_attributes(param, True, 0, 1)
    return param


def _expected_split_piece(tp_local_flat, fsdp_slice, lo, hi):
    """This rank's expected shard of the tp-local flat range [lo, hi).

    The split code slices the rank's own FSDP shard (tp_local_flat[fsdp_slice])
    by its intersection with [lo, hi) and then offsets the slice back into the
    shard, so in tp-local flat coordinates the expected values are exactly
    tp_local_flat[start:stop].
    """
    start = max(fsdp_slice.start, lo)
    stop = min(fsdp_slice.stop, hi)
    if start >= stop:
        return tp_local_flat[:0]
    return tp_local_flat[start:stop]


class TestHandleSwigluInStateDict:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        self.mesh = init_device_mesh(
            'cuda', (DP_SHARD_SIZE, TP_SIZE), mesh_dim_names=('dp_shard', 'tp')
        )
        self.dist_index = FSDPDistributedIndex(
            device_mesh=self.mesh, dp_shard_dim='dp_shard', tp_dim='tp'
        )

        glu_config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=16,
            gated_linear_unit=True,
            use_cpu_initialization=True,
        )
        plain_config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=16,
            gated_linear_unit=False,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec()
        glu_layer = TransformerLayer(glu_config, spec.submodules, layer_number=1)
        plain_layer = TransformerLayer(plain_config, spec.submodules, layer_number=1)

        self.fc1_full = torch.arange(float(32 * 32)).reshape(32, 32)
        (self.fc1_dtensor, self.fc1_fsdp_slice, self.fc1_tp_local) = _sharded_fsdp_tensor(
            self.fc1_full, self.dist_index, self.mesh
        )
        glu_layer.mlp.linear_fc1.weight = _fsdp_parameter(
            self.fc1_dtensor, self.dist_index, self.fc1_fsdp_slice
        )

        self.root = torch.nn.Module()
        self.root.module = torch.nn.ModuleDict({'language': glu_layer, 'vision': plain_layer})
        self.root.config = glu_config
        self.fc2_value = torch.full((16, 32), 5.0)
        self.vision_fc1_value = torch.full((32, 32), 7.0)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_splits_glu_fc1_into_w_and_v_and_skips_non_glu_layers(self):
        model_state_dict = {
            'language.mlp.linear_fc1.weight': self.fc1_dtensor,
            'language.mlp.linear_fc2.weight': self.fc2_value,
            'vision.mlp.linear_fc1.weight': self.vision_fc1_value,
        }
        new_state_dict, optimizer_state_dict = handle_swiglu_in_state_dict(
            self.root, model_state_dict, None
        )

        assert optimizer_state_dict is None
        assert set(new_state_dict) == {
            'language.mlp.linear_fc1.weight_w',
            'language.mlp.linear_fc1.weight_v',
            'language.mlp.linear_fc2.weight',
            'vision.mlp.linear_fc1.weight',
        }
        assert new_state_dict['vision.mlp.linear_fc1.weight'] is self.vision_fc1_value
        assert torch.equal(new_state_dict['language.mlp.linear_fc2.weight'], self.fc2_value)

        shard_len = self.fc1_tp_local.numel() // DP_SHARD_SIZE
        for suffix, lo in (('_w', 0), ('_v', shard_len)):
            piece = new_state_dict[f'language.mlp.linear_fc1.weight{suffix}']
            expected = _expected_split_piece(
                self.fc1_tp_local, self.fc1_fsdp_slice, lo, lo + shard_len
            )
            self._assert_component(piece, expected, (16, 32))

    def test_splits_optimizer_moments_for_glu_fc1_keys(self):
        exp_avg_full = torch.arange(100.0, 100.0 + 32 * 32).reshape(32, 32)
        exp_avg_sq_full = torch.arange(200.0, 200.0 + 32 * 32).reshape(32, 32)
        exp_avg, _, avg_tp_local = _sharded_fsdp_tensor(exp_avg_full, self.dist_index, self.mesh)
        exp_avg_sq, _, avg_sq_tp_local = _sharded_fsdp_tensor(
            exp_avg_sq_full, self.dist_index, self.mesh
        )
        optimizer_state_dict = {
            'state': {
                'module.module.language.mlp.linear_fc1.weight': {
                    'exp_avg': exp_avg,
                    'exp_avg_sq': exp_avg_sq,
                },
                'module.module.vision.mlp.linear_fc1.weight': {'step': 3},
                'module.module.language.mlp.linear_fc2.weight': {'step': 5},
            }
        }
        model_state_dict = {
            'language.mlp.linear_fc1.weight': self.fc1_dtensor,
            'language.mlp.linear_fc2.weight': self.fc2_value,
        }
        _, new_optimizer_state_dict = handle_swiglu_in_state_dict(
            self.root, model_state_dict, optimizer_state_dict
        )

        new_state = new_optimizer_state_dict['state']
        assert set(new_state) == {
            'module.module.language.mlp.linear_fc1.weight_w',
            'module.module.language.mlp.linear_fc1.weight_v',
            'module.module.vision.mlp.linear_fc1.weight',
            'module.module.language.mlp.linear_fc2.weight',
        }
        assert new_state['module.module.vision.mlp.linear_fc1.weight'] == {'step': 3}
        assert new_state['module.module.language.mlp.linear_fc2.weight'] == {'step': 5}

        shard_len = avg_tp_local.numel() // DP_SHARD_SIZE
        for subkey, tp_local in (('exp_avg', avg_tp_local), ('exp_avg_sq', avg_sq_tp_local)):
            for suffix, lo in (('_w', 0), ('_v', shard_len)):
                key = f'module.module.language.mlp.linear_fc1.weight{suffix}'
                piece = new_state[key][subkey]
                expected = _expected_split_piece(tp_local, self.fc1_fsdp_slice, lo, lo + shard_len)
                self._assert_component(piece, expected, (16, 32))

    def _assert_component(self, piece, expected_flat, global_shape):
        assert isinstance(piece, DTensor)
        assert tuple(piece.shape) == global_shape
        assert piece.placements == (Shard(0), Shard(0))
        assert torch.equal(piece.to_local(), expected_flat.view(-1, *global_shape[1:]))


class TestHandleGdnInStateDict:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        self.mesh = init_device_mesh(
            'cuda', (DP_SHARD_SIZE, TP_SIZE), mesh_dim_names=('dp_shard', 'tp')
        )
        self.dist_index = FSDPDistributedIndex(
            device_mesh=self.mesh, dp_shard_dim='dp_shard', tp_dim='tp'
        )

        self.hidden_size = 32
        self.gdn = _GdnFusedProjections(
            hidden_size=self.hidden_size,
            qk_dim=16,
            v_dim=16,
            num_value_heads=8,
            tp_size=TP_SIZE,
            conv_kernel=4,
        )
        in_proj_full = torch.arange(float(80 * 32)).reshape(80, 32)
        conv_weight_full = torch.arange(float(48 * 4)).reshape(48, 1, 4)
        conv_bias_full = torch.arange(float(48))
        (self.in_proj_dtensor, self.fsdp_slice, self.in_proj_tp_local) = _sharded_fsdp_tensor(
            in_proj_full, self.dist_index, self.mesh
        )
        (self.conv_weight_dtensor, self.conv_weight_fsdp_slice, self.conv_weight_tp_local) = (
            _sharded_fsdp_tensor(conv_weight_full, self.dist_index, self.mesh)
        )
        (self.conv_bias_dtensor, self.conv_bias_fsdp_slice, self.conv_bias_tp_local) = (
            _sharded_fsdp_tensor(conv_bias_full, self.dist_index, self.mesh)
        )
        self.gdn.in_proj.weight = _fsdp_parameter(
            self.in_proj_dtensor, self.dist_index, self.fsdp_slice
        )
        self.gdn.conv1d.weight = _fsdp_parameter(
            self.conv_weight_dtensor, self.dist_index, self.conv_weight_fsdp_slice
        )
        self.gdn.conv1d.bias = _fsdp_parameter(
            self.conv_bias_dtensor, self.dist_index, self.conv_bias_fsdp_slice
        )

        self.root = torch.nn.Module()
        self.root.module = torch.nn.ModuleDict({'gdn': self.gdn})
        self.other_value = torch.full((32, 32), 3.0)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_splits_fused_projections_into_named_components(self):
        model_state_dict = {
            'gdn.in_proj.weight': self.in_proj_dtensor,
            'gdn.conv1d.weight': self.conv_weight_dtensor,
            'gdn.conv1d.bias': self.conv_bias_dtensor,
            'other.mlp.linear_fc1.weight': self.other_value,
        }
        new_state_dict, optimizer_state_dict = handle_gdn_in_state_dict(
            self.root, model_state_dict, None
        )

        assert optimizer_state_dict is None
        expected_keys = {'other.mlp.linear_fc1.weight'}
        expected_keys |= {f'gdn.in_proj.weight.{name}' for name in GDN_IN_PROJ_NAMES}
        expected_keys |= {f'gdn.conv1d.weight.{name}' for name in GDN_CONV1D_NAMES}
        expected_keys |= {f'gdn.conv1d.bias.{name}' for name in GDN_CONV1D_NAMES}
        assert set(new_state_dict) == expected_keys
        assert new_state_dict['other.mlp.linear_fc1.weight'] is self.other_value

        in_proj_rows = [4, 4, 4, 4, 2, 2]
        elems = self.in_proj_tp_local.numel() // sum(in_proj_rows)
        assert elems == self.hidden_size
        flat_offset = 0
        for rows, name in zip(in_proj_rows, GDN_IN_PROJ_NAMES):
            piece = new_state_dict[f'gdn.in_proj.weight.{name}']
            expected = _expected_split_piece(
                self.in_proj_tp_local, self.fsdp_slice, flat_offset, flat_offset + rows * elems
            )
            self._assert_component(piece, expected, (rows * TP_SIZE, self.hidden_size))
            flat_offset += rows * elems

        conv_rows = [4, 4, 4]
        conv_elems = self.conv_weight_tp_local.numel() // sum(conv_rows)
        flat_offset = 0
        for rows, name in zip(conv_rows, GDN_CONV1D_NAMES):
            piece = new_state_dict[f'gdn.conv1d.weight.{name}']
            expected = _expected_split_piece(
                self.conv_weight_tp_local,
                self.conv_weight_fsdp_slice,
                flat_offset,
                flat_offset + rows * conv_elems,
            )
            self._assert_component(piece, expected, (rows * TP_SIZE, 1, 4))
            flat_offset += rows * conv_elems

        bias_elems = self.conv_bias_tp_local.numel() // sum(conv_rows)
        flat_offset = 0
        for rows, name in zip(conv_rows, GDN_CONV1D_NAMES):
            piece = new_state_dict[f'gdn.conv1d.bias.{name}']
            expected = _expected_split_piece(
                self.conv_bias_tp_local,
                self.conv_bias_fsdp_slice,
                flat_offset,
                flat_offset + rows * bias_elems,
            )
            self._assert_component(piece, expected, (rows * TP_SIZE,))
            flat_offset += rows * bias_elems

    def test_splits_optimizer_moments_for_fused_projections(self):
        exp_avg_full = torch.arange(100.0, 100.0 + 80 * 32).reshape(80, 32)
        exp_avg_sq_full = torch.arange(200.0, 200.0 + 80 * 32).reshape(80, 32)
        exp_avg, _, avg_tp_local = _sharded_fsdp_tensor(exp_avg_full, self.dist_index, self.mesh)
        exp_avg_sq, _, avg_sq_tp_local = _sharded_fsdp_tensor(
            exp_avg_sq_full, self.dist_index, self.mesh
        )
        optimizer_state_dict = {
            'state': {
                'module.module.gdn.in_proj.weight': {'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq},
                'module.module.other.weight': {'step': 9},
            }
        }
        model_state_dict = {
            'gdn.in_proj.weight': self.in_proj_dtensor,
            'other.mlp.linear_fc1.weight': self.other_value,
        }
        _, new_optimizer_state_dict = handle_gdn_in_state_dict(
            self.root, model_state_dict, optimizer_state_dict
        )

        new_state = new_optimizer_state_dict['state']
        expected_keys = {f'module.module.gdn.in_proj.weight.{name}' for name in GDN_IN_PROJ_NAMES}
        expected_keys.add('module.module.other.weight')
        assert set(new_state) == expected_keys
        assert new_state['module.module.other.weight'] == {'step': 9}

        in_proj_rows = [4, 4, 4, 4, 2, 2]
        elems = avg_tp_local.numel() // sum(in_proj_rows)
        flat_offset = 0
        for rows, name in zip(in_proj_rows, GDN_IN_PROJ_NAMES):
            for subkey, tp_local in (('exp_avg', avg_tp_local), ('exp_avg_sq', avg_sq_tp_local)):
                piece = new_state[f'module.module.gdn.in_proj.weight.{name}'][subkey]
                expected = _expected_split_piece(
                    tp_local, self.fsdp_slice, flat_offset, flat_offset + rows * elems
                )
                self._assert_component(piece, expected, (rows * TP_SIZE, self.hidden_size))
            flat_offset += rows * elems

    def test_returns_dicts_unchanged_when_model_has_no_gdn_modules(self):
        dense_layer = torch.nn.Linear(8, 8)
        root = torch.nn.Module()
        root.module = dense_layer
        model_state_dict = {'module.weight': torch.ones(8, 8)}
        optimizer_state_dict = {'state': {}}
        new_model, new_optimizer = handle_gdn_in_state_dict(
            root, model_state_dict, optimizer_state_dict
        )
        assert new_model is model_state_dict
        assert new_optimizer is optimizer_state_dict

    def _assert_component(self, piece, expected_flat, global_shape):
        assert isinstance(piece, DTensor)
        assert tuple(piece.shape) == global_shape
        assert piece.placements == (Shard(0), Shard(0))
        assert torch.equal(piece.to_local(), expected_flat.view(-1, *global_shape[1:]))
