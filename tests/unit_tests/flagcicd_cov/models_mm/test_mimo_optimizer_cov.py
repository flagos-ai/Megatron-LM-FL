# Copyright (c) 2026, BAAI. All rights reserved.
"""MimoOptimizer: per-module optimizers with global gradient norms and overflow skipping, sharded checkpoint state with extracted param groups and grad scaler, and process-group collections from module grids."""
import torch

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.optimizer import (
    MimoOptimizer,
    ModuleOptimizerInfo,
    _get_pg_collection_for_optimizer,
    _get_replica_id,
    get_mimo_optimizer,
)
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device

VOCAB_SIZE = 128
SEQ_LEN = 16


def _gpt_model():
    config = TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True, bf16=True
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
    )


def _optimizer_config():
    return OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        loss_scale=1.0,
        use_distributed_optimizer=False,
        log_num_zeros_in_grad=True,
    )


def _inner_optimizer(model, config):
    return get_megatron_optimizer(
        config=config, model_chunks=[model], use_gloo_process_groups=False
    )


def _mimo_optimizer(model, config, pg_collection=None):
    inner = _inner_optimizer(model, config)
    module_infos = {
        'language': ModuleOptimizerInfo(
            optimizer=inner, grid=None, pg_collection=pg_collection, is_active=True
        ),
        'images': ModuleOptimizerInfo(
            optimizer=None, grid=None, pg_collection=None, is_active=False
        ),
    }
    return MimoOptimizer(module_infos, config), inner


def _set_grads(model, value_fn):
    torch.manual_seed(3)
    for param in model.parameters():
        param.grad = value_fn(param)


class TestMimoOptimizerStep:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _gpt_model().to(get_current_device())
        self.model.bfloat16()
        self.config = _optimizer_config()
        self.optimizer, self.inner = _mimo_optimizer(self.model, self.config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_step_returns_global_grad_norm_and_zero_count(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))
        self.model.decoder.layers[0].self_attention.linear_qkv.weight.grad[0, 0] = 0.0

        success, grad_norm, num_zeros = self.optimizer.step()

        assert success is True
        expected_norm_sq = sum(
            param.grad.detach().float().pow(2).sum().item() for param in self.model.parameters()
        )
        torch.testing.assert_close(grad_norm, expected_norm_sq**0.5, rtol=1e-3, atol=1e-4)
        expected_zeros = sum(
            int((param.grad == 0).sum().item()) for param in self.model.parameters()
        )
        assert num_zeros == expected_zeros

    def test_step_skips_update_when_gradients_overflow(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))
        first_param = next(self.model.parameters())
        first_param.grad.fill_(float('inf'))
        params_before = [param.detach().clone() for param in self.model.parameters()]

        success, grad_norm, num_zeros = self.optimizer.step()

        assert success is False
        assert grad_norm is None
        assert num_zeros is None
        for before, after in zip(params_before, self.model.parameters()):
            assert torch.equal(before, after.detach())

    def test_zero_grad_clears_main_param_grads(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))
        self.optimizer.step()

        self.optimizer.zero_grad()

        assert all(param.grad is None for param in self.inner.get_parameters())

    def test_get_loss_scale_and_param_groups_delegate_to_active_optimizers(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))

        assert torch.equal(self.optimizer.get_loss_scale(), self.inner.get_loss_scale())
        assert self.optimizer.param_groups == self.inner.param_groups

    def test_stub_optimizer_returns_unit_loss_scale(self):
        stub = MimoOptimizer(
            {
                'language': ModuleOptimizerInfo(
                    optimizer=None, grid=None, pg_collection=None, is_active=False
                )
            },
            self.config,
        )

        assert stub.is_stub_optimizer is True
        assert stub.get_loss_scale().item() == 1.0

    def test_reload_model_params_picks_up_weight_changes(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))
        self.optimizer.step()
        model_param = self.model.decoder.layers[0].self_attention.linear_qkv.weight
        with torch.no_grad():
            model_param += 1.0

        self.optimizer.reload_model_params()

        main_params = [param.detach() for param in self.inner.get_parameters()]
        model_params = [param.detach() for param in self.model.parameters()]
        for main_param, model_p in zip(main_params, model_params):
            torch.testing.assert_close(main_param.float(), model_p.float(), rtol=1e-2, atol=1e-2)

    def test_state_dict_reports_per_module_state(self):
        _set_grads(self.model, lambda param: torch.randn_like(param))
        self.optimizer.step()

        state_dict = self.optimizer.state_dict()

        assert set(state_dict.keys()) == {'language', 'images'}
        assert state_dict['images'] is None
        assert 'optimizer' in state_dict['language']
        assert 'grad_scaler' in state_dict['language']


class TestMimoOptimizerCheckpoint:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = _gpt_model().to(get_current_device())
        self.model.bfloat16()
        self.config = _optimizer_config()
        from megatron.core.process_groups_config import ProcessGroupCollection

        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.optimizer, self.inner = _mimo_optimizer(
            self.model, self.config, pg_collection=self.pg_collection
        )
        _set_grads(self.model, lambda param: torch.randn_like(param))
        self.optimizer.step()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sharded_state_dict_extracts_param_groups_and_grad_scaler(self):
        model_sharded_sd = self.model.sharded_state_dict()

        sharded_state = self.optimizer.sharded_state_dict(model_sharded_sd, is_loading=False)

        language_sd = sharded_state['language']
        assert sharded_state['images'] == {}
        assert '_mimo_param_groups' in language_sd
        assert '_mimo_grad_scaler' in language_sd
        assert 'param_groups' not in language_sd.get('optimizer', {})
        param_groups = language_sd['_mimo_param_groups'].data
        assert len(param_groups) == len(self.inner.param_groups)
        assert all(group['params'] == [] for group in param_groups)
        assert language_sd['_mimo_param_groups'].key == 'optimizer.mimo.language.param_groups'
        assert language_sd['_mimo_grad_scaler'].key == 'optimizer.mimo.language.grad_scaler'
        tp_rank = self.pg_collection.tp.rank()
        pp_rank = self.pg_collection.pp.rank()
        dp_rank = self.pg_collection.dp.rank()
        assert language_sd['_mimo_param_groups'].replica_id == (tp_rank, pp_rank, dp_rank)

    def test_load_state_dict_restores_param_groups_and_optimizer_state(self):
        snapshot = self.optimizer.state_dict()['language']
        checkpoint = {
            'language': {
                'fp32_from_fp16_params': snapshot['fp32_from_fp16_params'],
                '_mimo_param_groups': [
                    dict(group, params=[]) for group in snapshot['optimizer']['param_groups']
                ],
                '_mimo_grad_scaler': snapshot['grad_scaler'],
            },
            'images': {},
        }

        self.optimizer.load_state_dict(checkpoint)

        restored = self.optimizer.state_dict()['language']
        current_groups = self.inner.optimizer.state_dict()['param_groups']
        assert restored['optimizer']['param_groups'] == current_groups
        for group_loaded, group_current in zip(
            checkpoint['language']['_mimo_param_groups'], current_groups
        ):
            assert group_loaded['params'] == group_current['params']
        assert restored['grad_scaler'] == snapshot['grad_scaler']
        loaded_state = restored['optimizer']['state']
        snapshot_state = snapshot['optimizer']['state']
        assert set(loaded_state.keys()) == set(snapshot_state.keys())
        for key in snapshot_state:
            for field, value in snapshot_state[key].items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(loaded_state[key][field], value)


class TestMimoOptimizerProcessGroups:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_pg_collection_for_optimizer_exposes_grid_groups(self):
        Utils.initialize_model_parallel(1, 1)
        grid = HyperCommGrid(
            shape=[1, 1, 1, 1, 8],
            dim_names=["tp", "cp", "ep", "pp", "dp"],
            rank_offset=0,
            backend=Utils.get_backend(),
        )
        for dims in (
            ["dp"],
            ["dp", "cp"],
            ["tp"],
            ["pp"],
            ["tp", "pp"],
            ["tp", "ep", "pp"],
            ["dp", "ep"],
            ["tp", "cp", "ep", "pp", "dp"],
        ):
            grid.create_pg(dims)

        pg = _get_pg_collection_for_optimizer(grid)

        assert pg.dp is grid.get_pg("dp")
        assert pg.dp_cp is grid.get_pg(["dp", "cp"])
        assert pg.tp is grid.get_pg("tp")
        assert pg.pp is grid.get_pg("pp")
        assert pg.mp is grid.get_pg(["tp", "pp"])
        assert pg.tp_ep_pp is grid.get_pg(["tp", "ep", "pp"])
        assert pg.expt_dp is grid.get_pg(["dp", "ep"])
        assert pg.intra_dist_opt is grid.get_pg(["tp", "cp", "ep", "pp", "dp"])

        grid.destroy()

    def test_replica_id_combines_tp_pp_dp_ranks(self):
        Utils.initialize_model_parallel(2, 1)
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg = ProcessGroupCollection.use_mpu_process_groups()

        replica_id = _get_replica_id(pg)

        assert replica_id == (pg.tp.rank(), pg.pp.rank(), pg.dp.rank())
        assert replica_id[1] == 0


class TestGetMimoOptimizer:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_get_mimo_optimizer_builds_per_module_optimizers(self):
        from torch import nn

        from megatron.core.models.mimo.config.base_configs import MimoModelConfig
        from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
        from megatron.core.models.mimo.model.base import MimoModel
        from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
        from megatron.core.transformer.spec_utils import ModuleSpec

        grid = HyperCommGrid(
            shape=[1, 1, 1, 1, 8],
            dim_names=["tp", "cp", "ep", "pp", "dp"],
            rank_offset=0,
            backend=Utils.get_backend(),
        )
        for dims in (
            ["dp"],
            ["dp", "cp"],
            ["tp"],
            ["pp"],
            ["tp", "pp"],
            ["tp", "ep", "pp"],
            ["dp", "ep"],
            ["tp", "cp", "ep", "pp", "dp"],
        ):
            grid.create_pg(dims)
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
        )
        language_spec = ModuleSpec(
            module=GPTModel,
            params={
                'config': language_config,
                'transformer_layer_spec': get_gpt_layer_with_transformer_engine_spec(),
                'vocab_size': VOCAB_SIZE,
                'max_sequence_length': SEQ_LEN,
                'pre_process': True,
                'post_process': True,
            },
        )
        vision_spec = ModuleSpec(
            module=VisionModalitySubmodules,
            submodules={
                'encoders': {
                    'enc': ModuleSpec(
                        module=nn.Linear, params={'in_features': 32, 'out_features': 64}
                    )
                }
            },
        )
        mimo_model = MimoModel(
            MimoModelConfig(
                language_model_spec=language_spec,
                modality_submodules_spec={'images': vision_spec},
                special_token_ids={'images': 50},
                module_to_grid_map={'images': grid, MIMO_LANGUAGE_MODULE_KEY: grid},
            )
        ).to(get_current_device())

        optimizer = get_mimo_optimizer(mimo_model, _optimizer_config())

        assert set(optimizer.module_infos.keys()) == {'images', MIMO_LANGUAGE_MODULE_KEY}
        for info in optimizer.module_infos.values():
            assert info.is_active is True
            assert isinstance(info.optimizer, MegatronOptimizer)
        assert optimizer.is_stub_optimizer is False
        assert optimizer.get_loss_scale().item() == 1.0

        mimo_model.destroy()
        grid.destroy()
