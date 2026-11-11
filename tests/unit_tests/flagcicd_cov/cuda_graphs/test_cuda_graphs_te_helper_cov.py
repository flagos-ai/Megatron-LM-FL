# Copyright (c) 2026, BAAI. All rights reserved.
"""TE cudagraph helpers: graphable-layer selection, layer discovery per model chunk, amax reduction groups, and post-capture lifecycle."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _layer_is_graphable,
    set_current_microbatch,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _te_helper_config(**overrides):
    base = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        cuda_graph_impl='transformer_engine',
        position_embedding_type='absolute',
    )
    base.update(overrides)
    return TransformerConfig(**base)


class TestLayerIsGraphableSelection:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        self.layer = TransformerLayer(
            _te_helper_config(), get_gpt_layer_local_spec().submodules, layer_number=1
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_attention_module_flag_selects_attention_layers(self):
        config = _te_helper_config(
            cuda_graph_impl='local', cuda_graph_modules=[CudaGraphModule.attn]
        )
        assert _layer_is_graphable(self.layer, config) is True

    def test_moe_flag_leaves_dense_layers_ungraphed(self):
        config = _te_helper_config(
            cuda_graph_impl='local', cuda_graph_modules=[CudaGraphModule.moe]
        )
        assert _layer_is_graphable(self.layer, config) is False

    def test_mlp_flag_selects_dense_mlp_layers(self):
        config = _te_helper_config(
            cuda_graph_impl='local', cuda_graph_modules=[CudaGraphModule.mlp]
        )
        assert _layer_is_graphable(self.layer, config) is True


class TestTECudaGraphHelperLayerDiscovery:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        self.seq_length = 32
        self.micro_batch_size = 2
        self.config = _te_helper_config()
        self.model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=64,
            max_sequence_length=64,
        ).cuda()
        init_num_microbatches_calculator(
            rank=Utils.rank,
            global_batch_size=self.micro_batch_size * Utils.world_size * 2,
            micro_batch_size=self.micro_batch_size,
            data_parallel_size=Utils.world_size,
            decrease_batch_size_if_needed=False,
        )

    def teardown_method(self, method):
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    def _make_helper(self, model):
        return TECudaGraphHelper(
            model=model,
            config=self.config,
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            optimizers=[],
        )

    def test_discovers_all_decoder_layers_of_a_single_chunk(self):
        helper = self._make_helper([self.model])
        assert helper.num_model_chunks == 1
        assert len(helper.flattened_callables) == 2
        assert helper.flattened_callables_is_mtp == [False, False]
        assert helper.num_layers_per_chunk == [2]
        assert helper.chunks_with_decoder[0] is self.model
        assert helper.flattened_callables[0] is self.model.decoder.layers[0]

    def test_chunk_without_decoder_yields_no_graphable_layers(self):
        helper = self._make_helper([torch.nn.Linear(8, 8)])
        assert helper.flattened_callables == []
        assert helper.num_layers_per_chunk == [0]
        assert helper.callables_per_chunk == [[]]
        assert helper.chunks_with_decoder == [None]
        assert helper.num_model_chunks == 1

    def test_sample_arguments_skip_rotary_for_absolute_embeddings(self):
        helper = self._make_helper([self.model])
        sample_args, kwargs = helper._get_cuda_graph_input_data()
        assert kwargs['allow_unused_input'] is True
        assert kwargs['_order'] == [1, -1]
        assert len(sample_args) == len(helper.flattened_callables)
        hidden_states = sample_args[0][0]
        assert hidden_states.shape == (
            self.seq_length,
            self.micro_batch_size,
            self.config.hidden_size,
        )
        assert hidden_states.dtype == torch.bfloat16
        assert 'rotary_pos_emb' not in kwargs['sample_kwargs'][0]


class TestTECudaGraphHelperAmaxGroups:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'tp_cp', 'tp_dp', 'tp_dp_cp', 'dp_cp', 'pp']
        )
        self.config = _te_helper_config(tensor_model_parallel_size=2, context_parallel_size=2)
        self.model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=64,
            max_sequence_length=64,
        ).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_amax_reduction_group_follows_context_parallel_and_tp_only_flags(self):
        helper = TECudaGraphHelper(
            model=[self.model],
            config=self.config,
            seq_length=32,
            micro_batch_size=2,
            optimizers=[],
            pg_collection=self.pg_collection,
        )
        assert (
            helper._get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False)
            is self.pg_collection.tp_dp
        )
        assert (
            helper._get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=True)
            is self.pg_collection.tp
        )
        assert (
            helper._get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=False)
            is self.pg_collection.tp_dp_cp
        )
        assert (
            helper._get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=True)
            is self.pg_collection.tp_cp
        )


class TestTECudaGraphHelperPostCaptureLifecycle:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        from megatron.core.distributed import DistributedDataParallel
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )

        self.config = _te_helper_config()
        self.model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=64,
            max_sequence_length=64,
        ).cuda()
        self.ddp_model = DistributedDataParallel(
            self.config,
            ddp_config=DistributedDataParallelConfig(check_for_nan_in_grad=True),
            module=self.model,
        )
        self.optimizer = torch.optim.SGD(self.ddp_model.parameters(), lr=0.01)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_reset_after_capture_zeroes_grad_buffers_and_optimizer_grads(self):
        helper = TECudaGraphHelper(
            model=[self.ddp_model],
            config=self.config,
            seq_length=32,
            micro_batch_size=2,
            optimizers=[self.optimizer],
        )
        assert len(helper.flattened_callables) == 2
        assert helper.chunks_with_decoder[0] is self.model

        seq_length = 32
        hidden = torch.ones(
            (seq_length, 2, self.config.hidden_size), device='cuda', requires_grad=True
        )
        mask = torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool).cuda()
        out = self.model(
            input_ids=None, position_ids=None, attention_mask=mask, decoder_input=hidden
        )
        out.sum().backward()
        main_grads = [p.main_grad for p in self.model.parameters() if hasattr(p, 'main_grad')]
        assert any(grad.abs().sum().item() > 0 for grad in main_grads)
        for param in self.ddp_model.parameters():
            param.grad = torch.ones_like(param)

        helper._reset_after_capture()

        assert all(torch.count_nonzero(grad).item() == 0 for grad in main_grads)
        assert all(param.grad is None for param in self.ddp_model.parameters())

    def test_manual_hooks_target_modules_with_direct_parameters(self):
        helper = TECudaGraphHelper(
            model=[self.ddp_model],
            config=self.config,
            seq_length=32,
            micro_batch_size=2,
            optimizers=[],
        )
        helper.cuda_graph_set_manual_hooks()
        for layer in helper.flattened_callables:
            assert any(args[0] is layer.mlp.linear_fc1 for _, args in layer.cuda_graph_manual_hooks)
            assert all(callable(hook) for hook, _ in layer.cuda_graph_manual_hooks)
            assert all(
                next(args[0].parameters(recurse=False), None) is not None
                for _, args in layer.cuda_graph_manual_hooks
            )


class TestSetCurrentMicrobatchOnVisionLayers:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        from copy import deepcopy

        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules,
        )
        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )
        from megatron.core.transformer.mlp import MLPSubmodules
        from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules

        language_config = _te_helper_config(num_layers=2, hidden_size=64, num_attention_heads=4)
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=16, num_attention_heads=2, use_cpu_initialization=True
        )
        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)
        vision_config.vision_model_type = 'clip'
        language_config.language_model_type = 'dummy'

        self.llava = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_microbatch_id_lands_on_language_and_vision_decoder_layers(self):
        set_current_microbatch(self.llava.language_model, 2)
        for layer in self.llava.language_model.decoder.layers:
            assert layer.current_microbatch == 2

        set_current_microbatch(self.llava, 3)
        for layer in self.llava.language_model.decoder.layers:
            assert layer.current_microbatch == 2
        for layer in self.llava.vision_model.decoder.layers:
            assert layer.current_microbatch == 3
