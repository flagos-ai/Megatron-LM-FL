# Copyright (c) 2026, BAAI. All rights reserved.
"""Inline inference cudagraph capture: static-batching cache keys, per-batch runners, and cross-layer buffer reuse."""

import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _GraphedFuncModule(MegatronModule):
    """Small numbered module whose method is wrapped by a CudaGraphManager."""

    def __init__(self, config, layer_number=None):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        if layer_number is not None:
            self.layer_number = layer_number
        self.register_buffer('scale', torch.ones(1))

    def my_op(self, hidden_states=None, flag=0, box=None, **kwargs):
        if hidden_states is None:
            return None
        out = self.linear(hidden_states) * self.scale
        return out + flag if flag else out


def _manager_config(**overrides):
    base = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=1,
        use_cpu_initialization=True,
        cuda_graph_impl='local',
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _reset_global_cudagraph_state():
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []
    CudaGraphManager.global_mempool = None


class TestInlineInferenceCapture:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )
        from megatron.core.inference.contexts.static_context import StaticInferenceContext

        self.static_context_cls = StaticInferenceContext
        self.config = _manager_config(inference_rng_tracker=True)
        self.hidden_size = self.config.hidden_size

    def teardown_method(self, method):
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def _make_module_and_manager(self, layer_number=None):
        module = _GraphedFuncModule(self.config, layer_number=layer_number).cuda().eval()
        manager = CudaGraphManager(
            self.config,
            base_module=module,
            function_name='my_op',
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )
        return module, manager

    @torch.inference_mode()
    def test_static_batching_cache_key_routes_runners_by_batch_and_phase(self):
        module, manager = self._make_module_and_manager()
        x4 = torch.randn(4, self.hidden_size, device='cuda')
        x5 = torch.randn(5, self.hidden_size, device='cuda')
        expected4 = module.linear(x4) * module.scale
        expected5 = module.linear(x5) * module.scale

        decode = self.static_context_cls(max_batch_size=8, max_sequence_length=64)
        decode.enable_decode_mode()
        prefill = self.static_context_cls(max_batch_size=8, max_sequence_length=64)
        prefill.enable_prefill_mode()

        decode_out = module.my_op(hidden_states=x4, inference_context=decode)
        decode_replay = module.my_op(hidden_states=x4, inference_context=decode)
        prefill_out = module.my_op(hidden_states=x4, inference_context=prefill)
        other_batch = module.my_op(hidden_states=x5, inference_context=decode)

        assert torch.equal(decode_out, expected4)
        assert torch.equal(decode_replay, expected4)
        assert torch.equal(prefill_out, expected4)
        assert torch.equal(other_batch, expected5)
        assert len(manager.cudagraph_runners) == 3
        assert manager.custom_cudagraphs_lookup_table[(4, True)] is manager.cudagraph_runners[0]
        assert manager.custom_cudagraphs_lookup_table[(4, False)] is manager.cudagraph_runners[1]
        assert manager.custom_cudagraphs_lookup_table[(5, True)] is manager.cudagraph_runners[2]

    @torch.inference_mode()
    def test_later_layer_scans_previous_runners_for_hidden_states_buffer(self):
        module1, _ = self._make_module_and_manager(layer_number=1)
        module2, manager2 = self._make_module_and_manager(layer_number=2)
        hidden = torch.randn(4, self.hidden_size, device='cuda')
        expected1 = module1.linear(hidden) * module1.scale
        expected2 = module2.linear(hidden) * module2.scale

        decode = self.static_context_cls(max_batch_size=8, max_sequence_length=64)
        decode.enable_decode_mode()
        out1 = module1.my_op(hidden_states=hidden, inference_context=decode)
        out2 = module2.my_op(hidden_states=hidden, inference_context=decode)

        assert torch.equal(out1, expected1)
        assert torch.equal(out2, expected2)
        runner2 = manager2.cudagraph_runners[0]
        assert runner2.is_first_layer is False
        recorded = _CudagraphGlobalRecord.cudagraph_inference_record
        assert len(recorded) == 2
        assert recorded[0][0].base_module is module1
        assert recorded[1][0].base_module is module2

    @torch.inference_mode()
    def test_layer_with_layer_number_gap_falls_back_to_its_own_input(self):
        module5, manager5 = self._make_module_and_manager(layer_number=5)
        hidden = torch.randn(4, self.hidden_size, device='cuda')
        expected5 = module5.linear(hidden) * module5.scale

        decode = self.static_context_cls(max_batch_size=8, max_sequence_length=64)
        decode.enable_decode_mode()
        out5 = module5.my_op(hidden_states=hidden, inference_context=decode)

        assert torch.equal(out5, expected5)
        runner5 = manager5.cudagraph_runners[0]
        assert runner5.is_first_layer is False
        assert runner5.fwd_graph_input_kwargs['hidden_states'] is hidden
