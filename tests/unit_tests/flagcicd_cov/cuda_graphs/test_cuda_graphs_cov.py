# Copyright (c) 2026, BAAI. All rights reserved.
"""CudaGraphManager: per-layer record/create/replay across microbatches, replay argument checks, and eager fallbacks."""

from dataclasses import dataclass

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    _ensure_generator_state_is_cudagraph_safe,
    create_cudagraphs,
)
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


class _ForwardModule(MegatronModule):
    """Plain forward module used with a CudaGraphManager without function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


@dataclass
class _TensorBox:
    values: torch.Tensor
    label: int = 0


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


def test_ensure_generator_state_is_cudagraph_safe_for_plain_generator():
    gen = torch.Generator()
    gen.manual_seed(1234)
    before = gen.get_state().clone()
    assert _ensure_generator_state_is_cudagraph_safe(gen) is gen
    assert torch.equal(gen.get_state(), before)
    assert gen.get_state().data_ptr() != before.data_ptr()


def test_create_cudagraphs_returns_early_when_no_graphs_were_recorded(monkeypatch):
    monkeypatch.setattr(_CudagraphGlobalRecord, 'cudagraph_created', False)
    monkeypatch.setattr(_CudagraphGlobalRecord, 'cudagraph_record', [])
    assert create_cudagraphs() is None


class TestLocalCudaGraphTrainingFlow:
    """Record, create and replay per-layer cudagraphs on a small TransformerBlock."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.transformer.transformer_block import TransformerBlock

        self.hidden_size = 64
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl='local',
            cuda_graph_warmup_steps=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.block = TransformerBlock(self.config, get_gpt_layer_local_spec()).cuda()

    def teardown_method(self, method):
        for layer in self.block.layers:
            manager = getattr(layer, 'cudagraph_manager', None)
            if manager is not None:
                for runner in manager.cudagraph_runners:
                    runner.fwd_graph = None
                    runner.bwd_graph = None
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def _run_step(self, hidden, mask):
        out = self.block(hidden_states=hidden, attention_mask=mask)
        out.sum().backward()
        return out

    def test_record_create_and_replay_two_microbatches_per_pipeline_stage(self):
        seq_length, micro_batch_size = 32, 2
        mask = torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool).cuda()
        inputs = [
            torch.ones(
                (seq_length, micro_batch_size, self.hidden_size), device='cuda', requires_grad=True
            )
            for _ in range(2)
        ]

        eager_outputs, eager_input_grads, eager_param_grads = [], [], []
        for hidden in inputs:
            self.block.zero_grad()
            eager_outputs.append(self._run_step(hidden, mask).detach().clone())
            eager_input_grads.append(hidden.grad.detach().clone())
            eager_param_grads.append([p.grad.detach().clone() for p in self.block.parameters()])

        for layer in self.block.layers:
            assert len(layer.cudagraph_manager.cudagraph_runners) == 2
        first_runner = self.block.layers[0].cudagraph_manager.cudagraph_runners[0]
        assert (
            str(first_runner)
            == f'TransformerLayer; hid ({seq_length}, {micro_batch_size}, {self.hidden_size})'
        )

        stats = create_cudagraphs()
        assert set(stats) == {'time', 'allocated_bytes', 'reserved_bytes'}
        assert _CudagraphGlobalRecord.cudagraph_created is True
        assert _CudagraphGlobalRecord.cudagraph_record == []

        for hidden, ref_out, ref_grad, ref_param_grads in zip(
            inputs, eager_outputs, eager_input_grads, eager_param_grads
        ):
            replay_hidden = hidden.detach().clone().requires_grad_(True)
            self.block.zero_grad()
            replay_out = self._run_step(replay_hidden, mask)
            assert torch.equal(replay_out, ref_out)
            assert torch.equal(replay_hidden.grad, ref_grad)
            for param, ref in zip(self.block.parameters(), ref_param_grads):
                assert torch.equal(param.grad, ref)

        assert create_cudagraphs() is None

    def test_replay_mismatched_arguments_raise_with_details(self):
        seq_length, micro_batch_size = 32, 2
        mask = torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool).cuda()
        hidden = torch.ones(
            (seq_length, micro_batch_size, self.hidden_size), device='cuda', requires_grad=True
        )
        self._run_step(hidden, mask)
        create_cudagraphs()

        bad_shape = torch.ones((seq_length // 2, micro_batch_size, self.hidden_size), device='cuda')
        with pytest.raises(AssertionError, match='CUDA graph argument mismatch'):
            self.block(hidden_states=bad_shape, attention_mask=mask)


class TestSingleMempoolReuseWhenPipelineParallelIsOne:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.transformer.transformer_block import TransformerBlock

        self.hidden_size = 64
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl='local',
            cuda_graph_warmup_steps=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.block = TransformerBlock(self.config, get_gpt_layer_local_spec()).cuda()

    def teardown_method(self, method):
        for layer in self.block.layers:
            manager = getattr(layer, 'cudagraph_manager', None)
            if manager is not None:
                for runner in manager.cudagraph_runners:
                    runner.fwd_graph = None
                    runner.bwd_graph = None
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def test_shared_mask_and_hidden_states_reuse_one_runner_across_microbatches(self):
        seq_length, micro_batch_size = 32, 2
        mask = torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool).cuda()
        hidden = torch.ones(
            (seq_length, micro_batch_size, self.hidden_size), device='cuda', requires_grad=True
        )
        eager_out = self.block(hidden_states=hidden, attention_mask=mask)
        eager_out.sum().backward()
        eager_grad = hidden.grad.detach().clone()

        hidden2 = torch.ones_like(hidden, requires_grad=True)
        out2 = self.block(hidden_states=hidden2, attention_mask=mask)
        out2.sum().backward()

        for layer in self.block.layers:
            assert len(layer.cudagraph_manager.cudagraph_runners) == 1

        first_layer_mask_meta = (
            self.block.layers[0]
            .cudagraph_manager.cudagraph_runners[0]
            .fwd_graph_input_kwarg_metas['attention_mask']
            .cg_buffer_metadata
        )
        assert first_layer_mask_meta.input_use_count == 2
        assert first_layer_mask_meta.is_cudagraph_input is True

        create_cudagraphs()

        replay_hidden = hidden.detach().clone().requires_grad_(True)
        replay_out = self.block(hidden_states=replay_hidden, attention_mask=mask)
        replay_out.sum().backward()
        assert torch.equal(replay_out, eager_out)
        assert torch.equal(replay_hidden.grad, eager_grad)

    def test_replay_after_creation_reports_no_matching_runner_for_new_shapes(self):
        seq_length, micro_batch_size = 32, 2
        mask = torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool).cuda()
        hidden = torch.ones(
            (seq_length, micro_batch_size, self.hidden_size), device='cuda', requires_grad=True
        )
        out = self.block(hidden_states=hidden, attention_mask=mask)
        out.sum().backward()
        create_cudagraphs()

        bad_shape = torch.ones((seq_length // 2, micro_batch_size, self.hidden_size), device='cuda')
        with pytest.raises(AssertionError, match='no matching cudagraph runners were found'):
            self.block(hidden_states=bad_shape, attention_mask=mask)


class TestManagerRecordAndMismatchErrors:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        self.config = _manager_config()
        self.module = _GraphedFuncModule(self.config).cuda()
        self.manager = CudaGraphManager(self.config, base_module=self.module, function_name='my_op')
        self.hidden_size = self.config.hidden_size
        self.recorded = torch.randn(4, self.hidden_size, device='cuda', requires_grad=True)
        self.box = _TensorBox(values=torch.randn(2, 3, device='cuda'), label=7)
        out = self.module.my_op(self.recorded, flag=3, box=self.box)
        out.sum().backward()

    def teardown_method(self, method):
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def _mismatch_errors(self, args, kwargs):
        runner = self.manager.cudagraph_runners[0]
        return runner.get_mismatch_errors(args, kwargs)

    def test_matching_arguments_produce_no_mismatch_errors(self):
        same = torch.randn(4, self.hidden_size, device='cuda')
        errors = self._mismatch_errors((same,), {'flag': 3, 'box': self.box})
        assert errors == []

    def test_get_mismatch_errors_reports_each_argument_kind(self):
        h = self.hidden_size
        cases = [
            (
                (torch.zeros(2, h, device='cuda'),),
                {'flag': 3, 'box': self.box},
                'Tensor mismatch at args[0]: Received shape',
            ),
            (
                (torch.zeros(4, h, device='cuda', dtype=torch.float64),),
                {'flag': 3, 'box': self.box},
                'Received dtype',
            ),
            ((torch.zeros(4, h, device='cpu'),), {'flag': 3, 'box': self.box}, 'Received device'),
            ((torch.zeros(4, h, device='cuda'), 'oops'), {'flag': 3}, 'Type mismatch at args[1]'),
            (
                (torch.zeros(4, h, device='cuda'),),
                {'flag': 4, 'box': self.box},
                "Value mismatch at kwargs['flag']: 4 vs 3",
            ),
            (
                (torch.zeros(4, h, device='cuda'),),
                {'flag': 3, 'box': _TensorBox(values=torch.zeros(2, 3, device='cuda'), label=7)},
                "Tensor mismatch at kwargs['box'].values",
            ),
            ((torch.zeros(4, h, device='cuda'),), {'flag': 3}, 'Missing kwargs'),
            (
                (torch.zeros(4, h, device='cuda'),),
                {'flag': 3, 'box': self.box, 'extra': 1},
                'Unexpected kwargs',
            ),
            (
                (torch.zeros(4, h, device='cuda'), torch.zeros(4, h, device='cuda')),
                {'flag': 3, 'box': self.box},
                'Argument count mismatch: 2 vs 1',
            ),
        ]
        for args, kwargs, fragment in cases:
            errors = self._mismatch_errors(args, kwargs)
            assert any(fragment in message for message in errors), (fragment, errors)

    def test_get_mismatch_errors_rejects_unsupported_argument_types(self):
        with pytest.raises(AssertionError, match='not supported'):
            self._mismatch_errors(([torch.zeros(4, self.hidden_size, device='cuda')],), {})

    def test_get_tensors_flattens_tensors_nested_in_dataclasses(self):
        flat = torch.zeros(4, self.hidden_size, device='cuda')
        box_tensor = torch.zeros(2, 3, device='cuda')
        runner = self.manager.cudagraph_runners[0]
        tensors = runner.get_tensors((flat, _TensorBox(values=box_tensor)), {'flag': 3})
        assert tensors == (flat, box_tensor)

    def test_get_mismatch_errors_dataclass_field_value_mismatch(self):
        same_shape = torch.zeros(2, 3, device='cuda')
        errors = self._mismatch_errors(
            (self.recorded,), {'flag': 3, 'box': _TensorBox(values=same_shape, label=9)}
        )
        assert any("Value mismatch at kwargs['box'].label: 9 vs 7" in message for message in errors)


class TestCudagraphWeakRefFallback:
    """Tensors whose dtype has no TE weak-ref representation keep a strong reference."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        self.config = _manager_config()
        self.module = _GraphedFuncModule(self.config).cuda().double()
        self.manager = CudaGraphManager(self.config, base_module=self.module, function_name='my_op')
        self.hidden_size = self.config.hidden_size
        self.box = _TensorBox(values=torch.randn(2, 3, device='cuda', dtype=torch.float64), label=7)
        recorded = torch.randn(
            4, self.hidden_size, device='cuda', dtype=torch.float64, requires_grad=True
        )
        out = self.module.my_op(recorded, flag=3, box=self.box)
        out.sum().backward()

    def teardown_method(self, method):
        for runner in self.manager.cudagraph_runners:
            runner.fwd_graph = None
            runner.bwd_graph = None
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def test_float64_inputs_replay_correctly_with_strong_references(self):
        stats = create_cudagraphs()
        assert set(stats) == {'time', 'allocated_bytes', 'reserved_bytes'}

        replay_input = torch.randn(
            4, self.hidden_size, device='cuda', dtype=torch.float64, requires_grad=True
        )
        expected = self.module.linear(replay_input) * self.module.scale
        out = self.module.my_op(replay_input, flag=3, box=self.box)
        out.sum().backward()
        assert torch.equal(out, expected)
        assert self.manager.cudagraph_runners[0].cudagraph_created is True


class TestManagerEagerFallback:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        self.config = _manager_config()
        self.hidden_size = self.config.hidden_size
        self.module = _GraphedFuncModule(self.config).cuda().eval()
        self.reference_input = torch.randn(4, self.hidden_size, device='cuda')

    def teardown_method(self, method):
        _reset_global_cudagraph_state()
        Utils.destroy_model_parallel()

    def test_wrapped_function_eager_flag_bypasses_the_manager(self):
        manager = CudaGraphManager(
            self.config, base_module=self.module, function_name='my_op', need_backward=False
        )
        expected = self.module.linear(self.reference_input) * self.module.scale
        with torch.no_grad():
            out = self.module.my_op(self.reference_input, eager=True)
        assert torch.equal(out, expected)
        assert manager.cudagraph_runners == []

    def test_manager_falls_back_to_wrapped_function_in_eval_without_grad(self):
        manager = CudaGraphManager(
            self.config, base_module=self.module, function_name='my_op', need_backward=False
        )
        manager.eval()
        expected = self.module.linear(self.reference_input) * self.module.scale
        with torch.no_grad():
            out = self.module.my_op(self.reference_input)
        assert torch.equal(out, expected)
        assert manager.cudagraph_runners == []

    def test_manager_falls_back_to_module_call_without_function(self):
        module = _ForwardModule(self.config).cuda().eval()
        manager = CudaGraphManager(self.config)
        manager.eval()
        expected = module.linear(self.reference_input)
        with torch.no_grad():
            out = manager(module, (self.reference_input,), {})
        assert torch.equal(out, expected)
        assert manager.cudagraph_runners == []
