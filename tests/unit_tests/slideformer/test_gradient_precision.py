"""Transport precision must not change Adam storage or silently change on resume."""
from types import SimpleNamespace
import pytest
import torch
from megatron.plugin.slideformer.config import MegatronSlideFormerConfig
from megatron.plugin.slideformer.engine import MegatronSlideFormerEngine, MegatronSlideFormerEngineConfig
from megatron.plugin.slideformer.layer_adam import LayerAdam


def test_precision_default_and_explicit_fallback(monkeypatch):
    monkeypatch.delenv('MEGATRON_SLIDEFORMER_GPU_REDUCE_DTYPE', raising=False)
    assert MegatronSlideFormerConfig.from_env().gpu_reduce_dtype == 'bf16'
    assert MegatronSlideFormerEngineConfig().gpu_reduce_dtype == 'bf16'
    monkeypatch.setenv('MEGATRON_SLIDEFORMER_GPU_REDUCE_DTYPE', 'fp32')
    assert MegatronSlideFormerConfig.from_env().gpu_reduce_dtype == 'fp32'
    with pytest.raises(ValueError, match='gpu_reduce_dtype'):
        MegatronSlideFormerEngineConfig(gpu_reduce_dtype='fp16')


def test_bf16_input_keeps_cpu_adam_weights_and_moments_fp32():
    torch.manual_seed(37)
    params = [torch.nn.Parameter(torch.randn(259)) for _ in range(2)]
    params[1].data.copy_(params[0])
    optimizers = [LayerAdam(lr=2e-4, weight_decay=0.01) for _ in params]
    for opt, p in zip(optimizers, params):
        opt.add_layer_params(0, [p])
    for step in range(5):
        grad = torch.randn_like(params[0]).bfloat16()
        for opt, p, g in zip(optimizers, params, (grad, grad.float())):
            opt.step_with_grad_views(0, {p:g}, {p:p})
            assert p.dtype == torch.float32
            assert opt.exp_avg_flat[0].dtype == opt.exp_avg_sq_flat[0].dtype == torch.float32
        torch.testing.assert_close(params[0], params[1], atol=0, rtol=0)
        for key in ('exp_avg_flat', 'exp_avg_sq_flat'):
            torch.testing.assert_close(getattr(optimizers[0], key)[0], getattr(optimizers[1], key)[0], atol=0, rtol=0)


class FakeEngine(MegatronSlideFormerEngine):
    def __init__(self, precision):
        self.layout = SimpleNamespace(model=SimpleNamespace(config=None))
        self.config = MegatronSlideFormerEngineConfig(gpu_reduce_dtype=precision)
        self._gpu_reduce = object()
        self.loaded = None
    def state_dict(self):return {'sentinel': 7}
    def _validate_checkpoint_model_state(self, state):assert state == {'sentinel':7}
    def load_state_dict(self, state):self.loaded=state


def test_checkpoint_precision_mismatch_rejected_before_mutation(tmp_path):
    source = FakeEngine('bf16')
    source.save_checkpoint(tmp_path, 10)
    target = FakeEngine('fp32')
    with pytest.raises(ValueError, match='gradient precision mismatch'):
        target.load_checkpoint(tmp_path)
    assert target.loaded is None
    assert source.load_checkpoint(tmp_path) == 10


def test_legacy_checkpoint_requires_explicit_fp32(tmp_path):
    source = FakeEngine('fp32')
    source.save_checkpoint(tmp_path, 10)
    latest = tmp_path/'slideformer/latest.pt'
    state=torch.load(latest, weights_only=False);state.pop('gradient_precision');torch.save(state, latest)
    target=FakeEngine('bf16')
    with pytest.raises(ValueError, match='Legacy GPU-reduce'):
        target.load_checkpoint(tmp_path)
    assert target.loaded is None
    assert source.load_checkpoint(tmp_path) == 10
