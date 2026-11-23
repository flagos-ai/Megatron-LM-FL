"""Two-rank CPU/Gloo check of real native replica/TP reductions and Adam.

Run with torchrun --standalone --nproc-per-node=2. No CUDA work or model files.
This checks ownership/scaling/update math, not CUDA streams or GPU performance.
"""
from datetime import timedelta
import json
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.plugin.slideformer.layer_update_pipeline import LayerUpdatePipeline


class Owner:
    def __init__(self, index, parameter):
        self.layer_idx, self.params = index, [parameter]
        self.cpu_params, self.cpu_grads = {parameter: parameter}, {}
        self.optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=.02, fused=True)

    def _record_lifecycle(self, event):
        pass

    def _step_layer_optimizer(self):
        for param, grad in self.cpu_grads.items():
            param.grad = grad
        self.optimizer.step()


def run_case(mode, per_token, early):
    rank = dist.get_rank()
    singles = [dist.new_group([r], backend="gloo") for r in range(2)]
    local, world = singles[rank], dist.group.WORLD
    tp = world if mode == "tp" else local
    replicas = local if mode == "tp" else world
    experts = local if mode == "ep" else replicas
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.linspace(-.5, .5, 17) + i) for i in range(3)])
    model.config = SimpleNamespace(calculate_per_token_loss=per_token,
                                   sequence_parallel=mode == "tp", qk_layernorm=False)
    if mode == "tp":
        model[1].sequence_parallel = True
        model[2].average_gradients_across_tp_domain = True
    if mode == "ep":
        model[1].allreduce = False
    owners = [Owner(i, p) for i, p in enumerate(model)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in model]
    optimizer = torch.optim.AdamW(reference, lr=.01, weight_decay=.02, fused=True)
    engine = SimpleNamespace(
        device=torch.device("cpu"), _sharded_opt=None, gpu_gradient_pipeline=None,
        unmanaged_params=[], layer_optimizer=SimpleNamespace(offloader=None),
        config=SimpleNamespace(defer_optimizer_step=True, skip_grad_d2h=False,
                               skip_cpu_adam_step=False, optimizer_pipeline_depth=2, chunk_size_mb=1),
        model=model, layout=SimpleNamespace(model=model), managed_layers=owners,
        _accumulated_grads={})
    engine._pooled_grad_accumulator = lambda idx, param, like, zero=False: torch.zeros_like(like)
    pipeline = LayerUpdatePipeline(engine, group=replicas, tp_group=tp,
                                   expert_group=experts, pp_group=local)
    try:
        for step in range(3):
            engine._accumulated_grads = {}
            pipeline.begin(2)
            local_tokens = 7 + rank if mode != "tp" else 7
            token_sum = local_tokens if mode == "tp" else 15
            if per_token and early:
                pipeline.set_expected_scale(1.0 / token_sum)
            oracle = {}
            for micro in range(2):
                for owner in reversed(owners):
                    param = owner.params[0]
                    grad = torch.arange(17).float() + step + micro + rank + owner.layer_idx
                    gradients = engine._accumulated_grads.setdefault(owner.layer_idx, {})
                    if param in gradients:
                        gradients[param].add_(grad)
                    else:
                        gradients[param] = grad.clone()
                    oracle[owner.layer_idx] = oracle.get(owner.layer_idx, torch.zeros_like(grad)) + grad
                    pipeline.capture(owner)
            pipeline.finalize(torch.tensor(local_tokens) if per_token else None)
            for i, param in enumerate(model):
                grad = oracle[i]
                if mode != "tp" and not (mode == "ep" and i == 1):
                    dist.all_reduce(grad)
                if mode == "tp" and i in (1, 2):
                    dist.all_reduce(grad)
                    if i == 2:
                        grad.div_(dist.get_world_size(tp))
                grad.mul_(1.0 / token_sum if per_token else 1.0 / dist.get_world_size(replicas))
                torch.testing.assert_close(engine._accumulated_grads[i][param], grad, rtol=0, atol=0)
                reference[i].grad = grad
            optimizer.step()
            for i, param in enumerate(model):
                torch.testing.assert_close(param, reference[i], rtol=0, atol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(owners[i].optimizer.state[param][key],
                                               optimizer.state[reference[i]][key], rtol=0, atol=0)
            assert pipeline.stats["gpu_full_model_accumulator_bytes"] == 0
        print("CPU_LAYER_PIPELINE " + json.dumps(dict(rank=rank, mode=mode, per_token=per_token,
              precomputed_tokens=early, steps=3, gradient_error=0, parameter_error=0,
              adam_state_error=0)), flush=True)
    finally:
        pipeline.close()


if __name__ == "__main__":
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        for mode in ("dp", "ep", "tp"):
            for per_token, early in ((False, False), (True, False), (True, True)):
                run_case(mode, per_token, early)
        if dist.get_rank() == 0:
            print("CPU_LAYER_PIPELINE_ALL_PASS", flush=True)
    finally:
        dist.destroy_process_group()
