"""CPU-only multi-process test of ShardedCPUOptimizer against unsharded LayerAdam.

torchrun --nproc_per_node=W test_sharded_cpu_opt.py --tp T      (W = T * dp)
Ranks are laid out tp-fastest like Megatron: tp_rank = g % T, dp_rank = g // T.
Replica (dp) groups are therefore {t, t+T, t+2T, ...}; TP groups are consecutive.
"""

import argparse
import io
import os
import sys
from types import SimpleNamespace

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.plugin.slideformer.layer_adam import LayerAdam
from megatron.plugin.slideformer.sharded_cpu_optimizer import ShardedCPUOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()

dist.init_process_group("gloo")
W, G = dist.get_world_size(), dist.get_rank()
T = args.tp
assert W % T == 0
D = W // T
t_rank, d_rank = G % T, G // T

# Collectively create every group, keep my own.
dp_groups = {t: dist.new_group([t + T * d for d in range(D)]) for t in range(T)}
tp_groups = {d: dist.new_group([d * T + t for t in range(T)]) for d in range(D)}
dp_group, tp_group = dp_groups[t_rank], tp_groups[d_rank]

torch.manual_seed(0)
# (name, numel, kind) — odd sizes exercise the 64-element alignment; kinds: dense | sum | avg
LAYER_SPECS = [
    [
        ("w_qkv", 70001, "dense"),
        ("norm_a", 129, "sum"),
        ("w_o", 33333, "dense"),
        ("norm_b", 77, "avg"),
    ],
    [("w_up", 90007, "dense"), ("norm_c", 64, "sum"), ("w_dn", 12345, "dense")],
    [("emb", 50021, "dense")],
    [("aligned_a", 256, "dense"), ("aligned_norm", 64, "sum"), ("aligned_b", 320, "dense")],
]
KIND = {}


class FakeOwner:
    def __init__(self, layer_idx, optimizer, spec, tp_rank, tag):
        self.layer_idx = layer_idx
        self.params, self._param_flat_offsets = [], {}
        total = sum(n for _, n, _ in spec)
        gen = torch.Generator().manual_seed(1000 + layer_idx)
        flat = torch.empty(total, dtype=torch.float32)
        offset = 0
        for name, n, kind in spec:
            p = nn.Parameter(torch.empty(n), requires_grad=False)
            if kind != "dense":
                p.sequence_parallel = True  # special-kind params
            KIND[(tag, id(p))] = kind
            self.params.append(p)
            self._param_flat_offsets[p] = (offset, offset + n)
            g = (
                gen
                if kind != "dense"
                else torch.Generator().manual_seed(2000 + 10 * layer_idx + tp_rank + offset)
            )
            flat[offset : offset + n] = torch.randn(n, generator=g)
            offset += n
        self.cpu_param_flat = flat
        self.cpu_params = {
            p: flat[s:e].view_as(p) for p, (s, e) in self._param_flat_offsets.items()
        }
        self.cpu_master = dict(self.cpu_params)
        self.optimizer_layer_idx = optimizer.add_layer_params(layer_idx, self.params)
        self.step = 0
        # engine attributes the exec cache rebinds (a non-None pool must be cleared)
        self.cpu_param_staging_pool = object()
        self.cpu_param_staging_flat = None
        self.cpu_execution_params = {}


def make_optimizer(num_layer):
    return LayerAdam(
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        adamw_mode=True,
        fp32_optimizer_state=True,
        num_layer=num_layer,
        nvme_offload_fraction=0.0,
        offload_dir=None,
        prefetch=True,
        distributed_cfg=SimpleNamespace(
            parallel_strategy="none", mode="single", world_size=1, gradient_average=True
        ),
        per_layer_cpu_adam=False,
    )


def build(tag):
    opt = make_optimizer(len(LAYER_SPECS))
    owners = [FakeOwner(i, opt, spec, t_rank, tag) for i, spec in enumerate(LAYER_SPECS)]
    return opt, owners


# ---------------- sharded side
s_opt, s_owners = build("shard")
engine = SimpleNamespace(
    managed_layers=s_owners,
    layer_optimizer=s_opt,
    unmanaged_params=[],
    gpu_gradient_pipeline=None,
    config=SimpleNamespace(skip_cpu_adam_step=False),
)
sharded = ShardedCPUOptimizer(
    engine,
    dp_group,
    tp_group if T > 1 else None,
    None,
    special_kind=lambda p: {"dense": None, "sum": "sum", "avg": "avg"}[KIND[("shard", id(p))]],
)
assert sharded.usable, sharded.reason
assert sharded.size == D and len(sharded.owners) == len(LAYER_SPECS)
assert sharded.exec_cache, "shared bf16 exec cache should be active"
assert all(o.cpu_param_staging_pool is None for o in s_owners)
for owner in s_owners:
    lo, hi = sharded.bounds[owner.layer_idx][sharded.rank]
    for name, active in (
        ("exp_avg_flat", sharded.exp_avg),
        ("exp_avg_sq_flat", sharded.exp_avg_sq),
    ):
        tensor = getattr(s_opt, name)[owner.optimizer_layer_idx]
        assert tensor is active[owner.layer_idx]
        assert tensor.untyped_storage().nbytes() == (hi - lo) * 4
    for state in s_opt.state.values():
        for moment in state.values():
            assert any(
                moment.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                for buffers in (s_opt.exp_avg_flat, s_opt.exp_avg_sq_flat)
                for tensor in buffers.values()
            )

# ---------------- unsharded reference
r_opt, r_owners = build("ref")


def grad_for(tp, dp, step, mb, layer, pidx, n, kind):
    seed = hash((tp, dp, step, mb, layer, pidx)) & 0x7FFFFFFF
    g = torch.Generator().manual_seed(seed)
    # bf16-rounded like the real D2H gradients
    return torch.randn(n, generator=g).to(torch.bfloat16)


MICRO = 2
failures = 0
for step in range(args.steps):
    divide = step == args.steps - 1  # last step: num_tokens=None style (divide by dp, no scale)
    scale = None if divide else 1.0 / (1000 + 37 * step)

    # sharded: accumulate micro-batches into the shared accumulators
    for owner in s_owners:
        for mb in range(MICRO):
            for pidx, p in enumerate(owner.params):
                n = p.numel()
                src = grad_for(
                    t_rank, d_rank, step, mb, owner.layer_idx, pidx, n, KIND[("shard", id(p))]
                )
                sharded.accumulate(owner, p, src, first=(mb == 0))
    dist.barrier(group=sharded.gloo_dp)
    sharded.reduce_slices(divide_by_size=divide, scale=scale)
    sharded.adam_slices()
    sharded.barrier()

    # reference: full computation, every rank regenerates all gradients it needs
    for owner in r_owners:
        views = {}
        for pidx, p in enumerate(owner.params):
            n, kind = p.numel(), KIND[("ref", id(p))]

            def acc_of(tp, dp):
                a = None
                for mb in range(MICRO):
                    g = grad_for(tp, dp, step, mb, owner.layer_idx, pidx, n, kind).float()
                    a = g.clone() if a is None else a.add_(g)
                return a

            def dp_sum(tp):
                s = torch.add(acc_of(tp, 0), acc_of(tp, 1))
                for d in range(2, D):
                    s.add_(acc_of(tp, d))
                if divide:
                    s.div_(D)
                return s

            if kind == "dense":
                red = dp_sum(t_rank)
            else:
                red = dp_sum(0)
                for t in range(1, T):
                    red = red + dp_sum(t) if T == 2 else red.add_(dp_sum(t))
                if kind == "avg":
                    red = red / T if T > 1 else red
            if scale is not None:
                red.mul_(scale)
            views[p] = red
        r_opt.step_with_grad_views(owner.optimizer_layer_idx, views, owner.cpu_params)

    # compare the complete updated masters (and the shared bf16 H2D source)
    for so, ro in zip(s_owners, r_owners):
        if not torch.equal(so.cpu_param_staging_flat, ro.cpu_param_flat.to(torch.bfloat16)):
            failures += 1
            print(
                f"[rank {G}] MISMATCH step {step} layer {so.layer_idx}: shared bf16 copy != cast(reference master)",
                flush=True,
            )
        for p in so.params:  # execution views must alias the shared bf16 buffer
            assert so.cpu_execution_params[p].data_ptr() >= so.cpu_param_staging_flat.data_ptr()
        for p_s, p_r in zip(so.params, ro.params):
            a, b = so.cpu_params[p_s], ro.cpu_params[p_r]
            if not torch.equal(a, b):
                failures += 1
                bad = (a != b).nonzero().numel()
                print(
                    f"[rank {G}] MISMATCH step {step} layer {so.layer_idx} kind "
                    f"{KIND[('shard', id(p_s))]} diff elems {bad} max {(a - b).abs().max().item():.3e}",
                    flush=True,
                )
    dist.barrier()

    if step == 1:
        # Save active moments, destroy their values, deserialize, and continue.
        # Comparing subsequent updates with the uninterrupted reference catches
        # saving the unused full moments instead of the optimizer's real state.
        stream = io.BytesIO()
        torch.save(s_opt.state_dict(), stream)
        for buffer in (s_opt.exp_avg_flat, s_opt.exp_avg_sq_flat):
            for tensor in buffer.values():
                tensor.fill_(float("nan"))
        for group in s_opt.param_groups:
            group["step"] = 1234
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        s_opt.load_state_dict(saved)
        # Legacy sharded checkpoints recorded unused full-sized moments. They
        # must fail explicitly, rather than silently resetting optimizer state.
        invalid = dict(saved, state_shards={})
        try:
            s_opt.load_state_dict(invalid)
        except ValueError as exc:
            assert "shard layout" in str(exc)
        else:
            raise AssertionError("Missing shard metadata was accepted")
        assert all(g["step"] == step + 1 for g in s_opt.param_groups)

total_fail = torch.tensor(failures)
dist.all_reduce(total_fail)
if G == 0:
    print(
        f"world={W} tp={T} dp={D}: {'PASS' if total_fail.item() == 0 else 'FAIL'} "
        f"({args.steps} steps, {MICRO} microbatches, divide-last-step)",
        flush=True,
    )
sys.exit(0 if total_fail.item() == 0 else 1)
