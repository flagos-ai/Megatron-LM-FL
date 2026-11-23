"""Regression: TP replicas must not multiply the eager optimizer's token count.

Run with torchrun on four idle GPUs and SLIDEFORMER_GPU_IDS set to their IDs.
Uses real TP2/DP2 collectives for source-only and distributed-aware iterators.
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

visible = os.environ["SLIDEFORMER_GPU_IDS"].split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = visible[int(os.environ["LOCAL_RANK"])]
os.environ["LOCAL_RANK"] = "0"
os.environ["MEGATRON_SLIDEFORMER_EAGER_SHARDED_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
import megatron.training.training as training

torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
assert dist.get_world_size() == 4
ps.initialize_model_parallel(tensor_model_parallel_size=2)
training.get_args = lambda: SimpleNamespace(
    calculate_per_token_loss=True,
    sft=False,
    context_parallel_size=1,
    pipeline_model_parallel_size=1,
)
training.get_num_microbatches = lambda: 2
engine = SimpleNamespace(_sharded_opt=object())
engine.set_expected_scale = lambda scale: setattr(engine, "scale", scale)
model = [SimpleNamespace(_megatron_slideformer_engine=engine)]

try:
    for layout in ("source_only", "replicated", "non_source_differs"):
        source = ps.get_tensor_model_parallel_rank() == 0
        # DP replica 0: two valid tokens per microbatch; replica 1: three.
        # Two microbatches -> ten global valid tokens, irrespective of TP.
        valid = 2 + ps.get_data_parallel_rank()
        if layout == "non_source_differs" and not source:
            valid = 1  # Source broadcast must take precedence over this count.
        batches = [{"loss_mask": torch.arange(3) < valid, "id": i} for i in range(3)]
        iterator = None if layout == "source_only" and not source else iter(batches)
        prepared = training._prepare_eager_sharded_update(model, iterator)
        assert engine.scale == 0.1, (layout, dist.get_rank(), engine.scale)
        if iterator is None:
            assert prepared is None
        else:
            assert [batch["id"] for batch in prepared] == [0, 1]
            assert next(iterator)["id"] == 2
        print(
            f"EAGER_TOKEN_PARITY rank={dist.get_rank()} layout={layout} scale={engine.scale}",
            flush=True,
        )
finally:
    ps.destroy_model_parallel()
    dist.destroy_process_group()
