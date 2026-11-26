"""Assign one exclusive GPU per worker before importing Megatron."""
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
gpu_ids = os.environ.get("SLIDEFORMER_GPU_IDS", "").split(",")
local_rank = int(os.environ["LOCAL_RANK"])
if not gpu_ids[0] or local_rank >= len(gpu_ids):
    raise ValueError("SLIDEFORMER_GPU_IDS must list one GPU for every local worker")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[local_rank]
os.environ["LOCAL_RANK"] = "0"
sys.path.insert(0, str(root))
sys.argv[0] = str(root / "pretrain_gpt.py")
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except BaseException:
    import traceback
    traceback.print_exc()  # print before destroy_process_group can block on peers
    sys.stderr.flush()
    raise
finally:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
