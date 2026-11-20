import os

from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer


def enable_forward_pre_hook(self: DistributedDataParallel) -> None:
    """Reuse DDP's removable readiness hooks when MC2 reads expert weights directly."""
    DistributedDataParallel.enable_forward_pre_hook.__wrapped__(self)
    if os.getenv("NPU_MOE_MC2", "0") == "1":
        for module in self.module.modules():
            if isinstance(module, MoELayer) and isinstance(module.experts, TEGroupedMLP):
                for linear in (module.experts.linear_fc1, module.experts.linear_fc2):
                    linear._npu_mc2_param_sync_hook_id = self.remove_forward_pre_hook_handles[
                        linear
                    ].id
