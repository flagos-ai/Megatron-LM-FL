import torch
from torch import _C
import torch_npu
from torch_npu.npu import _lazy_call, device as device_ctx_manager

from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax
from megatron.core.fusions.fused_softmax import ScaledMaskedSoftmax
from megatron.core.fusions.fused_softmax import ScaledSoftmax
from megatron.core.transformer.enums import AttnMaskType


class ScaledUpperTriangMaskedSoftmax(ScaledUpperTriangMaskedSoftmax):
    @staticmethod
    def apply(input_, scale):
        size = input_.size()
        input_ = input_.view((1,) + tuple(size)).contiguous()
        dummy_mask = torch.zeros(input_.size()).bool().npu()
        output = torch_npu.npu_scaled_masked_softmax(input_, dummy_mask, scale, True)
        return output.view(size).contiguous()


class ScaledMaskedSoftmax(ScaledMaskedSoftmax):
    @staticmethod
    def apply(input_, mask, scale):
        return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, False)

class ScaledSoftmax(ScaledSoftmax):
    @staticmethod
    def apply(input_, scale):
        dummy_mask = torch.zeros(input_.size()).bool().npu()
        return torch_npu.npu_scaled_masked_softmax(input_, dummy_mask, scale, False)

def is_kernel_available(self, mask, b, np, sq, sk):
    return (
        self.scaled_masked_softmax_fusion  # user want to fuse
        and self.input_in_float16  # input must be fp16
        and 32 < sk <= 4096  # sk must be 32 ~ 4096
        and sq % 16 == 0  # sq must be divisor of 16
        and sk % 16 == 0  # sk must be divisor of 16
    )

def forward_fused_softmax(self, input_, mask):
    b, np, sq, sk = input_.size()
    scale = self.scale if self.scale is not None else 1.0

    if self.attn_mask_type == AttnMaskType.causal:
        assert sq == sk, "causal mask is only for self attention"
        # if use upper triangular mask
        return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, True)
    else:
        if mask is not None:
            return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, False)
        else:
            return ScaledSoftmax.apply(input_, scale)