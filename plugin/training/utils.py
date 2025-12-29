import torch
import logging

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import get_args
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import (
    get_data_parallel_group_if_dtensor,
    to_local_if_dtensor,
)
from megatron.training.utils import calc_dtensor_params_l2_norm
from megatron.legacy.model.module import param_is_not_shared

from plugin.hetero.p2p_communication import get_device_type_for_comm
from plugin.decorators import plugin_implementation
logger = logging.getLogger(__name__)

@plugin_implementation("utils", "calc_params_l2_norm")
def calc_params_l2_norm(model, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    logger.debug(f"Megatron-LM-FL Plugin, calc_params_l2_norm")
    args = get_args()
    if not isinstance(model, list):
        model = [model]

    if getattr(args, 'use_megatron_fsdp', False):
        # All Megatron FSDP parameters are expected to be PyTorch DTensor.
        # params_data is a dict of device_mesh -> list of local tensors.
        params = []
        for model_chunk in model:
            model_chunk.stop_communication()
            for name, param in model_chunk.named_parameters():
                if not hasattr(param, "_local_tensor"):
                    raise RuntimeError(
                        f"Megatron FSDP requires parameters are PyTorch DTensor. "
                        f"Parameter {name} is not a DTensor."
                    )
                params.append(param)

        return calc_dtensor_params_l2_norm(params)

    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, 'allreduce', True):
                # TODO: Implement memory optimization for MoE parameters.
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                moe_params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if args.bf16:
                        if not force_create_fp32_copy and hasattr(param, 'main_param'):
                            if getattr(param, 'main_param_sharded', False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [params_data], False  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')

    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False,  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
    else:
        sharded_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    # Sum over all DP groups, including CP since distributed optimizer state is
    # sharded jointly over DP+CP.
    torch.distributed.all_reduce(
        sharded_norm_2,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_data_parallel_group(with_context_parallel=True)
    )
    norm_2 += sharded_norm_2

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False,  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm

    # Account for MoE norm even if current rank doesn't have any expert params to prevent
    # hang in models with un-even numbers of MoE layers.
    # See details in https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/409
    else:
        moe_norm_2 = torch.zeros_like(norm_2)

    ########## FlagScale Begin ##########
    # Sum across all model-parallel GPUs(tensor + pipeline).
    mp_groups = mpu.get_model_parallel_group()
    comm_device = get_device_type_for_comm(mp_groups)
    if comm_device == "cpu":
        norm_2 = norm_2.cpu()
    if isinstance(mp_groups, list):  # hetero
        original_norm_2 = norm_2.clone().detach()
        for mp_group in mp_groups:
            norm_2.copy_(original_norm_2)
            torch.distributed.all_reduce(
                norm_2, op=torch.distributed.ReduceOp.SUM, group=mp_group
            )
        if len(moe_params_data) > 0:
            emp_groups = mpu.get_expert_tensor_model_pipeline_parallel_group()
            comm_device = get_device_type_for_comm(emp_groups)
            if comm_device == "cpu":
                moe_norm_2 = moe_norm_2.cpu()

            assert isinstance(
                emp_groups, list
            ), "emp_groups should be a list if mp_groups is a list"
            original_norm_2 = moe_norm_2.clone().detach()
            for emp_group in emp_groups:
                moe_norm_2.copy_(original_norm_2)
                torch.distributed.all_reduce(
                    moe_norm_2, op=torch.distributed.ReduceOp.SUM, group=emp_group
                )
            norm_2 += moe_norm_2
    ########## FlagScale End ##########
    else:  # original code

        # Reduce norm across model parallel groups (dense and expert).
        # Dense params should sum across all model-parallel GPUs (tensor + pipeline).
        dense_reduce_group = mpu.get_model_parallel_group()
        ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
        # Expert params should sum across all model-parallel GPUs (expert + tensor + pipeline).
        expert_reduce_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
        ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    # If dense and expert reduce groups are the same, sum then reduce.
    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_norm_2
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
    # If dense and expert reduce groups are different, reduce then sum.
    else:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
        torch.distributed.all_reduce(
            moe_norm_2, op=torch.distributed.ReduceOp.SUM, group=expert_reduce_group
        )
        norm_2 += moe_norm_2

    if comm_device == "cpu":
        norm_2 = norm_2.cuda()
        moe_norm_2 = moe_norm_2.cuda()

    return norm_2.item() ** 0.5
