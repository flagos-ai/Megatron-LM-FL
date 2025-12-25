import copy
from logging import getLogger
import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding

logger = getLogger(__name__)

from plugin.decorators import plugin_implementation


@plugin_implementation("MixedPrecisionOptimizer", "_unscale_main_grads_and_check_for_nan")
def _unscale_main_grads_and_check_for_nan(self):
    print(f"Megatron-LM-FL Plugins: _unscale_main_grads_and_check_for_nan")
    # Collect main grads.
    if not self.is_stub_optimizer:
        main_grads = self._collect_main_grad_data_for_unscaling()

    # Reset found inf.
    self.found_inf.fill_(0.0)

    if not self.is_stub_optimizer:
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )

    # Update across all model parallel instances.
    groups = self.get_grad_stats_parallel_group()
    if isinstance(groups, list):
        if "cpu:gloo" == torch.distributed.get_backend(groups[0]):
            self.found_inf = self.found_inf.cpu()
    else:
        if "cpu:gloo" == torch.distributed.get_backend(groups):
            self.found_inf = self.found_inf.cpu()
    if not isinstance(groups, list):
        groups = [groups]
    for group in groups:
        torch.distributed.all_reduce(
            self.found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=group
        )
    if self.found_inf.device != torch.device('cuda'):
        self.found_inf = self.found_inf.cuda()

    # Check for nan.
    found_inf_flag = self.found_inf.item() > 0

    return found_inf_flag

@plugin_implementation("ChainedOptimizer", "sharded_state_dict")
def sharded_state_dict(
    self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, convert_to_ep: bool = False, **kwargs
):
    print(f"Megatron-LM-FL Plugins: sharded_state_dict")
    self.convert_to_ep = convert_to_ep ########## FlagScale Add ########
    metadata = kwargs.get('metadata') or {}
    # ChainedOptimizer should add its prefix to the tensor state keys only if
    # DistributedOptimizer is used (non-empty 'distrib_optim_sharding_type') and uses
    # a non fully-reshardable format. For backward compatibility we also add it
    # if `chained_optim_avoid_prefix` is False.
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    should_add_prefix = (
        "distrib_optim_sharding_type" in metadata
        and metadata["distrib_optim_sharding_type"]
        not in DistributedOptimizer.checkpoint_fully_reshardable_formats
    ) or not metadata.get('chained_optim_avoid_prefix', False)

    if len(self.chained_optimizers) == 1:
        return self.chained_optimizers[0].sharded_state_dict(
            model_sharded_state_dict, is_loading, **kwargs
        )
    else:
        ######### FlagScale Begin #########
        if convert_to_ep and is_loading:
            logger.info(
                "sharded_state_dict:convert tp/pp chained_optimizers to ep chained_optimizers!"
                )
            # convert tp/pp chained_optimizers to ep chained_optimizers
            sharded_state_dict = {}
            fake_sharded_state_dict = {
                'optimizer': {},
                'param_state': {},
                'param_state_sharding_type': {},
            }
            global_pid = 0
            mapping_idx = {}
            for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
                optim_state_dict = optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                tmp_dict = {}
                for local_pid, v in optim_state_dict['param_state'].items():
                    fake_sharded_state_dict['param_state'][global_pid] = v
                    tmp_dict[local_pid] = global_pid
                    global_pid += 1
                mapping_idx[optimizer_idx] = tmp_dict

                fake_sharded_state_dict['optimizer'] = optim_state_dict['optimizer']
                fake_sharded_state_dict['param_state_sharding_type'] = optim_state_dict[
                    'param_state_sharding_type'
                ]

                sharded_state_dict[optimizer_idx] = {}
                sharded_state_dict[optimizer_idx]['optimizer'] = copy.deepcopy(
                    optim_state_dict['optimizer']
                )
                sharded_state_dict[optimizer_idx]['param_state_sharding_type'] = copy.deepcopy(
                    optim_state_dict['param_state_sharding_type']
                )
                sharded_state_dict[optimizer_idx]['len_param_state'] = len(
                    optim_state_dict['param_state']
                )
                self.original_sharded_state_dict = sharded_state_dict
            self.mapping_idx = mapping_idx
            return fake_sharded_state_dict
        ######### FlagScale End #########
        else: # megatron source apply ep
            self._synchronize_steps()
            sharded_state_dict = {}
            for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
                optim_state_dict = optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                if should_add_prefix:
                    add_prefix_for_sharding(optim_state_dict, f'chained_{optimizer_idx}.')
                sharded_state_dict[optimizer_idx] = optim_state_dict
            return sharded_state_dict

@plugin_implementation("ChainedOptimizer", "load_state_dict")
def load_state_dict(self, state_dict):
    print(f"Megatron-LM-FL Plugins: load_state_dict")
    if self.convert_to_ep:  # convert tp/pp chained_optimizers to ep chained_optimizers
        logger.info(
            "load_state_dict:convert tp/pp chained_optimizers to ep chained_optimizers!"
        )
        new_state_dict = {}
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            new_state_dict[optimizer_idx] = {}
            new_state_dict[optimizer_idx]['optimizer'] = self.original_sharded_state_dict[
                optimizer_idx
            ]['optimizer']
            new_state_dict[optimizer_idx]['param_state_sharding_type'] = (
                self.original_sharded_state_dict[optimizer_idx]['param_state_sharding_type']
            )
            len_param_state = self.original_sharded_state_dict[optimizer_idx]['len_param_state']
            new_state_dict[optimizer_idx]['param_state'] = {}
            for i in range(len_param_state):
                new_state_dict[optimizer_idx]['param_state'][i] = state_dict['param_state'][
                    self.mapping_idx[optimizer_idx][i]
                ]
        if len(self.chained_optimizers) != len(new_state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(new_state_dict)}.'
            )
        if isinstance(new_state_dict, dict):
            new_state_dict = (v for k, v in sorted(new_state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, new_state_dict):
            optimizer.load_state_dict(state)
    else:  # megatron source apply ep
        # If there is only one optimizer, we read the state dict as a single optimizer.
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].load_state_dict(state_dict)
            return
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict(state)
        self._synchronize_steps()
