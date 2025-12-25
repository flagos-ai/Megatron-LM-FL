"""
fl_init branch implementation for LanguageModule._is_in_embd_group.

This file mirrors the path structure of the original file:
- Original: megatron/core/models/common/language_module/language_module.py
- Plugin:   plugins/core/models/common/language_module/language_module.py
"""

import torch
from plugins.decorators import plugin_implementation
from megatron.core.pipeline_parallel.utils import (
    is_vp_first_stage,
    is_vp_last_stage,
    is_pp_first_stage,
    is_pp_last_stage,
)


@plugin_implementation("LanguageModule", "_is_in_embd_group")
def _is_in_embd_group(self):
    """
    fl_init version of _is_in_embd_group.
    
    Supports both single process group and list of process groups
    (for heterogeneous mode).
    """
    print(f"Megatron-LM-FL Plugins: _is_in_embd_group called")
    if self.embd_group is None:
        return False
    
    # Original logic: handle single process group
    if not isinstance(self.embd_group, list):
        if torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(
            self.embd_group
        ):
            if (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group)[0]
            ):
                return is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(
                    self.pp_group
                )
            elif (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group)[-1]
            ):
                return is_vp_last_stage(self.vp_stage, self.vp_size) and is_pp_last_stage(
                    self.pp_group
                )
            else:
                return True
    
    # FlagScale Begin
    else:
        if torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(
            self.embd_group[0]
        ):
            if (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group[0])[0]
            ):
                return is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(
                    self.pp_group
                )
            elif (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group[0])[-1]
            ):
                return is_vp_last_stage(self.vp_stage, self.vp_size) and is_pp_last_stage(
                    self.pp_group
                )
            else:
                return True
    # FlagScale End

    return False

