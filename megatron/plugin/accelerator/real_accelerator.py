# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os


try:
    from accelerator.abstract_accelerator import MegatronAccelerator as mga
except ImportError as e:
    mga = None

SUPPORTED_ACCELERATOR_LIST = ['cuda', 'cpu']

mg_accelerator = None


def is_current_accelerator_supported():
    return get_accelerator().device_name() in SUPPORTED_ACCELERATOR_LIST


def get_accelerator():
    global mg_accelerator
    if mg_accelerator is not None:
        return mg_accelerator

    accelerator_name = None
    mg_set_method = None
    # 1. Detect whether there is override of Megatron-LM-FL accelerators from environment variable.
    if "MG_ACCELERATOR" in os.environ.keys():
        accelerator_name = os.environ["MG_ACCELERATOR"]
        if accelerator_name == "cpu":
            pass
        elif accelerator_name not in SUPPORTED_ACCELERATOR_LIST:
            raise ValueError(f'MG_ACCELERATOR must be one of {SUPPORTED_ACCELERATOR_LIST}. '
                             f'Value "{accelerator_name}" is not supported')
        mg_set_method = "override"

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name is None:
        # We need a way to choose among different accelerator types.
        # Currently we detect which accelerator extension is installed
        # in the environment and use it if the installing answer is True.
        # An alternative might be detect whether CUDA device is installed on
        # the system but this comes with two pitfalls:
        # 1. the system may not have torch pre-installed, so
        #    get_accelerator().is_available() may not work.
        # 2. Some scenario like install on login node (without CUDA device)
        #    and run on compute node (with CUDA device) may cause mismatch
        #    between installation time and runtime.

        if accelerator_name is None:
            try:
                import torch

                # Determine if we are on a GPU or x86 CPU with torch.
                # "torch.cuda.is_available()" provides a stronger guarantee,     #ignore-cuda
                # ensuring that we are free from CUDA initialization errors.
                # While "torch.cuda.device_count() > 0" check ensures that       #ignore-cuda
                # we won't try to do any CUDA calls when no device is available
                # For reference: https://github.com/deepspeedai/DeepSpeed/pull/6810
                if torch.cuda.device_count() > 0 and torch.cuda.is_available():  #ignore-cuda
                    accelerator_name = "cuda"
            except (RuntimeError, ImportError) as e:
                # TODO need a more decent way to detect which accelerator to use, consider using nvidia-smi command for detection
                pass
        if accelerator_name is None:
            # cpu added as catch-all when accelerator detection fails
            accelerator_name = "cpu"

        mg_set_method = "auto detect"

    # 3. Set mg_accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        mg_accelerator = CUDA_Accelerator()
    elif accelerator_name == "cpu":
        from .cpu_accelerator import CPU_Accelerator

        mg_accelerator = CPU_Accelerator()
    
    return mg_accelerator


def set_accelerator(accel_obj):
    global mg_accelerator
    mg_accelerator = accel_obj