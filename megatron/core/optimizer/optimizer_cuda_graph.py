# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph wrapper for ADAM optimizer."""

import logging

import torch

from megatron.core.full_cuda_graph import get_graph_pool, get_shared_capture_stream
from megatron.plugin.platform import get_platform

cur_platform = get_platform()

logger = logging.getLogger(__name__)


class OptimizerCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = 0
    cuda_graph = None
    result = None  # result of the optimizer.step() function

    def __init__(self, optimizer_step_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.optimizer_step_func = optimizer_step_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'optimizer.step() does not accept positional args'
        assert len(kwargs) == 0, 'optimizer.step() does not accept keyword args'

        curr_iteration = self.curr_iter()
        if curr_iteration == self.cuda_graph_warmup_steps:
            assert OptimizerCudaGraphWrapper.cuda_graph is None
            OptimizerCudaGraphWrapper.cuda_graph = cur_platform.create_graph()
            if OptimizerCudaGraphWrapper.cuda_graph is not None:
                logger.info(f'Capture CUDA graph for optimizer!!!')
                torch.distributed.barrier()
                cur_platform.synchronize()
                capture_stream = get_shared_capture_stream()
                with cur_platform.capture_to_graph(
                    OptimizerCudaGraphWrapper.cuda_graph,
                    stream=capture_stream,
                    pool=get_graph_pool(self.use_single_mempool),
                ):
                    OptimizerCudaGraphWrapper.result = self.optimizer_step_func()
                cur_platform.synchronize()
                torch.distributed.barrier()
                logger.info(f'Optimizer CUDA graph capture done!!!')
        if OptimizerCudaGraphWrapper.cuda_graph is None:
            OptimizerCudaGraphWrapper.result = self.optimizer_step_func()
        else:
            cur_platform.replay_graph(OptimizerCudaGraphWrapper.cuda_graph)
        OptimizerCudaGraphWrapper.curr_iteration += 1
        return OptimizerCudaGraphWrapper.result

    def curr_iter(self):
        """Return current training iteration."""
        return OptimizerCudaGraphWrapper.curr_iteration

    def next_iter(self):
        """Increment current training iteration."""
        OptimizerCudaGraphWrapper.curr_iteration += 1

    def __del__(self):
        logger.info(f"Destructor called for {type(self.optimizer_step_func).__name__} optimizer!!!")
        if OptimizerCudaGraphWrapper.cuda_graph is not None:
            del OptimizerCudaGraphWrapper.cuda_graph
            OptimizerCudaGraphWrapper.cuda_graph = None
        if OptimizerCudaGraphWrapper.result is not None:
            OptimizerCudaGraphWrapper.result = None
