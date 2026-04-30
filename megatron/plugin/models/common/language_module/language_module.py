import logging

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.models.common.language_module.language_module import LanguageModule

from megatron.plugin.decorators import override
from megatron.plugin.platform import get_platform
cur_platform = get_platform()

logger = logging.getLogger(__name__)


@override("LanguageModule", "_is_in_embd_group")
def _is_in_embd_group(self):
    """
    Plugin implementation of _is_in_embd_group.
    
    Supports both single process group and list of process groups
    (for heterogeneous mode).
    """
    logger.debug(f"Megatron-LM-FL Plugins: _is_in_embd_group")
    if self.embd_group is None:
        return False
    
    # Original logic: handle single process group
    if not isinstance(self.embd_group, list):
        if torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(
            self.embd_group
        ):
            if getattr(self, 'mtp_process', False):
                return True
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
            if getattr(self, 'mtp_process', False):
                return True
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


@override("LanguageModule", "setup_embeddings_and_output_layer")
def setup_embeddings_and_output_layer(self) -> None:
    """Sets up embedding layer in first stage and output layer in last stage.

    This function initalizes word embeddings in the final stage when we are
    using pipeline parallelism and sharing word embeddings, and sets up param
    attributes on the embedding and output layers.

    Parameter attributes set:
    - `is_embedding_or_output_parameter`: True for embedding + output layer weights.
      Used by decoupled_lr, Muon optimizer, and other Megatron features.
    - `is_embedding_parameter`: True for MuP "embedding-class" parameters.
      Used by MuP for table-8 style optimizer grouping (base LR/eps for vector-like params).
    """
    logger.debug(f"Megatron-LM-FL Plugins: setup_embeddings_and_output_layer")

    # Mark embedding and output layer for decoupled_lr and other features.
    if self.pre_process and hasattr(self, 'embedding'):
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    if (
        self.post_process
        and hasattr(self, 'output_layer')
        and self.output_layer.weight is not None
    ):
        self.output_layer.weight.is_embedding_or_output_parameter = True

    # Mark embedding-class parameters for MuP optimizer grouping.
    mtp_process = getattr(self, 'mtp_process', False)
    if self.config.use_mup and (self.pre_process or mtp_process) and hasattr(self, 'embedding'):
        for param in self.embedding.parameters():
            param.is_embedding_parameter = True
    if (
        self.config.use_mup
        and self.post_process
        and hasattr(self, 'output_layer')
        and self.output_layer.weight is not None
    ):
        self.output_layer.weight.is_embedding_parameter = True

    # If share_embeddings_and_output_weights is True, we need to maintain duplicated
    # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
    # we also need to maintain duplicated embedding weights in mtp process stage.
    # So we need to copy embedding weights from pre processing stage as initial parameters
    # in these cases.
    if not self.share_embeddings_and_output_weights and not getattr(
        self.config, 'mtp_num_layers', 0
    ):
        return

    # if self.config.pipeline_model_parallel_size == 1: # original code of Megatron
    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        self.shared_embedding_or_output_weight().zero_out_wgrad = True
        return

    if (
        is_vp_first_stage(self.vp_stage, self.vp_size)
        and is_pp_first_stage(self.pp_group)
        and self.pre_process
        and not self.post_process
    ):
        self.shared_embedding_or_output_weight().shared_embedding = True

    if ((self.post_process and self.share_embeddings_and_output_weights) or getattr(self, 'mtp_process', False)) and not self.pre_process:
        assert not (
            is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(self.pp_group)
        )
        # set weights of the duplicated embedding to 0 here,
        # then copy weights from pre processing stage using all_reduce below.
        weight = self.shared_embedding_or_output_weight()
        weight.data.fill_(0)
        weight.shared = True
        weight.shared_embedding = True
        # Keep optimizer grouping consistent for tied embedding/output copies.
        if self.config.use_mup:
            weight.is_embedding_parameter = True

    # Parameters are shared between the word embeddings layers, and the
    # heads at the end of the model. In a pipelined setup with more than
    # one stage, the initial embedding layer and the head are on different
    # workers, so we do the following:
    # 1. Create a second copy of word_embeddings on the last stage, with
    #    initial parameters of 0.0.
    # 2. Do an all-reduce between the first and last stage to ensure that
    #    the two copies of word_embeddings start off with the same
    #    parameter values.
    # 3. In the training loop, before an all-reduce between the grads of
    #    the two word_embeddings layers to ensure that every applied weight
    #    update is the same on both stages.

    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if self._is_in_embd_group() and not self.config.init_model_with_meta_device:
            weight = self.shared_embedding_or_output_weight()
            weight.data = weight.data.to(cur_platform.device())
            embedding_group = self.embd_group
            if not isinstance(embedding_group, list):
                torch.distributed.all_reduce(weight.data, group=self.embd_group)
            else: # for multiple embedding groups in heterogeneous mode
                with torch.no_grad():
                    original_dtype = weight.dtype
                    if (original_dtype == torch.bfloat16) and torch.distributed.get_backend(group=embedding_group[0])=="cpu:gloo": # gloo backend doesn't support bfloat16
                        weight = weight.to(torch.float32)
                        weight.data = weight.data.cpu()
                    original_weight = weight.clone().detach().data
                    for group in embedding_group:
                        weight.data.copy_(original_weight)
                        torch.distributed.all_reduce(weight.data, group=group)
                    if original_dtype != weight.dtype:
                        weight = weight.to(original_dtype)
                        weight.data = weight.data.to(cur_platform.device())

    elif not getattr(LanguageModule, "embedding_warning_printed", False):
        logging.getLogger(__name__).warning(
            "Distributed processes aren't initialized, so the output layer "
            "is not initialized with weights from the word embeddings. "
            "If you are just manipulating a model this is fine, but "
            "this needs to be handled manually. If you are training "
            "something is definitely wrong."
        )
        LanguageModule.embedding_warning_printed = True