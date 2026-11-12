# Copyright (c) 2026, BAAI. All rights reserved.
"""LanguageModule: fused cross-entropy parity, attention-backend env setup, shared embedding setup, tied state dict."""

import importlib
import logging
import os
import sys

import pytest
import torch
import torch.distributed

import megatron.core.extensions.transformer_engine as te_extensions
import megatron.core.models.common.language_module.language_module as language_module
from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def nvte_auto_env(monkeypatch):
    monkeypatch.setenv('NVTE_FLASH_ATTN', '1')
    monkeypatch.setenv('NVTE_FUSED_ATTN', '1')
    monkeypatch.setenv('NVTE_UNFUSED_ATTN', '1')


@pytest.fixture
def restore_language_module():
    yield
    importlib.reload(language_module)


def _config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _build_gpt(config, **kwargs):
    values = dict(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=8,
    )
    values.update(kwargs)
    return GPTModel(**values)


def test_language_module_falls_back_to_none_cross_entropy_when_te_import_fails(
    monkeypatch, restore_language_module
):
    with monkeypatch.context() as blocker:
        blocker.setitem(sys.modules, 'megatron.core.extensions.transformer_engine', None)
        importlib.reload(language_module)
        assert language_module.te_parallel_cross_entropy is None

    importlib.reload(language_module)
    assert language_module.te_parallel_cross_entropy is te_extensions.te_parallel_cross_entropy


def test_compute_language_model_loss_te_fusion_matches_vocabulary_parallel_loss(
    model_parallel, nvte_auto_env
):
    model_parallel()
    config = _config(
        cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te', tensor_model_parallel_size=1
    )
    module = LanguageModule(config)
    module.pre_process = True
    module.post_process = False
    device = torch.cuda.current_device()
    torch.manual_seed(1234)
    logits = torch.randn(4, 2, 128, device=device)
    labels = torch.randint(0, 128, (2, 4), device=device)

    fused_loss = module.compute_language_model_loss(labels, logits.clone())
    config.cross_entropy_loss_fusion = False
    reference_loss = module.compute_language_model_loss(labels, logits.clone())

    assert fused_loss.shape == (2, 4)
    torch.testing.assert_close(fused_loss, reference_loss, rtol=1e-5, atol=1e-6)


def test_compute_language_model_loss_native_fusion_matches_vocabulary_parallel_loss(
    model_parallel, nvte_auto_env
):
    model_parallel()
    config = _config(
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='native',
        tensor_model_parallel_size=1,
    )
    module = LanguageModule(config)
    module.pre_process = True
    module.post_process = False
    device = torch.cuda.current_device()
    torch.manual_seed(1234)
    logits = torch.randn(4, 2, 128, device=device)
    labels = torch.randint(0, 128, (2, 4), device=device)

    fused_loss = module.compute_language_model_loss(labels, logits.clone())
    config.cross_entropy_loss_fusion = False
    reference_loss = module.compute_language_model_loss(labels, logits.clone())

    assert fused_loss.shape == (2, 4)
    torch.testing.assert_close(fused_loss, reference_loss, rtol=1e-5, atol=1e-6)


def test_compute_language_model_loss_raises_for_cg_capturable_with_te_below_2_7(
    model_parallel, nvte_auto_env, monkeypatch
):
    model_parallel()
    config = _config(
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
        cuda_graph_impl='full_iteration',
        tensor_model_parallel_size=1,
    )
    module = LanguageModule(config)
    module.pre_process = True
    module.post_process = False
    torch.manual_seed(1234)
    logits = torch.randn(4, 2, 128)
    labels = torch.randint(0, 128, (2, 4))

    monkeypatch.setattr(language_module, 'is_te_min_version', lambda *args, **kwargs: False)
    with pytest.raises(AssertionError, match='CUDA graph compatible cross entropy requires'):
        module.compute_language_model_loss(labels, logits)


def test_compute_language_model_loss_raises_when_te_block_missing(
    model_parallel, nvte_auto_env, monkeypatch
):
    model_parallel()
    config = _config(
        cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te', tensor_model_parallel_size=1
    )
    module = LanguageModule(config)
    module.pre_process = True
    module.post_process = False
    logits = torch.randn(4, 2, 128)
    labels = torch.randint(0, 128, (2, 4))

    monkeypatch.setattr(language_module, 'te_parallel_cross_entropy', None)
    with pytest.raises(RuntimeError, match="Trying to use a TE block when it's not present"):
        module.compute_language_model_loss(labels, logits)


def test_scale_logits_multiplies_by_mup_output_mult_when_mup_enabled(model_parallel, nvte_auto_env):
    model_parallel()
    module = LanguageModule(_config(use_mup=True, mup_output_mult=2.0))
    logits = torch.randn(4, 2, 128)

    scaled = module._scale_logits(logits)

    torch.testing.assert_close(scaled, logits * 2.0)


def test_scale_logits_returns_logits_unchanged_when_mup_output_mult_is_one(
    model_parallel, nvte_auto_env
):
    model_parallel()
    module = LanguageModule(_config(use_mup=True, mup_output_mult=1.0))
    logits = torch.randn(4, 2, 128)

    assert module._scale_logits(logits) is logits


def test_shared_embedding_or_output_weight_returns_none_without_pre_or_post(
    model_parallel, nvte_auto_env
):
    model_parallel()
    module = LanguageModule(_config())
    module.pre_process = False
    module.post_process = False

    assert module.shared_embedding_or_output_weight() is None


def test_local_attention_backend_clears_te_env_variables(model_parallel, monkeypatch):
    model_parallel()
    for name in ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN'):
        monkeypatch.delenv(name, raising=False)

    LanguageModule(_config(attention_backend=AttnBackend.local))

    assert os.environ['NVTE_FLASH_ATTN'] == '0'
    assert os.environ['NVTE_FUSED_ATTN'] == '0'
    assert os.environ['NVTE_UNFUSED_ATTN'] == '0'


def test_fused_attention_backend_sets_fused_env_variable(model_parallel, monkeypatch):
    model_parallel()
    for name in ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN'):
        monkeypatch.delenv(name, raising=False)

    LanguageModule(_config(attention_backend=AttnBackend.fused))

    assert os.environ['NVTE_FLASH_ATTN'] == '0'
    assert os.environ['NVTE_FUSED_ATTN'] == '1'
    assert os.environ['NVTE_UNFUSED_ATTN'] == '0'


def test_setup_embeddings_zeroes_wgrad_when_sharing_on_single_stage(model_parallel, nvte_auto_env):
    model_parallel()
    model = _build_gpt(_config(), share_embeddings_and_output_weights=True)

    assert model.shared_embedding_or_output_weight().zero_out_wgrad is True


def test_setup_embeddings_syncs_shared_weight_between_first_and_last_stage(
    model_parallel, nvte_auto_env
):
    model_parallel(pipeline_model_parallel_size=2)
    config = _config(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16, use_mup=True)
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = _build_gpt(
        config,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=True,
    )
    weight = model.shared_embedding_or_output_weight()

    if pre_process:
        assert weight.shared_embedding is True
        assert weight.is_embedding_or_output_parameter is True
        assert weight.is_embedding_parameter is True
        assert torch.count_nonzero(weight).item() > 0
    else:
        assert weight.shared is True
        assert weight.shared_embedding is True
        assert weight.is_embedding_parameter is True
        rank_in_embd_group = (
            torch.distributed.get_rank()
            in torch.distributed.get_process_group_ranks(model.embd_group)
        )
        if rank_in_embd_group:
            assert torch.count_nonzero(weight).item() > 0
        else:
            assert torch.count_nonzero(weight).item() == 0


def test_setup_embeddings_all_reduces_over_listed_heterogeneous_groups(
    model_parallel, nvte_auto_env
):
    model_parallel(pipeline_model_parallel_size=2)
    rank = torch.distributed.get_rank()
    shared_group = torch.distributed.new_group(ranks=[0, 4, 5], backend='cpu:gloo')
    singleton_groups = [
        torch.distributed.new_group(ranks=[r], backend='cpu:gloo') for r in range(8)
    ]
    embd_list = [shared_group] if rank in (0, 4, 5) else [singleton_groups[rank]]
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = parallel_state.get_tensor_model_parallel_group()
    pg_collection.cp = parallel_state.get_context_parallel_group()
    pg_collection.pp = parallel_state.get_pipeline_model_parallel_group()
    pg_collection.embd = embd_list
    config = _config(
        pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16, params_dtype=torch.bfloat16
    )
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()

    model = _build_gpt(
        config,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=True,
        pg_collection=pg_collection,
    )

    assert model.embd_group is embd_list
    if rank in (0, 4, 5):
        assert model._is_in_embd_group() is True
    elif rank in (6, 7):
        assert model._is_in_embd_group() is False
    weight = model.shared_embedding_or_output_weight()
    # First-stage ranks hold the initialized word embeddings regardless of embd-group membership.
    # The last-stage ranks 4/5 are not asserted: language_module.py rebinds `weight` for the
    # bf16 + cpu:gloo all-reduce, so the synchronized copy never reaches the module parameter.
    if pre_process:
        assert torch.count_nonzero(weight).item() > 0
    elif rank in (6, 7):
        assert torch.count_nonzero(weight).item() == 0
    assert weight.dtype == torch.bfloat16


def test_setup_embeddings_warns_when_distributed_is_not_initialized(
    model_parallel, nvte_auto_env, monkeypatch, caplog
):
    # PP=2 is needed to reach the embedding sync (a single stage returns early). With
    # torch.distributed reported uninitialized the pipeline-rank helpers answer rank 0, so every
    # rank builds the first-stage half; a last-stage half would trip the stage-consistency assert.
    model_parallel(pipeline_model_parallel_size=2)
    # Reset and read the flag on the class the module namespace resolves: earlier reloads in
    # this file rebind language_module.LanguageModule, and setup_embeddings_and_output_layer
    # sets the flag on whatever class its own globals resolve to.
    monkeypatch.setattr(
        language_module.LanguageModule, 'embedding_warning_printed', False, raising=False
    )
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    config = _config(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)

    with caplog.at_level(logging.WARNING):
        model = _build_gpt(
            config, pre_process=True, post_process=False, share_embeddings_and_output_weights=True
        )

    assert "Distributed processes aren't initialized" in caplog.text
    assert language_module.LanguageModule.embedding_warning_printed is True
    assert model.shared_embedding_or_output_weight() is model.embedding.word_embeddings.weight


def test_tie_embeddings_state_dict_on_pp2_stages(model_parallel, nvte_auto_env):
    model_parallel(pipeline_model_parallel_size=2)
    config = _config(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = _build_gpt(
        config,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=True,
    )

    sharded_state_dict = model.sharded_state_dict()

    if pre_process:
        assert 'output_layer.weight' not in sharded_state_dict
        assert 'embedding.word_embeddings.weight' in sharded_state_dict
    else:
        tied = sharded_state_dict['output_layer.weight']
        assert tied.key == 'embedding.word_embeddings.weight'
        assert tied.replica_id == (
            1,
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )
