# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from megatron.training import arguments
from megatron.training import yaml_arguments


def test_pattern_argument_helpers_parse_ints_tuples_and_list_expressions():
    assert arguments.no_rope_freq_type(None) is None
    assert arguments.no_rope_freq_type(4) == 4
    assert arguments.no_rope_freq_type("4") == 4
    assert arguments.no_rope_freq_type("[0,1]*2") == [0, 1, 0, 1]

    assert arguments.moe_freq_type(2) == 2
    assert arguments.moe_freq_type("2") == 2
    assert arguments.moe_freq_type("([1]+[0])*2") == [1, 0, 1, 0]

    assert arguments.la_freq_type(None) is None
    assert arguments.la_freq_type(3) == 3
    assert arguments.la_freq_type("3") == 3
    assert arguments.la_freq_type("[1,0,0]") == [1, 0, 0]

    assert arguments.tuple_type(None) is None
    assert arguments.tuple_type((1, 2)) == (1, 2)
    assert arguments.tuple_type("(1,2,3)") == (1, 2, 3)


def test_pattern_argument_helpers_reject_unsafe_expression():
    with pytest.raises(ValueError, match="Invalid pattern"):
        arguments.moe_freq_type("[__import__('os').system('echo unsafe')]")


def test_add_megatron_arguments_parses_representative_core_options():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments.add_megatron_arguments(parser)

    args = parser.parse_args(
        [
            "--num-layers",
            "4",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "8",
            "--group-query-attention",
            "--num-query-groups",
            "4",
            "--micro-batch-size",
            "2",
            "--global-batch-size",
            "8",
            "--seq-length",
            "1024",
            "--max-position-embeddings",
            "1024",
            "--train-iters",
            "5",
            "--lr",
            "0.001",
            "--min-lr",
            "0.0001",
            "--lr-decay-style",
            "cosine",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--data-path",
            "dataset-prefix",
            "--split",
            "90,5,5",
            "--bf16",
            "--normalization",
            "RMSNorm",
            "--position-embedding-type",
            "rope",
            "--window-size",
            "(128,256)",
            "--window-attn-skip-freq",
            "2",
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
            "--fp8-param-gather",
            "--te-precision-config-file",
            "precision.yaml",
            "--inference-max-requests",
            "4",
            "--cuda-graph-scope",
            "attn",
            "mlp",
            "--wandb-project",
            "coverage",
        ]
    )

    assert args.num_layers == 4
    assert args.hidden_size == 128
    assert args.num_attention_heads == 8
    assert args.group_query_attention is True
    assert args.num_query_groups == 4
    assert args.lr_decay_style == "cosine"
    assert args.tokenizer_type == "GPT2BPETokenizer"
    assert args.position_embedding_type == "rope"
    assert args.window_size == (128, 256)
    assert args.window_attn_skip_freq == 2
    assert args.use_distributed_optimizer is True
    assert args.overlap_grad_reduce is True
    assert args.overlap_param_gather is True
    assert args.fp8_param_gather is True
    assert args.te_precision_config_file == "precision.yaml"
    assert [scope.name for scope in args.cuda_graph_scope] == ["attn", "mlp"]


def test_parse_args_accepts_extra_provider_and_environment_rank(monkeypatch):
    def extra_args_provider(parser):
        parser.add_argument("--custom-coverage-flag", default="unset")
        return parser

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--disable-msc", "--custom-coverage-flag", "enabled"],
    )

    with (
        patch.object(arguments.MultiStorageClientFeature, "disable") as mock_disable,
        patch.object(arguments.MultiStorageClientFeature, "is_enabled", return_value=False),
        patch.object(arguments, "warn_rank_0") as mock_warn_rank_0,
    ):
        parsed_args = arguments.parse_args(
            extra_args_provider=extra_args_provider,
            ignore_unknown_args=False,
        )

    assert parsed_args.rank == 3
    assert parsed_args.world_size == 8
    assert parsed_args.custom_coverage_flag == "enabled"
    assert parsed_args.enable_msc is False
    mock_disable.assert_called_once()
    mock_warn_rank_0.assert_called_once()


def test_yaml_env_constructor_and_load_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/tmp/data")

    env_loaded = yaml.load("data_path: ${DATA_ROOT}/dataset", Loader=yaml.Loader)
    assert env_loaded["data_path"] == "/tmp/data/dataset"

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "data_path": "/tmp/data/dataset",
                "model_parallel": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                },
            }
        )
    )

    loaded = yaml_arguments.load_yaml(yaml_path)

    assert loaded.yaml_cfg == yaml_path
    assert loaded.data_path == "/tmp/data/dataset"
    assert loaded.model_parallel.tensor_model_parallel_size == 2


def test_yaml_env_constructor_requires_existing_environment_variable(monkeypatch):
    monkeypatch.delenv("MISSING_DATA_ROOT", raising=False)

    with pytest.raises(AssertionError, match="environment variable MISSING_DATA_ROOT"):
        yaml.load("data_path: ${MISSING_DATA_ROOT}/dataset", Loader=yaml.Loader)


def test_yaml_helpers_validate_required_fields_and_core_config_mapping():
    @dataclass
    class TinyConfig:
        hidden_size: int = 16
        num_layers: int = 2

    args = SimpleNamespace(hidden_size=32, num_layers=4)
    assert yaml_arguments.core_config_from_args(args, TinyConfig) == {
        "hidden_size": 32,
        "num_layers": 4,
    }

    with pytest.raises(Exception, match="Missing argument num_layers"):
        yaml_arguments.core_config_from_args(SimpleNamespace(hidden_size=32), TinyConfig)

    yaml_arguments._check_arg_is_not_none(SimpleNamespace(foo="bar"), "foo")
    with pytest.raises(AssertionError, match="foo argument is None"):
        yaml_arguments._check_arg_is_not_none(SimpleNamespace(foo=None), "foo")
