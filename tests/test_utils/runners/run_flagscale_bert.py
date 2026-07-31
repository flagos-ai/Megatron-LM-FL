# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run the stock BERT pretraining path through FlagScale's generic runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_ROOT_TEXT = str(_REPOSITORY_ROOT)
sys.path[:] = [
    entry for entry in sys.path if entry != _REPOSITORY_ROOT_TEXT
]
sys.path.insert(0, _REPOSITORY_ROOT_TEXT)

def add_flagscale_runner_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Consume the run-directory argument injected by FlagScale."""

    parser.add_argument("--straggler-log-dir", type=str, default=None)
    return parser


def main() -> None:
    """Delegate training to the unchanged BERT providers and pretrain loop."""

    from megatron.core.enums import ModelType
    from megatron.training import pretrain
    from pretrain_bert import (
        forward_step,
        model_provider,
        train_valid_test_datasets_provider,
    )

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_flagscale_runner_arguments,
        args_defaults={"tokenizer_type": "BertWordPieceLowerCase"},
    )


if __name__ == "__main__":
    main()
