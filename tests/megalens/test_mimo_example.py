# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import torch

from examples.mimo.data.mock import MockVLMDataset, _collate_fn
from examples.mimo import train
from examples.mimo.model_providers import mock as mock_model_provider


def test_mock_data_matches_mimo_modality_schema_and_label_contract() -> None:
    dataset = MockVLMDataset(
        size=1,
        image_size=224,
        seq_len=256,
        image_seq_length=197,
    )

    sample = dataset[0]
    image = sample["modality_inputs"]["images"]["clip_encoder"]["x"]
    masked_labels = sample["labels"][sample["loss_mask"] == 0]

    assert image.shape == (3, 224, 224)
    assert masked_labels.numel() == 197
    assert torch.all(masked_labels == dataset.pad_token_id)

    batch = _collate_fn([sample, sample])
    assert batch["modality_inputs"]["images"]["clip_encoder"]["x"].shape == (
        2,
        3,
        224,
        224,
    )


def test_mock_provider_import_does_not_load_optional_energon_modules() -> None:
    script = """
import sys
from examples.mimo import train

assert train._get_dataset_provider("mock") is not None
assert train._get_model_provider("mock") is not None
assert "examples.mimo.data.energon_avlm_task_encoder" not in sys.modules
assert "examples.mimo.data.energon_vlm_task_encoder" not in sys.modules
"""

    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_model_provider_routes_runtime_image_token_to_mock_builder(monkeypatch) -> None:
    calls = []
    process_groups = object()

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return "mock-model"

    monkeypatch.setattr(
        train,
        "get_args",
        lambda: SimpleNamespace(model_provider="mock", image_token_id=42000),
    )
    monkeypatch.setattr(train, "_get_model_provider", lambda name: builder)

    result = train.model_provider(pg_collection=process_groups)

    assert result == "mock-model"
    assert calls == [
        (
            (True, True, True, True),
            {"special_token_id": 42000, "pg_collection": process_groups},
        )
    ]


def test_mock_builder_preserves_framework_process_group_collection(monkeypatch) -> None:
    calls = []
    process_groups = SimpleNamespace(cp=object(), tp=object())
    layer_spec = SimpleNamespace(submodules=object())

    def build_mimo(config, **kwargs):
        calls.append((config, kwargs))
        return "mock-model"

    monkeypatch.setattr(mock_model_provider, "MimoModel", build_mimo)
    monkeypatch.setattr(
        mock_model_provider, "get_mock_language_layer_spec", lambda: layer_spec
    )
    monkeypatch.setattr(
        mock_model_provider, "get_mock_vision_layer_spec", lambda: layer_spec
    )
    monkeypatch.setattr(
        mock_model_provider, "get_mock_projection_layer_spec", lambda: layer_spec
    )

    result = mock_model_provider.model_provider_mock_vlm_single_encoder(
        pg_collection=process_groups
    )

    assert result == "mock-model"
    assert len(calls) == 1
    assert calls[0][1] == {
        "cp_group": process_groups.cp,
        "tp_group": process_groups.tp,
        "pg_collection": process_groups,
    }
