# Copyright (c) 2026, BAAI. All rights reserved.
"""Audio modality submodules: single output projection enforced; decoding rejected until implemented."""
import pytest
import torch
from torch import nn

from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules


def test_more_than_one_output_projection_is_rejected():
    with pytest.raises(AssertionError, match="only one output projection"):
        AudioModalitySubmodules(output_projections=[nn.Linear(4, 4), nn.Linear(4, 4)])


def test_decode_raises_not_implemented():
    submodules = AudioModalitySubmodules()
    with pytest.raises(NotImplementedError, match="Audio decoding not implemented yet"):
        submodules.decode(torch.zeros(1, 4), {})
