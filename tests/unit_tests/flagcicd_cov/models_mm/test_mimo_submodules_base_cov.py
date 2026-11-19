# Copyright (c) 2026, BAAI. All rights reserved.
"""Modality submodules: spec-driven encoder/decoder/projection construction per pipeline stage, embedding encoding and projection."""
import pytest
import torch
from torch import nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec


def _vision_spec(with_decoders=False, with_input_projection=True, with_output_projection=False):
    submodules = {
        'encoders': {
            'enc': ModuleSpec(module=nn.Linear, params={'in_features': 8, 'out_features': 8})
        }
    }
    if with_decoders:
        submodules['decoders'] = {
            'dec': ModuleSpec(module=nn.Linear, params={'in_features': 8, 'out_features': 8})
        }
    if with_input_projection:
        submodules['input_projections'] = [
            ModuleSpec(module=nn.Linear, params={'in_features': 8, 'out_features': 6})
        ]
    if with_output_projection:
        submodules['output_projections'] = [
            ModuleSpec(module=nn.Linear, params={'in_features': 6, 'out_features': 8})
        ]
    return ModuleSpec(module=VisionModalitySubmodules, submodules=submodules)


def test_from_spec_builds_decoders_when_present():
    spec = _vision_spec(with_decoders=True)
    submodules = VisionModalitySubmodules.from_spec(spec, is_first_stage=True, is_last_stage=True)

    assert set(submodules.encoders.keys()) == {'enc'}
    assert set(submodules.decoders.keys()) == {'dec'}
    assert len(submodules.input_projections) == 1
    assert len(submodules.output_projections) == 0


def test_from_spec_builds_output_projections_only_on_first_stage():
    spec = _vision_spec(with_output_projection=True)
    first = VisionModalitySubmodules.from_spec(spec, is_first_stage=True, is_last_stage=False)
    later = VisionModalitySubmodules.from_spec(spec, is_first_stage=False, is_last_stage=True)

    assert len(first.output_projections) == 1
    assert len(later.output_projections) == 0


def test_from_spec_builds_input_projections_only_on_last_stage():
    spec = _vision_spec(with_input_projection=True)
    last = VisionModalitySubmodules.from_spec(spec, is_first_stage=False, is_last_stage=True)
    not_last = VisionModalitySubmodules.from_spec(spec, is_first_stage=True, is_last_stage=False)

    assert len(last.input_projections) == 1
    assert len(not_last.input_projections) == 0


def test_encode_rejects_missing_encoder_inputs():
    submodules = VisionModalitySubmodules.from_spec(_vision_spec())
    with pytest.raises(ValueError, match="No inputs found for encoder 'enc'"):
        submodules.encode({'other_encoder': {}})


def test_encode_rejects_one_dimensional_encoder_output():
    submodules = VisionModalitySubmodules(encoders={'enc': nn.Identity()})
    with pytest.raises(ValueError, match="is not supported"):
        submodules.encode({'enc': {'input': torch.zeros(4)}})


def test_project_embeddings_returns_combined_without_projections():
    encoder = nn.Linear(8, 8)
    submodules = VisionModalitySubmodules(encoders={'enc': encoder})
    combined = torch.ones(3, 8)

    projected = submodules.project_embeddings([combined], is_input=True)

    assert torch.equal(projected, combined)


def test_forward_returns_none_when_first_stage_has_no_encoder_inputs():
    submodules = VisionModalitySubmodules.from_spec(
        _vision_spec(), is_first_stage=True, is_last_stage=True
    )
    assert submodules.forward(encoder_inputs=None, hidden_states=None) is None


def test_forward_returns_none_when_non_first_stage_has_no_hidden_states():
    submodules = VisionModalitySubmodules.from_spec(
        _vision_spec(), is_first_stage=False, is_last_stage=True
    )
    assert submodules.forward(encoder_inputs={'enc': {}}, hidden_states=None) is None


def test_forward_returns_combined_when_not_last_stage():
    encoder = nn.Linear(8, 8)
    submodules = VisionModalitySubmodules(
        encoders={'enc': encoder}, is_first_stage=False, is_last_stage=False
    )
    hidden = torch.randn(2, 8)

    output = submodules.forward(encoder_inputs=None, hidden_states=hidden)

    assert torch.equal(output, hidden)


def test_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError, match="Can't instantiate abstract class ModalitySubmodules"):
        ModalitySubmodules()
