# Copyright (c) 2026, BAAI. All rights reserved.
"""Config-logger serialisation and on-disk naming."""

import dataclasses
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core import config_logger
from megatron.core.config_logger import JSONEncoderWithMcoreTypes, log_config_to_disk


@dataclasses.dataclass
class Settings:
    hidden_size: int
    name: str


class UniqueDescriptor:
    def __init__(self):
        self.backend = 'local'


class Float16Module(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def encode(value):
    return json.loads(json.dumps(value, cls=JSONEncoderWithMcoreTypes))


def test_enabled_only_when_a_directory_is_configured():
    assert not config_logger.has_config_logger_enabled(SimpleNamespace())
    assert not config_logger.has_config_logger_enabled(SimpleNamespace(config_logger_dir=''))
    assert config_logger.has_config_logger_enabled(SimpleNamespace(config_logger_dir='/logs'))


def test_path_counter_is_tracked_per_path(tmp_path):
    first, second = str(tmp_path / 'first'), str(tmp_path / 'second')
    assert config_logger.get_path_with_count(first) == f'{first}.iter0'
    assert config_logger.get_path_with_count(first) == f'{first}.iter1'
    assert config_logger.get_path_with_count(second) == f'{second}.iter0'


def test_encoder_stringifies_callables_types_and_dtypes():
    def provider():
        return None

    assert encode({'fn': provider}) == {'fn': str(provider)}
    assert encode({'cls': Settings}) == {'cls': str(Settings)}
    assert encode({'dtype': torch.float16}) == {'dtype': 'torch.float16'}


def test_encoder_expands_dataclasses_and_descriptors():
    assert encode({'settings': Settings(8, 'tiny')}) == {
        'settings': {'hidden_size': 8, 'name': 'tiny'}
    }
    assert encode({'descriptor': UniqueDescriptor()}) == {'descriptor': {'backend': 'local'}}


def test_encoder_walks_module_trees_down_to_leaf_reprs():
    leaf = nn.Linear(2, 3)
    tree = nn.Sequential(OrderedDict(body=nn.ModuleList([leaf]), head=nn.Identity()))
    assert encode({'model': tree}) == {'model': {'body': [str(leaf)], 'head': 'Identity()'}}
    assert encode({'model': Float16Module(leaf)}) == {
        'model': {'Float16Module': {'module': str(leaf)}}
    }


def test_encoder_default_recurses_into_plain_containers():
    encoder = JSONEncoderWithMcoreTypes()
    assert encoder.default({'dtype': torch.int64}) == {'dtype': 'torch.int64'}
    assert encoder.default([torch.int64]) == ['torch.int64']
    assert encoder.default(OrderedDict(dtype=torch.int64)) == {'dtype': 'torch.int64'}


def test_encoder_falls_back_to_str_for_unknown_objects():
    assert encode({'value': 1 + 2j}) == {'value': '(1+2j)'}


def test_json_dump_uses_the_owner_class_as_prefix_and_counts_repeats(tmp_path):
    config = SimpleNamespace(config_logger_dir=str(tmp_path / 'logs'))
    owner = Settings(8, 'tiny')

    log_config_to_disk(config, {'self': owner, 'dtype': torch.float32}, rank_str='r0')
    log_config_to_disk(config, {'self': owner, 'dtype': torch.float64}, rank_str='r0')

    first = tmp_path / 'logs' / 'Settings.rank_r0.iter0.json'
    second = tmp_path / 'logs' / 'Settings.rank_r0.iter1.json'
    assert json.loads(first.read_text()) == {'dtype': 'torch.float32'}
    assert json.loads(second.read_text()) == {'dtype': 'torch.float64'}


def test_explicit_prefix_wins_and_self_is_never_serialised(tmp_path):
    config = SimpleNamespace(config_logger_dir=str(tmp_path))
    data = {'self': Settings(8, 'tiny'), 'layers': 2}

    log_config_to_disk(config, data, prefix='custom', rank_str='r1')

    assert 'self' not in data
    assert json.loads((tmp_path / 'custom.rank_r1.iter0.json').read_text()) == {'layers': 2}


def test_ordered_dict_is_saved_as_a_torch_checkpoint(tmp_path):
    config = SimpleNamespace(config_logger_dir=str(tmp_path))
    state = OrderedDict(weight=torch.arange(3))

    log_config_to_disk(config, state, prefix='state', rank_str='r0')

    loaded = torch.load(tmp_path / 'state.rank_r0.iter0.pth')
    assert torch.equal(loaded['weight'], torch.arange(3))


def test_default_rank_string_comes_from_parallel_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config_logger.parallel_state, 'get_all_ranks', lambda: '1_2_3_4_5')
    config = SimpleNamespace(config_logger_dir=str(tmp_path))

    log_config_to_disk(config, {'layers': 2}, prefix='model')

    assert (tmp_path / 'model.rank_1_2_3_4_5.iter0.json').exists()
