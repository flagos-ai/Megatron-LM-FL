# Copyright (c) 2026, BAAI. All rights reserved.
"""Rank and world-size fallbacks when torch.distributed is not initialised."""

import pytest
import torch

from megatron.core import _rank_utils


@pytest.fixture
def bare_process(monkeypatch):
    """No process group and none of the launcher or SLURM variables."""
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    for name in ('RANK', 'WORLD_SIZE', 'SLURM_NTASKS', 'SLURM_PROCID'):
        monkeypatch.delenv(name, raising=False)


def test_rank_defaults_to_zero_with_a_warning(bare_process):
    with pytest.warns(UserWarning, match='Defaulting to rank 0'):
        rank = _rank_utils.safe_get_rank()
    assert rank == 0


def test_rank_and_world_size_come_from_slurm(bare_process, monkeypatch):
    monkeypatch.setenv('SLURM_NTASKS', '4')
    monkeypatch.setenv('SLURM_PROCID', '2')
    assert _rank_utils.safe_get_rank() == 2
    assert _rank_utils.safe_get_world_size() == 4


def test_launcher_rank_wins_over_slurm(bare_process, monkeypatch):
    monkeypatch.setenv('SLURM_NTASKS', '4')
    monkeypatch.setenv('SLURM_PROCID', '2')
    monkeypatch.setenv('RANK', '5')
    assert _rank_utils.safe_get_rank() == 5


@pytest.mark.parametrize('variables', [{'RANK': 'bad'}, {'SLURM_NTASKS': '4', 'SLURM_PROCID': 'x'}])
def test_unparsable_rank_falls_back_to_zero(bare_process, monkeypatch, variables):
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert _rank_utils.safe_get_rank() == 0


def test_world_size_defaults_to_one_with_a_warning(bare_process):
    with pytest.warns(UserWarning, match='Defaulting to world size 1'):
        size = _rank_utils.safe_get_world_size()
    assert size == 1
