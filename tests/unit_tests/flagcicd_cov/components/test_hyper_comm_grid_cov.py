# Copyright (c) 2026, BAAI. All rights reserved.
"""HyperCommGrid construction: world-size resolution from WORLD_SIZE or an initialized process group, and rejection of negative rank offsets."""

import pytest
import torch.distributed

from megatron.core.hyper_comm_grid import HyperCommGrid


def test_hyper_comm_grid_reads_world_size_from_initialized_distributed(monkeypatch):
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    grid = HyperCommGrid([2, 4], ['dp', 'mp'])
    assert grid.shape == [2, 4]
    assert grid.size == 8
    assert grid.dim_names == ['dp', 'mp']
    assert grid.rank_offset == 0
    assert grid.backend is None


def test_hyper_comm_grid_raises_when_world_size_is_unknowable(monkeypatch):
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    with pytest.raises(RuntimeError, match='Cannot determine world size'):
        HyperCommGrid([2, 2], ['dp', 'mp'])


def test_hyper_comm_grid_rejects_negative_rank_offset(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '16')
    with pytest.raises(ValueError, match='rank_offset must be non-negative, got -1'):
        HyperCommGrid([2, 4], ['dp', 'mp'], rank_offset=-1)
