# Copyright (c) 2026, BAAI. All rights reserved.

"""Checkpointing config helpers: maybe_load_config returns None for non-checkpoints, save/load roundtrips, and rejection of unknown config fields."""

import pytest

from megatron.core.dist_checkpointing.core import (
    CheckpointingConfig,
    maybe_load_config,
    save_config,
)


def test_maybe_load_config_returns_none_when_checkpoint_dir_is_empty():
    assert maybe_load_config('') is None


def test_maybe_load_config_returns_none_when_metadata_missing(tmp_path):
    ckpt_dir = tmp_path / 'not_a_dist_ckpt'
    ckpt_dir.mkdir()
    assert maybe_load_config(str(ckpt_dir)) is None


def test_maybe_load_config_roundtrips_saved_config(tmp_path):
    ckpt_dir = tmp_path / 'dist_ckpt'
    ckpt_dir.mkdir()
    save_config(CheckpointingConfig('torch_dist', 2, 'torch', 3), str(ckpt_dir))

    config = maybe_load_config(str(ckpt_dir))

    assert config == CheckpointingConfig('torch_dist', 2, 'torch', 3)
    assert config.sharded_backend == 'torch_dist'
    assert config.sharded_backend_version == 2
    assert config.common_backend == 'torch'
    assert config.common_backend_version == 3


def test_maybe_load_config_rejects_config_with_unknown_fields(tmp_path):
    ckpt_dir = tmp_path / 'broken_dist_ckpt'
    ckpt_dir.mkdir()
    (ckpt_dir / 'metadata.json').write_text('{"sharded_backend": "torch_dist", "bogus": 1}')

    with pytest.raises(TypeError, match='unexpected keyword'):
        maybe_load_config(str(ckpt_dir))
