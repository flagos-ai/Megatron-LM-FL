# Copyright (c) 2026, BAAI. All rights reserved.

"""Common-state loading: a missing common.pt raises CheckpointingException naming the checkpoint directory and its leftover files."""

import logging

import pytest

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.strategies.common import load_common


def test_load_common_raises_when_common_file_missing(tmp_path):
    ckpt_dir = tmp_path / 'incomplete_ckpt'
    ckpt_dir.mkdir()
    (ckpt_dir / 'some_tensor.distcp').write_bytes(b'not common.pt')

    with pytest.raises(CheckpointingException, match='does not exist'):
        load_common(str(ckpt_dir))


def test_load_common_error_names_checkpoint_directory_contents(tmp_path, caplog):
    ckpt_dir = tmp_path / 'incomplete_ckpt_with_files'
    ckpt_dir.mkdir()
    (ckpt_dir / 'leftover.distcp').write_bytes(b'x')

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(CheckpointingException) as exc_info:
            load_common(str(ckpt_dir))

    assert str(ckpt_dir / 'common.pt') in str(exc_info.value)
    assert 'leftover.distcp' in caplog.text
