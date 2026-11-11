# Copyright (c) 2026, BAAI. All rights reserved.
'''Last-dimension tensor splitting: contiguous chunk materialization versus zero-copy views into the parent storage.'''

import torch

from megatron.core.tensor_parallel.utils import split_tensor_along_last_dim


def test_split_tensor_along_last_dim_returns_contiguous_chunks():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    chunks = split_tensor_along_last_dim(x, 2, contiguous_split_chunks=True)

    assert isinstance(chunks, tuple)
    assert len(chunks) == 2
    assert all(chunk.is_contiguous() for chunk in chunks)
    assert torch.equal(chunks[0], x[:, :2])
    assert torch.equal(chunks[1], x[:, 2:])


def test_split_tensor_along_last_dim_keeps_views_by_default():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    chunks = split_tensor_along_last_dim(x, 2)

    assert isinstance(chunks, tuple)
    assert len(chunks) == 2
    assert not chunks[0].is_contiguous()
    assert chunks[0].data_ptr() == x.data_ptr()
    assert torch.equal(chunks[0], x[:, :2])
    assert torch.equal(chunks[1], x[:, 2:])
