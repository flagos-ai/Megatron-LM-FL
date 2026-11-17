# Copyright (c) 2026, BAAI. All rights reserved.
"""CUDA graph module scope normalisation and deprecated-scope migration to new-style settings."""

import pytest

from megatron.core.transformer.cuda_graph_config import (
    get_deprecated_cuda_graph_modules_migration,
    normalize_cuda_graph_modules,
    validate_deprecated_cuda_graph_modules_migration_inputs,
)
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope


def test_normalize_returns_empty_scopes_for_none_input():
    assert normalize_cuda_graph_modules(None) == ([], [], False)


def test_normalize_accepts_a_single_enum_member_without_a_list_wrapper():
    normalized, deprecated, uses_full_scope = normalize_cuda_graph_modules(CudaGraphModule.moe)

    assert normalized == [CudaGraphModule.moe]
    assert deprecated == []
    assert not uses_full_scope


def test_normalize_maps_a_deprecated_scope_to_its_migration_tuple():
    normalized, deprecated, uses_full_scope = normalize_cuda_graph_modules('full_iteration')

    assert normalized == []
    assert deprecated == [('full_iteration', 'cuda_graph_impl', 'full_iteration')]
    assert not uses_full_scope


def test_migration_validation_rejects_multiple_deprecated_scopes_at_once():
    _, deprecated, _ = normalize_cuda_graph_modules('full_iteration,full_iteration_inference')

    with pytest.raises(
        AssertionError, match="cannot contain multiple deprecated values at the same time"
    ):
        validate_deprecated_cuda_graph_modules_migration_inputs(deprecated, 'none', None)


@pytest.mark.parametrize(
    ('scopes', 'cuda_graph_impl', 'inference_cuda_graph_scope', 'expected_migration'),
    [
        ('full_iteration', 'none', None, ('cuda_graph_impl', 'full_iteration')),
        (
            'full_iteration',
            'full_iteration',
            InferenceCudaGraphScope.none,
            ('cuda_graph_impl', 'full_iteration'),
        ),
        ('full_iteration', 'local', 'none', ('cuda_graph_impl', 'full_iteration')),
        ('full_iteration_inference', 'none', None, None),
        ('full_iteration_inference', 'none', InferenceCudaGraphScope.block, None),
        (
            'full_iteration_inference',
            'local',
            'block',
            ('inference_cuda_graph_scope', InferenceCudaGraphScope.block),
        ),
    ],
    ids=[
        'full_iteration_default_impl',
        'full_iteration_matching_impl',
        'full_iteration_local_impl',
        'full_iteration_inference_default_impl',
        'full_iteration_inference_block_scope',
        'full_iteration_inference_local_impl',
    ],
)
def test_deprecated_scope_with_compatible_impl_migrates_to_its_documented_setting(
    scopes, cuda_graph_impl, inference_cuda_graph_scope, expected_migration
):
    _, deprecated, _ = normalize_cuda_graph_modules(scopes)

    result = validate_deprecated_cuda_graph_modules_migration_inputs(
        deprecated, cuda_graph_impl, inference_cuda_graph_scope
    )
    migration = get_deprecated_cuda_graph_modules_migration(
        *deprecated[0], cuda_graph_impl=cuda_graph_impl
    )

    assert result is None
    assert migration == expected_migration
