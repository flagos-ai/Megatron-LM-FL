# Copyright (c) 2026, BAAI. All rights reserved.
"""ShardedTensor and ShardedObject: metadata validation errors, global slicing, chunk limits, and ShardedTensorFactory merge rules."""

import pytest
import torch

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import (
    ShardedBase,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
    apply_factory_merges,
)


def test_sharded_base_without_data_is_abstract():
    tensor = ShardedTensor.from_rank_offsets('keyA', torch.arange(4.0), (0, 0, 2))
    with pytest.raises(NotImplementedError) as caught:
        ShardedBase.without_data(tensor)
    assert caught.type is NotImplementedError


def test_sharded_tensor_rejects_data_dtype_mismatch():
    with pytest.raises(CheckpointingException, match='Data dtype should match `dtype` attribute'):
        ShardedTensor('keyA', torch.ones(2), torch.float64, (2,), (2,), (0,), (1,))


def test_sharded_tensor_rejects_local_shape_with_prepended_axes():
    with pytest.raises(
        CheckpointingException,
        match='Local shape together with `prepend_axis_num` dimensions should be',
    ):
        ShardedTensor(
            'keyA', None, torch.float32, (2, 2), (2, 2), (0, 0), (1, 1), prepend_axis_num=1
        )


def test_sharded_tensor_rejects_offset_not_divisible_by_local_shape():
    with pytest.raises(
        CheckpointingException, match=r'Global offset \(1\) must be divisible by local shape \(2\)'
    ):
        ShardedTensor('keyA', None, torch.float32, (2,), (2,), (1,), (1,))


def test_sharded_tensor_rejects_flattened_range():
    with pytest.raises(
        CheckpointingException, match='ShardedTensor.flattened_range is not supported.'
    ):
        ShardedTensor(
            'keyA', None, torch.float32, (2,), (2,), (0,), (1,), flattened_range=slice(0, 1)
        )


def test_sharded_tensor_global_slice_includes_prepended_axes():
    data = torch.ones(2, 3)
    tensor = ShardedTensor.from_rank_offsets('keyA', data, (1, 1, 4), prepend_axis_num=1)
    assert tensor.global_slice() == (0, slice(2, 4), slice(0, 3))


def test_sharded_tensor_max_allowed_chunks_returns_chunk_size_per_axis():
    tensor = ShardedTensor('keyA', None, torch.float32, (3,), (6,), (0,), (2,))
    assert tensor.max_allowed_chunks() == (3,)


def test_sharded_tensor_max_allowed_chunks_rejects_indivisible_axis():
    tensor = ShardedTensor('keyA', None, torch.float32, (3,), (5,), (0,), (2,))
    with pytest.raises(
        CheckpointingException, match=r'Axis shape \(5\) not divisible by axis fragmentation \(2'
    ):
        tensor.max_allowed_chunks()


def test_from_rank_offsets_rejects_flattened_range():
    with pytest.raises(ValueError, match='Cannot instantiate a flat ShardedTensor'):
        ShardedTensor.from_rank_offsets(
            'keyA', torch.arange(4.0), (0, 0, 2), flattened_range=slice(0, 2)
        )


def test_from_rank_offsets_rejects_invalid_rank_offsets():
    with pytest.raises(CheckpointingException, match='Invalid rank offsets'):
        ShardedTensor.from_rank_offsets('keyA', torch.arange(4.0), (0, 2, 2))


def test_sharded_object_rejects_offset_dimension_mismatch():
    with pytest.raises(
        CheckpointingException,
        match='Global offset dimensions should be equal to global shape dimensions',
    ):
        ShardedObject('keyA', object(), (1,), (0, 0))


def test_sharded_object_str_shows_key_only():
    obj = ShardedObject('keyA', object(), (1,), (0,))
    assert str(obj) == "ShardedObject(key='keyA')"


def test_sharded_object_empty_from_unique_key_appends_unknown_last_shape():
    obj = ShardedObject.empty_from_unique_key('keyA/shard_0.1.2_2.3')
    assert obj.key == 'keyA'
    assert obj.global_offset == (0, 1, 2)
    assert obj.global_shape == (2, 3, -1)
    assert obj.data is None


def test_apply_factory_merges_applies_factory_merge_function():
    factory = ShardedTensorFactory('a', torch.arange(3.0), lambda *args: {}, lambda sd: sd['x'] + 1)
    merged = apply_factory_merges({'x': torch.tensor(1.0)}, factory)
    assert torch.equal(merged, torch.tensor(2.0))


def test_apply_factory_merges_rejects_missing_dict_key():
    with pytest.raises(
        ValueError, match='Different dict keys encountered in `apply_factory_merges`'
    ):
        apply_factory_merges({'a': 1}, {'b': 2})


def test_apply_factory_merges_merges_factory_lists_in_place():
    factory_a = ShardedTensorFactory('a', torch.arange(2.0), lambda *args: {}, lambda sd: sd * 10)
    factory_b = ShardedTensorFactory('b', torch.arange(2.0), lambda *args: {}, lambda sd: sd * 20)
    x1 = [1, 2]
    merged = apply_factory_merges(x1, [factory_a, factory_b])
    assert merged == [10, 40]
    assert merged is x1


def test_apply_factory_merges_rejects_list_length_mismatch():
    with pytest.raises(ValueError, match='Cannot merge two lists with different lengths'):
        apply_factory_merges([1, 2], [1, 2, 3])


def test_apply_factory_merges_rejects_non_integer_key_in_list_dict_merge():
    with pytest.raises(ValueError, match='Invalid dict key a non-integer type encountered'):
        apply_factory_merges([1], {'a': 2})


def test_apply_factory_merges_rejects_out_of_bound_index_in_list_dict_merge():
    with pytest.raises(ValueError, match=r'Dict key 2 out of bound for list of length1'):
        apply_factory_merges([1], {2: 3})


def test_apply_factory_merges_rejects_duplicate_leaf_values():
    with pytest.raises(ValueError, match='Duplicate non-dict and non-list values encountered'):
        apply_factory_merges(1, 2)
