# Copyright (c) 2026, BAAI. All rights reserved.
"""Nested state-dict helpers: diff, merge, mapping and type inspection edge cases."""

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import (
    dict_map,
    dict_map_with_key,
    diff,
    extract_matching_values,
    inspect_types,
    merge,
    nested_items_iter,
)


class Replica:
    def __init__(self, data):
        self.replica_id = 0
        self.data = data


class Incomparable:
    def __ne__(self, other):
        raise RuntimeError('cannot compare')


class Unprintable:
    def __str__(self):
        raise RuntimeError('cannot print')


def test_extract_matching_values_rejects_scalar_top_level():
    with pytest.raises(ValueError, match=r"Unexpected top-level object type: <class 'int'>"):
        extract_matching_values(5, lambda value: True)


def test_diff_reports_tensors_of_different_shapes_with_both_shapes():
    assert diff({'a': torch.zeros(2)}, {'a': torch.zeros(3)}) == (
        [],
        [],
        [(('a',), torch.Tensor, torch.Tensor, torch.Size([2]), torch.Size([3]))],
    )


def test_diff_descends_into_replica_data():
    assert diff({'k': Replica({'x': 1, 'y': 2})}, {'k': Replica({'x': 1, 'z': 3})}) == (
        [('k', Replica, 'y')],
        [('k', Replica, 'z')],
        [],
    )


def test_diff_reports_mismatch_inside_replica_data():
    assert diff(Replica([1, 2]), Replica([1, 3])) == ([], [], [((Replica, 1), int, int)])


def test_diff_treats_failing_comparison_as_mismatch():
    left, right = Incomparable(), Incomparable()
    assert diff({'k': left}, {'k': right}) == ([], [], [(('k',), Incomparable, Incomparable)])


def test_inspect_types_prints_nested_structure(capsys):
    inspect_types({'a': torch.zeros(2, 3), 'b': [1, 'x' * 40], 'c': {'d': Unprintable()}})
    expected = (
        '\n'
        '> a: Tensor of shape torch.Size([2, 3])\n'
        '> b: \n'
        "    - 0: [<class 'int'>]: 1\n"
        "    - 1: [<class 'str'>]: " + 'x' * 30 + '... (truncated)\n'
        '> c: \n'
        f'    > d: [{Unprintable}]: <no string repr>\n'
    )
    assert capsys.readouterr().out == expected


def test_nested_items_iter_yields_leaf_container_key_value():
    data = {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}
    items = list(nested_items_iter(data))
    assert [(key, value) for _, key, value in items] == [('a', 1), ('c', 2), (0, 3), (1, 4)]
    assert items[-1][0] is data['b']['d']


def test_dict_map_rewrites_leaves_in_place():
    data = {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}
    dict_map(lambda value: value * 10, data)
    assert data == {'a': 10, 'b': {'c': 20, 'd': [30, 40]}}


def test_dict_map_with_key_passes_the_leaf_key():
    data = {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}
    dict_map_with_key(lambda key, value: f'{key}={value}', data)
    assert data == {'a': 'a=1', 'b': {'c': 'c=2', 'd': ['0=3', '1=4']}}


def test_merge_combines_dicts_and_equal_length_lists():
    assert merge({'a': {'x': 1}, 'l': [{'p': 1}]}, {'a': {'y': 2}, 'b': 3, 'l': [{'q': 2}]}) == {
        'a': {'x': 1, 'y': 2},
        'l': [{'p': 1, 'q': 2}],
        'b': 3,
    }


def test_merge_rejects_lists_of_different_lengths():
    with pytest.raises(
        ValueError,
        match=r'Cannot merge two lists with different lengths \(2 and 1, encountered at level \(\)\)',
    ):
        merge([1, 2], [1])


@pytest.mark.parametrize(
    ('left', 'right', 'message'),
    [
        ({'a': 1}, {'a': 2}, r"`1` and `2` \(at level \('a',\)\)"),
        ({'a': [1]}, {'a': {'b': 1}}, r"`\[1\]` and `\{'b': 1\}` \(at level \('a',\)\)"),
    ],
)
def test_merge_rejects_conflicting_leaves(left, right, message):
    with pytest.raises(ValueError, match=message):
        merge(left, right)
