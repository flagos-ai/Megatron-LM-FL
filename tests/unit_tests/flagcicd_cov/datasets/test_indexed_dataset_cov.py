# Copyright (c) 2026, BAAI. All rights reserved.
"""IndexedDataset: dtype codes, file/mmap reads, slicing, pickling, modes, builder round trips."""

import pickle

import numpy
import pytest
import torch

from megatron.core.datasets.indexed_dataset import (
    DType,
    IndexedDataset,
    IndexedDatasetBuilder,
    _BinReader,
    _FileBinReader,
    _MMapBinReader,
)


def _build_dataset(tmp_path, name, dtype=numpy.uint16, multimodal=False):
    prefix = str(tmp_path / name)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype, multimodal=multimodal)
    if multimodal:
        builder.add_item(torch.tensor([1, 2], dtype=torch.int64), mode=4)
        builder.end_document()
        builder.add_document(torch.tensor([3, 4, 5], dtype=torch.int64), [1, 2], modes=[5, 6])
    else:
        builder.add_item(torch.tensor([1, 2], dtype=torch.int64))
        builder.end_document()
        builder.add_document(torch.tensor([3, 4, 5], dtype=torch.int64), [1, 2])
    builder.finalize(prefix + ".idx")
    return prefix


def test_dtype_maps_between_codes_dtypes_and_sizes():
    assert DType.code_from_dtype(numpy.uint16) == DType.uint16.value
    assert DType.dtype_from_code(DType.uint16.value) is numpy.uint16
    assert DType.size(DType.uint16.value) == 2
    assert DType.size(numpy.int32) == 4
    assert DType.optimal_dtype(100) is numpy.uint16
    assert DType.optimal_dtype(65500) is numpy.int32


def test_dtype_size_rejects_keys_that_are_neither_codes_nor_dtypes():
    with pytest.raises(ValueError) as caught:
        DType.size(str)

    assert caught.value.args == ()


class _DelegatingBinReader(_BinReader):
    def read(self, dtype, count, offset):
        return super().read(dtype, count, offset)


def test_bin_reader_base_read_is_an_unimplemented_contract():
    assert _DelegatingBinReader().read(numpy.uint16, 1, 0) is None


def test_indexed_dataset_reads_sequences_via_file_and_mmap_readers(tmp_path):
    prefix = _build_dataset(tmp_path, "sample")
    file_dataset = IndexedDataset(prefix, mmap=False)
    mmap_dataset = IndexedDataset(prefix, mmap=True)

    assert isinstance(file_dataset.bin_reader, _FileBinReader)
    assert isinstance(mmap_dataset.bin_reader, _MMapBinReader)
    assert len(file_dataset) == 3
    numpy.testing.assert_array_equal(file_dataset[0], numpy.array([1, 2], dtype=numpy.uint16))
    numpy.testing.assert_array_equal(file_dataset[2], mmap_dataset[2])
    numpy.testing.assert_array_equal(file_dataset.sequence_lengths, numpy.array([2, 1, 2]))
    assert IndexedDataset.exists(prefix)
    assert not IndexedDataset.exists(str(tmp_path / "missing"))


def test_indexed_dataset_slice_reads_contiguous_ranges(tmp_path):
    prefix = _build_dataset(tmp_path, "sliced")
    dataset = IndexedDataset(prefix, mmap=False)

    first_two = dataset[0:2]
    numpy.testing.assert_array_equal(first_two[0], numpy.array([1, 2], dtype=numpy.uint16))
    numpy.testing.assert_array_equal(first_two[1], numpy.array([3], dtype=numpy.uint16))
    tail = dataset[1:]
    assert [sequence.tolist() for sequence in tail] == [[3], [4, 5]]


def test_indexed_dataset_rejects_non_contiguous_slices(tmp_path):
    dataset = IndexedDataset(_build_dataset(tmp_path, "strided"), mmap=False)

    with pytest.raises(ValueError, match="Slices into indexed_dataset must be contiguous"):
        dataset[::2]


def test_indexed_dataset_rejects_unexpected_index_types(tmp_path):
    dataset = IndexedDataset(_build_dataset(tmp_path, "typed"), mmap=False)

    with pytest.raises(TypeError, match="Unexpected type received for idx"):
        dataset["0"]


def test_multimodal_dataset_returns_tokens_and_mode_per_sequence(tmp_path):
    prefix = _build_dataset(tmp_path, "multimodal", multimodal=True)
    dataset = IndexedDataset(prefix, multimodal=True, mmap=False)

    sequence, mode = dataset[0]
    numpy.testing.assert_array_equal(sequence, numpy.array([1, 2], dtype=numpy.uint16))
    assert int(mode) == 4

    sequences, modes = dataset[:]
    assert [int(sequence.size) for sequence in sequences] == [2, 1, 2]
    numpy.testing.assert_array_equal(modes, numpy.array([4, 5, 6], dtype=numpy.int8))
    numpy.testing.assert_array_equal(
        dataset.index.sequence_modes, numpy.array([4, 5, 6], dtype=numpy.int8)
    )


def test_sequence_modes_property_returns_the_modes_of_a_single_sequence(tmp_path):
    prefix = str(tmp_path / "single")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.uint16, multimodal=True)
    builder.add_item(torch.tensor([9], dtype=torch.int64), mode=7)
    builder.end_document()
    builder.finalize(prefix + ".idx")

    dataset = IndexedDataset(prefix, multimodal=True, mmap=False)

    assert dataset.sequence_modes.tolist() == [7]
    assert dataset.sequence_modes.dtype == numpy.int8
    numpy.testing.assert_array_equal(dataset.sequence_modes, numpy.array([7], dtype=numpy.int8))


def test_fast_cache_load_index_reader_uses_the_supplied_counts(tmp_path):
    prefix = _build_dataset(tmp_path, "fast")
    dataset = IndexedDataset(
        prefix,
        mmap=False,
        fast_cache_load=True,
        sequences_per_dataset=(3, 3),
        dtype_code=DType.uint16.value,
    )

    assert len(dataset) == 3
    assert dataset[2].tolist() == [4, 5]
    assert dataset.sequence_lengths.tolist() == [2, 1, 2]
    numpy.testing.assert_array_equal(dataset[2], numpy.array([4, 5], dtype=numpy.uint16))


def test_indexed_dataset_pickling_round_trips_the_reader_configuration(tmp_path):
    original = IndexedDataset(_build_dataset(tmp_path, "pickled"), mmap=False)

    restored = pickle.loads(pickle.dumps(original))

    assert restored.__getstate__() == original.__getstate__()
    numpy.testing.assert_array_equal(restored[1], original[1])


def test_document_indices_accessors_return_and_replace_the_index(tmp_path):
    dataset = IndexedDataset(_build_dataset(tmp_path, "documents"), mmap=False)
    assert dataset.document_indices.tolist() == [0, 1, 3]
    assert dataset.get_document_indices().tolist() == [0, 1, 3]

    replacement = numpy.array([0, 1, 3], dtype=numpy.int64)
    dataset.set_document_indices(replacement)

    assert dataset.get_document_indices().tolist() == [0, 1, 3]
    assert dataset.get_document_indices() is replacement
    numpy.testing.assert_array_equal(dataset.get_document_indices(), replacement)


def test_builder_round_trips_items_documents_and_modes(tmp_path):
    prefix = str(tmp_path / "roundtrip")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32, multimodal=True)
    builder.add_item(torch.tensor([10, 11], dtype=torch.int64), mode=2)
    builder.end_document()
    builder.add_document(torch.tensor([12, 13], dtype=torch.int64), [1, 1], modes=[3, 4])
    builder.finalize(prefix + ".idx")

    dataset = IndexedDataset(prefix, multimodal=True, mmap=False)
    numpy.testing.assert_array_equal(dataset.sequence_lengths, numpy.array([2, 1, 1]))
    numpy.testing.assert_array_equal(dataset.document_indices, numpy.array([0, 1, 3]))
    numpy.testing.assert_array_equal(
        dataset.index.sequence_modes, numpy.array([2, 3, 4], dtype=numpy.int8)
    )
    numpy.testing.assert_array_equal(dataset[0][0], numpy.array([10, 11], dtype=numpy.int32))
    assert int(dataset[2][1]) == 4


def test_builder_add_index_concatenates_another_dataset(tmp_path):
    source = _build_dataset(tmp_path, "source", multimodal=True)

    destination = str(tmp_path / "destination")
    builder = IndexedDatasetBuilder(destination + ".bin", dtype=numpy.uint16, multimodal=True)
    builder.add_index(source)
    builder.finalize(destination + ".idx")

    dataset = IndexedDataset(destination, multimodal=True, mmap=False)
    assert len(dataset) == 3
    numpy.testing.assert_array_equal(dataset.sequence_lengths, numpy.array([2, 1, 2]))
    numpy.testing.assert_array_equal(dataset.document_indices, numpy.array([0, 1, 3]))
    numpy.testing.assert_array_equal(
        dataset.index.sequence_modes, numpy.array([4, 5, 6], dtype=numpy.int8)
    )
    sequence, mode = dataset[2]
    numpy.testing.assert_array_equal(sequence, numpy.array([4, 5], dtype=numpy.uint16))
    assert int(mode) == 6


def test_builder_add_index_requires_matching_dtypes(tmp_path):
    source = _build_dataset(tmp_path, "uint16_source")

    builder = IndexedDatasetBuilder(str(tmp_path / "int32_dest.bin"), dtype=numpy.int32)

    with pytest.raises(AssertionError):
        builder.add_index(source)
