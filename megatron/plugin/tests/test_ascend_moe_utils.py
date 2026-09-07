import pytest
import torch

from megatron.plugin.Ascend.transformer.moe.moe_utils import (
    _indexed_sort_chunks_by_idxs,
    _sort_chunks_by_idxs,
)


def _reference(input, split_sizes, sorted_idxs):
    chunks = torch.split(input, split_sizes.tolist(), dim=0)
    return torch.cat([chunks[index] for index in sorted_idxs.tolist()], dim=0)


@pytest.mark.parametrize("with_probs", [False, True])
def test_indexed_chunk_reorder_matches_reference_forward_and_backward(with_probs):
    split_sizes = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    sorted_idxs = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    input = torch.randn(6, 4, requires_grad=True)
    probs = torch.randn(6, 2, requires_grad=True) if with_probs else None

    output, permuted_probs = _indexed_sort_chunks_by_idxs(input, split_sizes, sorted_idxs, probs)
    torch.testing.assert_close(output, _reference(input, split_sizes, sorted_idxs))
    if probs is not None:
        torch.testing.assert_close(
            permuted_probs, _reference(probs, split_sizes, sorted_idxs)
        )

    loss = output.square().sum()
    if permuted_probs is not None:
        loss = loss + permuted_probs.square().sum()
    loss.backward()

    torch.testing.assert_close(input.grad, 2 * input.detach())
    if probs is not None:
        torch.testing.assert_close(probs.grad, 2 * probs.detach())


@pytest.mark.parametrize(
    "split_sizes,sorted_idxs,message",
    [
        ([2, 1], [0], "exactly one entry"),
        ([2, -1], [0, 1], "non-negative"),
        ([2, 1], [0, 0], "permutation"),
        ([2, 2], [0, 1], "input.shape"),
    ],
)
def test_indexed_chunk_reorder_rejects_invalid_metadata(split_sizes, sorted_idxs, message):
    with pytest.raises(ValueError, match=message):
        _indexed_sort_chunks_by_idxs(
            torch.randn(3, 2),
            torch.tensor(split_sizes),
            torch.tensor(sorted_idxs),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_probs", [False, True])
def test_512_chunks_bitwise_forward_and_backward(dtype, with_probs):
    generator = torch.Generator().manual_seed(42)
    sizes = torch.randint(0, 5, (512,), generator=generator)
    order = torch.randperm(512, generator=generator)
    rows = int(sizes.sum())
    # Transpose to exercise non-contiguous inputs and upstream gradients.
    values = torch.randn(8, rows, generator=generator, dtype=dtype).t()
    input = values.detach().requires_grad_()
    ref_input = values.clone().requires_grad_()
    probs = torch.randn(rows, generator=generator, requires_grad=True) if with_probs else None
    ref_probs = probs.detach().clone().requires_grad_() if with_probs else None
    output, output_probs = _indexed_sort_chunks_by_idxs(input, sizes, order, probs)
    reference = _reference(ref_input, sizes, order)
    assert torch.equal(output, reference)
    grad = torch.randn(8, rows, generator=generator, dtype=dtype).t()
    output.backward(grad, retain_graph=with_probs)
    reference.backward(grad)
    assert torch.equal(input.grad, ref_input.grad)
    if with_probs:
        reference_probs = _reference(ref_probs, sizes, order)
        assert torch.equal(output_probs, reference_probs)
        grad_probs = torch.randn(rows, generator=generator)
        output_probs.backward(grad_probs)
        reference_probs.backward(grad_probs)
        assert torch.equal(probs.grad, ref_probs.grad)


def test_all_empty_chunks():
    input = torch.empty(0, 8, requires_grad=True)
    sizes = torch.zeros(512, dtype=torch.long)
    order = torch.arange(511, -1, -1)
    output, probs = _indexed_sort_chunks_by_idxs(input, sizes, order)
    assert probs is None and torch.equal(output, _reference(input, sizes, order))
    output.sum().backward()
    assert input.grad.shape == input.shape


def test_rejects_float_metadata():
    with pytest.raises(TypeError, match="integer dtype"):
        _indexed_sort_chunks_by_idxs(torch.randn(3, 2), torch.tensor([1.5, 1.5]), torch.tensor([1, 0]))


def test_non_npu_keeps_core_reference():
    input = torch.randn(3, 2, requires_grad=True)
    sizes, order = torch.tensor([1, 2]), torch.tensor([1, 0])
    actual, probs = _sort_chunks_by_idxs(input, sizes, order)
    assert probs is None and torch.equal(actual, _reference(input, sizes, order))
