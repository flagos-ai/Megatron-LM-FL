import pytest
import torch

from megatron.core.tensor_parallel import mappings
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
def test_CopyToModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones((1)).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    class Ctx:
        group = tp_group

    output_data, _ = mappings._CopyToModelParallelRegion.backward(Ctx(), input_data)
    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert torch.equal(output_data, result)
    assert torch.equal(input_data, mappings.copy_to_tensor_model_parallel_region(input_data))
    assert torch.equal(
        input_data, mappings._CopyToModelParallelRegion.symbolic(None, input_data, tp_group)
    )
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ReduceFromModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones((1)).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings._ReduceFromModelParallelRegion.symbolic(None, input_data, tp_group)

    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert torch.equal(output_data, result)

    input_data = torch.ones((1)).cuda() * Utils.rank
    assert torch.equal(mappings.reduce_from_tensor_model_parallel_region(input_data), result)

    class Ctx:
        group = tp_group

    output_data, _ = mappings._ReduceFromModelParallelRegion.backward(Ctx(), input_data)
    assert torch.equal(input_data, output_data)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ScatterToModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)

    req_dim = int(Utils.rank % (Utils.world_size / 2))
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))
    output_data = mappings._ScatterToModelParallelRegion.symbolic(None, input_data, tp_group)
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = torch.ones(8).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    actual_output_data, _ = mappings._ScatterToModelParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.cat(
        (torch.ones(8) * 0, torch.ones(8) * 1, torch.ones(8) * 2, torch.ones(8) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(actual_output_data, expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_GatherFromModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = int(Utils.rank % (Utils.world_size / 2))

    class Ctx:
        group = tp_group

    output_data, _ = mappings._GatherFromModelParallelRegion.backward(Ctx(), input_data)
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = torch.ones(8).cuda() * Utils.rank
    actual_output_data = mappings.gather_from_tensor_model_parallel_region(input_data)
    expected_output = torch.cat(
        (torch.ones(8) * 0, torch.ones(8) * 1, torch.ones(8) * 2, torch.ones(8) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(actual_output_data, expected_output)
    assert torch.equal(
        mappings._GatherFromModelParallelRegion.symbolic(None, input_data, tp_group),
        expected_output,
    )
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ScatterToSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = int(Utils.rank % (Utils.world_size / 2)) * 2
    output_data = mappings._ScatterToSequenceParallelRegion.symbolic(None, input_data, tp_group)
    assert torch.equal(output_data, input_data[req_dim : req_dim + 2, :])
    output_data = mappings.scatter_to_sequence_parallel_region(input_data)
    assert torch.equal(output_data, input_data[req_dim : req_dim + 2, :])

    input_data = torch.ones(4).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    output_data, _ = mappings._ScatterToModelParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data, expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_GatherFromSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones(4).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.gather_from_sequence_parallel_region(input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data, expected_output)
    assert torch.equal(
        mappings._GatherFromSequenceParallelRegion.symbolic(None, input_data, tp_group),
        expected_output,
    )
    input_data = torch.vstack(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()

    class Ctx:
        tensor_parallel_output_grad = True
        output_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._GatherFromSequenceParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.ones((1, 4)).cuda() * 4 * int(Utils.rank % 4)
    assert torch.equal(output_data[0], expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ReduceScatterToSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.vstack(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.reduce_scatter_to_sequence_parallel_region(input_data)
    expected_output = torch.ones(4).cuda() * 4 * int(Utils.rank % 4)
    assert torch.equal(output_data[0], expected_output)
    assert torch.equal(
        mappings._ReduceScatterToSequenceParallelRegion.symbolic(None, input_data, tp_group),
        expected_output.reshape((1, 4)),
    )
    input_data = torch.ones(4).cuda() * Utils.rank

    class Ctx:
        input_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._ReduceScatterToSequenceParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data[0], expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("tp,partitions", [(2, 1), (4, 1), (4, 2), (8, 2), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided,recompute", [(False, False), (True, False), (False, True)])
def test_all_gather_last_dim_partition_gradients(tp, partitions, dtype, strided, recompute):
    """Compare output and linear input/weight gradients with gather-then-slice."""
    Utils.initialize_model_parallel(tp, 1)
    group = get_tensor_model_parallel_group_if_none(tp_group=None)
    device = mappings.cur_platform.current_device()
    torch.manual_seed(123 + Utils.rank)
    data = torch.randn(16, 2, 12, device=device, dtype=dtype)
    if strided:
        data = data.transpose(0, 1)
    x = data.detach().requires_grad_()
    weight = torch.randn(12, 8, device=device, dtype=dtype, requires_grad=True)
    expected = mappings.all_gather_last_dim_from_tensor_parallel_region.__wrapped__(
        x @ weight, group=group, output_split_count=partitions
    )

    def gather(inputs, weights):
        return mappings.all_gather_last_dim_from_tensor_parallel_region(
            inputs @ weights, group=group, output_split_count=partitions
        )

    if recompute:
        from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

        with set_checkpoint_early_stop(False):
            actual = checkpoint(gather, x, weight, use_reentrant=False)
    else:
        actual = gather(x, weight)
    comparisons = [(actual, expected, 0)]
    grad = torch.randn_like(actual)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    for _ in range(2):
        expected_grads = torch.autograd.grad(expected, (x, weight), grad, retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (x, weight), grad, retain_graph=True)
        for result, reference in zip(actual_grads, expected_grads):
            comparisons.append((result, reference, tol))
    # Rank-local gradient flags must not select different forward collectives.
    mixed = x.detach().requires_grad_(group.rank() % 2 == 0)
    comparisons.append(
        (
            mappings.all_gather_last_dim_from_tensor_parallel_region(
                mixed, group=group, output_split_count=partitions
            ),
            mappings.all_gather_last_dim_from_tensor_parallel_region.__wrapped__(
                mixed, group=group, output_split_count=partitions
            ),
            0,
        )
    )
    Utils.destroy_model_parallel()
    for result, reference, tolerance in comparisons:
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.internal
@pytest.mark.parametrize("partitions", [1, 2])
def test_all_gather_last_dim_cpu_fallback(monkeypatch, partitions):
    """The Ascend override preserves native full and partitioned CPU AG/RS gradients."""
    import torch.distributed as dist

    from megatron.plugin.Ascend.tensor_parallel.mappings import (
        all_gather_last_dim_from_tensor_parallel_region,
    )

    if Utils.world_size % partitions:
        pytest.skip("Output partitions must divide the distributed world size")
    Utils.initialize_distributed()
    group = dist.new_group(backend="gloo")
    try:
        # Native mappings allocate on the active platform; exercise the CPU path with Gloo.
        monkeypatch.setattr(mappings.cur_platform, "current_device", lambda: "cpu")
        x = torch.full((4, 2, 3), float(Utils.rank), requires_grad=True)
        expected = mappings.all_gather_last_dim_from_tensor_parallel_region.__wrapped__(
            x, group=group, output_split_count=partitions
        )
        actual = all_gather_last_dim_from_tensor_parallel_region(
            x, group=group, output_split_count=partitions
        )
        grad = torch.full_like(actual, float(Utils.rank + 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual, x, grad)[0],
            torch.autograd.grad(expected, x, grad)[0],
            rtol=0,
            atol=0,
        )
    finally:
        dist.destroy_process_group(group)
