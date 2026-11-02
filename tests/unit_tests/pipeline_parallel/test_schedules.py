# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import (
    convert_schedule_table_to_order,
    get_overlap_moe_expert_parallel_comm_order,
)
from tests.unit_tests.test_utilities import Utils, get_current_device

rank = Utils.rank


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def test_get_forward_backward_func():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()


def test_deallocate_output_tensor():
    out = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schedule.deallocate_output_tensor(out)
    assert out.nelement() == 6


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "pipeline_model_parallel_size,microbatch_group_size_per_vp_stage",
    [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("num_microbatches", [8, 32])
@pytest.mark.parametrize("virtual_pipeline_model_parallel_size", [None, 2, 4, 8])
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
    if pipeline_model_parallel_size == 1 and virtual_pipeline_model_parallel_size is not None:
        return

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    num_model_chunks = (
        virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is not None
        else 1
    )

    _, _, num_warmup_microbatches, _ = schedule.get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, False
    )
    schedule_table = schedule.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )

    assert max(order) == num_model_chunks
    assert len(order) == num_microbatches * num_model_chunks * 2
    order_cnt = {}
    accumulated_order = 0
    for o in order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0
    assert 0 not in order_cnt
    for k, v in order_cnt.items():
        assert -k in order_cnt and order_cnt[-k] == v

    layers_per_chunk = 2
    num_layers_per_chunk = [layers_per_chunk] * num_model_chunks
    # disable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, False
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk
    assert len(chunk_id_list) == len(overlapped_order)
    order_cnt = {}
    accumulated_order = 0
    for o in overlapped_order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0

    # enable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, True
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk * 3 // 2
    assert len(chunk_id_list) == len(overlapped_order)
    from math import ceil

    order_cnt = {}
    accumulated_order = 0
    prev_o = 0
    for o in overlapped_order:
        if ceil(o) != o:
            assert prev_o - 0.5 == o
        else:
            order_cnt[o] = order_cnt.get(o, 0) + 1
            if o < 0:
                assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
            elif -o in order_cnt:
                assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        prev_o = o
    assert accumulated_order < 0

    Utils.destroy_model_parallel()


def test_forward_backward_func_without_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_forward_backward_func_with_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256, device=get_current_device()), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=[model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]
    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256, device=get_current_device()), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_uneven_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256, device=get_current_device()), loss_func

    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(8, 1)
    model_a.vp_stage = 0
    model_b.vp_stage = 1

    def set_input_tensor(input_tensor):
        return None

    model_a.set_input_tensor = set_input_tensor
    model_b.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model_a.config = config
    model_b.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_without_interleaving produces the same output
    with and without explicit process group parameters."""

    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256, device=get_current_device()), loss_func

    # Create model
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    # Mock custom_backward to avoid actual computation
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    # Common arguments for both calls
    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': None,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    # First call: without providing process group parameters (they'll be created internally)
    losses_reduced_default = schedule.forward_backward_pipelining_without_interleaving(
        **common_args
    )

    grid = HyperCommGrid([2, 1, 4, 1], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = dp_cp_group
    pg_collection.cp = grid.create_pg("cp")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=p2p_communicator, pg_collection=pg_collection, **common_args
    )

    assert len(losses_reduced_default) == len(
        losses_reduced_explicit
    ), "Output lengths should be identical"

    for i, (default_loss, explicit_loss) in enumerate(
        zip(losses_reduced_default, losses_reduced_explicit)
    ):
        assert (
            default_loss == explicit_loss
        ), f"Loss at index {i} should be identical between default and explicit PG calls"
    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_with_interleaving produces the same output
    with and without explicit process group parameters."""

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256, device=get_current_device()), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.cp = grid.create_pg("cp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = grid.create_pg(["dp", "cp"])

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_no_pipelining_with_custom_pgs(mocker):
    """Validate no-pipeline schedule when explicit custom PGs are provided."""

    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank_local = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank_local, {'loss_reduced': rank_local}

        dummy_inp = torch.ones(1, 4)
        return model(dummy_inp), loss_func

    # Simple model.
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'
    model.set_input_tensor = lambda _tensor: None  # type: ignore[assignment]

    # Minimal config.
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    grid = HyperCommGrid([2, 1, 1, 4], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.pp = pp_group
    pg_collection.dp_cp = dp_cp_group

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 10),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
        pg_collection=pg_collection,
    )

    expected = {'loss_reduced': Utils.rank}
    for l in losses_reduced:
        assert l['loss_reduced'] == expected['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.experimental
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_dsv4_pp2_real_forward_backward():
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires a multiple of two ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(240)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = _make_config(
            num_layers=2,
            csa_compress_ratios=[0, 4],
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_window_size=16,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=128,
            dsa_indexer_loss_coeff=0.2,
            dsa_indexer_use_sparse_loss=True,
            dsa_kernel_backend="cudnn",
            qk_layernorm=True,
            apply_rope_fusion=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            variable_seq_lengths=True,
            gradient_accumulation_fusion=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
                    config
                ),
                vocab_size=128,
                max_sequence_length=128,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
                pg_collection=pg,
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
            )
            .cuda()
            .to(torch.bfloat16)
        )
        cu = torch.tensor([0, 32, 64, 96, 128], device="cuda", dtype=torch.int32)
        real = torch.tensor([0, 31, 62, 93, 124], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=32,
            max_seqlen_kv=32,
        )
        positions = torch.arange(32, device="cuda").repeat(4).view(1, -1)
        ids = (torch.arange(128, device="cuda") % 127).view(1, -1)
        mask = positions.eq(31)

        def loss_func(output):
            loss = output.float().masked_fill(mask, 0).sum()
            count = (~mask).sum()
            return loss, count, {"lm loss": torch.stack((loss.detach(), count.float()))}

        def forward_step(iterator, module):
            output = module(
                ids if module.pre_process else None,
                positions,
                None,
                labels=ids if module.post_process else None,
                packed_seq_params=packed,
                padding_mask=mask,
            )
            return output, loss_func

        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=None,
            model=[model],
            num_microbatches=2,
            seq_length=64,
            micro_batch_size=2,
            forward_only=False,
        )
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        if model.post_process:
            assert len(losses) == 2
            assert all(torch.isfinite(item['lm loss']).all() for item in losses)
    finally:
        Utils.destroy_model_parallel()
