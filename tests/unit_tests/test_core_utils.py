# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core import config
from megatron.core import num_microbatches_calculator
from megatron.core import optimizer_param_scheduler
from megatron.core import parallel_state
from megatron.core import process_groups_config
from megatron.core import utils
from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.pipeline_parallel import schedules
from megatron.core.transformer.moe import moe_utils


def test_null_decorator_direct_and_factory_modes():
    def fn():
        return "ok"

    direct_context = utils.null_decorator(fn)
    assert direct_context.gen is fn

    factory_context = utils.null_decorator("unused")
    with pytest.raises(TypeError, match="not an iterator"):
        with factory_context:
            pass


def test_experimental_fn_respects_global_flag(monkeypatch):
    calls = []

    @utils.experimental_fn("0.1.0")
    def experimental(value):
        calls.append(value)
        return value + 1

    monkeypatch.setattr(config, "ENABLE_EXPERIMENTAL", True)
    assert experimental(3) == 4
    assert calls == [3]

    monkeypatch.setattr(config, "ENABLE_EXPERIMENTAL", False)
    with pytest.raises(utils.ExperimentalNotEnabledError):
        experimental(3)


def test_process_group_helpers_handle_uninitialized_none_and_list_groups(monkeypatch):
    class FakeGroup:
        def __init__(self, size, rank):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: False)
    assert utils.get_pg_size(FakeGroup(8, 3)) == 1
    assert utils.get_pg_rank(FakeGroup(8, 3)) == 0
    assert utils.get_pg_src_rank(FakeGroup(8, 3)) == 0

    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.torch.distributed, "get_process_group_ranks", lambda group: [5, 6])
    assert utils.get_pg_size([FakeGroup(4, 2)]) == 4
    assert utils.get_pg_rank([FakeGroup(4, 2)]) == 2
    assert utils.get_pg_src_rank(FakeGroup(4, 2)) == 5


def test_tensor_parallel_group_helper_warns_and_selects_default_groups(monkeypatch):
    calls = []
    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(utils.parallel_state, "is_initialized", lambda: True)
    monkeypatch.setattr(
        utils.parallel_state,
        "get_tensor_model_parallel_group",
        lambda check_initialized=True: calls.append(("tp", check_initialized)) or "tp-group",
    )
    monkeypatch.setattr(
        utils.parallel_state,
        "get_expert_tensor_parallel_group",
        lambda check_initialized=True: calls.append(("expert", check_initialized))
        or "expert-group",
    )

    with pytest.warns(DeprecationWarning, match="tp_group is None"):
        assert utils.get_tensor_model_parallel_group_if_none(None) == "tp-group"
    with pytest.warns(DeprecationWarning, match="tp_group is None"):
        assert (
            utils.get_tensor_model_parallel_group_if_none(None, is_expert=True, check_initialized=False)
            == "expert-group"
        )
    assert ("tp", True) in calls
    assert ("expert", False) in calls


def test_wrapped_model_attribute_helpers_find_nested_attributes():
    config_obj = object()
    leaf = SimpleNamespace(model_type="decoder", xattn_needed=True, config=config_obj)
    wrapped = SimpleNamespace(module=SimpleNamespace(module=leaf))

    assert utils.get_attr_wrapped_model(wrapped, "model_type") == "decoder"
    assert utils.get_attr_wrapped_model(wrapped, "model_type", return_model_obj=True) is leaf
    assert utils.get_model_type(wrapped) == "decoder"
    assert utils.get_model_xattn(wrapped) is True
    assert utils.get_model_xattn(SimpleNamespace()) is False
    assert utils.get_model_config(wrapped) is config_obj

    with pytest.raises(RuntimeError, match="given a list"):
        utils.get_attr_wrapped_model([wrapped], "model_type")
    with pytest.raises(RuntimeError, match="couldn't find"):
        utils.get_attr_wrapped_model(SimpleNamespace(), "missing")


def test_viewless_tensor_and_wrapped_tensor_paths():
    base = torch.arange(6.0)
    view = base.view(2, 3)

    viewless = utils.make_viewless_tensor(view, requires_grad=True, keep_graph=False)
    assert viewless._base is None
    assert viewless.requires_grad
    assert torch.equal(viewless, view)

    autograd_viewless = utils.make_viewless_tensor(view, requires_grad=False, keep_graph=True)
    assert autograd_viewless._base is None
    assert not autograd_viewless.requires_grad

    wrapper = utils.WrappedTensor(viewless)
    assert wrapper.unwrap() is viewless
    with pytest.raises(RuntimeError, match="already been unwrapped"):
        wrapper.unwrap()

    with pytest.raises(AssertionError, match="Ensure tensor._base is None"):
        utils.assert_viewless_tensor(view)


def test_init_method_helpers_capture_expected_standard_deviations():
    normal = utils.init_method_normal(0.02)
    scaled = utils.scaled_init_method_normal(0.02, num_layers=2, multiplier=2.0)
    mup_scaled = utils.mup_scaled_init_method_normal(0.02, num_layers=2, width_mult=4.0)

    assert normal.keywords["std"] == pytest.approx(0.02)
    assert scaled.keywords["std"] == pytest.approx(0.01)
    assert mup_scaled.keywords["std"] == pytest.approx(0.005)


def test_log_on_each_pipeline_stage_uses_explicit_groups(monkeypatch):
    records = []
    logger = logging.getLogger("test-log-on-each-pipeline-stage")

    class FakeGroup:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(logger, "log", lambda *args, **kwargs: records.append((args, kwargs)))

    utils.log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        "visible",
        tp_group=FakeGroup(0),
        dp_cp_group=FakeGroup(0),
        extra={"key": "value"},
    )
    utils.log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        "hidden",
        tp_group=FakeGroup(1),
        dp_cp_group=FakeGroup(0),
    )

    assert len(records) == 1
    assert records[0][0][:2] == (logging.INFO, "visible")

    with pytest.raises(ValueError, match="must be provided"):
        utils.log_on_each_pipeline_stage(logger, logging.INFO, "bad", tp_group=FakeGroup(0))


def test_checkpoint_sharded_tensor_helpers_build_offsets(monkeypatch):
    calls = []

    class FakeGroup:
        def __init__(self, rank, size):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    def fake_from_rank_offsets(key, tensor, *offsets, **kwargs):
        calls.append((key, tuple(offsets), kwargs))
        return {"key": key, "offsets": tuple(offsets), "kwargs": kwargs}

    monkeypatch.setattr(
        utils.ShardedTensor, "from_rank_offsets", staticmethod(fake_from_rank_offsets)
    )
    tensor = torch.ones(2, 3)

    tp = utils.make_tp_sharded_tensor_for_checkpoint(
        tensor,
        "tp",
        tp_axis=1,
        prepend_offsets=((0, 7, 9),),
        tp_group=FakeGroup(rank=1, size=4),
        dp_cp_group=FakeGroup(rank=2, size=8),
    )
    replicated = utils.make_sharded_tensor_for_checkpoint(
        tensor,
        "replicated",
        tp_group=FakeGroup(rank=3, size=4),
        dp_cp_group=FakeGroup(rank=2, size=8),
    )

    assert tp["offsets"] == ((0, 7, 9), (2, 1, 4))
    assert tp["kwargs"]["replica_id"] == (0, 0, 2)
    assert replicated["offsets"] == ()
    assert replicated["kwargs"]["replica_id"] == (0, 3, 2)
    assert [call[0] for call in calls] == ["tp", "replicated"]


def test_prepare_input_tensors_for_wgrad_compute_flattens_3d_inputs():
    grad_output = torch.arange(24.0).view(2, 3, 4).transpose(0, 1)
    gathered_input = torch.arange(24.0).view(2, 3, 4).transpose(0, 1)

    grad_output, gathered_input = utils.prepare_input_tensors_for_wgrad_compute(
        grad_output, gathered_input
    )

    assert grad_output.shape == (6, 4)
    assert gathered_input.shape == (6, 4)
    assert grad_output.is_contiguous()
    assert gathered_input.is_contiguous()


def test_local_multi_tensor_applier_scale_and_value_with_rank():
    src = torch.tensor([1.0, 2.0])
    dst = torch.zeros(2)
    utils.local_multi_tensor_applier(
        utils.local_multi_tensor_scale,
        torch.zeros(1),
        [[src], [dst]],
        3.0,
    )
    assert dst.tolist() == [3.0, 6.0]

    low = utils._ValueWithRank(1.25, 3, "ms")
    high = utils._ValueWithRank(2.5, 7, "ms")
    assert low < high
    assert high > low
    assert low() == (1.25, 3, "ms")
    assert str(low) == "1.25ms/3"


def test_context_parallel_batch_slicing_with_explicit_group():
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    batch = {
        "tokens": torch.arange(16).view(1, 16),
        "labels": torch.arange(16).view(1, 16),
        "attention_mask": torch.arange(16).view(1, 1, 16, 1),
        "optional": None,
    }

    sliced = utils.get_batch_on_this_cp_rank(batch, cp_group=FakeGroup())

    assert sliced["tokens"].tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
    assert sliced["labels"].tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
    assert sliced["attention_mask"].shape == (1, 1, 8, 1)
    assert sliced["optional"] is None


def test_nvtx_stack_and_decorator_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(utils.cur_platform, "range_push", lambda msg: calls.append(("push", msg)))
    monkeypatch.setattr(utils.cur_platform, "range_pop", lambda: calls.append("pop"))
    monkeypatch.setattr(utils, "HAVE_NVTX", False)
    utils._nvtx_range_messages.clear()
    utils.configure_nvtx_profiling(True)

    utils.nvtx_range_push("section", "suffix")
    assert calls == [("push", "section.suffix")]

    with pytest.raises(ValueError, match="last range"):
        utils.nvtx_range_pop("wrong")
    utils._nvtx_range_messages.clear()

    utils.nvtx_range_push("section")
    utils.nvtx_range_pop("section")
    assert calls[-1] == "pop"

    def fn():
        return "decorated"

    utils.configure_nvtx_profiling(False)
    assert utils.nvtx_decorator(message="msg")(fn) is fn


def test_unwrap_model_custom_wrappers_and_async_loop(monkeypatch):
    class Wrapper:
        def __init__(self, module):
            self.module = module

    leaf = object()
    assert utils.unwrap_model(Wrapper(Wrapper(leaf)), module_instances=(Wrapper,)) is leaf
    assert utils.unwrap_model([Wrapper(leaf)], module_instances=(Wrapper,)) == [leaf]

    loop = asyncio.new_event_loop()
    try:
        assert utils.get_asyncio_loop(loop) is loop
    finally:
        loop.close()


def test_trace_async_exceptions_success_paths():
    @utils.trace_async_exceptions(verbose=True)
    async def coro(value):
        return value + 1

    @utils.trace_async_exceptions
    async def agen():
        yield "a"
        yield "b"

    async def run():
        values = []
        async for item in agen():
            values.append(item)
        return await coro(4), values

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()

    assert result == (5, ["a", "b"])

    with pytest.raises(TypeError, match="async functions or generators"):
        utils.trace_async_exceptions(lambda: None)


def test_deprecated_decorator_adds_metadata_and_warning():
    @utils.deprecated("1.0", removal_version="2.0", alternative="new_fn", reason="cleanup")
    def old_fn(value):
        return value * 2

    assert old_fn._deprecated is True
    assert old_fn._deprecated_version == "1.0"
    assert old_fn._removal_version == "2.0"
    assert old_fn._alternative == "new_fn"
    assert old_fn._deprecation_reason == "cleanup"
    with pytest.warns(DeprecationWarning, match="Use new_fn instead"):
        assert old_fn(3) == 6


def test_submodule_detection_respects_strict_mode():
    parent = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    child = parent[0]

    assert utils.is_submodule(child, parent)
    assert utils.is_submodule(parent, parent, strict=False)
    assert not utils.is_submodule(parent, parent, strict=True)


def test_misc_compatibility_helpers_and_quantization_flags():
    @utils.internal_api
    def internal():
        return "internal"

    @utils.experimental_api
    def experimental():
        return "experimental"

    @utils.deprecate_args("old")
    def accepts_new(**kwargs):
        return kwargs

    assert internal._internal_api is True
    assert experimental._experimental_api is True
    assert accepts_new(new=1) == {"new": 1}
    with pytest.raises(TypeError, match="old"):
        accepts_new(old=1)

    assert utils.deprecate_inference_params("context", "params") == "context"
    with pytest.warns(UserWarning, match="renamed to `inference_context`"):
        assert utils.deprecate_inference_params(None, "params") == "params"

    assert utils.is_using_quantization_scales(SimpleNamespace(fp8=True)) is True
    assert utils.is_using_quantization_scales(SimpleNamespace(fp4=True)) is True
    assert utils.is_using_quantization_scales(SimpleNamespace()) is False


def test_dtensor_local_helpers_return_plain_tensor_paths():
    tensor = torch.ones(2)

    assert utils.to_local_if_dtensor(tensor) is tensor
    assert utils.get_data_parallel_group_if_dtensor(tensor) is None


def test_straggler_detector_disabled_and_min_max_paths(monkeypatch):
    detector = utils.StragglerDetector()
    monkeypatch.setattr(utils.StragglerDetector, "_configured", False)
    monkeypatch.setattr(utils.cur_platform, "is_available", lambda: False)

    detector.configure(world=2, rank=1, enabled=True)

    assert detector.enabled is False
    assert detector.elapsed() == (0, 0, 0, 0, 0, 0)
    with detector(bdata=True):
        pass

    monkeypatch.setattr(detector, "_off", False)
    detector.rank = 0
    detector.world = 3
    gathered = []

    def fake_gather_object(value, object_gather_list=None, dst=0):
        gathered.append(value)
        if object_gather_list is not None:
            object_gather_list[:] = [
                {
                    "rank": 2,
                    "time": 5.0,
                    "btime": 50.0,
                    "temp": 35.0,
                    "power": 100.0,
                    "util": 80.0,
                    "clock": 900.0,
                    "flops": 1.0,
                },
                {
                    "rank": 0,
                    "time": 1.0,
                    "btime": 10.0,
                    "temp": 30.0,
                    "power": 90.0,
                    "util": 70.0,
                    "clock": 800.0,
                    "flops": 2.0,
                },
                {
                    "rank": 1,
                    "time": 3.0,
                    "btime": 30.0,
                    "temp": 40.0,
                    "power": 110.0,
                    "util": 85.0,
                    "clock": 950.0,
                    "flops": 0.5,
                },
            ]

    monkeypatch.setattr(utils.torch.distributed, "gather_object", fake_gather_object)
    monkeypatch.setattr(utils.torch.distributed, "barrier", lambda: None)

    data = detector._min_max(2.0, 20.0, 33.0, 95.0, 75.0, 850.0, 1.5)

    assert gathered == [
        {
            "rank": 0,
            "time": 2.0,
            "btime": 20.0,
            "temp": 33.0,
            "power": 95.0,
            "util": 75.0,
            "clock": 850.0,
            "flops": 1.5,
        }
    ]
    assert data.min_elapsed() == (1.0, 0, "ms")
    assert data.max_elapsed() == (5.0, 2, "ms")
    assert data.aflops[0]() == (0.5, 1, "")


def test_parallel_state_rank_group_generation_and_defaults():
    assert parallel_state.generate_masked_orthogonal_rank_groups(
        world_size=8, parallel_size=[2, 2, 2], mask=[False, True, False]
    ) == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert parallel_state.generate_masked_orthogonal_rank_groups(
        world_size=8, parallel_size=[2, 2, 2], mask=[True, False, True]
    ) == [[0, 1, 4, 5], [2, 3, 6, 7]]

    generator = parallel_state.RankGenerator(
        tp=2, ep=1, dp=2, pp=2, cp=1, order="tp-dp-pp", rank_offset=8
    )
    assert generator.get_mask(generator.order, "tp-pp") == [True, False, True, False, False]
    assert generator.get_ranks("tp") == [[8, 9], [10, 11], [12, 13], [14, 15]]
    assert generator.get_ranks("dp") == [[8, 10], [9, 11], [12, 14], [13, 15]]

    with pytest.raises(RuntimeError, match="specified the order"):
        parallel_state.RankGenerator(tp=2, ep=1, dp=1, pp=1, cp=1, order="dp-pp")
    with pytest.raises(AssertionError, match="Both EP and CP"):
        parallel_state.RankGenerator(tp=1, ep=2, dp=1, pp=1, cp=2, order="tp-cp-ep-dp-pp")

    assert parallel_state.default_embedding_ranks([3]) == [3]
    assert parallel_state.default_embedding_ranks([0, 2, 4]) == [0, 4]
    assert parallel_state.default_position_embedding_ranks([5, 7]) == [5]

    comm_cfgs = {}
    parallel_state.overwrite_nccl_comm_cfgs(comm_cfgs, "dp", ("max_ctas", 8))
    parallel_state.overwrite_nccl_comm_cfgs(comm_cfgs, "dp", ("min_ctas", 2))
    assert comm_cfgs == {"dp": {"max_ctas": 8, "min_ctas": 2}}


def test_parallel_state_rank_accessors_use_manual_overrides(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_parallel_context", lambda: None)
    monkeypatch.setattr(parallel_state, "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(parallel_state, "_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(parallel_state, "_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(parallel_state, "_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE", None)
    monkeypatch.setattr(parallel_state, "_MPU_TENSOR_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(parallel_state, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(parallel_state, "_MPU_EXPERT_MODEL_PARALLEL_RANK", None)
    monkeypatch.setattr(parallel_state, "_MPU_EXPERT_TENSOR_PARALLEL_RANK", None)

    parallel_state.set_tensor_model_parallel_world_size(4)
    parallel_state.set_pipeline_model_parallel_world_size(2)
    parallel_state.set_expert_model_parallel_world_size(3)
    parallel_state.set_expert_tensor_parallel_world_size(5)
    parallel_state.set_tensor_model_parallel_rank(1)
    parallel_state.set_pipeline_model_parallel_rank(0)
    parallel_state.set_expert_model_parallel_rank(2)
    parallel_state.set_expert_tensor_parallel_rank(4)

    assert parallel_state.get_tensor_model_parallel_world_size() == 4
    assert parallel_state.get_pipeline_model_parallel_world_size() == 2
    assert parallel_state.get_expert_model_parallel_world_size() == 3
    assert parallel_state.get_expert_tensor_parallel_world_size() == 5
    assert parallel_state.get_tensor_model_parallel_rank() == 1
    assert parallel_state.get_pipeline_model_parallel_rank() == 0
    assert parallel_state.get_expert_model_parallel_rank() == 2
    assert parallel_state.get_expert_tensor_parallel_rank() == 4


def test_process_group_collection_repr_validation_and_mpu_mapping(monkeypatch):
    class FakeGroup:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    collection = process_groups_config.ProcessGroupCollection(
        tp=FakeGroup(2),
        dp=None,
    )
    assert repr(collection) == "ProcessGroupCollection(tp(2), dp(None))"

    with pytest.raises(ValueError, match="Unknown attribute"):
        process_groups_config.ProcessGroupCollection(unknown=FakeGroup(1))

    monkeypatch.setattr(
        parallel_state,
        "get_tensor_model_parallel_group",
        lambda check_initialized=False: "tp",
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pipeline_model_parallel_group",
        lambda check_initialized=False: "pp",
    )
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=False, partial_data_parallel=False: (
            "dp_cp" if with_context_parallel else "dp"
        ),
    )

    from_mpu = process_groups_config.ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "pp", "dp", "dp_cp"]
    )

    assert from_mpu.tp == "tp"
    assert from_mpu.pp == "pp"
    assert from_mpu.dp == "dp"
    assert from_mpu.dp_cp == "dp_cp"
    with pytest.raises(ValueError, match="Invalid process groups"):
        process_groups_config.ProcessGroupCollection.use_mpu_process_groups(["missing"])


def test_process_group_collection_optimizer_fallbacks_without_global_groups(monkeypatch):
    model = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1),
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=1,
            use_distributed_optimizer=False,
        ),
    )
    collection = process_groups_config.ProcessGroupCollection(
        dp="dp",
        expt_dp=None,
        mp="mp",
        tp_ep_pp="tp_ep_pp",
    )

    groups = process_groups_config.ProcessGroupCollection.setup_process_groups_for_optimizer(
        collection,
        [model],
        use_gloo_process_groups=False,
    )

    assert groups["dp_group"] == "dp"
    assert groups["dp_cp_group"] == "dp"
    assert groups["intra_dp_cp_group"] == "dp"
    assert groups["expt_dp_group"] is None
    assert groups["intra_expt_dp_group"] is None
    assert groups["mp_group"] == "mp"
    assert groups["expt_tp_pp_group"] == "tp_ep_pp"
    assert groups["inter_dist_opt_group"] is None
    assert groups["intra_dist_opt_group"] is None
    assert collection.engram_dp is None
    assert collection.engram_embed is None
    assert collection.engram_mp is None

    with pytest.raises(ValueError, match="Gloo process groups are not supported"):
        process_groups_config.ProcessGroupCollection.setup_process_groups_for_optimizer(
            collection,
            [model],
            use_gloo_process_groups=True,
        )

    incomplete = process_groups_config.ProcessGroupCollection(dp="dp")
    with pytest.raises(ValueError, match="expt_dp process group is required"):
        process_groups_config.ProcessGroupCollection.setup_process_groups_for_optimizer(
            incomplete,
            [model],
            use_gloo_process_groups=False,
        )

    model.config.context_parallel_size = 2
    with pytest.raises(ValueError, match="dp_cp process group is required"):
        process_groups_config.ProcessGroupCollection.setup_process_groups_for_optimizer(
            collection,
            [model],
            use_gloo_process_groups=False,
        )


def test_num_microbatches_constant_rampup_and_global_helpers():
    num_microbatches_calculator.destroy_num_microbatches_calculator()
    assert num_microbatches_calculator._round(15, 4) == 12

    constant = num_microbatches_calculator.ConstantNumMicroBatchesCalculator(
        global_batch_size=10,
        micro_batch_size=2,
        data_parallel_size=4,
        decrease_batch_size_if_needed=True,
        rank=0,
    )
    assert constant.get() == 1
    assert constant.get_current_global_batch_size() == 10
    assert constant.get_current_running_global_batch_size() == 8
    assert constant.get_micro_batch_size() == 2

    with pytest.raises(AssertionError, match="not divisible"):
        num_microbatches_calculator.ConstantNumMicroBatchesCalculator(
            global_batch_size=10,
            micro_batch_size=2,
            data_parallel_size=4,
            decrease_batch_size_if_needed=False,
            rank=0,
        )

    rampup = num_microbatches_calculator.RampupBatchsizeNumMicroBatchesCalculator(
        global_batch_size=16,
        micro_batch_size=2,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        rank=0,
        start_global_batch_size=8,
        batch_size_increment=4,
        ramup_samples=100,
    )
    assert rampup.get() == 2
    rampup.update(consumed_samples=50, consistency_check=True, verbose=True)
    assert rampup.get_current_global_batch_size() == 12
    assert rampup.get() == 3
    rampup.update(consumed_samples=101, consistency_check=True, verbose=True)
    assert rampup.get_current_global_batch_size() == 16
    assert rampup.get() == 4

    num_microbatches_calculator.init_num_microbatches_calculator(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=16,
        micro_batch_size=2,
        data_parallel_size=4,
    )
    assert num_microbatches_calculator.get_num_microbatches() == 2
    assert num_microbatches_calculator.get_current_global_batch_size() == 16
    assert num_microbatches_calculator.get_micro_batch_size() == 2
    num_microbatches_calculator.update_num_microbatches(consumed_samples=8)
    num_microbatches_calculator.destroy_num_microbatches_calculator()


def test_optimizer_param_scheduler_helpers_and_state_paths():
    assert optimizer_param_scheduler.get_canonical_lr_for_logging(
        [{"lr": 0.1}, {"default_config": True, "lr": 0.2}]
    ) == 0.2
    assert optimizer_param_scheduler.get_canonical_lr_for_logging([{"lr": 0.1}]) is None
    assert optimizer_param_scheduler.param_group_override_to_tuple({"min_lr": 0.01, "wd_mult": 2}) == (
        ("min_lr", 0.01),
        ("wd_mult", 2),
    )
    assert optimizer_param_scheduler.param_group_override_to_tuple(None) is None
    assert optimizer_param_scheduler.combine_param_group_overrides(
        [None, {"min_lr": 0.01}, {"wd_mult": 2}]
    ) == {"min_lr": 0.01, "wd_mult": 2}
    with pytest.raises(ValueError, match="Conflicting overrides"):
        optimizer_param_scheduler.combine_param_group_overrides(
            [{"min_lr": 0.01}, {"min_lr": 0.02}]
        )

    optimizer = SimpleNamespace(
        param_groups=[
            {"lr": torch.tensor(0.0), "default_config": True, "wd_mult": 0.5},
            {"lr": 0.0, "max_lr": 0.2, "min_lr": 0.02, "wd_mult": 2.0},
        ]
    )
    scheduler = optimizer_param_scheduler.OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.01,
        lr_warmup_steps=2,
        lr_decay_steps=10,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=4,
        wd_incr_style="linear",
    )

    scheduler.step(2)
    assert optimizer.param_groups[0]["lr"].item() == pytest.approx(0.1)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.025)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.2)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.1)

    state = scheduler.state_dict()
    assert state["num_steps"] == 2
    scheduler.load_state_dict(
        {
            "max_lr": 0.1,
            "min_lr": 0.01,
            "lr_warmup_steps": 2,
            "lr_decay_steps": 10,
            "lr_decay_style": "linear",
            "num_steps": 3,
            "start_wd": 0.0,
            "end_wd": 0.1,
            "wd_incr_steps": 4,
            "wd_incr_style": "linear",
        }
    )
    assert scheduler.num_steps == 5

    with pytest.raises(Exception, match="not supported"):
        scheduler.num_steps = 2
        scheduler.wd_incr_style = "bad"
        scheduler.get_wd()


def test_pipeline_schedule_selection_and_shape_helpers():
    assert schedules.get_forward_backward_func(pp_size=1, vp_size=None) is (
        schedules.forward_backward_no_pipelining
    )
    assert schedules.get_forward_backward_func(pp_size=2, vp_size=None) is (
        schedules.forward_backward_pipelining_without_interleaving
    )
    assert schedules.get_forward_backward_func(pp_size=2, vp_size=2) is (
        schedules.forward_backward_pipelining_with_interleaving
    )

    assert schedules.check_first_val_step(first_val_step=True, forward_only=True, cond=True)
    assert not schedules.check_first_val_step(first_val_step=True, forward_only=True, cond=False)
    assert schedules.check_first_val_step(first_val_step=False, forward_only=False, cond=True)

    assert schedules.get_schedule_table(
        num_microbatches=5, num_model_chunks=2, microbatch_group_size_per_vp_stage=3
    ) == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (3, 1), (4, 1)]

    class FakeGroup:
        def __init__(self, size, rank=0):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    class FakeP2P:
        def __init__(self, size, rank, virtual_size=None):
            self.pp_group = FakeGroup(size, rank)
            self.virtual_pipeline_model_parallel_size = virtual_size

    assert schedules.get_pp_rank_microbatches(
        num_microbatches=3,
        num_model_chunks=1,
        microbatch_group_size_per_vp_stage=1,
        p2p_communicator=FakeP2P(size=4, rank=1),
    ) == (3, False, 2, 1)
    assert schedules.get_pp_rank_microbatches(
        num_microbatches=2,
        num_model_chunks=3,
        microbatch_group_size_per_vp_stage=2,
        forward_only=True,
        p2p_communicator=FakeP2P(size=4, rank=2, virtual_size=3),
    ) == (6, True, 6, 0)

    config_obj = SimpleNamespace(variable_seq_lengths=True, sequence_parallel=False, hidden_size=16)
    assert schedules.get_tensor_shapes(
        seq_length=8, micro_batch_size=2, decoder_seq_length=None, config=config_obj
    ) == [()]

    config_obj.variable_seq_lengths = False
    config_obj.sequence_parallel = True
    assert schedules.get_tensor_shapes(
        seq_length=16,
        micro_batch_size=2,
        decoder_seq_length=8,
        config=config_obj,
        tp_group=FakeGroup(2),
        cp_group=FakeGroup(2),
    ) == [(2, 2, 16)]


def test_pipeline_deallocation_recurses_and_rejects_views():
    first = torch.ones(2, 3)
    second = torch.ones(4, 5)
    nested = {"first": first, "nested": [second]}

    schedules.deallocate_output_tensor(nested, deallocate_pipeline_outputs=True)

    assert first.shape == (1,)
    assert second.shape == (1,)

    untouched = torch.ones(2, 2)
    schedules.deallocate_output_tensor(untouched, deallocate_pipeline_outputs=False)
    assert untouched.shape == (2, 2)

    with pytest.raises(AssertionError, match="counter-productive"):
        schedules.deallocate_output_tensor(torch.ones(4).view(2, 2), True)


def test_moe_auxiliary_losses_capacity_and_routing_cpu_paths(monkeypatch):
    probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    tokens_per_expert = torch.tensor([1.0, 1.0])
    assert moe_utils.switch_load_balancing_loss_func(
        probs, tokens_per_expert, total_num_tokens=2, topk=1, num_experts=2, moe_aux_loss_coeff=0.5
    ).item() == pytest.approx(0.5)

    logits = torch.tensor([[1.0, 2.0], [0.5, -0.5]])
    assert moe_utils.z_loss_func(logits, z_loss_coeff=0.1).item() > 0
    masked_z = moe_utils.z_loss_func(
        logits, z_loss_coeff=0.1, padding_mask=torch.tensor([False, True])
    )
    assert masked_z.item() > 0

    balanced = moe_utils.sinkhorn(torch.zeros(2, 2), tol=1e-6)
    assert torch.allclose(balanced.sum(dim=0), torch.full((2,), 0.5), atol=1e-5)
    assert torch.allclose(balanced.sum(dim=1), torch.full((2,), 0.5), atol=1e-5)
    assert moe_utils.get_capacity(5, 2, 1.2) == 3
    assert moe_utils.get_capacity(2, 4, 0.5, min_capacity=2) == 2

    class FakeReduceGroup:
        def size(self):
            return 3

    monkeypatch.setattr(
        moe_utils,
        "reduce_from_tensor_model_parallel_region",
        lambda tensor, group: tensor + 1,
    )
    global_counts, local_tokens, total_tokens = moe_utils.get_tokens_per_expert_and_token_count(
        torch.tensor([[True, False], [True, True]]), FakeReduceGroup(), topk=2, with_padding_mask=True
    )
    assert global_counts.tolist() == [3, 2]
    assert local_tokens.item() == pytest.approx(1.5)
    assert total_tokens.item() == pytest.approx(2.5)


def test_moe_permute_unpermute_and_topk_cpu_paths():
    tokens = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    routing_map = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
    probs = torch.tensor([[0.9, 0.0], [0.0, 0.8], [0.7, 0.0]])

    permuted, permuted_probs, sorted_indices, _, _ = moe_utils.permute(
        tokens, routing_map, probs=probs, num_out_tokens=3
    )
    assert sorted_indices.tolist() == [0, 2, 1]
    assert permuted.tolist() == [[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]]
    assert permuted_probs.tolist() == pytest.approx([0.9, 0.7, 0.8])

    restored = moe_utils.unpermute(permuted, sorted_indices, tokens.shape)
    assert torch.equal(restored, tokens)

    sorted_chunks, sorted_probs = moe_utils.sort_chunks_by_idxs(
        torch.arange(10.0).view(5, 2),
        split_sizes=torch.tensor([2, 1, 2]),
        sorted_idxs=torch.tensor([2, 0, 1]),
        probs=torch.arange(5.0),
    )
    assert sorted_chunks.tolist() == [[6.0, 7.0], [8.0, 9.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert sorted_probs.tolist() == [3.0, 4.0, 0.0, 1.0, 2.0]

    padded = moe_utils.pad_routing_map(
        torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.int), pad_multiple=2
    )
    assert padded.sum(dim=0).tolist() == [2, 2]

    logits = torch.tensor([[3.0, 1.0, 2.0, 0.0], [0.0, 2.0, 1.0, 3.0]])
    dense_probs, dense_indices = moe_utils.topk_routing_with_score_function(
        logits, topk=2, score_function="softmax", dense_output=True
    )
    assert dense_indices.tolist() == [[0, 2], [3, 1]]
    assert torch.allclose(dense_probs.sum(dim=1), torch.ones(2))

    routing_probs, routing_mask = moe_utils.topk_routing_with_score_function(
        logits, topk=1, score_function="sigmoid", scaling_factor=2.0
    )
    assert routing_mask.sum(dim=1).tolist() == [1, 1]
    assert torch.all(routing_probs[routing_mask] > 0)

    group_probs, group_indices = moe_utils.group_limited_topk(
        torch.tensor([[0.1, 0.9, 0.8, 0.2]]),
        topk=2,
        num_tokens=1,
        num_experts=4,
        num_groups=2,
        group_topk=1,
    )
    assert group_indices.shape == (1, 2)
    assert group_probs.shape == (1, 2)

    aux_map, aux_scores = moe_utils.compute_routing_scores_for_aux_loss(
        logits, topk=2, score_function="softmax", padding_mask=torch.tensor([False, True])
    )
    assert aux_map[0].sum().item() == 2
    assert aux_map[1].sum().item() == 0
    assert aux_scores[1].sum().item() == 0

    final_probs, final_map = moe_utils.apply_router_token_dropping(
        probs,
        routing_map,
        router_topk=1,
        capacity_factor=0.5,
        drop_policy="probs",
    )
    assert final_map.sum(dim=0).tolist() == [1, 1]
    assert torch.equal(final_probs > 0, final_map)
    with pytest.raises(ValueError, match="Invalid drop_policy"):
        moe_utils.apply_router_token_dropping(probs, routing_map, 1, 1.0, drop_policy="bad")


def test_optimizer_zero_grad_helper_handles_grad_and_decoupled_grad():
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones(2, requires_grad=True) * 2
    optimizer_module._zero_grad_group_helper([param], set_to_none=False)
    assert param.grad.tolist() == [0.0, 0.0]
    assert not param.grad.requires_grad

    param.decoupled_grad = torch.ones(2)
    optimizer_module._zero_grad_group_helper([param], set_to_none=True, use_decoupled_grad=True)
    assert param.decoupled_grad is None


def test_optimizer_multi_tensor_copy_falls_back_without_overflow_buffer():
    source = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    target = [torch.zeros(2), torch.zeros(1)]

    optimizer_module._multi_tensor_copy_this_to_that(source, target)

    assert target[0].tolist() == [1.0, 2.0]
    assert target[1].tolist() == [3.0]


def test_optimizer_common_step_extract_restore_and_mismatch_detection():
    state_dict = {
        "state": {
            0: {"step": 3, "exp_avg": torch.tensor([1.0])},
            1: {"step": 3, "exp_avg": torch.tensor([2.0])},
            2: {},
        }
    }

    assert optimizer_module.MegatronOptimizer._extract_common_per_param_step(state_dict) == 3
    optimizer_module.MegatronOptimizer._restore_common_per_param_step(state_dict, 9)
    assert [state["step"] for state in state_dict["state"].values()] == [9, 9, 9]

    with pytest.raises(ValueError, match="differs per parameter"):
        optimizer_module.MegatronOptimizer._extract_common_per_param_step(
            {"state": {0: {"step": 1}, 1: {"step": 2}}}
        )


def test_optimizer_filter_and_reorder_param_groups_matches_identifier_keys():
    current_groups = [
        {
            "wd_mult": 1.0,
            "lr_mult": 2.0,
            "is_expert_parallel": False,
            "is_decoupled_lr": False,
            "is_vision_model_param": False,
            "is_engram_parallel": False,
        },
        {
            "pre_wd_mult": 0.5,
            "pre_lr_mult": 1.0,
            "pre_is_expert_parallel": True,
            "pre_is_decoupled_lr": False,
            "pre_is_vision_model_param": False,
            "pre_is_engram_parallel": False,
        },
    ]
    state_groups = [
        {
            "wd_mult": 0.5,
            "lr_mult": 1.0,
            "is_expert_parallel": True,
            "is_decoupled_lr": False,
            "is_vision_model_param": False,
            "is_engram_parallel": False,
            "params": ["expert"],
        },
        {
            "wd_mult": 1.0,
            "lr_mult": 2.0,
            "is_expert_parallel": False,
            "is_decoupled_lr": False,
            "is_vision_model_param": False,
            "is_engram_parallel": False,
            "params": ["dense"],
        },
    ]

    reordered = optimizer_module.MegatronOptimizer._filter_and_reorder_param_groups(
        current_groups, state_groups
    )

    assert [group["params"] for group in reordered] == [["expert"], ["dense"]]
    assert reordered[0]["wd_mult"] == 1.0
    assert reordered[1]["wd_mult"] == 0.5

    with pytest.raises(ValueError, match="Could not find parameter group"):
        optimizer_module.MegatronOptimizer._filter_and_reorder_param_groups(
            current_groups[:1], state_groups[:1]
        )
