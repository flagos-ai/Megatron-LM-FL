# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.dist_checkpointing.exchange_utils as exchange_utils
import megatron.core.dist_checkpointing.validation as validation
import megatron.core.timers as timers_module
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject


def _patch_timer_dist(monkeypatch, rank=1, world_size=2, backend="cpu:gloo"):
    monkeypatch.setattr(timers_module.cur_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: backend)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)

    def _fake_all_gather(output, input_tensor):
        local_values = input_tensor.clone()
        output.zero_()
        output.view(world_size, -1)[rank].copy_(local_values)

    monkeypatch.setattr(timers_module, "dist_all_gather_func", _fake_all_gather)


def test_timers_dummy_real_elapsed_gather_log_and_writer_paths(monkeypatch, caplog):
    _patch_timer_dist(monkeypatch)
    clock = {"value": 10.0}

    def _time():
        value = clock["value"]
        clock["value"] += 0.25
        return value

    monkeypatch.setattr(timers_module.time, "time", _time)

    dummy = timers_module.DummyTimer()
    dummy.start()
    dummy.stop()
    dummy.reset()
    with pytest.raises(Exception, match="dummy timer"):
        dummy.elapsed()
    with pytest.raises(Exception, match="active timer"):
        dummy.active_time()

    timer = timers_module.Timer("core")
    timer.set_barrier_group("group")
    timer.start(barrier=True)
    with pytest.raises(AssertionError, match="already been started"):
        timer.start()
    timer.stop(barrier=True)
    assert timer.elapsed(reset=False) == pytest.approx(0.25)
    assert timer.active_time() == pytest.approx(0.25)
    timer.set_elapsed(3.5)
    assert timer.elapsed(reset=True) == pytest.approx(3.5)
    with pytest.raises(AssertionError, match="timer is not started"):
        timer.stop()

    with pytest.raises(AssertionError, match="invalid"):
        timers_module.Timers(log_level=1, log_option="median")

    timers = timers_module.Timers(log_level=1, log_option="max")
    fast = timers("fast", log_level=1)
    assert timers("fast", log_level=1) is fast
    with pytest.raises(AssertionError, match="does not match"):
        timers("fast", log_level=2)
    with pytest.raises(AssertionError, match="larger than max"):
        timers("too-high", log_level=3)
    assert timers("disabled", log_level=2) is timers._dummy_timer

    fast.set_elapsed(2.0)
    gathered = timers._get_elapsed_time_all_ranks(["fast"], reset=False, barrier=True)
    assert gathered.shape == (2, 1)
    assert gathered[1, 0] == pytest.approx(2.0)
    assert timers._get_elapsed_time_all_ranks([], reset=False, barrier=False) is None

    max_string = timers.get_all_timers_string(["fast"], reset=False)
    assert "max time across ranks" in max_string
    timers_minmax = timers_module.Timers(log_level=1, log_option="minmax")
    timers_minmax("fast", log_level=1).set_elapsed(4.0)
    assert "(min, max)" in timers_minmax.get_all_timers_string(["fast"], reset=False)
    timers_all = timers_module.Timers(log_level=1, log_option="all")
    timers_all("fast", log_level=1).set_elapsed(5.0)
    assert "times across ranks" in timers_all.get_all_timers_string(["fast"], reset=False)
    with pytest.raises(AssertionError):
        timers_all.get_all_timers_string(["fast"], normalizer=0.0)

    caplog.set_level("INFO", logger=timers_module.logger.name)
    timers.log(["fast"], rank=1, reset=False)
    assert "max time across ranks" in caplog.text

    class _Writer:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, name, value, iteration):
            self.scalars.append((name, value, iteration))

    monkeypatch.setattr(timers_module, "SummaryWriter", _Writer)
    writer = _Writer()
    timers.write(["fast"], writer, iteration=9, reset=False)
    assert writer.scalars == [("fast-time", pytest.approx(2.0), 9)]

    logged = []
    wandb = SimpleNamespace(log=lambda payload, step: logged.append((payload, step)))
    monkeypatch.setattr(timers_module, "wandb", wandb)
    timers.write(["fast"], wandb, iteration=10, reset=False)
    assert logged == [({"fast-time": pytest.approx(2.0)}, 10)]


def test_checkpoint_validation_strict_flags_reporting_and_metadata_paths(monkeypatch, caplog):
    assert validation.parse_strict_flag(validation.StrictHandling.LOG_ALL) is validation.StrictHandling.LOG_ALL
    assert validation.parse_strict_flag("raise_all") is validation.StrictHandling.RAISE_ALL
    with pytest.raises(ValueError, match="Invalid strict flag"):
        validation.parse_strict_flag("bad")

    explicit = {
        flag: validation.StrictHandling.requires_explicit_ckpt_mismatch_check(flag)
        for flag in validation.StrictHandling
    }
    assert explicit[validation.StrictHandling.ASSUME_OK_UNEXPECTED] is False
    assert all(v is True for k, v in explicit.items() if k != validation.StrictHandling.ASSUME_OK_UNEXPECTED)
    assert validation.StrictHandling.requires_global_app_metadata(validation.StrictHandling.LOG_ALL)
    assert validation.StrictHandling.requires_returning_mismatch_keys(validation.StrictHandling.RETURN_ALL)

    validation.maybe_report_missing_and_unexpected_keys(set(), set(), raise_error=True)
    caplog.set_level("WARNING", logger=validation.logger.name)
    validation.maybe_report_missing_and_unexpected_keys({"missing"}, {"unexpected"}, raise_error=False)
    assert "Missing keys" in caplog.text
    with pytest.raises(CheckpointingException, match="Unexpected keys"):
        validation.maybe_report_missing_and_unexpected_keys(set(), {"unexpected"}, raise_error=True)

    keep = ShardedTensor.from_rank_offsets("keep", torch.ones(2))
    drop = ShardedTensor.from_rank_offsets("drop", torch.ones(2))
    adjusted = validation.adjust_non_strict_load({"keep": keep, "drop": drop}, {"drop"})
    assert adjusted == {"keep": keep}

    ckpt_meta = SimpleNamespace(values=lambda: [keep.without_data(), drop.without_data()])
    local_meta = [keep.without_data(), ShardedTensor.from_rank_offsets("local-only", torch.ones(1)).without_data()]
    missing, unexpected = validation._determine_missing_and_unexpected_keys(
        ckpt_meta,
        local_meta,
        global_metadata=[[keep.without_data()]],
    )
    assert missing == {"drop"}
    assert unexpected == {"local-only"}

    with pytest.raises(CheckpointingException, match="ckpt_sharded_metadata=None"):
        validation.validate_integrity_and_strict_load(
            {"keep": keep},
            validation.StrictHandling.RAISE_UNEXPECTED,
            validate_access_integrity=False,
        )
    filtered, missing_keys, unexpected_keys = validation.validate_integrity_and_strict_load(
        {"keep": keep, "local-only": ShardedTensor.from_rank_offsets("local-only", torch.ones(1))},
        validation.StrictHandling.RETURN_ALL,
        validate_access_integrity=False,
        ckpt_sharded_metadata=ckpt_meta,
        global_metadata=[[keep.without_data()]],
    )
    assert set(filtered) == {"keep"}
    assert missing_keys == {"drop"}
    assert unexpected_keys == {"local-only"}
    ignored = validation.validate_integrity_and_strict_load(
        {"keep": keep, "local-only": ShardedTensor.from_rank_offsets("local-only", torch.ones(1))},
        validation.StrictHandling.IGNORE_ALL,
        validate_access_integrity=False,
        ckpt_sharded_metadata=ckpt_meta,
        global_metadata=[[keep.without_data()]],
    )
    assert ignored[1:] == (set(), set())

    monkeypatch.setattr(validation.Path, "exists", lambda self: False)
    with pytest.raises(CheckpointingException, match="does not exist"):
        validation.verify_checkpoint("/missing")
    monkeypatch.setattr(validation.Path, "exists", lambda self: True)
    monkeypatch.setattr(validation, "check_is_distributed_checkpoint", lambda path: False)
    with pytest.raises(CheckpointingException, match="not a distributed checkpoint"):
        validation.verify_checkpoint("/plain")
    monkeypatch.setattr(validation, "check_is_distributed_checkpoint", lambda path: True)
    validation.verify_checkpoint("/dist")


def test_checkpoint_validation_sharding_object_common_and_global_metadata_paths(monkeypatch, caplog):
    shard0 = ShardedTensor.from_rank_offsets(
        "tensor",
        torch.ones(1),
        (0, 0, 2),
        replica_id=0,
    )
    shard1 = ShardedTensor.from_rank_offsets(
        "tensor",
        torch.ones(1),
        (0, 1, 2),
        replica_id=0,
    )
    access = validation._compute_shards_access([(0, shard0), (1, shard1)])
    assert torch.equal(access, torch.ones(2, dtype=torch.int))
    assert validation._validate_sharding_for_key([(0, shard0), (1, shard1)]) == []
    assert validation._validate_sharding_for_key([(0, shard0)]) != []

    uneven = ShardedTensor.from_rank_offsets("uneven", torch.ones(2), allow_shape_mismatch=True)
    assert validation._validate_sharding_for_key([(0, uneven)]) == []

    obj0 = ShardedObject("obj", "a", (2,), (0,), replica_id=0)
    obj1 = ShardedObject("obj", "b", (2,), (1,), replica_id=0)
    assert validation._validate_objects_for_key([(0, obj0), (1, obj1)]) == []
    duplicate = ShardedObject("obj", "dup", (2,), (0,), replica_id=0)
    assert validation._validate_objects_for_key([(0, duplicate), (1, duplicate)])

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    validation.validate_sharding_integrity([[shard0], [shard1]])
    with pytest.raises(CheckpointingException, match="Invalid sharding pattern"):
        validation.validate_sharding_integrity([[shard0]])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    validation.validate_sharding_integrity([[shard0]])

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda objects, src: objects.__setitem__(0, {"rank0": 1}),
    )
    caplog.set_level("WARNING", logger=validation.logger.name)
    validation._validate_common_state_dict({"rank1": 2})
    assert "common state dict differs" in caplog.text
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    validation._validate_common_state_dict({"rank": 0})

    local = [shard0.without_data()]
    monkeypatch.setattr(validation, "nested_values", lambda state_dict: local)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value) or output.__setitem__(1, []),
    )
    assert validation.determine_global_metadata({"tensor": shard0}) == (local, [local, []])


class _FakeShard:
    def __init__(
        self,
        key,
        data=None,
        dtype=torch.float32,
        local_shape=(2,),
        replica_id=0,
        shard_id=None,
    ):
        self.key = key
        self.data = data
        self.dtype = dtype
        self.local_shape = local_shape
        self.replica_id = replica_id
        self._shard_id = shard_id if shard_id is not None else (key, local_shape, replica_id)

    def init_data(self, device):
        self.data = torch.empty(*self.local_shape, dtype=self.dtype, device="cpu")

    def without_data(self):
        return _FakeShard(
            self.key,
            None,
            dtype=self.dtype,
            local_shape=self.local_shape,
            replica_id=self.replica_id,
            shard_id=self._shard_id,
        )


def test_exchange_utils_distribution_empty_tensor_object_and_dispatch_paths(monkeypatch):
    monkeypatch.setattr(exchange_utils.cur_platform, "device_name", lambda: "cpu")
    monkeypatch.setattr(exchange_utils.cur_platform, "device", lambda: torch.device("cpu"))
    assert exchange_utils.is_float8tensor(torch.ones(1)) is False

    shard_id = ("weight", 0)
    needed = {shard_id: _FakeShard("weight", data=None)}
    loaded = {}
    tensor, orig_device = exchange_utils._get_empty_tensor_for_exchange(shard_id, needed, {}, loaded)
    assert tensor.shape == (2,)
    assert orig_device == torch.device("cpu")
    assert loaded[shard_id] is tensor

    unneeded = {shard_id: _FakeShard("weight", data=torch.ones(2))}
    tensor, orig_device = exchange_utils._get_empty_tensor_for_exchange(shard_id, {}, unneeded, {})
    assert tensor.shape == (2,)
    assert orig_device is None
    assert exchange_utils._shard_size(_FakeShard("weight", dtype=torch.float32, local_shape=(2, 3))) == 24

    assignment = exchange_utils.distribute_shards_to_ranks(
        {"a": [0, 1], "b": [0], "c": [1]},
        {"a": 8, "b": 4, "c": 12},
        2,
        cross_parallelization_group_loads={"a"},
    )
    assert assignment["b"] == 0
    assert assignment["c"] == 1
    assert assignment["a"] in (0, 1)

    gathered_payloads = [
        {_FakeShard("a", shard_id=("a", 0))._shard_id: torch.tensor([1.0])},
        {_FakeShard("b", shard_id=("b", 0))._shard_id: torch.tensor([2.0])},
    ]
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, gathered_payloads[0])
        or output.__setitem__(1, gathered_payloads[1]),
    )
    merged = exchange_utils.exchange_loaded_tensors_gather_object({}, {}, None)
    assert sorted(merged) == [("a", 0), ("b", 0)]

    duplicate = {("dup", 0): torch.tensor([1.0])}
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, duplicate)
        or output.__setitem__(1, duplicate),
    )
    with pytest.raises(CheckpointingException, match="Duplicate shard ids"):
        exchange_utils.exchange_loaded_tensors_gather_object({}, {}, None)
    with pytest.raises(CheckpointingException, match="Duplicate shard ids"):
        exchange_utils.exchange_loaded_objects_gather_object({})

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, gathered_payloads[0])
        or output.__setitem__(1, gathered_payloads[1]),
    )
    distribution = exchange_utils.ShardDistribution(
        main_rank_for_shard={("a", 0): 0},
        shards_in_this_group={("a", 0)},
        shard_to_metadata={("a", 0): _FakeShard("a", shard_id=("a", 0))},
        all_ranks_for_shard={("a", 0): [0]},
    )
    assert exchange_utils.exchange_by_distribution(
        {("a", 0): torch.ones(1)},
        {},
        distribution,
        exchange_algo="gather_object",
    )
    with pytest.raises(NotImplementedError, match="Unrecognized"):
        exchange_utils.exchange_by_distribution({}, {}, distribution, exchange_algo="bad")
    with pytest.raises(AssertionError, match="Expecting distribution"):
        exchange_utils.exchange_by_distribution({}, {}, None)


def test_exchange_utils_main_replica_distribution_and_broadcast_paths(monkeypatch):
    monkeypatch.setattr(exchange_utils.cur_platform, "device_name", lambda: "cpu")
    monkeypatch.setattr(exchange_utils.cur_platform, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(exchange_utils, "ShardedTensor", _FakeShard)
    monkeypatch.setattr(exchange_utils, "_sharded_tensor_shard_id", lambda shard: shard._shard_id)
    monkeypatch.setattr(exchange_utils, "_shard_size", lambda shard: int(torch.tensor(shard.local_shape).prod()))
    monkeypatch.setattr(exchange_utils, "nested_values", lambda state: state.values())

    rank0_main = _FakeShard("a", replica_id=0, shard_id=("a", 0))
    rank1_non_main = _FakeShard("a", replica_id=1, shard_id=("a", 0))
    class _Group:
        def size(self):
            return 2

        def rank(self):
            return 0

    group = _Group()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, [rank0_main])
        or output.__setitem__(1, [rank1_non_main]),
    )
    distribution = exchange_utils.determine_main_replica_uniform_distribution({"a": rank0_main}, group)
    assert distribution.main_rank_for_shard[("a", 0)] in (0, 1)
    assert distribution.shards_in_this_group == {("a", 0)}
    class _TrivialGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    trivial = _TrivialGroup()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    assert exchange_utils.determine_main_replica_uniform_distribution({"a": rank0_main}, trivial) is None

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src, group=None, async_op=False: broadcasts.append((tensor.clone(), src, async_op)),
    )
    dist = exchange_utils.ShardDistribution(
        main_rank_for_shard={("a", 0): 0, ("b", 0): 1},
        shards_in_this_group={("a", 0), ("b", 0)},
        shard_to_metadata={
            ("a", 0): _FakeShard("a", local_shape=(2,), shard_id=("a", 0)),
            ("b", 0): _FakeShard("b", local_shape=(2,), shard_id=("b", 0)),
        },
        all_ranks_for_shard={("a", 0): [0, 1], ("b", 0): [1]},
    )
    out = exchange_utils.exchange_loaded_tensors_broadcast(
        {("b", 0): torch.tensor([2.0, 3.0])},
        {("a", 0): _FakeShard("a", local_shape=(2,), shard_id=("a", 0))},
        dist,
    )
    assert ("a", 0) in out and ("b", 0) in out
    assert broadcasts[0][1] == 0
