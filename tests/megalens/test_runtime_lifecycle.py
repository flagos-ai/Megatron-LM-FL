from __future__ import annotations

import ast
import builtins
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from megatron.core.observability import reset_trace_sink, scoped_forward
from megatron.megalens.runtime import MegaLensRuntime
from megatron.megalens.trace import Tracer, _trace_filename


def _args(**overrides: Any) -> SimpleNamespace:
    values = {
        "trace": True,
        "hardware_monitor": False,
        "trace_mode": 1,
        "trace_dir": "trace_output",
        "trace_interval": 1000,
        "continuous_trace_iterations": 1,
        "trace_granularity": "full",
        "sentinel_hw_sample_ms": 100.0,
        "sentinel_flush_interval": 100,
        "trace_gather_to_rank0": False,
        "trace_cupti_kernels": "off",
        "cuda_graph_impl": "none",
        "profile": False,
        "use_pytorch_profiler": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeScope:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def get(self, key):
        return None

    def set(self, key, value):
        return True


class _FakeTracer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.enabled = False

    def configure(self, args) -> None:
        self.calls.append(("configure", args))

    def iteration_begin(self, iteration_id, enable_hw_monitor=False) -> None:
        self.calls.append(("begin", iteration_id, enable_hw_monitor))
        self.enabled = True

    def iteration_end(self, enable_hw_monitor=False) -> None:
        self.calls.append(("end", enable_hw_monitor))
        self.enabled = False

    def abort_iteration(self) -> None:
        self.calls.append(("abort",))
        self.enabled = False

    def shutdown(self, *, graceful: bool) -> None:
        self.calls.append(("shutdown", graceful))
        self.enabled = False

    def is_event_enabled(self, name: str) -> bool:
        return self.enabled

    def scope(self, name, **kwargs):
        self.calls.append(("scope", name))
        return _FakeScope()


@pytest.fixture(autouse=True)
def _reset_sink():
    reset_trace_sink()
    yield
    reset_trace_sink()


def test_runtime_closes_successful_iteration_and_uses_one_based_id() -> None:
    tracer = _FakeTracer()
    runtime = MegaLensRuntime(_args(), tracer=tracer)

    with runtime.iteration(1, enable_hw_monitor=True):
        pass

    assert tracer.calls[1:] == [("begin", 1, True), ("end", True)]


def test_runtime_aborts_iteration_and_preserves_training_exception() -> None:
    tracer = _FakeTracer()
    runtime = MegaLensRuntime(_args(), tracer=tracer)
    failure = RuntimeError("model failure")

    with pytest.raises(RuntimeError) as raised:
        with runtime.iteration(7):
            raise failure

    assert raised.value is failure
    assert tracer.calls[-1] == ("abort",)


def test_runtime_aborts_when_iteration_finalization_fails() -> None:
    class _FinalizationFailureTracer(_FakeTracer):
        def iteration_end(self, enable_hw_monitor=False) -> None:
            self.calls.append(("end", enable_hw_monitor))
            raise RuntimeError("trace finalization failed")

    tracer = _FinalizationFailureTracer()
    runtime = MegaLensRuntime(_args(), tracer=tracer)

    with pytest.raises(RuntimeError, match="trace finalization failed"):
        with runtime.iteration(9):
            pass

    assert tracer.calls[-2:] == [("end", False), ("abort",)]
    assert not tracer.enabled


def test_training_iteration_scope_matches_the_source_envelope() -> None:
    training_path = Path(__file__).resolve().parents[2] / "megatron/training/training.py"
    module = ast.parse(training_path.read_text(encoding="utf-8"))
    train = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    scopes = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Name)
        and node.items[0].context_expr.id == "megalens_iteration"
    ]

    assert len(scopes) == 1
    scope = scopes[0]
    calls = [node for node in ast.walk(scope) if isinstance(node, ast.Call)]

    def named_calls(name: str) -> list[ast.Call]:
        return sorted(
            [
                call
                for call in calls
                if (
                    (isinstance(call.func, ast.Name) and call.func.id == name)
                    or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
                )
            ],
            key=lambda call: call.lineno,
        )

    profiler_step = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "prof"
        and call.func.attr == "step"
    )
    async_finalize = named_calls("maybe_finalize_async_save")
    microbatch_updates = named_calls("update_num_microbatches")
    train_steps = named_calls("train_step")
    checkpoints = named_calls("save_checkpoint_and_time")
    pre_hook_enables = named_calls("enable_forward_pre_hook")
    graph_captures = named_calls("create_cudagraphs")

    assert len(async_finalize) == 1
    assert len(microbatch_updates) == 2
    assert len(train_steps) == 1
    assert len(checkpoints) == 2
    assert len(pre_hook_enables) == 2
    assert len(graph_captures) == 1
    assert (
        scope.lineno
        < profiler_step.lineno
        < async_finalize[0].lineno
        < microbatch_updates[0].lineno
        < microbatch_updates[1].lineno
        < train_steps[0].lineno
        < checkpoints[-1].lineno
        < pre_hook_enables[-1].lineno
        <= scope.end_lineno
    )

    should_exit = next(
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "should_exit"
    )
    assert any(isinstance(node, ast.Break) for node in ast.walk(should_exit))

    normal_iteration_advance = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "iteration"
        and node.lineno > scope.end_lineno
    )
    assert normal_iteration_advance.lineno > scope.end_lineno
    for name in (
        "training_log",
        "evaluate_and_print_results",
        "post_training_step_callbacks",
        "checkpoint_and_decide_exit",
    ):
        call = next(
            node
            for node in ast.walk(train)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )
        assert call.lineno > scope.end_lineno


def test_training_iteration_admission_restores_the_source_mode1_barrier() -> None:
    training_path = Path(__file__).resolve().parents[2] / "megatron/training/training.py"
    module = ast.parse(training_path.read_text(encoding="utf-8"))
    train = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    loop = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.While)
        and any(
            isinstance(statement, ast.Assign)
            and ast.unparse(statement.targets[0]) == "skip_iteration"
            for statement in node.body
        )
    )
    assignments = {
        ast.unparse(statement.targets[0]): statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
    }
    assert ast.unparse(assignments["skip_iteration"].value) == (
        "iteration + 1 in args.iterations_to_skip"
    )
    assert ast.unparse(assignments["trace_iteration"].value) == (
        "getattr(args, 'trace', False) and (not skip_iteration)"
    )

    trace_branch = next(
        statement
        for statement in loop.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "trace_iteration"
    )
    mode_branch = next(
        statement
        for statement in trace_branch.body
        if isinstance(statement, ast.If)
    )
    assert ast.unparse(mode_branch.test) == "not megalens_runtime.tracer.is_mode0()"
    barriers = [
        node
        for node in ast.walk(mode_branch)
        if isinstance(node, ast.Call) and ast.unparse(node) == "torch.distributed.barrier()"
    ]
    assert len(barriers) == 1
    barrier = barriers[0]
    assert barrier.args == []
    assert barrier.keywords == []
    assert "is_tracing_active" not in ast.unparse(trace_branch)

    runtime_assignment = next(
        statement
        for statement in trace_branch.body
        if isinstance(statement, ast.Assign)
        and ast.unparse(statement.targets[0]) == "megalens_iteration"
    )
    assert ast.unparse(runtime_assignment.value) == (
        "megalens_runtime.iteration(iteration + 1, "
        "enable_hw_monitor=getattr(args, 'hardware_monitor', False))"
    )
    assert [ast.unparse(statement) for statement in trace_branch.orelse] == [
        "megalens_iteration = nullcontext()"
    ]

    scope = next(
        node
        for node in loop.body
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Name)
        and node.items[0].context_expr.id == "megalens_iteration"
    )
    assert barrier.lineno < scope.lineno


def test_tracer_discards_partial_records_when_iteration_finalization_fails() -> None:
    tracer = Tracer()
    tracer.configure(_args())
    tracer._records = [{"iteration": 8}, {"iteration": 9, "partial": True}]
    tracer._iteration_record_start = 1
    tracer._iteration_open = True
    tracer._iteration_end_impl = lambda enable_hw_monitor=False: (_ for _ in ()).throw(
        RuntimeError("trace finalization failed")
    )

    with pytest.raises(RuntimeError, match="trace finalization failed"):
        tracer.iteration_end()

    assert tracer._records == [{"iteration": 8}]
    assert not tracer._iteration_open
    tracer.shutdown(graceful=False)


def test_runtime_shutdown_is_idempotent_and_restores_null_sink() -> None:
    tracer = _FakeTracer()
    runtime = MegaLensRuntime(_args(), tracer=tracer)
    runtime.shutdown(graceful=False)
    runtime.shutdown(graceful=True)

    @scoped_forward("after-shutdown")
    def after_shutdown() -> int:
        return 1

    assert after_shutdown() == 1
    assert tracer.calls.count(("shutdown", False)) == 1
    assert not any(call[0] == "scope" for call in tracer.calls)


def test_tracer_event_window_closes_before_eval_or_postprocessing() -> None:
    tracer = Tracer()
    tracer.configure(_args(trace_interval=2, continuous_trace_iterations=1))
    tracer.iter = 1
    tracer._pendings = []
    tracer._iteration_open = True
    assert tracer.is_event_enabled("attention")

    tracer._iteration_open = False
    assert not tracer.is_event_enabled("attention")
    tracer.iter = 2
    tracer._iteration_open = True
    tracer._pendings = []
    assert not tracer.is_event_enabled("attention")
    tracer._iteration_open = False
    tracer.shutdown(graceful=False)


def test_tracer_abort_discards_only_incomplete_iteration_records() -> None:
    tracer = Tracer()
    tracer.configure(_args())
    tracer._records = [{"iteration": 1}, {"iteration": 2, "partial": True}]
    tracer._iteration_record_start = 1
    tracer._iteration_open = True
    tracer.iter = 2
    tracer._stop_kernel_profiler_and_extract = lambda: tracer._records.extend(
        (
            {"iteration": 1, "record_type": "cuda_kernel"},
            {"iteration": 2, "record_type": "cuda_kernel"},
        )
    )

    tracer.abort_iteration()

    assert tracer._records == [
        {"iteration": 1},
        {"iteration": 1, "record_type": "cuda_kernel"},
    ]
    tracer.shutdown(graceful=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"trace": False, "hardware_monitor": True},
        {"trace_interval": 0},
        {"continuous_trace_iterations": 0},
        {"trace_interval": 2, "continuous_trace_iterations": 3},
        {"sentinel_hw_sample_ms": 0},
        {"sentinel_flush_interval": 0},
        {"trace_cupti_kernels": "on", "profile": True, "use_pytorch_profiler": True},
    ],
)
def test_tracer_rejects_unsafe_runtime_configuration(overrides) -> None:
    tracer = Tracer()
    with pytest.raises(ValueError):
        tracer.configure(_args(**overrides))
    tracer.shutdown(graceful=False)


def test_continuous_kernel_window_attributes_both_iterations(monkeypatch) -> None:
    from torch.autograd import DeviceType

    from megatron.megalens import trace as trace_module
    from megatron.training.arguments import _validate_megalens_args

    args = _args(
        trace_interval=2,
        continuous_trace_iterations=2,
        trace_cupti_kernels="on",
    )
    _validate_megalens_args(args)

    tracer = Tracer()
    tracer.configure(args)
    calls: list[str] = []
    flushed: list[dict[str, Any]] = []
    events = [
        SimpleNamespace(
            device_type=DeviceType.CUDA,
            device_index=0,
            dur=10,
            name=f"kernel-{iteration}",
            time_range=SimpleNamespace(start=start, end=start + 10),
        )
        for iteration, start in ((1, 150), (2, 250))
    ]
    profiler = SimpleNamespace(
        __enter__=lambda: calls.append("start"),
        __exit__=lambda *_: calls.append("stop"),
        events=lambda: events,
    )

    def add_cuda_event(*_args, **_kwargs) -> None:
        assert tracer._pendings is not None
        tracer._pendings.append(
            SimpleNamespace(event=SimpleNamespace(synchronize=lambda: None))
        )

    def process_pending(*_args) -> int:
        tracer._records.append({"iteration": tracer.iter, "rel_ts": 1})
        assert tracer._pendings is not None
        return len(tracer._pendings)

    def log() -> None:
        calls.append("log")
        flushed.extend(tracer._records)
        tracer._records = []

    clock = iter((1_000_000, 1_100_000, 1_200_000))

    def time_ns() -> int:
        assert calls and calls[0] == "start"
        return next(clock)

    monkeypatch.setattr(trace_module.time, "time_ns", time_ns)
    monkeypatch.setattr("torch.profiler.profile", lambda **_kwargs: profiler)
    tracer._calibrate = lambda: 0
    tracer._add_cuda_event = add_cuda_event
    tracer._process_pending_scope = process_pending
    tracer._cached_dp_rank = 0
    tracer._cached_pp_rank = 0
    tracer._cached_tp_rank = 0
    tracer._cached_device = 0
    tracer._cached_global_rank = 0
    tracer._cache_ranks = lambda: None
    tracer.log = log

    tracer.iteration_begin(1)
    tracer.iteration_end()
    assert calls == ["start"]

    tracer.iteration_begin(2)
    tracer.iteration_end()

    kernels = [row for row in flushed if row.get("record_type") == "cuda_kernel"]
    assert calls == ["start", "stop", "log"]
    assert [row["iteration"] for row in kernels] == [1, 2]
    assert [row["duration_us"] for row in kernels] == [10, 10]
    assert [row["iter_rel_start_us"] for row in kernels] == [50, 50]
    assert [row["iter_rel_end_us"] for row in kernels] == [60, 60]
    tracer.shutdown(graceful=False)


@pytest.mark.parametrize("cupti_mode", ("auto", "on", "off"))
def test_mode0_ignores_kernel_capture_settings(cupti_mode: str) -> None:
    from megatron.training.arguments import _validate_megalens_args

    args = _args(
        trace_mode=0,
        trace_cupti_kernels=cupti_mode,
        trace_interval=2,
        continuous_trace_iterations=2,
        profile=True,
        use_pytorch_profiler=True,
    )
    _validate_megalens_args(args)

    tracer = Tracer()
    tracer.configure(args)
    assert not tracer._resolve_kernel_capture_mode()
    tracer.shutdown(graceful=False)


def test_trace_off_global_vars_import_does_not_load_megalens_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import megatron.training.global_vars; "
            "assert 'megatron.megalens.trace' not in sys.modules",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_tracer_accessor_is_safe_when_tracing_is_disabled() -> None:
    from megatron.training import global_vars

    global_vars.shutdown_megalens_runtime(graceful=False)

    assert global_vars.get_tracer() is None


def test_global_runtime_teardown_allows_fresh_reinitialization() -> None:
    from megatron.training import global_vars

    global_vars.shutdown_megalens_runtime(graceful=False)
    global_vars._set_megalens_runtime(_args())
    first = global_vars.get_megalens_runtime().tracer
    global_vars.shutdown_megalens_runtime(graceful=False)
    global_vars._set_megalens_runtime(_args())
    second = global_vars.get_megalens_runtime().tracer
    global_vars.shutdown_megalens_runtime(graceful=False)

    assert first is not second


def test_training_argument_validation_matches_runtime_contract() -> None:
    from megatron.training.arguments import _validate_megalens_args

    _validate_megalens_args(_args())
    with pytest.raises(ValueError, match="continuous-trace-iterations"):
        _validate_megalens_args(_args(trace_interval=2, continuous_trace_iterations=3))
    with pytest.warns(RuntimeWarning, match="graph-external"):
        _validate_megalens_args(
            _args(cuda_graph_impl="local", trace_cupti_kernels="on")
        )
    with pytest.raises(ValueError, match="hardware-monitor requires --trace"):
        _validate_megalens_args(_args(trace=False, hardware_monitor=True))


def test_training_argument_validation_ignores_trace_only_values_when_disabled() -> None:
    from megatron.training.arguments import _validate_megalens_args

    disabled_configurations = (
        {"trace_interval": 0},
        {"continuous_trace_iterations": 0},
        {"trace_interval": 2, "continuous_trace_iterations": 3},
        {"sentinel_hw_sample_ms": 0},
        {"sentinel_flush_interval": 0},
        {"trace_mode": 0, "trace_cupti_kernels": "on"},
        {"cuda_graph_impl": "local", "trace_cupti_kernels": "on"},
        {"trace_cupti_kernels": "on", "profile": True, "use_pytorch_profiler": True},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for overrides in disabled_configurations:
            _validate_megalens_args(_args(trace=False, **overrides))

    assert caught == []


def test_cuda_graph_framework_trace_warns_about_probe_coverage() -> None:
    from megatron.training.arguments import _validate_megalens_args

    with pytest.warns(RuntimeWarning, match="graph-external"):
        _validate_megalens_args(_args(cuda_graph_impl="local", trace_cupti_kernels="off"))


def test_cuda_graph_kernel_capture_is_allowed_in_mode1() -> None:
    tracer = Tracer()
    tracer.configure(_args(cuda_graph_impl="local", trace_cupti_kernels="on"))

    assert tracer._resolve_kernel_capture_mode()

    tracer.shutdown(graceful=False)


def test_flushed_trace_window_reopens_kernel_profiler() -> None:
    tracer = Tracer()
    tracer.configure(
        _args(
            trace_interval=1,
            continuous_trace_iterations=1,
            trace_cupti_kernels="on",
        )
    )
    calls: list[str] = []
    tracer.iter = 1
    tracer._pendings = [
        SimpleNamespace(event=SimpleNamespace(synchronize=lambda: None))
    ]
    tracer._pending_pad_before = 0
    tracer._add_cuda_event = lambda *args, **kwargs: None
    tracer._cache_ranks = lambda: None
    tracer._calibrate = lambda: 1
    tracer._process_pending_scope = lambda *args: tracer._records.append(
        {"rel_ts": 1}
    )
    tracer._stop_kernel_profiler_and_extract = lambda: calls.append("stop")
    tracer.log = lambda: calls.append("log")

    tracer._iteration_end_impl()
    assert tracer._pendings is None

    tracer._start_kernel_profiler = lambda: calls.append("start")
    tracer._iteration_begin_impl(2)

    assert calls == ["stop", "log", "start"]


def test_runtime_warns_when_hardware_monitor_has_no_metric_provider(monkeypatch) -> None:
    from megatron.megalens import hardware_monitor as hardware_monitor_module

    monkeypatch.setattr(hardware_monitor_module, "_HAS_PSUTIL", False)
    monkeypatch.setattr(hardware_monitor_module, "_HAS_NVML", False)
    tracer = _FakeTracer()

    with pytest.warns(RuntimeWarning, match="no metric provider"):
        runtime = MegaLensRuntime(_args(hardware_monitor=True), tracer=tracer)

    assert not runtime.hardware_monitor_capabilities.any_provider_available
    runtime.shutdown(graceful=False)


def test_hardware_monitor_initializes_nvml_only_on_first_start(monkeypatch) -> None:
    from megatron.megalens import hardware_monitor as hardware_monitor_module

    calls: list[str] = []

    class _FakeNvml:
        class NVMLError(Exception):
            pass

        NVML_CLOCK_SM = 0

        @staticmethod
        def nvmlInit() -> None:
            calls.append("init")

        @staticmethod
        def nvmlShutdown() -> None:
            calls.append("shutdown")

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> object:
            calls.append(f"handle:{index}")
            return object()

        @staticmethod
        def nvmlDeviceGetMaxClockInfo(handle: object, clock: int) -> int:
            return 0

    monkeypatch.setattr(hardware_monitor_module, "_HAS_NVML", True)
    monkeypatch.setattr(hardware_monitor_module, "pynvml", _FakeNvml, raising=False)
    monkeypatch.setattr(hardware_monitor_module.torch.cuda, "current_device", lambda: 3)
    monitor = hardware_monitor_module.HardwareMonitor()
    monitor._monitor_loop = lambda: None

    assert calls == []
    monitor.start(0)
    monitor.stop()
    monitor.start(0)
    monitor.stop()
    monitor.shutdown()

    assert calls.count("init") == 1
    assert calls.count("handle:3") == 1
    assert calls.count("shutdown") == 1


def test_writer_initialization_failure_is_synchronous(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace-output"
    trace_path.write_text("occupied", encoding="utf-8")
    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(trace_path)))

    with pytest.raises(OSError):
        tracer._initialize_save_thread(
            rank_local_filename="benchmark-global-0-data-0-pipeline-0-tensor-0.json"
        )

    assert tracer._save_thread is None
    assert tracer._work_queue is None
    tracer.shutdown(graceful=False)


def test_cleanup_removes_only_megalens_owned_trace_files(tmp_path: Path) -> None:
    owned = tmp_path / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    legacy_owned = tmp_path / "benchmark-data-0-pipeline-0-tensor-0.json"
    unrelated = tmp_path / "analysis.json"
    for path in (owned, legacy_owned, unrelated):
        path.write_text("[]", encoding="utf-8")
    tracer = Tracer()

    tracer._cleanup_trace_dir(str(tmp_path))

    assert not owned.exists()
    assert not legacy_owned.exists()
    assert unrelated.exists()
    tracer.shutdown(graceful=False)


def test_rank_cache_survives_missing_cuda_device_query(monkeypatch) -> None:
    from megatron.megalens import trace as trace_module

    monkeypatch.setattr(trace_module.parallel_state, "get_data_parallel_rank", lambda: 2)
    monkeypatch.setattr(trace_module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(trace_module.parallel_state, "get_tensor_model_parallel_rank", lambda: 4)
    monkeypatch.setattr(trace_module.torch.distributed, "get_rank", lambda: 7)

    def _missing_device() -> int:
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(trace_module.torch.cuda, "current_device", _missing_device)
    tracer = Tracer()

    tracer._cache_ranks()

    assert (
        tracer._cached_dp_rank,
        tracer._cached_pp_rank,
        tracer._cached_tp_rank,
        tracer._cached_global_rank,
        tracer._cached_device,
    ) == (2, 3, 4, 7, 0)
    tracer.shutdown(graceful=False)


def test_trace_log_uses_cached_fallback_without_model_parallel(
    tmp_path: Path, monkeypatch
) -> None:
    from megatron.megalens import trace as trace_module

    def _missing_parallel_state() -> int:
        raise RuntimeError("model parallel is not initialized")

    monkeypatch.setattr(
        trace_module.parallel_state, "get_data_parallel_rank", _missing_parallel_state
    )
    monkeypatch.setattr(
        trace_module.parallel_state,
        "get_pipeline_model_parallel_rank",
        _missing_parallel_state,
    )
    monkeypatch.setattr(
        trace_module.parallel_state,
        "get_tensor_model_parallel_rank",
        _missing_parallel_state,
    )
    monkeypatch.setattr(trace_module.torch.distributed, "get_rank", lambda: 3)

    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(tmp_path)))
    tracer.iter = 1
    tracer._records = [{"name": "iteration", "ph": "B", "iteration": 1}]

    tracer.log()
    tracer.shutdown(graceful=True)

    output = tmp_path / "benchmark-global-3-data-0-pipeline-0-tensor-0.json"
    assert output.exists()
    assert '"iteration": 1' in output.read_text(encoding="utf-8")


def test_trace_shard_filename_is_unique_by_global_rank() -> None:
    rank0 = _trace_filename(global_rank=0, dp_rank=0, pp_rank=0, tp_rank=0, mode0=False)
    rank1 = _trace_filename(global_rank=1, dp_rank=0, pp_rank=0, tp_rank=0, mode0=False)

    assert rank0 != rank1
    assert rank0 == "benchmark-global-0-data-0-pipeline-0-tensor-0.json"


def test_background_writer_failure_propagates_from_shutdown(tmp_path: Path, monkeypatch) -> None:
    from megatron.megalens import trace as trace_module

    trace_dir = tmp_path / "trace-output"
    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(trace_dir)))
    tracer._initialize_save_thread(
        rank_local_filename="benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    )
    assert tracer._work_queue is not None

    real_open = builtins.open

    def _failing_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".json") and "w" in mode:
            raise OSError("simulated trace write failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(trace_module, "open", _failing_open, raising=False)
    tracer._work_queue.put(
        [("benchmark-global-0-data-0-pipeline-0-tensor-0.json", [{"name": "iteration", "ph": "B"}])]
    )

    with pytest.raises(RuntimeError, match="trace writer failed") as raised:
        tracer.shutdown(graceful=False)

    assert isinstance(raised.value.__cause__, OSError)
    assert tracer._closed
    assert tracer._save_thread is None
    assert tracer._work_queue is None


def test_single_process_gather_mode_writes_global_rank_trace(tmp_path: Path, monkeypatch) -> None:
    from megatron.megalens import trace as trace_module

    monkeypatch.setattr(trace_module.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trace_module.torch.distributed, "is_initialized", lambda: False)

    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(tmp_path), trace_gather_to_rank0=True))
    tracer.iter = 1
    tracer._records = [{"name": "iteration", "ph": "B", "iteration": 1}]

    tracer.log()
    tracer.shutdown(graceful=True)

    output = tmp_path / "benchmark-global-0-data-0-pipeline-0-tensor-0.json"
    assert output.exists()
    assert '"iteration": 1' in output.read_text(encoding="utf-8")


def test_rank_local_graceful_shutdown_flushes_without_world_barrier(
    tmp_path: Path, monkeypatch
) -> None:
    from megatron.megalens import trace as trace_module

    monkeypatch.setattr(trace_module.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trace_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_module.torch.distributed, "get_rank", lambda: 3)

    def _unexpected_barrier() -> None:
        raise AssertionError("rank-local trace shutdown must not synchronize WORLD")

    monkeypatch.setattr(trace_module.torch.distributed, "barrier", _unexpected_barrier)
    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(tmp_path)))
    tracer.iter = 1
    tracer._records = [{"name": "shutdown-flush", "ph": "B", "iteration": 1}]

    tracer.shutdown(graceful=True)

    output = tmp_path / _trace_filename(global_rank=3, dp_rank=0, pp_rank=0, tp_rank=0, mode0=False)
    assert tracer._closed
    assert output.exists()
    assert '"name": "shutdown-flush"' in output.read_text(encoding="utf-8")


def test_rank_local_writer_cleanup_preserves_peer_shards(tmp_path: Path, monkeypatch) -> None:
    from megatron.megalens import trace as trace_module

    monkeypatch.setattr(trace_module.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(trace_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trace_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_module.torch.distributed, "get_rank", lambda: 3)

    local_output = tmp_path / _trace_filename(
        global_rank=3, dp_rank=0, pp_rank=0, tp_rank=0, mode0=False
    )
    peer_output = tmp_path / _trace_filename(
        global_rank=4, dp_rank=0, pp_rank=0, tp_rank=0, mode0=False
    )
    local_output.write_text('[{"name": "stale-local"}]', encoding="utf-8")
    peer_output.write_text('[{"name": "peer-record"}]', encoding="utf-8")

    tracer = Tracer()
    tracer.configure(_args(trace_dir=str(tmp_path)))
    tracer.iter = 1
    tracer._records = [{"name": "fresh-local", "ph": "B", "iteration": 1}]

    tracer.log()
    tracer.shutdown(graceful=True)

    local_contents = local_output.read_text(encoding="utf-8")
    assert '"name": "fresh-local"' in local_contents
    assert "stale-local" not in local_contents
    assert peer_output.read_text(encoding="utf-8") == '[{"name": "peer-record"}]'
