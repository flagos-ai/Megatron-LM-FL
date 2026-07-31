import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

COLOR_UNKNOWN = "thread_state_unknown"
COLOR_FORWARD = "thread_state_running"
COLOR_BACKWARD = "thread_state_iowait"
COLOR_RECV = "rail_response"
COLOR_SEND = "rail_animation"
COLOR_EXCHANGE_NEXT = "thread_state_runnable"
COLOR_EXCHANGE_PREV = "thread_state_uninterruptible"
COLOR_ALLREDUCE = "light_memory_dump"
COLOR_OPTIMIZER = "detailed_memory_dump"

COLOR_MAP = {
    "forward": COLOR_FORWARD,
    "forward-warmup": COLOR_FORWARD,
    "backward": COLOR_BACKWARD,
    "backward-cooldown": COLOR_BACKWARD,
    "recv-extra": COLOR_RECV,
    "recv-warmup": COLOR_RECV,
    "recv-forward": COLOR_RECV,
    "recv-backward": COLOR_RECV,
    "recv-cooldown": COLOR_RECV,
    "send-extra": COLOR_SEND,
    "send-warmup": COLOR_SEND,
    "send-forward": COLOR_SEND,
    "send-backward": COLOR_SEND,
    "send-cooldown": COLOR_SEND,
    "exchange-next": COLOR_EXCHANGE_NEXT,
    "exchange-prev": COLOR_EXCHANGE_PREV,
    "allreduce": COLOR_ALLREDUCE,
    "optimizer": COLOR_OPTIMIZER,
}

DATA_PARALLELISM: int = 0
PIPELINE_PARALLELISM: int = 0
TENSOR_PARALLELISM: int = 0

# TP_ALIGNMENT_EVENTS: List[str] = ["_reduce"]
# PP_ALIGNMENT_EVENTS: List[Tuple[str, str]] = [("recv_backward", "recv_forward")]


@dataclass
class Rank:
    data: int
    pipeline: int
    tensor: int
    global_rank: Optional[int] = None

    def __str__(self) -> str:
        coordinates = f"{self.data}-{self.pipeline}-{self.tensor}"
        return coordinates if self.global_rank is None else f"g{self.global_rank}-{coordinates}"

    def __hash__(self) -> int:
        return hash((self.data, self.pipeline, self.tensor, self.global_rank))

    def to_pid(self, pipeline_paralellism: int, tensor_parallelism: int) -> int:
        if self.global_rank is not None:
            return self.global_rank
        return (
            self.data * pipeline_paralellism * tensor_parallelism
            + self.pipeline * tensor_parallelism
            + self.tensor
        )


@dataclass
class Event:
    rel_ts: int
    rank: Rank
    name: str
    ph: str
    attrs: Any
    cat: Optional[str] = None


@dataclass
class Iteration:
    pad_before: int
    events: List[Event]
    duration: int
    iteration_id: Optional[int] = None
    ranks: Tuple[Rank, ...] = ()


def _validate_iteration_id(value: Any, *, record: str) -> Optional[int]:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{record} has a non-integer iteration ID: {value!r}")
    return value


def _canonicalize_rank_attrs(attrs: Dict[str, Any], rank: Rank, *, event_name: str) -> None:
    expected = {"dp_rk": rank.data, "pp_rk": rank.pipeline, "tp_rk": rank.tensor}
    if rank.global_rank is not None:
        expected["g_rk"] = rank.global_rank
    for field, value in expected.items():
        if field in attrs and attrs[field] != value:
            raise ValueError(
                f"Trace event {event_name!r} has {field}={attrs[field]!r}, "
                f"but its shard identity requires {value}"
            )
        attrs[field] = value


def collect_benchmark_files(dir: os.PathLike) -> List[Tuple[Rank, str]]:
    """Collect benchmark.json files from the given directory.
    Args:
        dir: path to the trace output dir
    Return:
        files: list[(rank, f.read())]
    """
    files = []
    with os.scandir(dir) as it:
        for entry in it:
            file: str = entry.name
            if file.startswith("benchmark-") and file.endswith(".json"):
                desc: str = file[len("benchmark-") : -len(".json")]
                # disc is "data-*-pipeline-*-tensor-*"
                fields = desc.split("-")
                chunks = dict((fields[i], int(fields[i + 1])) for i in range(0, len(fields), 2))
                global_rank = chunks.pop("global", None)
                unknown = set(chunks) - {"data", "pipeline", "tensor"}
                if unknown:
                    raise ValueError(
                        f"Unknown rank fields in benchmark filename {file!r}: {sorted(unknown)}"
                    )
                rank = Rank(**chunks, global_rank=global_rank)
                with open(os.path.join(dir, file), "r") as f:
                    files.append((rank, f.read()))
    return files


def read_benchmark_file(rank: Rank, content: str) -> List[Iteration]:
    """Returns events in each iteration.(one file is one element in contents list)"""
    result = []
    rows: List[Dict[str, Any]] = json.loads(content)
    current_iteration = None
    current_iteration_id = None
    # Round 4: tier-2 cuda_kernel records get appended in batches by
    # _stop_kernel_profiler_and_extract() AFTER iteration E in the source
    # JSON. We buffer them by their own ``iteration`` field and attach
    # them to the matching Iteration after all rows have been read.
    pending_kernel_rows: List[Dict[str, Any]] = []
    for row in rows:
        # Tier-2 CUDA kernel records: keep them aside (no rel_ts / B/E pairing)
        if row.get("record_type") == "cuda_kernel":
            pending_kernel_rows.append(row)
            continue
        if row["name"] == "iteration" and row["ph"] == "B":
            if current_iteration is not None:
                raise ValueError("Nested iteration begin record in benchmark trace")
            pad_before = row["pad_before"]
            current_iteration_id = _validate_iteration_id(
                row.get("iteration"), record="Iteration begin record"
            )
            current_iteration = []
        elif row["name"] == "iteration" and row["ph"] == "E":
            if current_iteration is None:
                raise ValueError("Iteration end record has no matching begin")
            end_iteration_id = _validate_iteration_id(
                row.get("iteration"), record="Iteration end record"
            )
            if end_iteration_id != current_iteration_id:
                raise ValueError(
                    "Iteration boundary IDs do not match: "
                    f"begin={current_iteration_id}, end={end_iteration_id}"
                )
            duration = row["duration_wall"]
            result.append(
                Iteration(
                    pad_before=pad_before if pad_before is not None else 0,
                    events=current_iteration if current_iteration is not None else [],
                    duration=duration,
                    iteration_id=current_iteration_id,
                    ranks=(rank,),
                )
            )
            pad_before = None
            current_iteration = None
            current_iteration_id = None
        else:
            if current_iteration is None:
                # In evaluation, so ignore.
                continue
            attrs = dict(row)
            name = attrs.pop("name")
            rel_ts = attrs.pop("rel_ts")
            ph = attrs.pop("ph")
            cat = attrs.pop("cat", None)
            event_iteration_id = _validate_iteration_id(
                attrs.get("iteration"), record=f"Trace event {name!r}"
            )
            if "iteration" in attrs and event_iteration_id != current_iteration_id:
                raise ValueError(
                    f"Trace event {name!r} has iteration={event_iteration_id}, "
                    f"but its enclosing iteration is {current_iteration_id}"
                )
            if current_iteration_id is not None:
                attrs["iteration"] = current_iteration_id
            _canonicalize_rank_attrs(attrs, rank, event_name=name)
            event = Event(rel_ts=rel_ts, rank=rank, name=name, ph=ph, attrs=attrs, cat=cat)
            current_iteration.append(event)

    if current_iteration is not None:
        raise ValueError(f"Iteration {current_iteration_id} has no matching end record")

    # Attach pending cuda_kernel records to Iterations by their `iteration` field.
    if pending_kernel_rows:
        if not result:
            raise ValueError("CUDA kernel records have no matching iterations")
        iter_index: Dict[int, Iteration] = {
            it.iteration_id: it for it in result if it.iteration_id is not None
        }
        for kr in pending_kernel_rows:
            kr = dict(kr)
            kr_iter = kr.get("iteration")
            if kr_iter is None or kr_iter not in iter_index:
                raise ValueError(f"CUDA kernel record has unknown iteration ID: {kr_iter}")
            _canonicalize_rank_attrs(kr, rank, event_name=str(kr.get("name", "cuda_kernel")))
            target_iter = iter_index[kr_iter]
            # Wrap as Event with name="cuda_kernel" and full record as attrs.
            # rel_ts is unused (kernel uses its own start_us/wall_start_us).
            target_iter.events.append(
                Event(rel_ts=0, rank=rank, name="cuda_kernel", ph="X", attrs=kr, cat="cuda_kernel")
            )
    return result


def aggregate_benchmark_data(
    contents: List[List[Iteration]],
) -> Tuple[List[Iteration], int, int, int]:
    """Sort and aggregate benchmark data.
    Args:
        contents: List[List[Iteration]], dim1 is each rank's file, dim2 is iteration id
    Return:
        iterations: List[Iteration]
    """
    if not contents:
        raise ValueError("No per-rank benchmark contents were provided")
    if any(not content for content in contents):
        empty_ranks = [index for index, content in enumerate(contents) if not content]
        raise ValueError(f"Per-rank benchmark contents are empty at indices {empty_ranks}")

    all_iterations = [iteration for content in contents for iteration in content]
    with_ids = [iteration.iteration_id is not None for iteration in all_iterations]

    iteration_groups: List[Tuple[Optional[int], List[Iteration]]] = []
    if all(with_ids):
        per_rank_by_id: List[Dict[int, Iteration]] = []
        for rank_index, content in enumerate(contents):
            by_id: Dict[int, Iteration] = {}
            for iteration in content:
                iteration_id = iteration.iteration_id
                assert iteration_id is not None
                if iteration_id in by_id:
                    raise ValueError(
                        f"Duplicate iteration ID {iteration_id} in rank content "
                        f"at index {rank_index}"
                    )
                by_id[iteration_id] = iteration
            per_rank_by_id.append(by_id)

        iteration_ids = sorted(per_rank_by_id[0])
        expected_ids = set(iteration_ids)
        for rank_index, by_id in enumerate(per_rank_by_id[1:], start=1):
            actual_ids = set(by_id)
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                unexpected = sorted(actual_ids - expected_ids)
                raise ValueError(
                    "Mismatched iteration IDs in rank content at index "
                    f"{rank_index}: missing={missing}, unexpected={unexpected}"
                )

        iteration_groups = [
            (iteration_id, [by_id[iteration_id] for by_id in per_rank_by_id])
            for iteration_id in iteration_ids
            if iteration_id is not None
        ]
    elif not any(with_ids):
        warnings.warn(
            "Aggregating legacy benchmark contents without iteration IDs; "
            "rank alignment falls back to file position and may be ambiguous",
            RuntimeWarning,
            stacklevel=2,
        )
        num_iterations = len(contents[0])
        if any(len(content) != num_iterations for content in contents):
            raise ValueError("Mismatched number of iterations without iteration IDs")
        iteration_groups = [
            (None, [content[index] for content in contents]) for index in range(num_iterations)
        ]
    else:
        raise ValueError("Iteration IDs are present for only part of the benchmark contents")

    ranks = {rank for iteration in all_iterations for rank in iteration.ranks} | {
        event.rank for iteration in all_iterations for event in iteration.events
    }
    if not ranks:
        raise ValueError("Cannot derive parallel topology from benchmark data without events")

    data_parallelism = max(rank.data for rank in ranks) + 1
    pipeline_parallelism = max(rank.pipeline for rank in ranks) + 1
    tensor_parallelism = max(rank.tensor for rank in ranks) + 1
    iterations: List[Iteration] = []

    for iteration_id, rank_iterations in iteration_groups:
        # TODO: CHECK -> Align the iteration start time of each rank to the latest timestamp.
        # Earlier starters will have their initial period truncated.
        iter_pad_before = min(iteration.pad_before for iteration in rank_iterations)
        # update event rel_ts according to iter_pad_before
        # event.rel_ts = event.rel_ts + (iteration.pad_before - iter_pad_before)
        rank_pad_before = [iteration.pad_before - iter_pad_before for iteration in rank_iterations]
        # update event rel_ts
        events: List[Event] = []
        for content_idx, iteration in enumerate(rank_iterations):
            pad_before = rank_pad_before[content_idx]
            for event in iteration.events:
                updated_event = Event(
                    rel_ts=event.rel_ts + pad_before,
                    rank=event.rank,
                    name=event.name,
                    ph=event.ph,
                    attrs=event.attrs,
                    cat=event.cat,
                )
                events.append(updated_event)

        events.sort(key=lambda event: event.rel_ts)
        duration = max(
            pad_before + iteration.duration
            for pad_before, iteration in zip(rank_pad_before, rank_iterations)
        )
        iterations.append(
            Iteration(
                pad_before=iter_pad_before,
                events=events,
                duration=duration,
                iteration_id=iteration_id,
                ranks=tuple(
                    sorted(
                        {
                            rank
                            for rank_iteration in rank_iterations
                            for rank in rank_iteration.ranks
                        }
                        | {
                            event.rank
                            for rank_iteration in rank_iterations
                            for event in rank_iteration.events
                        },
                        key=lambda rank: (
                            rank.global_rank is None,
                            rank.global_rank if rank.global_rank is not None else -1,
                            rank.data,
                            rank.pipeline,
                            rank.tensor,
                        ),
                    )
                ),
            )
        )

    return (iterations, data_parallelism, pipeline_parallelism, tensor_parallelism)


def transform(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a sequence of trace events (including interval events with ph "B"/"E")
    into a final event list in Chrome/Perfetto style with ph "X"
    """
    transformed: List[Dict[str, Any]] = (
        []
    )  # 构建的输出事件数组（开始时只存放 B 事件的引用，后续被转换成 X）。
    on_flight: Dict[Tuple[Any, Any], List[int]] = {}
    rank_map: Dict[int, Tuple[int, int, int]] = {}
    for trace in traces:
        # Round 4: cuda_kernel records (record_type=cuda_kernel, ph=X) pass
        # through unchanged; transform() pairs B/E into X for framework events
        # but kernel events are already complete.
        if trace.get("record_type") == "cuda_kernel":
            transformed.append(trace)
            continue
        if trace["ph"] in ("B", "E"):
            try:
                dp_rank = trace["args"]["dp_rk"]
                pp_rank = trace["args"]["pp_rk"]
                tp_rank = trace["args"]["tp_rk"]
            except KeyError as error:
                raise ValueError(
                    f"Trace event {trace.get('name')!r} lacks topology field " f"{error.args[0]!r}"
                ) from error
            pid = trace["pid"]
            coordinates = (dp_rank, pp_rank, tp_rank)
            previous = rank_map.get(pid)
            if previous is not None and previous != coordinates:
                raise ValueError(
                    f"Trace pid {pid} has conflicting topology: {previous} and {coordinates}"
                )
            rank_map[pid] = coordinates
        flight_key = (trace.get("pid", -1), trace.get("tid", 0))
        if trace["ph"] == "B":
            transformed.append(trace)
            on_flight.setdefault(flight_key, []).append(len(transformed) - 1)
        elif trace["ph"] == "E":
            pending = on_flight.get(flight_key)
            if not pending:
                raise ValueError(
                    f"Trace end {trace.get('name')!r} has no matching begin "
                    f"for pid/tid {flight_key}"
                )
            idx = pending.pop()
            begin_name = transformed[idx].get("name")
            if begin_name != trace.get("name"):
                raise ValueError(
                    f"Trace end {trace.get('name')!r} closes begin "
                    f"{begin_name!r} for pid/tid {flight_key}"
                )
            if trace["ts"] < transformed[idx]["ts"]:
                raise ValueError(
                    f"Trace end {trace.get('name')!r} precedes its begin "
                    f"for pid/tid {flight_key}"
                )
            # 计算 dur = E.ts - B.ts 并写入 transformed[idx]["dur"]。
            transformed[idx]["dur"] = trace["ts"] - transformed[idx]["ts"]
            # 将该 begin 事件的 ph 改为 "X"（表示 complete event）
            # 并把 E event args 的键合入 transformed[idx]["args"] (event B)
            transformed[idx]["ph"] = "X"
            for key in trace["args"]:
                if (
                    key not in transformed[idx]["args"]
                    or transformed[idx]["args"][key] == trace["args"][key]
                ):
                    transformed[idx]["args"][key] = trace["args"][key]
                else:
                    raise ValueError(
                        f"Trace attribute {key!r} conflicts between begin and end: "
                        f"{transformed[idx]['args'][key]!r} != {trace['args'][key]!r}"
                    )
        elif trace["ph"] == "C":
            # # 1. 提取嵌套的 args 数据源
            # outer_args = trace.get("args", {})
            # # 兼容处理：如果 outer_args 里还有 args 则取内部的，否则就用 outer_args
            # metrics_dict = outer_args.get("args", outer_args)

            # if isinstance(metrics_dict, dict):
            #     # 获取时间戳
            #     current_ts = trace.get("ts", trace.get("rel_ts", 0))

            #     # 2. 遍历每一个指标
            #     for key, value in metrics_dict.items():
            #         # --- 【关键步骤 A】过滤黑名单 ---
            #         # 直接跳过 id, pid, tid, iteration 等无关字段，不生成对应的 Trace
            #         if key in ["id", "iteration", "pid", "tid"]:
            #             continue

            #         # --- 【关键步骤 B】过滤非数值 ---
            #         # 只有数值才能画出 Counter 曲线
            #         if not isinstance(value, (int, float)):
            #             continue

            #         # 3. 创建新 Event
            #         new_event = trace.copy()

            #         # 修改 Event 名称
            #         new_event["name"] = f"{trace['name']}: {key}"

            #         # --- 【关键步骤 C】彻底重写 args ---
            #         # 不要直接赋值 metrics_dict，也不要 del keys
            #         # 而是直接创建一个全新的字典，只包含当前的一个 Key-Value
            #         new_event["args"] = {key: value}

            #         # 修正时间戳
            #         new_event["ts"] = current_ts
            #         # 清理旧的 rel_ts 以免混淆
            #         if "rel_ts" in new_event:
            #             del new_event["rel_ts"]

            transformed.append(trace)
        else:
            transformed.append(trace)

    unclosed = [
        transformed[index].get("name") for pending in on_flight.values() for index in pending
    ]
    if unclosed:
        raise ValueError(f"Trace begin records have no matching end: {unclosed}")

    sorted_ranks = sorted(rank_map.items(), key=lambda item: (*item[1], item[0]))
    for sort_index, (pid, (d, p, t)) in enumerate(sorted_ranks):
        transformed.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "args": {"name": f"DP{d}-PP{p}-TP{t}-G{pid}"},
            }
        )
        transformed.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": pid,
                "args": {"sort_index": sort_index},
            }
        )

    return transformed


def benchmark_to_chrome_trace(iterations: List[Iteration]) -> List[Dict[str, Any]]:
    """Convert benchmark data to Chrome trace format."""
    traces = []
    timeline = 0
    for i, iteration in enumerate(iterations):
        timeline += iteration.pad_before
        actual_iter = iteration.iteration_id if iteration.iteration_id is not None else i

        # 寻找当前 Iteration 涉及的所有 Rank，注入 iteration 的 B 事件
        iteration_ranks = set(iteration.ranks) | {event.rank for event in iteration.events}
        n_pp = max((rank.pipeline for rank in iteration_ranks), default=0) + 1
        n_tp = max((rank.tensor for rank in iteration_ranks), default=0) + 1
        rank_meta: Dict[int, Rank] = {}
        rank_pid: Dict[Rank, int] = {}
        for rank_obj in iteration_ranks:
            pid = rank_obj.to_pid(n_pp, n_tp)
            if pid in rank_meta and rank_meta[pid] != rank_obj:
                raise ValueError(
                    f"Multiple rank identities map to trace pid {pid}: "
                    f"{rank_meta[pid]} and {rank_obj}"
                )
            rank_meta[pid] = rank_obj
            rank_pid[rank_obj] = pid
        for event in iteration.events:
            pid = event.attrs.get("g_rk", rank_pid[event.rank])
            if pid in rank_meta and rank_meta[pid] != event.rank:
                raise ValueError(
                    f"Trace pid {pid} is shared by rank identities "
                    f"{rank_meta[pid]} and {event.rank}"
                )
            rank_meta[pid] = event.rank
            rank_pid[event.rank] = pid

        for pid, rank_obj in rank_meta.items():
            traces.append(
                {
                    "name": "iteration",
                    "cname": "thread_state_runnable",  # 赋予 iteration 一种特定的颜色
                    "ph": "B",
                    "ts": int(timeline / 1e3),
                    "pid": pid,
                    "tid": 0,
                    # 显式填入 dp_rk, pp_rk, tp_rk 以满足 transform 函数的要求
                    "args": {
                        "iteration": actual_iter,
                        "dp_rk": rank_obj.data,
                        "pp_rk": rank_obj.pipeline,
                        "tp_rk": rank_obj.tensor,
                    },
                }
            )

        for event in iteration.events:
            event_pid = event.attrs.get("g_rk", rank_pid[event.rank])
            # Round 4: cuda_kernel records pass through verbatim (carry their
            # own start_us / wall_start_us; no need to splice into chrome ts).
            if event.cat == "cuda_kernel" or event.name == "cuda_kernel":
                kr = dict(event.attrs)
                kr.setdefault("pid", kr.get("g_rk", -1))
                kr.setdefault("tid", "cuda_kernel")
                kr.setdefault("iteration", actual_iter)
                traces.append(kr)
                continue
            if event.ph == "C":
                counter_args = dict(event.attrs.get("args", {}))
                counter_args["iteration"] = actual_iter
                trace = {
                    "name": event.name,
                    "cname": COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                    "ph": event.ph,
                    "ts": int((event.rel_ts + timeline) / 1e3),
                    "pid": event_pid,
                    "tid": "Hardware Monitor",
                    "args": counter_args,
                }
            elif event.ph in ("B", "E"):
                args = dict(event.attrs)
                args.update(
                    iteration=actual_iter,
                    dp_rk=event.rank.data,
                    pp_rk=event.rank.pipeline,
                    tp_rk=event.rank.tensor,
                )
                trace = {
                    "name": event.name,
                    "cname": COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                    "ph": event.ph,
                    "ts": int((event.rel_ts + timeline) / 1e3),
                    "pid": event_pid,
                    "tid": 0,
                    "args": args,
                }
            else:
                args = dict(event.attrs)
                args.update(
                    iteration=actual_iter,
                    dp_rk=event.rank.data,
                    pp_rk=event.rank.pipeline,
                    tp_rk=event.rank.tensor,
                )
                trace = {
                    "name": event.name,
                    "cname": COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                    "ph": event.ph,
                    "ts": int((event.rel_ts + timeline) / 1e3),
                    "pid": event_pid,
                    "tid": 0,
                    "args": args,
                }
            if event.cat is not None:
                trace["cat"] = event.cat
            traces.append(trace)

        for pid, rank_obj in rank_meta.items():
            traces.append(
                {
                    "name": "iteration",
                    "cname": "thread_state_runnable",
                    "ph": "E",
                    "ts": int((timeline + iteration.duration) / 1e3),
                    "pid": pid,
                    "tid": 0,
                    "args": {
                        "iteration": actual_iter,
                        "dp_rk": rank_obj.data,
                        "pp_rk": rank_obj.pipeline,
                        "tp_rk": rank_obj.tensor,
                    },
                }
            )

        timeline += iteration.duration

    traces = transform(traces)

    return traces
