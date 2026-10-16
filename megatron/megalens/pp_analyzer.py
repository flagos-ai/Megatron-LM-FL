import collections
import os
import statistics
from typing import List, Dict, Any, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from megatron.megalens.trace import Tracer
from megatron.megalens.utils import get_tensor_bytes, get_gpu_p2p_theory_bw_gbps
from megatron.megalens.paper_style import (
    FIG_W_SINGLE, FIG_W_DOUBLE, FIG_H, FIG_H_TALL, FIG_H_SHORT,
)


_CORRELATION_EPSILON = 1e-6


def _average_ranks(values: Sequence[float]) -> Sequence[float]:
    """Return average-tie ranks for the scipy-free Spearman fallback."""
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        for rank_index in range(index, next_index):
            ranks[indexed[rank_index][0]] = average_rank
        index = next_index
    return ranks


def _pearsonr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson correlation without requiring scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = sum((x - mean_x) ** 2 for x in xs)
    denominator_y = sum((y - mean_y) ** 2 for y in ys)
    if denominator_x <= _CORRELATION_EPSILON or denominator_y <= _CORRELATION_EPSILON:
        return 0.0
    return numerator / ((denominator_x * denominator_y) ** 0.5)


def _spearmanr_fallback(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Spearman correlation without requiring scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearsonr(_average_ranks(xs), _average_ranks(ys))


def _spearmanr_compat(xs, ys):
    """Spearman ρ that works with or without scipy.

    Returns (rho, pvalue). pvalue is 0.0 in the fallback path.
    """
    try:
        import scipy.stats as scipy_stats
    except ImportError:
        return _spearmanr_fallback(list(xs), list(ys)), 0.0
    return scipy_stats.spearmanr(xs, ys)

# ============================================================================
# Configuration & Constants
# ============================================================================

# --- Bubble Analysis Whitelist ---
# Events considered as "Active" (doing real work, not idling)
ACTIVE_EVENTS_WHITELIST = {
    "forward-step", 
    "backward-step",
    "grad-sync", 
    "optimizer",
}

def is_tp_comm_event(name: str) -> bool:
    """判断是否是 TP 组内的同步集体通信事件"""
    name_lower = name.lower()
    return any(k in name_lower for k in ["allreduce", "reduce-scatter", "all-gather", "reducescatter", "allgather"])

# --- Global Thresholds ---
P2P_THEORY_BW_GBPS = 450.0  # Unidirectional reference; overridden at runtime by NVML probe
COMPUTE_JITTER_CV_THRESHOLD = 0.1  # Coefficient of Variation (std/mean) > 10% is jitter
# Paper sec:design-decoupling uses max(T_fwd)/min(T_fwd) > 1.2 for stage-level skew detection.
COMPUTE_SKEW_MAX_MIN_RATIO = 1.2  # Max/Min duration ratio across PP stages

# Re-export for backward compat — some callers may still reference the old name.
COMPUTE_SKEW_MAX_MEAN_RATIO = COMPUTE_SKEW_MAX_MIN_RATIO

# Kept local so PP analysis remains usable when the optional PIG layer is absent.
PP_P2P_STALL_THRESHOLD_US = 100.0


def _load_pandas():
    """Load the optional tabular reporting dependency when first required."""
    import pandas as pd

    return pd


def _load_reporting_dependencies():
    """Load optional plotting dependencies when first required."""
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


# ============================================================================
# 1. Object-Oriented Data Model (OOP Design)
# ============================================================================
class ReportLogger:
    """Handles writing dual outputs to both Console and a Report TXT file."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" MegaLens Pipeline Parallelism Diagnostic Report\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, end: str = "\n"):
        print(message, end=end)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + end)


class TracedWaitHandle:
    """包装 PyTorch 的 distributed request
    在调用 wait() 发生真实阻塞时记录 Trace
    示例1
    通信流 (NCCL): |-------- 真实网络传输耗时 (比如 5ms) --------|
    计算流 (CUDA): |---------- 跑其他的 Forward 算子 (比如 8ms) ----------| req.wait()
    示例2
    通信流 (NCCL): |-------- 真实网络传输耗时 (比如 5ms) --------|
    计算流 (CUDA): |-- 算子 (2ms) --| req.wait() 开始阻塞 ........| wait() 结束
                                   |---Tracer 记录的耗时 (3ms)---|
    """
    def __init__(self, req: Any, name: str, peer_rank: int, tracer: "Tracer", tensor_bytes: int):
        self.req = req
        self.name = name
        self.peer_rank = peer_rank
        self.tracer = tracer
        self.tensor_bytes = tensor_bytes

    def wait(self):
        if self.tracer is not None and self.tracer.is_tracing():
            with self.tracer.scope(
                self.name, 
                ctx={
                    "peer_rank": self.peer_rank, 
                    "comm_type": "p2p", 
                    "is_blocking": True,
                    "data_bytes": self.tensor_bytes
                }
            ):
                self.req.wait()
        else:
            self.req.wait()


def wrap_p2p_communicate_reqs(reqs: Dict, communicator: Any, tensors_dict: Dict) -> Dict:
    """Low-coupling hook function: automatically infers the flow direction 
    and wraps it based on the standard keys used inside _communicate.
    """
    from megatron.training.global_vars import get_args, get_tracer
    args = get_args()
    tracer = get_tracer()

    if not (tracer and getattr(args, "trace", False) and reqs):
        return reqs

    if not isinstance(reqs, dict):
        return reqs

    # mapping dict to determine the name and peer rank for each request key
    meta_map = {
        "send_next": {"name": "send-forward", "peer": communicator.next_rank},
        "recv_prev": {"name": "recv-forward", "peer": communicator.prev_rank},
        "send_prev": {"name": "send-backward", "peer": communicator.prev_rank},
        "recv_next": {"name": "recv-backward", "peer": communicator.next_rank},
    }

    wrapped_reqs = {}
    for k, req in reqs.items():
        if k in meta_map:
            tb = get_tensor_bytes(tensors_dict.get(k))
            wrapped_reqs[k] = TracedWaitHandle(
                req=req,
                name=meta_map[k]["name"],
                peer_rank=meta_map[k]["peer"],
                tracer=tracer,
                tensor_bytes=tb
            )
        else:
            wrapped_reqs[k] = req
            
    return wrapped_reqs


class PPEvent:
    """Base class for all Pipeline Parallel events."""
    def __init__(self, raw_dict: Dict[str, Any]):
        self.raw = raw_dict  # Reference to original dict. Mutating this modifies the trace!
        self.name: str = raw_dict.get("name", "")
        self.ph: str = raw_dict.get("ph", "")
        self.ts: int = raw_dict.get("ts", 0)
        self.dur: int = raw_dict.get("dur", 0)
        self.rank: int = raw_dict.get("pid", -1)  # global_rank
        self.args: Dict[str, Any] = raw_dict.get("args", {})
        self.iteration: int = self.args.get("iteration", -1)
        
    @property
    def end_time(self) -> int:
        return self.ts + self.dur
    
class IterationEvent(PPEvent):
    """Represents the boundary of a full training iteration."""
    pass

class MiscActiveEvent(PPEvent):
    """Represents other active events like Optimizer, AllReduce."""
    pass

class TPCommEvent(PPEvent):
    """Represents TP collective communication events."""
    pass

class CommEvent(PPEvent):
    """Communication Event (Send/Recv)."""
    def __init__(self, raw_dict: Dict[str, Any]):
        super().__init__(raw_dict)
        self.peer_rank: Optional[int] = self.args.get("peer_rank")
        self.bytes: int = self.args.get("data_bytes", 0)

        self.is_atomic_send = self.name in {"send-forward", "send-backward"}
        self.is_atomic_recv = self.name in {"recv-forward", "recv-backward"}
        
        # [Core Feature] Object pointer to the corresponding sender event on the peer rank
        self.direction = "forward" if "forward" in self.name else "backward"
        
        self.paired_event: Optional['CommEvent'] = None 
        self.launch_event: Optional['PPEvent'] = None
        self.wait_time_us: float = 0.0
        self.real_dur_us: float = float(self.dur)


class ComputeEvent(PPEvent):
    """Computation Event (Forward/Backward)."""
    def __init__(self, raw_dict: Dict[str, Any]):
        super().__init__(raw_dict)
        self.num_tokens: int = self.args.get("num_tokens", 0)
        self.microbatch_id: int = self.args.get("current_microbatch", -1)


class P2PLaunchEvent(PPEvent):
    """Records when isend/irecv was enqueued to NCCL comm stream.
    The p2p-launch scope in _communicate() wraps p2p_func() which calls
    batch_isend_irecv or individual isend/irecv. This event captures the
    CUDA launch overhead, NOT the actual network transfer time.
    """
    pass


class TraceGraph:
    """
    Manager class to build relationships between events.
    Parses raw dictionaries into Objects and automatically links paired P2P events.
    """
    def __init__(self, traces: List[Dict[str, Any]]):
        self.raw_traces = traces
        self.comm_events: List[CommEvent] = []
        self.tp_comm_events: List[TPCommEvent] = []
        self.compute_events: List[ComputeEvent] = []
        self.iter_events: List[IterationEvent] = []
        self.misc_events: List[MiscActiveEvent] = []
        self.p2p_launch_events: List[P2PLaunchEvent] = []
        self.tp_comm_by_rank = collections.defaultdict(list)
        self.rank_topology: Dict[int, Dict[str, int]] = {}
        
        self._build_graph()
        self._pair_p2p_events()
        self._link_launch_events()

    def _build_graph(self):
        """Parse raw dicts into Event Objects."""
        for t in self.raw_traces:
            if t.get("ph") != "X": continue
            name = t.get("name", "")
            
            if name == "iteration":
                ev = IterationEvent(t)
                self.iter_events.append(ev)
                if ev.dur > 0 and ev.rank not in self.rank_topology:
                    self.rank_topology[ev.rank] = {
                        'dp': ev.args.get("dp_rk", 0),
                        'tp': ev.args.get("tp_rk", 0),
                        'pp': ev.args.get("pp_rk", 0),
                    }
            elif name == "p2p-launch":
                self.p2p_launch_events.append(P2PLaunchEvent(t))
            elif any(k in name for k in ["recv", "send", "exchange"]):
                self.comm_events.append(CommEvent(t))
            elif "forward" in name or "backward" in name:
                self.compute_events.append(ComputeEvent(t))
            elif name in ACTIVE_EVENTS_WHITELIST:
                self.misc_events.append(MiscActiveEvent(t))
            elif is_tp_comm_event(name):
                tp_ev = TPCommEvent(t)
                self.tp_comm_events.append(tp_ev)
                self.tp_comm_by_rank[tp_ev.rank].append((tp_ev.ts, tp_ev.end_time))

    def _pair_p2p_events(self):
        """Automatically find and link Sender and Receiver events across ranks."""
        send_queues = collections.defaultdict(collections.deque)

        # 1. 严格按照事件的时间戳排序，这是保证 FIFO 物理保序的前提
        sorted_comm_events = sorted(self.comm_events, key=lambda e: e.ts)

        # 2. 一次遍历完成生产和消费的精准匹配
        for ev in sorted_comm_events:
            if not (ev.is_atomic_send or ev.is_atomic_recv) or ev.peer_rank is None:
                continue

            if ev.is_atomic_send:
                # 生产通道：发送给对端
                queue_key = (ev.rank, ev.peer_rank, ev.direction)
                send_queues[queue_key].append(ev)
                
            elif ev.is_atomic_recv:
                # 消费通道：接收来自对端
                queue_key = (ev.peer_rank, ev.rank, ev.direction)
                if queue_key in send_queues and len(send_queues[queue_key]) > 0:
                    matched_sender = send_queues[queue_key].popleft()
                    
                    # 建立强引用双向链接，用于计算 Dependency Stall
                    ev.paired_event = matched_sender
                    matched_sender.paired_event = ev

    def _link_launch_events(self):
        """Associate each atomic P2P comm event with its preceding p2p-launch on the same rank.
        
        In _communicate(), the p2p-launch scope fires first (enqueuing isend/irecv),
        then TracedWaitHandle.wait() fires (blocking until completion). By linking
        these, we get precise sender readiness timestamps for cross-rank analysis.
        """
        launches_by_rank = collections.defaultdict(list)
        for ev in self.p2p_launch_events:
            launches_by_rank[ev.rank].append(ev)
        for rank in launches_by_rank:
            launches_by_rank[rank].sort(key=lambda e: e.ts)
        
        for ev in self.comm_events:
            if not (ev.is_atomic_send or ev.is_atomic_recv):
                continue
            rank_launches = launches_by_rank.get(ev.rank, [])
            for launch in reversed(rank_launches):
                if launch.end_time <= ev.ts + 50:
                    ev.launch_event = launch
                    break

    def get_tp_groups(self) -> Dict[Tuple[int, int], List[int]]:
        """Build TP groups from topology: key=(dp_rk, pp_rk), value=list of global ranks."""
        tp_groups = collections.defaultdict(list)
        for rank, topo in self.rank_topology.items():
            tp_groups[(topo['dp'], topo['pp'])].append(rank)
        return dict(tp_groups)

# ============================================================================
# 2. Analysis Modules
# ============================================================================
def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Standard LeetCode Interval Merging. Returns list of merged intervals."""
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1]: # Overlap
                merged[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged.append(curr)
        return merged

def get_pure_compute_dur(ev: PPEvent, graph:TraceGraph) -> float:
        """从 Compute Event 总时长中，剥离出内部发生的所有 TP 同步时间"""
        comms = graph.tp_comm_by_rank[ev.rank]
        overlaps = []
        for s, e in comms:
            # 找到在 Compute Event 期间发生的通信
            if e > ev.ts and s < ev.end_time:
                overlaps.append((max(s, ev.ts), min(e, ev.end_time)))
        if not overlaps: return float(ev.dur)
        
        # 合并重叠的通信区间
        merged_comms = merge_intervals(overlaps)
        comm_dur = sum(e - s for s, e in merged_comms)
        
        # 纯计算 = 总时间 - 通信干等时间
        # 限制下限，防止极端追踪误差导致纯计算为负
        return max(1.0, float(ev.dur - comm_dur))

def bubble_analysis(graph: TraceGraph, logger: ReportLogger) -> List[Dict[str, Any]]:
    """
    Calculates precise Per-Rank Bubble Rate using Interval Merging.
    """
    logger.log("[PP Bubble Analysis] Running Precision Bubble Rate Analysis...")
    
    iter_boundaries = collections.defaultdict(dict)
    rank_topology = graph.rank_topology
    for ev in graph.iter_events:
        if ev.dur > 0:
            iter_boundaries[ev.rank][ev.iteration] = [ev.ts, ev.end_time]

    # if not graph.iter_events:
    #     logger.log("  -> No explicit 'iteration' scopes found. Auto-inferring boundaries...")
    #     iter_bounds = collections.defaultdict(lambda: [float('inf'), 0])
    #     for ev in graph.compute_events + graph.misc_events + graph.comm_events:
    #         if ev.iteration >= 0:
    #             iter_bounds[(ev.rank, ev.iteration)][0] = min(iter_bounds[(ev.rank, ev.iteration)][0], ev.ts)
    #             iter_bounds[(ev.rank, ev.iteration)][1] = max(iter_bounds[(ev.rank, ev.iteration)][1], ev.end_time)
    #     for (rank, iteration), (ts, end_time) in iter_bounds.items():
    #         if end_time > ts:
    #             graph.iter_events.append(IterationEvent({"name": "iteration", "ph": "X", "ts": ts, "dur": end_time - ts, "pid": rank, "args": {"iteration": iteration}}))
    # for ev in graph.iter_events:
    #     if ev.dur > 0:
    #         iter_boundaries[ev.rank][ev.iteration] = (ev.ts, ev.end_time)

    optimizer_end_time_per_iter = collections.defaultdict(lambda: collections.defaultdict(int))

        
    # 2. 收集所有有效工作区间 (只看纯计算和 Collective 通信)
    # format: active_intervals[rank] = [ (start, end), (start, end) ]
    active_intervals_global = collections.defaultdict(list)
    
    def add_interval(ev: PPEvent):
        if ev.dur <= 0 or ev.name not in ACTIVE_EVENTS_WHITELIST: return
        active_intervals_global[ev.rank].append((ev.ts, ev.end_time))

    for ev in graph.compute_events + graph.misc_events:
        add_interval(ev)
        # TODO Optimizer time is not normal use all-grads-sync instead temporary solution
        if ev.name == "optimizer":
            current_iter = ev.iteration
            if current_iter >= 0:
                optimizer_end_time_per_iter[ev.rank][current_iter] = max(
                    optimizer_end_time_per_iter[ev.rank][current_iter],
                    ev.ts
                )

    # 3. 区间合并算法
    bubble_stats = []
    def get_active_time_in_window(merged_intervals: List[Tuple[int, int]], window_start: int, window_end: int) -> int:
        """计算某个时间窗口 (Iteration) 内的有效时长，精准裁剪越界部分。"""
        active_time = 0
        for start, end in merged_intervals:
            # 区间与窗口无交集
            if end <= window_start or start >= window_end:
                continue
            # 计算交集长度
            overlap_start = max(start, window_start)
            overlap_end = min(end, window_end)
            active_time += (overlap_end - overlap_start)
        return active_time

    # 4. 执行计算
    for rank, iterations in iter_boundaries.items():
        # 提前对这个 Rank 的所有全局区间进行一次合并，极大地提升后续计算效率
        merged_global = merge_intervals(active_intervals_global[rank])
        topo = rank_topology.get(rank, {'dp': 0, 'tp': 0, 'pp': 0})
        group_name = f"DP={topo['dp']} | TP={topo['tp']}"
        
        for iteration, (w_start, w_end) in iterations.items():
            w_end = optimizer_end_time_per_iter[rank].get(iteration, w_end)
            if w_end <= w_start: continue
            t_actual = w_end - w_start
            t_active = get_active_time_in_window(merged_global, w_start, w_end)
            if rank == 0 and iteration == 0:
                logger.log(f"[PP Bubble Analysis] Iteration {iteration} on Rank {rank}")
                logger.log(f"t_active{t_active}, w_start{w_start}, w_end{w_end}")
            
            bubble_rate = 1.0 - (t_active / t_actual)
            bubble_rate = max(0.0, min(1.0, bubble_rate)) # 限制在 0~1 之间
            
            bubble_stats.append({
                "Rank": rank,
                "DP_Rank": topo['dp'],
                "TP_Rank": topo['tp'],
                "PP_Rank": topo['pp'],
                "Group": group_name,
                "Iteration": iteration,
                "T_actual_ms": t_actual / 1000.0,
                "T_active_ms": t_active / 1000.0,
                "Bubble_Rate": bubble_rate
            })

    # 将数据写回原始 Iteration Event，供 Chrome Trace UI 显示
    for ev in graph.iter_events:
        if ev.dur <= 0: continue
        
        # 寻找对应的统计数据
        stat = next((s for s in bubble_stats if s["Rank"] == ev.rank and s["Iteration"] == ev.iteration), None)
        if stat:
            ev.raw["args"]["Analysis_Bubble_Rate"] = f"{stat['Bubble_Rate']:.2%}"
            ev.raw["args"]["Analysis_Active_Time_ms"] = f"{stat['T_active_ms']:.2f}"
            
            # 颜色预警逻辑
            if stat['Bubble_Rate'] > 0.5:
                ev.raw["cname"] = "terrible" # 红灯：气泡极度严重
            elif stat['Bubble_Rate'] > 0.3:
                ev.raw["cname"] = "bad" # 黄灯：气泡偏高

    logger.log(f"[PP Bubble Analysis] Calculated precision Bubble Rates for {len(bubble_stats)} iteration spans.")
    return bubble_stats

def p2p_comm_analysis(graph: TraceGraph, theory_bw_gbps: float, logger: ReportLogger) -> List[Dict[str, Any]]:
    """Analyzes P2P communication with cross-rank pairing to decompose wait time.
    
    Key insight: NCCL isend/irecv run on a separate comm stream, so CUDA events
    on the compute stream can only measure req.wait() blocking time. This conflates:
      (1) Dependency Stall - receiver ready before sender launches send
      (2) Actual network transfer time
      (3) NCCL scheduling overhead
    
    By pairing the sender's send event (from TracedWaitHandle) with the receiver's
    recv event cross-rank, we decompose the wait time:
      - sender.ts ≈ when the isend was enqueued (sender readiness)
      - If sender.ts > recv.ts: dependency stall = sender.ts - recv.ts
      - transfer_upper_bound = recv.end_time - max(sender.ts, recv.ts)
      - corrected_bw = bytes / transfer_upper_bound (much more accurate)
    """
    pd = _load_pandas()
    logger.log("\n[PP P2P Analysis] Starting Cross-Rank P2P Decomposition...")
    plot_data_list = []
    count_dep_stalls = 0
    count_bw_issues = 0
    paired_count = 0
    
    for ev in graph.comm_events:
        if not ev.is_atomic_recv: continue
        wait_us = float(ev.dur)
        if wait_us <= 0: continue
        
        transferred_mb = ev.bytes / (1024 * 1024)
        ideal_transfer_us = (ev.bytes / (theory_bw_gbps * 1e3)) if theory_bw_gbps > 0 else 0.0
        
        # --- Cross-rank decomposition using paired sender event ---
        sender = ev.paired_event
        dependency_stall_us = 0.0
        transfer_upper_bound_us = wait_us
        has_pairing = sender is not None
        
        if has_pairing:
            paired_count += 1
            # Best estimate of "sender readiness": when isend was enqueued.
            # Prefer launch_event.end_time (p2p-launch scope end) for precision;
            # fallback to sender.ts (start of send req.wait, slightly later).
            sender_ready_ts = sender.ts
            if sender.launch_event is not None:
                sender_ready_ts = sender.launch_event.end_time
            
            recv_start = ev.ts
            recv_end = ev.end_time
            
            if sender_ready_ts > recv_start:
                # Case A: Receiver was waiting before sender had data ready.
                # This is a genuine dependency stall (sender's preceding
                # forward/backward compute was the bottleneck).
                dependency_stall_us = float(sender_ready_ts - recv_start)
                transfer_upper_bound_us = max(1.0, float(recv_end - sender_ready_ts))
            else:
                # Case B: Sender had already launched before receiver started waiting.
                # The full wait time is transfer + NCCL scheduling.
                dependency_stall_us = 0.0
                transfer_upper_bound_us = wait_us
        
        # Corrected bandwidth: uses transfer upper bound (excludes dependency stall)
        corrected_bw_gbps = 0.0
        if transfer_upper_bound_us > 0 and ev.bytes > 0:
            corrected_bw_gbps = (ev.bytes / 1e9) / (transfer_upper_bound_us / 1e6)
            corrected_bw_gbps = min(corrected_bw_gbps, theory_bw_gbps * 1.2)
        
        # Naive bandwidth (uses total wait time, for backward compatibility)
        naive_bw_gbps = 0.0
        if wait_us > 0 and ev.bytes > 0:
            naive_bw_gbps = (ev.bytes / 1e9) / (wait_us / 1e6)
            naive_bw_gbps = min(naive_bw_gbps, theory_bw_gbps)
        
        eff_bw_gbps = corrected_bw_gbps if has_pairing else naive_bw_gbps
        
        # --- Diagnosis with cross-rank awareness ---
        diagnosis, new_color = "Normal Transfer", "good"
        
        if has_pairing and dependency_stall_us > wait_us * 0.5 and dependency_stall_us > 500:
            # Dominant factor is waiting for the sender (peer compute was slow)
            diagnosis = "Dependency Stall (Peer Compute Slow)"
            new_color = "terrible" if dependency_stall_us > 5000 else "bad"
            count_dep_stalls += 1
        elif transfer_upper_bound_us > ideal_transfer_us * 3.0 and transfer_upper_bound_us > 500:
            if corrected_bw_gbps < theory_bw_gbps * 0.3 and ev.bytes > 1024 * 1024:
                # Achieved BW far below theoretical even after removing dep stall
                diagnosis = "Bandwidth Bottleneck (Network/PCIe Congestion)"
                new_color = "bad"
                count_bw_issues += 1
            elif corrected_bw_gbps < theory_bw_gbps * 0.6:
                diagnosis = "Sub-optimal Bandwidth"
                new_color = "yellow"
        elif not has_pairing and wait_us > ideal_transfer_us * 3.0 + 500:
            diagnosis = "High Wait (Unpaired - Cannot Decompose)"
            new_color = "bad" if wait_us > 5000 else "yellow"

        ev.raw["args"].update({
            "Wait_Total_us": int(wait_us),
            "Dependency_Stall_us": int(dependency_stall_us),
            "Transfer_Upper_Bound_us": int(transfer_upper_bound_us),
            "Transferred_MB": f"{transferred_mb:.2f}",
            "Corrected_BW_GBps": f"{corrected_bw_gbps:.2f}",
            "Naive_BW_GBps": f"{naive_bw_gbps:.2f}",
            "Has_Cross_Rank_Pair": has_pairing,
            "Analysis_Diagnosis": diagnosis,
        })
        
        plot_data_list.append({
            "ts_s": ev.ts / 1e6, "rank": ev.rank,
            "sender_rank": sender.rank if sender else -1,
            "event": ev.name, "direction": ev.direction,
            "stall_us": dependency_stall_us,
            "transfer_ub_us": transfer_upper_bound_us,
            "mb": transferred_mb,
            "eff_bw_gbps": eff_bw_gbps,
            "corrected_bw_gbps": corrected_bw_gbps,
            "naive_bw_gbps": naive_bw_gbps,
            "diagnosis": diagnosis,
            "iteration": ev.iteration,
        })
        if new_color != "good": ev.raw["cname"] = new_color

    logger.log(f"\n[PP P2P Analysis] Analyzed {len(plot_data_list)} Recv events.")
    logger.log(f"  -> Successfully paired {paired_count}/{len(plot_data_list)} events cross-rank.")
    logger.log(f"  -> Found {count_dep_stalls} Dependency Stalls (sender slow), "
                f"{count_bw_issues} Bandwidth Issues.")

    # --- Per-GPU-pair bandwidth summary ---
    if plot_data_list:
        pdf = pd.DataFrame(plot_data_list)
        paired_df = pdf[pdf["sender_rank"] >= 0]
        if not paired_df.empty:
            pair_summary = paired_df.groupby(["sender_rank", "rank", "direction"]).agg(
                mean_bw=("corrected_bw_gbps", "mean"),
                min_bw=("corrected_bw_gbps", "min"),
                mean_stall=("stall_us", "mean"),
                mean_mb=("mb", "mean"),
                n_events=("corrected_bw_gbps", "count"),
            ).reset_index()
            pair_summary.rename(columns={"rank": "receiver_rank"}, inplace=True)
            pair_summary["util_pct"] = (pair_summary["mean_bw"] / theory_bw_gbps * 100
                                        ).clip(upper=100.0)

            logger.log("\n  Per-GPU-Pair Bandwidth Summary (corrected, unidirectional):")
            logger.log("  " + "-" * 95)
            logger.log(f"  {'Sender':>8}  {'Recv':>8}  {'Dir':>8}  {'Mean BW':>10}  "
                        f"{'Min BW':>10}  {'Util%':>8}  {'Mean MB':>8}  {'Mean Stall':>12}  {'N':>6}")
            logger.log("  " + "-" * 95)
            for _, row in pair_summary.iterrows():
                is_small_msg = row["mean_mb"] < 1.0
                flag = " *** SMALL_MSG" if is_small_msg else (" *** LOW" if row["util_pct"] < 30 else "")
                logger.log(
                    f"  {int(row['sender_rank']):>8d}  {int(row['receiver_rank']):>8d}  "
                    f"{row['direction']:>8}  {row['mean_bw']:>8.1f} GB/s  "
                    f"{row['min_bw']:>8.1f} GB/s  {row['util_pct']:>7.1f}%  "
                    f"{row['mean_mb']:>7.2f}  {row['mean_stall']:>10.0f} us  "
                    f"{int(row['n_events']):>6d}{flag}")
            logger.log("  " + "-" * 95)

            # Flag GPU pairs with consistently low bandwidth as potential link issues
            bad_pairs = pair_summary[(pair_summary["util_pct"] < 30) & (pair_summary["mean_mb"] >= 1.0)]
            small_pairs = pair_summary[pair_summary["mean_mb"] < 1.0]
            if not small_pairs.empty:
                logger.log("\n  [INFO] Some GPU pairs carry sub-1MB payloads; low utilization is expected for latency-dominated messages:")
                for _, row in small_pairs.iterrows():
                    logger.log(
                        f"    Rank {int(row['sender_rank'])} -> Rank {int(row['receiver_rank'])} "
                        f"({row['direction']}): mean size {row['mean_mb']:.2f} MB, "
                        f"mean BW {row['mean_bw']:.1f} GB/s")
            if not bad_pairs.empty:
                logger.log("\n  [WARNING] The following GPU pairs show < 30% BW utilization on >=1MB payloads.")
                logger.log("  This may indicate a faulty NVLink/IB link or cross-node bottleneck:")
                for _, row in bad_pairs.iterrows():
                    logger.log(f"    Rank {int(row['sender_rank'])} -> Rank {int(row['receiver_rank'])} "
                                f"({row['direction']}): {row['mean_bw']:.1f} GB/s "
                                f"({row['util_pct']:.0f}% of {theory_bw_gbps} GB/s)")

    return plot_data_list

def compute_load_analysis(graph: TraceGraph, logger: ReportLogger) -> Tuple[List[Dict[str, Any]], Dict[int, "pd.DataFrame"]]:
    """Analyzes Forward compute times to identify Data Skew vs Hardware Jitter."""
    pd = _load_pandas()
    logger.log("[PP Compute Analysis] Starting Load Imbalance & Jitter Analysis...")
    
    compute_data = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for ev in graph.compute_events + graph.misc_events:
        if ev.dur <= 10:  # 过滤极小噪点
            continue
            
        phase = None
        if ev.name == "forward-step": phase = "forward"
        elif ev.name == "backward-step": phase = "backward"
        elif ev.name in ["optimizer", "optimizer-step"]: phase = "optimizer"
        
        if not phase: continue

        # [核心特征提取] 
        # N = 总 Token 数
        # S_sq = 序列长度的平方和 (对应 VarLen Flash Attention 计算量)
        n_tokens = ev.args.get("num_tokens", 0)
        # 优先读取 sum_sq_seq_len，如果没有，则降级兼容传统的 seq_len 平方
        s_sq = ev.args.get("sum_sq_seq_len", ev.args.get("seq_len", ev.args.get("max_seq_len", 0)) ** 2)
        if phase != "optimizer" and (n_tokens is None or s_sq is None):
            continue
        
        pure_dur_ms = get_pure_compute_dur(ev, graph) / 1000.0
        compute_data[phase][ev.rank].append({
            "tokens_N": n_tokens,
            "seq_len_sq_L2": s_sq,
            "dur_ms": pure_dur_ms,
            "iteration": ev.iteration,
            "event_ref": ev 
        })

    multi_stats = {"forward": [], "backward": [], "optimizer": []}
    multi_dfs = {"forward": {}, "backward": {}, "optimizer": {}}

    for phase in ["forward", "backward", "optimizer"]:
        if not compute_data[phase]: continue
            
        ranks = sorted(compute_data[phase].keys())
        for rank in ranks:
            df = pd.DataFrame(compute_data[phase][rank])
            multi_dfs[phase][rank] = df
            
            N = df['tokens_N'].values
            Ssq = df['seq_len_sq_L2'].values
            T = df['dur_ms'].values
            
            mean_T, max_T, std_T = np.mean(T), np.max(T), np.std(T)
            cv_T = std_T / mean_T if mean_T > 0 else 0
            
            diagnosis = "Normal"
            is_straggler = False
            r_N, rho_N, r_Ssq, rho_Ssq = np.nan, np.nan, np.nan, np.nan

            if phase == "optimizer":
                # Optimizer 是静态 Element-wise，靠 CV 抓显存/系统抖动
                if cv_T > 0.10: 
                    diagnosis, is_straggler = "Hardware Jitter (Mem/PCIe)", True
                else:
                    diagnosis = "Stable (Optimizer)"
            else:
                # --- 严密的 4D 相关性分析 (Forward / Backward) ---
                is_N_static = len(np.unique(N)) <= 1
                is_Ssq_static = len(np.unique(Ssq)) <= 1
                
                # 安全计算相关系数 (防除零)
                if not is_N_static and len(np.unique(T)) > 1:
                    r_N = np.corrcoef(N, T)[0, 1]
                    rho_N, _ = _spearmanr_compat(N, T)
                if not is_Ssq_static and len(np.unique(T)) > 1:
                    r_Ssq = np.corrcoef(Ssq, T)[0, 1]
                    rho_Ssq, _ = _spearmanr_compat(Ssq, T)
                    
                # 处理 NaN 变为 0.0 便于数值比较
                r_N = r_N if pd.notna(r_N) else 0.0
                rho_N = rho_N if pd.notna(rho_N) else 0.0
                r_Ssq = r_Ssq if pd.notna(r_Ssq) else 0.0
                rho_Ssq = rho_Ssq if pd.notna(rho_Ssq) else 0.0

                # 辅助判断条件
                N_high = r_N > 0.7 or rho_N > 0.7
                N_low = r_N < 0.6 and rho_N < 0.6
                N_very_low = r_N < 0.4 and rho_N < 0.4
                
                Ssq_high = r_Ssq > 0.8 or rho_Ssq > 0.8
                Ssq_low = r_Ssq < 0.6 and rho_Ssq < 0.6
                Ssq_very_low = r_Ssq < 0.4 and rho_Ssq < 0.4

                # === 诊断矩阵分类器 (Diagnostic Matrix) ===
                if is_N_static and is_Ssq_static:
                    if cv_T > 0.10:
                        diagnosis, is_straggler = "Hardware Jitter (Static)", True
                    elif cv_T < 0.05:
                        diagnosis = "Stable (Static Load)"
                    else:
                        diagnosis = "Minor Jitter (Static)"
                        
                elif N_high and Ssq_low and cv_T > 0.05:
                    diagnosis = "Data Skew (FFN/Linear Bound)"
                    
                elif N_low and Ssq_high and cv_T > 0.05:
                    diagnosis = "Data Skew (Attention/VarLen Bound)"
                    
                elif N_high and Ssq_high and cv_T > 0.05:
                    diagnosis = "Stable (Perfect Dynamic Scaling)"
                    
                elif N_very_low and Ssq_very_low and cv_T > 0.10:
                    diagnosis, is_straggler = "Hardware Jitter (Dynamic)", True
                    
                else:
                    diagnosis = "Mixed Workload Variance"

            # 记录数据
            multi_stats[phase].append({
                "Rank": rank, "Mean_Dur_ms": mean_T, "Max_Dur_ms": max_T, "CV_T": cv_T,
                "r_N": r_N, "rho_N": rho_N, "r_Ssq": r_Ssq, "rho_Ssq": rho_Ssq,
                "Diagnosis": diagnosis
            })

            # 回写 Trace 数据
            for item in compute_data[phase][rank]:
                ev_obj = item["event_ref"]
                ev_obj.raw["args"]["Analysis_Rank_Mean_ms"] = f"{mean_T:.2f}"
                ev_obj.raw["args"]["Analysis_Diagnosis"] = diagnosis
                if is_straggler and item["dur_ms"] > mean_T * 1.3:
                    ev_obj.raw["cname"] = "terrible"

    # --- 打印诊断报告与静态负载不均识别 ---
    for phase in ["forward", "backward"]:
        if not multi_stats[phase]: continue
        stats_df = pd.DataFrame(multi_stats[phase])
        logger.log(f"\n" + "=" * 100)
        logger.log(f"[{phase.upper()}] VarLen-Aware Compute Load Imbalance Report")
        logger.log("=" * 100)
        display_cols = ['Rank', 'Mean_Dur_ms', 'Max_Dur_ms', 'CV_T', 'r_N', 'rho_Ssq', 'Diagnosis']
        logger.log(stats_df[display_cols].to_string(index=False, float_format="%.3f", na_rep="Static"))
        logger.log("-" * 100)
        
        # Step 4: 静态负载不均识别 (paper sec:design-decoupling: max/min > 1.2 triggers stage skew)
        mean_durs = stats_df['Mean_Dur_ms'].values
        min_mean = np.min(mean_durs)
        max_min_ratio = (np.max(mean_durs) / min_mean) if min_mean > 0 else 0
        static_imbalance_ratio = max_min_ratio - 1.0  # for display continuity

        if max_min_ratio > COMPUTE_SKEW_MAX_MIN_RATIO:
            slowest = stats_df.iloc[np.argmax(mean_durs)]['Rank']
            fastest = stats_df.iloc[np.argmin(mean_durs)]['Rank']
            logger.log(f"[Static Imbalance]: max/min = {max_min_ratio:.2f}x (> {COMPUTE_SKEW_MAX_MIN_RATIO:.2f}x).")
            logger.log(f"Slowest: Rank {slowest} | Fastest: Rank {fastest}")
            logger.log("Action: Check pipeline layer partitioning or operator assignment.")
        else:
            logger.log("[Static Imbalance]: Pipeline stages are well-balanced.")
        logger.log("=" * 100 + "\n")

    return multi_stats, multi_dfs


def hardware_jitter_analysis(graph: TraceGraph, logger: ReportLogger) -> List[Dict[str, Any]]:
    """Workload-Normalized Hardware Jitter Detection with TP-Group Contamination Analysis.
    
    Phase 1: Per-rank jitter detection using workload bucketing and slowdown ratio.
    Phase 2: TP-group contamination detection. When TP > 1, a single slow GPU
             causes TP AllReduce to block all peers in the group. This phase
             correlates simultaneous spikes within TP groups to identify the
             true root-cause GPU vs innocent victims of TP synchronization.
    """
    logger.log("\n[Hardware Jitter Analysis] Running Workload-Normalized Jitter Detection (TP-Decoupled)...")

    jitter_data = []
    spike_threshold = 1.15
    
    compute_events = [ev for ev in graph.compute_events if ev.name in ["forward-step", "backward-step"] and ev.dur > 10]
    
    # === Phase 1: Per-Rank Jitter Detection ===
    for phase in ["forward-step", "backward-step"]:
        phase_events = [ev for ev in compute_events if ev.name == phase]
        if not phase_events: continue
        
        ranks = sorted(list(set(ev.rank for ev in phase_events)))
        for rank in ranks:
            rank_events = [
                ev
                for ev in phase_events
                if ev.rank == rank
                and ev.args.get("num_tokens", 0) is not None
                and ev.args.get("sum_sq_seq_len", 0.0) is not None
            ]
            if not rank_events:
                continue
            
            workload_buckets = collections.defaultdict(list)
            for ev in rank_events:
                n = ev.args.get("num_tokens", 0)
                s_sq = ev.args.get("sum_sq_seq_len", 0.0)
                if s_sq == 0.0 and "seq_len" in ev.args:
                    s_sq = float(ev.args.get("seq_len", ev.args.get("max_seq_len", 0)) ** 2)
                
                sig = (n, round(s_sq / 10000) * 10000)
                pure_dur_ms = get_pure_compute_dur(ev, graph) / 1000.0
                workload_buckets[sig].append(pure_dur_ms)
                
            baselines = {}
            for sig, pure_durs in workload_buckets.items():
                if len(pure_durs) < 3: baselines[sig] = np.min(pure_durs)
                else: baselines[sig] = np.percentile(pure_durs, 10) 
            
            rank_spike_count = 0
            for ev in rank_events:
                n = ev.args.get("num_tokens", 0)
                s_sq = ev.args.get("sum_sq_seq_len", 0.0)
                if s_sq == 0.0 and "seq_len" in ev.args:
                    s_sq = float(ev.args.get("seq_len", ev.args.get("max_seq_len", 0)) ** 2)
                
                sig = (n, round(s_sq / 10000) * 10000)
                pure_dur_ms = get_pure_compute_dur(ev, graph) / 1000.0
                base_t = baselines[sig]
                
                ratio = (pure_dur_ms / base_t) if base_t > 0 else 1.0
                is_spike = ratio > spike_threshold
                if is_spike: rank_spike_count += 1
                
                topo = graph.rank_topology.get(rank, {'dp': 0, 'tp': 0, 'pp': 0})
                jitter_data.append({
                    "ts_s": ev.ts / 1e6,
                    "rank": rank,
                    "phase": "Forward" if "forward" in phase else "Backward",
                    "total_dur_ms": ev.dur / 1000.0,
                    "pure_dur_ms": pure_dur_ms,
                    "baseline_ms": base_t,
                    "slowdown_ratio": ratio,
                    "is_spike": is_spike,
                    "iteration": ev.iteration,
                    "microbatch_id": ev.args.get("current_microbatch", -1),
                    "dp_rk": topo['dp'],
                    "tp_rk": topo['tp'],
                    "pp_rk": topo['pp'],
                    "is_tp_root_cause": False,
                    "is_tp_victim": False,
                    "event_ref": ev,
                })
                
                ev.raw["args"]["Analysis_Pure_Compute_ms"] = f"{pure_dur_ms:.2f}"
                ev.raw["args"]["Analysis_Slowdown_Ratio"] = f"{ratio:.2f}x"
                if is_spike:
                    ev.raw["cname"] = "terrible"
                    ev.raw["args"]["Analysis_Diagnosis"] = "True Hardware Jitter Spike"
            
            if rank_spike_count > 0:
                logger.log(f"  -> Rank {rank} [{phase}]: Found {rank_spike_count} hardware jitter spikes (>15% slower).")

    # === Phase 2: TP-Group Contamination Detection ===
    tp_groups = graph.get_tp_groups()
    tp_size = max((len(v) for v in tp_groups.values()), default=1)
    
    if tp_size > 1 and jitter_data:
        logger.log("\n  [TP Contamination Detection] Analyzing cross-rank correlation within TP groups...")
        
        spike_events = [j for j in jitter_data if j.get("is_spike", False)]
        
        if spike_events:
            # Index spikes by (rank, iteration, microbatch) for precise matching
            spike_index = collections.defaultdict(list)
            for j in spike_events:
                key = (j["rank"], j["iteration"], j["microbatch_id"], j["phase"])
                spike_index[key].append(j)
            
            tp_root_counts = collections.defaultdict(int)
            tp_victim_counts = collections.defaultdict(int)
            processed = set()
            
            for j in spike_events:
                tp_key = (j["dp_rk"], j["pp_rk"])
                match_key = (tp_key, j["iteration"], j["microbatch_id"], j["phase"])
                if match_key in processed:
                    continue
                processed.add(match_key)
                
                group_ranks = tp_groups.get(tp_key, [j["rank"]])
                
                # Find which ranks in this TP group also spiked at this exact computation step
                group_spikes = {}
                for gr in group_ranks:
                    gr_topo = graph.rank_topology.get(gr, {})
                    candidates = spike_index.get(
                        (gr, j["iteration"], j["microbatch_id"], j["phase"]), []
                    )
                    if candidates:
                        group_spikes[gr] = max(c["slowdown_ratio"] for c in candidates)
                
                if len(group_spikes) >= 2:
                    # Multiple ranks spiked simultaneously in the same TP group.
                    # Root cause = rank with highest pure-compute slowdown ratio,
                    # because it arrived latest at the TP AllReduce, blocking others.
                    root_rank = max(group_spikes, key=group_spikes.get)
                    tp_root_counts[root_rank] += 1
                    
                    for gr in group_spikes:
                        if gr != root_rank:
                            tp_victim_counts[gr] += 1
                    
                    # Tag the individual jitter_data entries
                    for jd in jitter_data:
                        if (jd["iteration"] == j["iteration"] 
                                and jd["microbatch_id"] == j["microbatch_id"]
                                and jd["phase"] == j["phase"]
                                and jd.get("is_spike")):
                            ev_ref = jd.get("event_ref")
                            if jd["rank"] == root_rank:
                                jd["is_tp_root_cause"] = True
                                if ev_ref:
                                    ev_ref.raw["args"]["TP_Jitter_Role"] = "ROOT_CAUSE"
                            elif jd["rank"] in group_spikes:
                                jd["is_tp_victim"] = True
                                jd["is_spike"] = False
                                if ev_ref:
                                    ev_ref.raw["cname"] = "bad"
                                    ev_ref.raw["args"]["TP_Jitter_Role"] = "VICTIM (TP sync propagation)"
                                    ev_ref.raw["args"]["Analysis_Diagnosis"] = f"TP Victim (Root: Rank {root_rank})"
            
            if tp_root_counts:
                logger.log("  TP Contamination Summary:")
                for rank, count in sorted(tp_root_counts.items(), key=lambda x: -x[1]):
                    logger.log(f"    -> Rank {rank}: ROOT CAUSE of {count} TP-correlated spike events")
                for rank, count in sorted(tp_victim_counts.items(), key=lambda x: -x[1]):
                    logger.log(f"    -> Rank {rank}: VICTIM of {count} TP-correlated spikes (TP sync propagation)")
            else:
                logger.log("  -> No TP-correlated contamination detected (spikes are independent).")
    
    # Clean up event_ref before returning (not serializable)
    for j in jitter_data:
        j.pop("event_ref", None)

    return jitter_data


# ============================================================================
# 3. Visualization Modules
# ============================================================================
def generate_p2p_bw_plots(plot_data: List[Dict[str, Any]], theory_bw_gbps: float, output_dir: str):
    """[NEW] Generates plots specifically for Effective P2P Bandwidth analysis."""
    if not plot_data: return
    plt, pd = _load_reporting_dependencies()
    df = pd.DataFrame(plot_data)
    df["ts_s"] = df["ts_s"] - df["ts_s"].min()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle('P2P Communication: Effective Bandwidth Analysis', fontsize=22, fontweight='bold')
    
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))

    # --- Left: Effective BW over Time ---
    ax = axes[0]
    for idx, rank in enumerate(ranks):
        subset = df[df["rank"] == rank]
        ax.plot(subset["ts_s"], subset["eff_bw_gbps"], marker='o', markersize=4, linestyle='-', 
                alpha=0.7, label=f"Rank {rank}", color=colors[idx % 10])
    ax.axhline(y=theory_bw_gbps, color='red', linestyle='--', linewidth=2, label=f'Theoretical Peak ({theory_bw_gbps} GB/s)')
    ax.set_title("Effective Bandwidth over Time\n(Low = Dependency Stall / PCIe Contention)", fontsize=14)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.legend(loc='upper right', fontsize='small')

    # --- Right: Effective BW Distribution (Violin/Boxplot equivalent) ---
    ax = axes[1]
    data_to_plot = [df[df["rank"] == r]["eff_bw_gbps"].values for r in ranks]
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    ax.set_xticks(range(1, len(ranks) + 1))
    ax.set_xticklabels([str(r) for r in ranks])
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    
    ax.axhline(y=theory_bw_gbps, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title("Bandwidth Distribution per Rank\n(Variance indicates System Jitter)", fontsize=14)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    bw_path = os.path.join(output_dir, "p2p_actual_bandwidth_analysis.pdf")
    plt.savefig(bw_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[PP P2P Analysis] Bandwidth plot saved to {bw_path}")


def generate_p2p_plots(plot_data: List[Dict[str, Any]], output_dir: str):
    """Generates insightful plots focused on Exposed Communication Bubbles."""
    if not plot_data: 
        print("[PP P2P Analysis] No data available for P2P plots.")
        return
    plt, pd = _load_reporting_dependencies()

    df = pd.DataFrame(plot_data)
    df = df.sort_values(by="ts_s")
    df["ts_s"] = df["ts_s"] - df["ts_s"].min()
    
    # Calculate cumulative stall time per rank
    df['cum_stall_s'] = df.groupby('rank')['stall_us'].cumsum() / 1e6

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_DOUBLE, FIG_H_TALL))
    fig.suptitle('Pipeline Parallelism: Exposed Communication Bubble Analysis', fontsize=20)

    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(ranks)))
    rank_color_map = dict(zip(ranks, colors))

    # --- Plot 1: Exposed Stall Time over Time (Scatter/Line) ---
    ax = axes[0, 0]
    for rank in ranks:
        subset = df[df["rank"] == rank]
        ax.plot(subset["ts_s"], subset["stall_us"], 
                marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.7, 
                label=f"Rank {rank}", color=rank_color_map[rank])
    ax.set_title("Exposed Bubble Time (Dependency Stall)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Stall Time (us)", fontsize=12)
    ax.set_xlabel("Time elapsed (seconds)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='small', frameon=True)

    # --- Plot 2: Cumulative Stall Time (Line) ---
    ax = axes[0, 1]
    for rank in ranks:
        subset = df[df["rank"] == rank]
        ax.plot(subset["ts_s"], subset["cum_stall_s"], 
                marker='', linestyle='-', linewidth=2, alpha=0.9, 
                label=f"Rank {rank}", color=rank_color_map[rank])
    ax.set_title("Cumulative Wasted Time (GPU Idle)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Total Wasted Time (Seconds)", fontsize=12)
    ax.set_xlabel("Time elapsed (seconds)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize='small', frameon=True)

    # --- Plot 3: Stall Distribution (Histogram) ---
    ax = axes[1, 0]
    # Filter out near-zero stalls (perfect overlaps) to see the actual problems
    problematic_stalls = df[df['stall_us'] > 100] 
    if not problematic_stalls.empty:
        for rank in ranks:
            subset = problematic_stalls[problematic_stalls["rank"] == rank]
            ax.hist(subset["stall_us"], bins=30, alpha=0.5, 
                    label=f"Rank {rank}", color=rank_color_map[rank])
    ax.set_title("Distribution of Problematic Stalls (>100us)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlabel("Stall Time (us)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='small', frameon=True)

    # --- Plot 4: Transferred Data Size over Time ---
    ax = axes[1, 1]
    for rank in ranks:
        subset = df[df["rank"] == rank]
        ax.plot(subset["ts_s"], subset["mb"], 
                marker='s', markersize=3, linestyle='-', linewidth=1, alpha=0.7, 
                label=f"Rank {rank}", color=rank_color_map[rank])
    ax.set_title("P2P Payload Size", fontsize=14, fontweight='bold')
    ax.set_ylabel("Size (MB)", fontsize=12)
    ax.set_xlabel("Time elapsed (seconds)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "p2p_exposed_bubble_analysis.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[PP P2P Analysis] Plot saved to {output_path}")

def generate_compute_plots(multi_stats: Dict[str, List[Dict]], multi_dfs: Dict[str, Dict[int, "pd.DataFrame"]], output_dir: str):
    """Generates scatter and bar charts for Compute Load Imbalance."""
    if "forward" not in multi_stats or not multi_stats["forward"]: 
        return
    plt, pd = _load_reporting_dependencies()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle('Forward Compute Analysis: Data Skew vs Hardware Jitter (VarLen Aware)', fontsize=22, fontweight='bold')

    phase = "forward"
    stats_df = pd.DataFrame(multi_stats[phase])
    ranks = sorted(stats_df['Rank'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))

    # --- 左图: X轴为 N (Total Tokens) ---
    ax = axes[0]
    for idx, rank in enumerate(ranks):
        df = multi_dfs[phase][rank]
        r_val = stats_df[stats_df['Rank'] == rank]['r_N'].values[0]
        r_str = f"r={r_val:.2f}" if r_val != 0.0 else "Static"
        ax.scatter(df['tokens_N'], df['dur_ms'], label=f"Rank {rank} ({r_str})", 
                   color=colors[idx % 10], alpha=0.6, edgecolors='none', s=40)
    ax.set_title('Duration vs Total Tokens (N)\n[FFN / Linear Check]', fontsize=14)
    ax.set_xlabel('Number of Tokens (N)', fontsize=12)
    ax.set_ylabel('Forward Duration (ms)', fontsize=12)
    ax.legend(title="Pearson(N)", loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 中图: X轴为 S_sq (Sum of Squared SeqLen) ---
    ax = axes[1]
    for idx, rank in enumerate(ranks):
        df = multi_dfs[phase][rank]
        rho_val = stats_df[stats_df['Rank'] == rank]['rho_Ssq'].values[0]
        rho_str = f"ρ={rho_val:.2f}" if rho_val != 0.0 else "Static"
        # 归一化 S_sq 以便横轴显示 (如 x 10^6)
        ax.scatter(df['seq_len_sq_L2'] / 1e6, df['dur_ms'], label=f"Rank {rank} ({rho_str})", 
                   color=colors[idx % 10], alpha=0.6, marker='^', edgecolors='none', s=45)
    ax.set_title('Duration vs VarLen Workload (S_sq)\n[Attention Check]', fontsize=14)
    ax.set_xlabel('Sum of Squared SeqLen (Millions)', fontsize=12)
    ax.set_ylabel('Forward Duration (ms)', fontsize=12)
    ax.legend(title="Spearman(S_sq)", loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 右图: 静态均值柱状图 ---
    ax = axes[2]
    mean_durs = stats_df['Mean_Dur_ms'].values
    min_mean = np.min(mean_durs)
    max_mean = np.max(mean_durs)
    max_min_ratio = (max_mean / min_mean) if min_mean > 0 else 0

    bars = ax.bar(stats_df['Rank'].astype(str), mean_durs, color=colors[:len(ranks)], alpha=0.85, edgecolor='black')
    ax.set_title(f'Average Forward Compute per Rank\n(Max/Min Ratio: {max_min_ratio:.2f}x, threshold {COMPUTE_SKEW_MAX_MIN_RATIO:.2f}x)', fontsize=14)
    ax.set_xlabel('Global Rank', fontsize=12)
    ax.set_ylabel('Mean Duration (ms)', fontsize=12)
    
    # 高亮诊断
    min_idx, max_idx = np.argmin(mean_durs), np.argmax(mean_durs)
    bars[max_idx].set_edgecolor('red'); bars[max_idx].set_linewidth(2.5)
    bars[min_idx].set_edgecolor('green'); bars[min_idx].set_linewidth(2.5)
    
    ax.axhline(y=min_mean, color='green', linestyle='--', alpha=0.8, label='Fastest Stage')
    ax.axhline(y=max_mean, color='red', linestyle='--', alpha=0.8, label='Slowest Stage')
    ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "forward_compute_diagnostics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[PP Compute Analysis] Plot saved to {output_path}")


def generate_bubble_plots(bubble_stats: List[Dict[str, Any]], output_dir: str):
    """Generates charts for Bubble Rate Analysis, structurally organized by DP and TP."""
    if not bubble_stats: return
    plt, pd = _load_reporting_dependencies()
    
    df = pd.DataFrame(bubble_stats)
    
    # 将 DP_Rank, TP_Rank, PP_Rank 等拓扑信息加入聚合，保持上下文
    df_avg = df.groupby(['Group', 'DP_Rank', 'TP_Rank', 'PP_Rank', 'Rank']).agg({
        'T_actual_ms': 'mean',
        'T_active_ms': 'mean',
        'Bubble_Rate': 'mean'
    }).reset_index()
    
    df_avg['T_bubble_ms'] = df_avg['T_actual_ms'] - df_avg['T_active_ms']
    
    # [核心修改 2]: 按照 DP Rank 和 TP Rank 对所有的子图 Group 进行逻辑排序
    unique_groups_df = df_avg[['Group', 'DP_Rank', 'TP_Rank']].drop_duplicates().sort_values(by=['DP_Rank', 'TP_Rank'])
    groups = unique_groups_df['Group'].tolist()
    num_groups = len(groups)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 动态创建画布，每条独立的流水线 (固定 DP 和 TP) 占据一整行
    fig, axes = plt.subplots(num_groups, 2, figsize=(FIG_W_DOUBLE, FIG_H * num_groups), squeeze=False)
    
    for i, group_name in enumerate(groups):
        # 取出该 Group 内的所有 Rank，并严格按照 PP Rank (流水线前后顺序) 从左到右排序！
        group_df = df_avg[df_avg['Group'] == group_name].sort_values(by="PP_Rank")
        
        # 创新展示：X 轴同时显示全局 Rank 和 PP Rank (让用户直接看出这是第几层)
        x_labels = [f"Rank {r}\n(PP={p})" for r, p in zip(group_df['Rank'], group_df['PP_Rank'])]
        
        # 计算该流水线的整体 Bubble Rate 平均值
        group_mean_bubble_rate = group_df['Bubble_Rate'].mean()
        
        # --- Left Plot: Bubble Rate % ---
        ax_left = axes[i, 0]
        bars = ax_left.bar(x_labels, group_df['Bubble_Rate'] * 100, color='coral', edgecolor='black', alpha=0.85)
        
        # 绘制流水线平均标准线
        ax_left.axhline(y=group_mean_bubble_rate * 100, color='red', linestyle='--', linewidth=2, 
                        label=f'Pipeline Avg ({group_mean_bubble_rate * 100:.1f}%)')
        
        ax_left.set_title(f'Pipeline Bubble Rate [{group_name}]', fontsize=14, fontweight='bold')
        ax_left.set_ylabel('Bubble Rate (%)', fontsize=12)
        ax_left.set_ylim(0, 100)
        ax_left.legend(loc='upper right')
        
        for bar in bars:
            height = bar.get_height()
            ax_left.annotate(f'{height:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)

        # --- Right Plot: Active vs Bubble Breakdown (Stacked Bar) ---
        ax_right = axes[i, 1]
        ax_right.bar(x_labels, group_df['T_active_ms'], label='Active Time (Compute + Comm)', color='mediumseagreen', edgecolor='black')
        ax_right.bar(x_labels, group_df['T_bubble_ms'], bottom=group_df['T_active_ms'], label='Bubble Time (Wait/Idle)', color='lightgrey', hatch='//', edgecolor='black')
        
        ax_right.set_title(f'Iteration Time Breakdown [{group_name}]', fontsize=14, fontweight='bold')
        ax_right.set_ylabel('Time (ms)', fontsize=12)
        ax_right.legend(loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pipeline_bubble_analysis_grouped.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Bubble Analysis Plots -> Saved to {output_path}")

def generate_jitter_plots(jitter_data: List[Dict[str, Any]], output_dir: str):
    """[NEW] Generates True Hardware Jitter visualizations."""
    if not jitter_data: return
    plt, pd = _load_reporting_dependencies()
    
    df = pd.DataFrame(jitter_data)
    df["ts_s"] = df["ts_s"] - df["ts_s"].min()
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    fig.suptitle('True Hardware Jitter Analysis (TP-Wait Decoupled)', fontsize=22, fontweight='bold')
    
    ranks = sorted(df["rank"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    rank_color_map = dict(zip(ranks, colors))
    
    # --- Left: Timeline Scatter ---
    ax = axes[0]
    has_tp_info = "is_tp_root_cause" in df.columns and "is_tp_victim" in df.columns
    for rank in ranks:
        subset = df[df["rank"] == rank]
        subset = subset[subset["slowdown_ratio"] >= 1.0] 
        ax.scatter(subset["ts_s"], subset["slowdown_ratio"], 
                   label=f"Rank {rank}", color=rank_color_map[rank], alpha=0.6, s=30, edgecolors='none')
        if has_tp_info:
            root_pts = subset[subset["is_tp_root_cause"] == True]
            if not root_pts.empty:
                ax.scatter(root_pts["ts_s"], root_pts["slowdown_ratio"],
                           marker='X', s=120, color='red', edgecolors='black', linewidth=1.0, zorder=10)
                   
    ax.axhline(y=1.0, color='green', linestyle='-', linewidth=2, label='Ideal Baseline (1.0x)')
    ax.axhline(y=1.15, color='red', linestyle='--', linewidth=2, label='Spike Threshold (1.15x)')
    title = "Pure Compute Slowdown Ratio over Time\n(Spikes indicate True Hardware Throttling)"
    if has_tp_info:
        title += "\n(X marks = TP Root Cause)"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Slowdown Ratio (Pure Time / Baseline)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylim(0.95, min(df["slowdown_ratio"].max() * 1.05, 2.5))
    ax.legend(loc='upper right', fontsize='small')

    # --- Right: Jitter Spike Count (distinguishes TP root cause vs victim) ---
    ax = axes[1]
    spike_counts = df[df["is_spike"] == True].groupby("rank").size()
    counts_to_plot = [spike_counts.get(r, 0) for r in ranks]
    
    if has_tp_info:
        root_counts = df[df["is_tp_root_cause"] == True].groupby("rank").size()
        victim_counts = df[df["is_tp_victim"] == True].groupby("rank").size()
        root_to_plot = [root_counts.get(r, 0) for r in ranks]
        victim_to_plot = [victim_counts.get(r, 0) for r in ranks]
        independent_to_plot = [max(0, c - root_counts.get(r, 0) - victim_counts.get(r, 0)) 
                               for c, r in zip(counts_to_plot, ranks)]
        
        ax.bar([str(r) for r in ranks], root_to_plot, color='red', alpha=0.85, 
               edgecolor='black', label='TP Root Cause')
        ax.bar([str(r) for r in ranks], independent_to_plot, 
               bottom=root_to_plot, color=colors[:len(ranks)], alpha=0.85, 
               edgecolor='black', label='Independent Spike')
        ax.bar([str(r) for r in ranks], victim_to_plot, 
               bottom=[a + b for a, b in zip(root_to_plot, independent_to_plot)],
               color='lightgrey', alpha=0.85, edgecolor='black', hatch='//', label='TP Victim')
        ax.legend(loc='upper right', fontsize='small')
    else:
        bars = ax.bar([str(r) for r in ranks], counts_to_plot, color=colors[:len(ranks)], alpha=0.85, edgecolor='black')
    
    ax.set_title("Hardware Jitter Spikes per Rank\n(Red = TP Root Cause, Grey = TP Victim)", fontsize=14)
    ax.set_ylabel("Number of Spikes (>15% slowdown)", fontsize=12)
    ax.set_xlabel("Global Rank", fontsize=12)
    
    if sum(counts_to_plot) > 0 and not has_tp_info:
        worst_idx = np.argmax(counts_to_plot)
        bars[worst_idx].set_edgecolor('red')
        bars[worst_idx].set_linewidth(3)
        bars[worst_idx].set_hatch('//')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "hardware_jitter_analysis.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  -> Saved {out_path}")


def generate_compute_plots2(multi_stats: Dict[str, List[Dict]], multi_dfs: Dict[str, Dict[int, "pd.DataFrame"]], output_dir: str):
    """Generates scatter and bar charts for Compute Load Imbalance."""
    if "forward" not in multi_stats or not multi_stats["forward"]: 
        return
    plt, pd = _load_reporting_dependencies()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 恢复原始的超宽画布 (24, 7)，但只切分为 1 行 2 列
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))

    phase = "forward"
    stats_df = pd.DataFrame(multi_stats[phase])
    ranks = sorted(stats_df['Rank'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(ranks))))
    
    # 图例固定为两列显示
    legend_ncol = 2

    # 图例内部紧密排列的参数
    legend_tight_kwargs = {
        'labelspacing': 0.2,     # 行距
        'columnspacing': 0.8,    # 列距
        'handletextpad': 0.4,    # 图标和文字间距
        'borderpad': 0.3         # 边框内边距
    }

    # --- 左图: X轴为 N (Total Tokens) ---
    ax = axes[0]
    for idx, rank in enumerate(ranks):
        df = multi_dfs[phase][rank]
        r_val = stats_df[stats_df['Rank'] == rank]['r_N'].values[0]
        r_str = f"r={r_val:.2f}" if r_val != 0.0 else "Static"
        ax.scatter(df['tokens_N'], df['dur_ms'], label=f"Rank {rank} ({r_str})", 
                   color=colors[idx % 10], alpha=0.6, edgecolors='none', s=40)
    ax.set_xlabel('Number of Tokens (N)', fontsize=24)
    ax.set_ylabel('Forward Duration (ms)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 【美化重点】：loc='upper center' (顶部居中), shadow=True (阴影), fancybox=True (圆角)
    leg1 = ax.legend(title="Pearson(N)", loc='upper right', fontsize=17, 
                     ncol=legend_ncol, facecolor='white', framealpha=1.0, edgecolor='black',
                     frameon=True, shadow=True, fancybox=True, markerscale=1.5, **legend_tight_kwargs)
    leg1.set_zorder(100)
    leg1.get_title().set_fontsize(18) 
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 中图: X轴为 S_sq (Sum of Squared SeqLen) ---
    ax = axes[1]
    for idx, rank in enumerate(ranks):
        df = multi_dfs[phase][rank]
        rho_val = stats_df[stats_df['Rank'] == rank]['rho_Ssq'].values[0]
        rho_str = f"ρ={rho_val:.2f}" if rho_val != 0.0 else "Static"
        ax.scatter(df['seq_len_sq_L2'] / 1e6, df['dur_ms'], label=f"Rank {rank} ({rho_str})", 
                   color=colors[idx % 10], alpha=0.6, marker='^', edgecolors='none', s=45)
    ax.set_xlabel('Sum of Squared SeqLen (Millions)', fontsize=24)
    ax.set_ylabel('Forward Duration (ms)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 【美化重点】：loc='upper center' (顶部居中), shadow=True (阴影), fancybox=True (圆角)
    leg2 = ax.legend(title="Spearman(S_sq)", loc='upper right', fontsize=17, 
                     ncol=legend_ncol, facecolor='white', framealpha=1.0, edgecolor='black',
                     frameon=True, shadow=True, fancybox=True, markerscale=1.5, **legend_tight_kwargs)
    leg2.set_zorder(100)
    leg2.get_title().set_fontsize(18)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局铺满画布
    plt.tight_layout(pad=1.5, w_pad=2.0)
    
    output_path = os.path.join(output_dir, "forward_compute_diagnostics.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[PP Compute Analysis] Plot saved to {output_path}")


def generate_bubble_plots2(bubble_stats: List[Dict[str, Any]], output_dir: str,
                           pp_config: Optional[Dict[str, Any]] = None):
    """Generates charts for Bubble Rate Analysis, structurally organized by DP and TP.

    ``pp_config`` is an optional dict with keys ``n_pp``, ``n_microbatches``
    (and optionally ``n_dp``, ``n_tp``) extracted from the actual trace data.
    If not provided, topology is inferred from *bubble_stats* and no
    theoretical line is drawn.
    """
    if not bubble_stats: return
    plt, pd = _load_reporting_dependencies()
    
    df = pd.DataFrame(bubble_stats)
    
    df_avg = df.groupby(['Group', 'DP_Rank', 'TP_Rank', 'PP_Rank', 'Rank']).agg({
        'T_actual_ms': 'mean',
        'T_active_ms': 'mean',
        'Bubble_Rate': 'mean'
    }).reset_index()
    
    df_avg['T_bubble_ms'] = df_avg['T_actual_ms'] - df_avg['T_active_ms']

    # --- Derive 1F1B theoretical bubble rate from trace topology ---
    if pp_config is None:
        pp_config = {}
    n_pp = pp_config.get("n_pp", int(df["PP_Rank"].max()) + 1 if len(df) else 1)
    n_micro = pp_config.get("n_microbatches", 0)

    has_theory = n_pp > 1 and n_micro > 0
    if has_theory:
        theo_bubble_rate = (n_pp - 1) / (n_micro + n_pp - 1)
    else:
        theo_bubble_rate = 0.0
    theo_bubble_rate_pct = theo_bubble_rate * 100.0
    
    unique_groups_df = df_avg[['Group', 'DP_Rank', 'TP_Rank']].drop_duplicates().sort_values(by=['DP_Rank', 'TP_Rank'])
    groups = unique_groups_df['Group'].tolist()
    num_groups = len(groups)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 将宽度系数从 7 缩小为 5.5，让子图适当缩窄
    fig, axes = plt.subplots(2, num_groups, figsize=(max(FIG_W_DOUBLE, 3.4 * num_groups), FIG_H_TALL * 1.25), squeeze=False)
    
    for i, group_name in enumerate(groups):
        # 取出该 Group 内的所有 Rank，并严格按照 PP Rank (流水线前后顺序) 从左到右排序！
        group_df = df_avg[df_avg['Group'] == group_name].sort_values(by="PP_Rank")
        
        # X 轴同时显示全局 Rank 和 PP Rank
        x_labels = [f"Rank {r}\n(PP={p})" for r, p in zip(group_df['Rank'], group_df['PP_Rank'])]
        
        # 计算该流水线的整体 Bubble Rate 实际平均值
        group_mean_bubble_rate_pct = group_df['Bubble_Rate'].mean() * 100.0
        
        # ============================================================
        # --- Top Plot: Bubble Rate % (横向排列的第 0 行) ---
        # ============================================================
        ax_top = axes[0, i]
        bars = ax_top.bar(x_labels, group_df['Bubble_Rate'] * 100, color='coral', edgecolor='black', alpha=0.85)
        
        ax_top.axhline(y=group_mean_bubble_rate_pct, color='red', linestyle='-', linewidth=2.5, 
                        label=f'Actual Avg ({group_mean_bubble_rate_pct:.1f}%)')
        if has_theory:
            ax_top.axhline(y=theo_bubble_rate_pct, color='royalblue', linestyle='--', linewidth=2.5, 
                            label=f'Theory 1F1B ({theo_bubble_rate_pct:.1f}%)')
        
        ax_top.set_xlim(-0.6, len(x_labels) + 0.2)
        text_x_pos = len(x_labels) - 0.4
        
        ax_top.text(text_x_pos, group_mean_bubble_rate_pct, f' {group_mean_bubble_rate_pct:.1f}%', 
                    color='red', va='bottom', ha='left', fontweight='bold', fontsize=16)
        if has_theory:
            ax_top.text(text_x_pos, theo_bubble_rate_pct, f' {theo_bubble_rate_pct:.1f}%', 
                        color='royalblue', va='bottom', ha='left', fontweight='bold', fontsize=16)
        
        # 使用 ax.text 将简化的表头 (DP=x, TP=x) 放在子图内部的正上方
        ax_top.text(0.5, 0.97, group_name, transform=ax_top.transAxes, ha='center', va='top', fontsize=22, fontweight='bold', zorder=10)
        
        if i == 0:
            ax_top.set_ylabel('Bubble Rate (%)', fontsize=22)
        
        ax_top.tick_params(axis='both', which='major', labelsize=21)
        
        # 👈 [核心修改点]：去除强制显示 100% 的逻辑，改为自适应最大高度。
        # 取本组柱子最高值、平均值、理论值中的最大者，乘以 1.45 留出上方空间放图例和表头。
        local_max = max((group_df['Bubble_Rate'] * 100).max(), group_mean_bubble_rate_pct, theo_bubble_rate_pct)
        ax_top.set_ylim(0, local_max * 1.70)
        
        # 锚定到 (X=1.0, Y=0.88)，即图例右上角贴着坐标系右边缘，并在表头 (0.97) 的下方
        ax_top.legend(loc='upper right', bbox_to_anchor=(1.0, 0.88), fontsize=17, frameon=True)
        
        # 给每个柱子头顶标上具体值
        for bar in bars:
            height = bar.get_height()
            ax_top.annotate(f'{height:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontsize=18, fontweight='bold')

        # ============================================================
        # --- Bottom Plot: Active vs Bubble Breakdown (横向排列的第 1 行) ---
        # ============================================================
        ax_bottom = axes[1, i]
        ax_bottom.bar(x_labels, group_df['T_active_ms'], label='Active Time (Compute + Comm)', color='mediumseagreen', edgecolor='black', alpha=0.9)
        ax_bottom.bar(x_labels, group_df['T_bubble_ms'], bottom=group_df['T_active_ms'], label='Bubble Time (Wait/Idle)', color='lightgrey', hatch='//', edgecolor='black', alpha=0.9)
        
        # 同样使用 ax.text 将简化的表头放在底部子图内部的正上方
        ax_bottom.text(0.5, 0.97, group_name, transform=ax_bottom.transAxes, ha='center', va='top', fontsize=22, fontweight='bold', zorder=10)
        
        if i == 0:
            ax_bottom.set_ylabel('Time (ms)', fontsize=24)
            
        ax_bottom.tick_params(axis='both', which='major', labelsize=21)
        
        # 动态获取 T_actual_ms 的最大值，并增加 25% 的 Y 轴余量，防止柱子撞到内部标题
        max_time_ms = group_df['T_actual_ms'].max()
        ax_bottom.set_ylim(0, max_time_ms * 1.25)

    # 底部 4 张子图配置共享图例。axes[1, -1] 代表第 2 行最右边的子图
    axes[1, -1].legend(loc='upper right', bbox_to_anchor=(1.0, 0.92), fontsize=18, frameon=True)

    # 恢复紧凑的排版，因为标题已经放回了内部，外部不再需要额外的空间
    plt.tight_layout(rect=[0, 0.04, 1, 0.96], h_pad=2.2, w_pad=2.0)
    output_path = os.path.join(output_dir, "pipeline_bubble_analysis_grouped.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Bubble Analysis Plots -> Saved to {output_path}")


# ============================================================================
# 3b. PP Straggler Extraction Helper
# ============================================================================

def _extract_pp_straggler_data(
    jitter_data: List[Dict[str, Any]],
    p2p_data: List[Dict[str, Any]],
    graph: "TraceGraph",
) -> List[Dict[str, Any]]:
    """Produce a unified straggler record list consumable by hybrid_analyzer.

    Two PP straggler signals are combined:

    1. **Jitter ROOT_CAUSE** — the GPU that arrived latest at a TP AllReduce
       (identified by :func:`hardware_jitter_analysis`).  This rank directly
       stalls the PP forward pass by holding up the whole TP group.

    2. **P2P Dependency Stall** — the *sender* rank whose compute was so slow
       that the downstream receiver was blocked waiting.  The sender rank
       (``sender_rank`` in p2p_data) is the true PP straggler, not the
       receiver.

    Returns:
        List of dicts, one per (rank, iteration) straggler event::

            {
                "rank": int,
                "iteration": int,
                "signal": str,          # "jitter_root_cause" | "p2p_dep_stall"
                "stall_us": float,      # delay caused to peers (μs)
                "slowdown_ratio": float | None,  # jitter-based ratio
                "pp_rank": int | None,
                "tp_rank": int | None,
                "dp_rank": int | None,
            }
    """
    results: List[Dict[str, Any]] = []

    # Signal 1: Jitter ROOT_CAUSE ranks
    for j in jitter_data:
        if not j.get("is_tp_root_cause", False):
            continue
        rank = j["rank"]
        topo = graph.rank_topology.get(rank, {})
        results.append({
            "rank": rank,
            "iteration": j.get("iteration", -1),
            "signal": "jitter_root_cause",
            "stall_us": 0.0,  # stall propagated via TP sync, not directly measured
            "slowdown_ratio": j.get("slowdown_ratio"),
            "pp_rank": topo.get("pp"),
            "tp_rank": topo.get("tp"),
            "dp_rank": topo.get("dp"),
        })

    # Signal 2: P2P dependency stall — attribute to the *sender* (upstream stage)
    for p in p2p_data:
        stall_us = float(p.get("stall_us", 0.0))
        if stall_us < PP_P2P_STALL_THRESHOLD_US:
            continue
        sender_rank = p.get("sender_rank", -1)
        if sender_rank < 0:
            continue
        topo = graph.rank_topology.get(sender_rank, {})
        results.append({
            "rank": sender_rank,
            "iteration": p.get("iteration", -1),
            "signal": "p2p_dep_stall",
            "stall_us": stall_us,
            "slowdown_ratio": None,
            "pp_rank": topo.get("pp"),
            "tp_rank": topo.get("tp"),
            "dp_rank": topo.get("dp"),
        })

    return results


# ============================================================================
# 4. Main Orchestrator
# ============================================================================

def analyze_pp_traces(
    traces: List[Dict[str, Any]],
    theory_bw_gbps: Optional[float] = None,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Master function to run all Pipeline Parallelism trace analyses."""
    pd = _load_pandas()
    os.makedirs(output_dir, exist_ok=True)
    # Paper-mode figure styling: fonts, dpi, font embedding.
    from megatron.megalens.paper_style import apply_global_rcparams
    apply_global_rcparams()
    report_file = os.path.join(output_dir, "pp_diagnostic_report.txt")
    logger = ReportLogger(report_file)

    if theory_bw_gbps is None:
        theory_bw_gbps = get_gpu_p2p_theory_bw_gbps()
        logger.log(f"[Hardware Spec] Auto-detected P2P Theoretical BW (unidirectional): {theory_bw_gbps} GB/s")
    else:
        logger.log(f"[Hardware Spec] Using user-provided P2P BW (unidirectional): {theory_bw_gbps} GB/s")

    logger.log("[PP Analyzer] Building Event Object Graph...")
    graph = TraceGraph(traces)

    # --- Derive pp_config from actual trace topology ---
    pp_config: Dict[str, Any] = {}
    if graph.rank_topology:
        pp_config["n_dp"] = len(set(t['dp'] for t in graph.rank_topology.values()))
        pp_config["n_tp"] = len(set(t['tp'] for t in graph.rank_topology.values()))
        pp_config["n_pp"] = len(set(t['pp'] for t in graph.rank_topology.values()))
    if graph.compute_events:
        fwd_counts = collections.Counter()
        for ev in graph.compute_events:
            if ev.name == "forward-step" and ev.iteration >= 0:
                fwd_counts[(ev.rank, ev.iteration)] += 1
        if fwd_counts:
            pp_config["n_microbatches"] = max(fwd_counts.values())
    logger.log(f"[PP Config] Derived from traces: {pp_config}")

    # 1. P2P Communication Analysis (Calculates Wait Times)
    p2p_plot_data = p2p_comm_analysis(graph, theory_bw_gbps, logger)

    # 2. Bubble Rate Analysis (Depends on P2P Wait Times)
    bubble_stats = bubble_analysis(graph, logger)

    # 3. Compute Load Imbalance Analysis
    compute_stats, raw_dfs = compute_load_analysis(graph, logger)

    # 4. Hardware Jitter Analysis
    jitter_data = hardware_jitter_analysis(graph, logger)

    # 5. Generate Visualizations
    print("\n[PP Analyzer] Generating Visualizations...")
    generate_p2p_bw_plots(p2p_plot_data, theory_bw_gbps, output_dir)
    generate_p2p_plots(p2p_plot_data, output_dir)
    generate_compute_plots2(compute_stats, raw_dfs, output_dir)
    generate_bubble_plots2(bubble_stats, output_dir, pp_config=pp_config)
    generate_jitter_plots(jitter_data, output_dir)

    # 6. CSV export
    logger.log("[PP Analyzer] Exporting CSV stats...")
    if p2p_plot_data:
        pd.DataFrame(p2p_plot_data).to_csv(
            os.path.join(output_dir, "pp_p2p_stats.csv"), index=False
        )
    if bubble_stats:
        pd.DataFrame(bubble_stats).to_csv(
            os.path.join(output_dir, "pp_bubble_stats.csv"), index=False
        )
    compute_rows: List[Dict[str, Any]] = []
    for phase, rows in compute_stats.items():
        for row in rows:
            compute_rows.append({"Phase": phase, **row})
    if compute_rows:
        pd.DataFrame(compute_rows).to_csv(
            os.path.join(output_dir, "pp_compute_stats.csv"), index=False
        )
    if jitter_data:
        pd.DataFrame(jitter_data).drop(columns=["event_ref"], errors="ignore").to_csv(
            os.path.join(output_dir, "pp_jitter_stats.csv"), index=False
        )

    # --- Extract standardised PP straggler data for hybrid_analyzer ---
    # Combines two PP straggler signals into a unified format:
    #   1. Hardware jitter ROOT_CAUSE ranks (slowest GPU in TP group, blocks PP pipeline)
    #   2. P2P dependency-stall victims (ranks with highest recv wait = upstream stage slow)
    pp_straggler_data = _extract_pp_straggler_data(
        jitter_data, p2p_plot_data, graph
    )
    if pp_straggler_data:
        pd.DataFrame(pp_straggler_data).to_csv(
            os.path.join(output_dir, "pp_straggler_stats.csv"), index=False
        )

    print(f"\nAll analysis complete. Report saved to: {report_file}")
    return {
        "p2p_stats": p2p_plot_data,
        "bubble_stats": bubble_stats,
        "compute_stats": compute_stats,
        "jitter_data": jitter_data,
        "pp_config": pp_config,
        "straggler_data": pp_straggler_data,
    }
