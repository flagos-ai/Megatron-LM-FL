from typing import TYPE_CHECKING, Any, Dict, List

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

if TYPE_CHECKING:
    from megatron.megalens.data_loader import TraceDataLoader


def get_tensor_bytes(obj):
    """calculate the number of bytes of a tensor or a list/tuple of tensors"""
    if obj is None:
        return 0
    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(get_tensor_bytes(x) for x in obj)
    return 0


# All NVLink versions (2.0 V100, 3.0 A100, 4.0 H100) provide 25 GB/s per
# direction per sub-link.  Full-duplex means 50 GB/s bidirectional per link,
# but a single P2P send can only use the 25 GB/s unidirectional half.
_NVLINK_UNI_PER_LINK_GBPS = 25.0

# GPU name → NVLink unidirectional P2P bandwidth (GB/s).
# These are NVSwitch numbers (full bisection: any pair can use ALL sub-links).
_NAME_UNI_BW_MAP = {
    "B200": 900.0,  # 36 links × 25  (NVLink 5.0)
    "B100": 900.0,
    "H200": 450.0,  # 18 links × 25  (NVLink 4.0)
    "H100": 450.0,
    "H800": 450.0,
    "A100": 300.0,  # 12 links × 25  (NVLink 3.0)
    "A800": 300.0,
    "V100": 150.0,  #  6 links × 25  (NVLink 2.0)
}

# PCIe per-lane unidirectional bandwidth (GB/s), after 128b/130b encoding etc.
_PCIE_UNI_PER_LANE_GBPS = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}


def get_gpu_p2p_theory_bw_gbps() -> float:
    """Return the theoretical **unidirectional** P2P bandwidth (GB/s) for a
    single point-to-point transfer between two adjacent GPUs.

    Strategy (highest-priority first):
      1. NVML probe — count active NVLink sub-links on GPU 0 and identify
         NVSwitch vs direct-connect topology using
         ``nvmlDeviceGetNvLinkRemotePciInfo``.
         • **NVSwitch** (remote PCI device is a bridge, not a GPU): any GPU
           pair can saturate *all* sub-links → per-pair BW = ``n_links × 25``.
         • **Direct NVLink** (remote is another GPU): per-pair BW =
           ``links_to_that_peer × 25``.
      2. PCIe fallback — ``Gen × Width`` unidirectional.
      3. GPU name fallback — table lookup if pynvml is unavailable.
    """
    fallback = _PCIE_UNI_PER_LANE_GBPS.get(4, 2.0) * 16  # ~31.5 GB/s
    if _HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).upper()
        for key, bw in _NAME_UNI_BW_MAP.items():
            if key in name:
                fallback = bw
                break

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        active_links = 0
        peer_link_counts: dict = {}  # pci_bus_id → count (GPU peers only)
        nvswitch_detected = False

        for link_idx in range(24):
            try:
                state = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                if state != 1:
                    continue
                active_links += 1

                # Identify the remote device at the other end of this sub-link.
                try:
                    remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link_idx)
                    bus_id = getattr(remote_pci, "busId", None)
                    if isinstance(bus_id, bytes):
                        bus_id = bus_id.decode("utf-8", errors="ignore")
                    if bus_id:
                        bus_id = bus_id.strip("\x00").strip()
                    if bus_id:
                        try:
                            bid = bus_id.encode() if isinstance(bus_id, str) else bus_id
                            pynvml.nvmlDeviceGetHandleByPciBusId(bid)
                            peer_link_counts[bus_id] = peer_link_counts.get(bus_id, 0) + 1
                        except pynvml.NVMLError:
                            nvswitch_detected = True
                except (pynvml.NVMLError, AttributeError):
                    pass
            except pynvml.NVMLError:
                break

        if active_links > 0:
            if nvswitch_detected or not peer_link_counts:
                bw = active_links * _NVLINK_UNI_PER_LINK_GBPS
            else:
                bw = max(peer_link_counts.values()) * _NVLINK_UNI_PER_LINK_GBPS
            pynvml.nvmlShutdown()
            return bw

        # No NVLink — fall back to PCIe
        try:
            gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
            width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
            bw = _PCIE_UNI_PER_LANE_GBPS.get(gen, 1.0) * width
            pynvml.nvmlShutdown()
            return bw if bw > 0 else fallback
        except pynvml.NVMLError:
            pass

        pynvml.nvmlShutdown()
        return fallback

    except ImportError:
        return fallback
    except Exception:
        return fallback


def get_gpu_memory_mb() -> float:
    """Return total GPU HBM capacity in MB, auto-detecting from hardware.

    Tries NVML first; falls back to GPU name lookup.  Returns a
    conservative default (40960 MB / A100 40GB) if detection fails.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return mem_info.total / (1024 * 1024)
    except Exception:
        pass
    # Fallback: name-based lookup
    _NAME_MEM_MAP = {
        "B200": 192 * 1024,
        "B100": 192 * 1024,
        "H200": 141 * 1024,
        "H100": 80 * 1024,
        "H800": 80 * 1024,
        "A100": 40 * 1024,
        "A800": 80 * 1024,
        "V100": 32 * 1024,
    }
    if _HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).upper()
        for key, mem in _NAME_MEM_MAP.items():
            if key in name:
                return float(mem)
    return 40960.0


def infer_parallel_sizes_from_traces(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Infer DP/PP/TP/EP sizes from trace event metadata."""
    dp_ranks: set[int] = set()
    pp_ranks: set[int] = set()
    tp_ranks: set[int] = set()
    ep_size = 1
    has_ep_events = False

    for ev in traces:
        args = ev.get("args")
        if isinstance(args, dict):
            dp = args.get("dp_rk")
            pp = args.get("pp_rk")
            tp = args.get("tp_rk")
            if isinstance(dp, int):
                dp_ranks.add(dp)
            if isinstance(pp, int):
                pp_ranks.add(pp)
            if isinstance(tp, int):
                tp_ranks.add(tp)

            if ev.get("name") == "moe-router":
                ep_raw = args.get("ep_size")
                try:
                    if ep_raw is not None:
                        ep_size = max(ep_size, int(ep_raw))
                except (TypeError, ValueError):
                    pass

        name = str(ev.get("name", ""))
        if name.startswith("moe-") or name.startswith("ep-"):
            has_ep_events = True

    return {
        "dp": len(dp_ranks) if dp_ranks else 1,
        "pp": len(pp_ranks) if pp_ranks else 1,
        "tp": len(tp_ranks) if tp_ranks else 1,
        "ep": ep_size if has_ep_events else 1,
    }


def infer_parallel_sizes_from_loader(loader: "TraceDataLoader") -> Dict[str, int]:
    """Infer DP/PP/TP/EP sizes using indexed loader topology + events."""
    topo_vals = list(loader.topology.values())
    dp_size = len({t.get("dp", -1) for t in topo_vals}) if topo_vals else 1
    pp_size = len({t.get("pp", -1) for t in topo_vals}) if topo_vals else 1
    tp_size = len({t.get("tp", -1) for t in topo_vals}) if topo_vals else 1

    ep_size = 1
    has_ep_events = False
    for ev in loader.span_events:
        if ev.name.startswith("moe-") or ev.name.startswith("ep-"):
            has_ep_events = True
        if ev.name == "moe-router":
            ep_raw = ev.args.get("ep_size")
            try:
                if ep_raw is not None:
                    ep_size = max(ep_size, int(ep_raw))
            except (TypeError, ValueError):
                pass

    return {
        "dp": max(1, dp_size),
        "pp": max(1, pp_size),
        "tp": max(1, tp_size),
        "ep": ep_size if has_ep_events else 1,
    }
