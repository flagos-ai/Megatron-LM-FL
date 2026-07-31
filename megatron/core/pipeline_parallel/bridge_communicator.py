# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.observability import open_trace_scope, prepare_trace_scope, trace_is_enabled

########## FlagScale Begin ##########
from megatron.plugin.platform import get_platform

cur_platform = get_platform()
########## FlagScale End ##########

_BRIDGE_BATCH_SEQUENCE = count(1)


def _bridge_payload_context(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    peer_rank: int,
    *,
    direction: str,
    pipeline_direction: str,
) -> dict[str, Any]:
    return {
        "backend": str(dist.get_backend()),
        "comm_type": "p2p",
        "communicator_kind": "bridge",
        "completion_guarantee": "api_return_observed",
        "completion_included": True,
        "completion_kind": "inline_api_return",
        "completion_mode": "inline",
        "data_bytes": int(tensor.numel() * tensor.element_size()),
        "device_completion_guaranteed": False,
        "direction": direction,
        "duration_attribution": "per_operation",
        "message_kind": "payload",
        "operation_count": 1,
        "payload_role": "activation" if pipeline_direction == "forward" else "gradient",
        "peer_rank": peer_rank,
        "pipeline_direction": pipeline_direction,
        "request_id": None,
        "request_pairing": "none",
        "src_module": communicator.src_module_name,
        "dest_module": communicator.dest_module_name,
        "timing_phase": "inline_api_call",
        "transport_api": "send_recv",
    }


def _run_bridge_payload(
    scope: Any,
    tensor: torch.Tensor,
    peer_rank: int,
    *,
    direction: str,
):
    with scope as observation:
        try:
            if direction == "send":
                result = dist.send(tensor, dst=peer_rank)
            else:
                result = dist.recv(tensor, src=peer_rank)
        except BaseException as transport_error:
            observation.set("completed", False)
            observation.set("error_type", type(transport_error).__name__)
            raise
        observation.set("completed", True)
        return result


def _bridge_send_forward(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    peer_rank: int,
):
    gate = prepare_trace_scope("bridge-send-forward")
    if gate is None:
        return dist.send(tensor, dst=peer_rank)
    scope = open_trace_scope(
        gate,
        "bridge-send-forward",
        ctx=_bridge_payload_context(
            communicator,
            tensor,
            peer_rank,
            direction="send",
            pipeline_direction="forward",
        ),
        slots=("completed", "error_type"),
    )
    return _run_bridge_payload(scope, tensor, peer_rank, direction="send")


def _bridge_recv_forward(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    peer_rank: int,
):
    gate = prepare_trace_scope("bridge-recv-forward")
    if gate is None:
        return dist.recv(tensor, src=peer_rank)
    scope = open_trace_scope(
        gate,
        "bridge-recv-forward",
        ctx=_bridge_payload_context(
            communicator,
            tensor,
            peer_rank,
            direction="recv",
            pipeline_direction="forward",
        ),
        slots=("completed", "error_type"),
    )
    return _run_bridge_payload(scope, tensor, peer_rank, direction="recv")


def _bridge_send_backward(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    peer_rank: int,
):
    gate = prepare_trace_scope("bridge-send-backward")
    if gate is None:
        return dist.send(tensor, dst=peer_rank)
    scope = open_trace_scope(
        gate,
        "bridge-send-backward",
        ctx=_bridge_payload_context(
            communicator,
            tensor,
            peer_rank,
            direction="send",
            pipeline_direction="backward",
        ),
        slots=("completed", "error_type"),
    )
    return _run_bridge_payload(scope, tensor, peer_rank, direction="send")


def _bridge_recv_backward(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    peer_rank: int,
):
    gate = prepare_trace_scope("bridge-recv-backward")
    if gate is None:
        return dist.recv(tensor, src=peer_rank)
    scope = open_trace_scope(
        gate,
        "bridge-recv-backward",
        ctx=_bridge_payload_context(
            communicator,
            tensor,
            peer_rank,
            direction="recv",
            pipeline_direction="backward",
        ),
        slots=("completed", "error_type"),
    )
    return _run_bridge_payload(scope, tensor, peer_rank, direction="recv")


@dataclass(frozen=True, slots=True)
class _BridgeBatchOperation:
    batch_id: str
    index: int
    event_name: str
    direction: str
    pipeline_direction: str
    peer_rank: int
    data_bytes: int
    message_kind: str
    backend: str

    @property
    def operation_id(self) -> str:
        return f"{self.batch_id}:{self.index}"

    def trace_fields(self) -> dict[str, Any]:
        fields = {
            "backend": self.backend,
            "comm_type": "p2p",
            "completion_mode": "internal_wait",
            "data_bytes": self.data_bytes,
            "direction": self.direction,
            "message_kind": self.message_kind,
            "microbatch": None,
            "operation_id": self.operation_id,
            "peer_rank": self.peer_rank,
            "pipeline_direction": self.pipeline_direction,
            "request_id": self.operation_id,
            "transport_api": "batch_isend_irecv",
        }
        semantic_role = "activation" if self.pipeline_direction == "forward" else "gradient"
        if self.message_kind == "shape":
            fields["shape_of"] = semantic_role
        else:
            fields["payload_role"] = semantic_role
        return fields


_BridgeBatchSpec = Tuple[str, str, str, int, int, str]


def _bridge_batch_observation_requested(launch_gate: Any) -> bool:
    return (
        launch_gate is not None
        or trace_is_enabled("bridge-send-forward")
        or trace_is_enabled("bridge-recv-forward")
        or trace_is_enabled("bridge-send-backward")
        or trace_is_enabled("bridge-recv-backward")
    )


def _build_bridge_batch_operations(
    specs: List[_BridgeBatchSpec],
) -> List[_BridgeBatchOperation]:
    batch_id = f"bridge-p2p:{next(_BRIDGE_BATCH_SEQUENCE)}"
    backend = str(dist.get_backend())
    return [
        _BridgeBatchOperation(
            batch_id=batch_id,
            index=index,
            event_name=event_name,
            direction=direction,
            pipeline_direction=pipeline_direction,
            peer_rank=peer_rank,
            data_bytes=data_bytes,
            message_kind=message_kind,
            backend=backend,
        )
        for index, (
            event_name,
            direction,
            pipeline_direction,
            peer_rank,
            data_bytes,
            message_kind,
        ) in enumerate(specs)
    ]


def _bridge_batch_launch_context(
    communicator: "BridgeCommunicator",
    operations: List[_BridgeBatchOperation],
) -> dict[str, Any]:
    first = operations[0]
    return {
        "backend": first.backend,
        "batch_id": first.batch_id,
        "comm_type": "p2p-launch",
        "communicator_kind": "bridge",
        "completion_included": False,
        "completion_mode": "internal_wait",
        "dest_module": communicator.dest_module_name,
        "message_kind": first.message_kind,
        "operation_count": len(operations),
        "operations": [operation.trace_fields() for operation in operations],
        "request_pairing": "position",
        "src_module": communicator.src_module_name,
        "timing_phase": "launch",
        "transport_api": "batch_isend_irecv",
    }


def _launch_bridge_batch(
    communicator: "BridgeCommunicator",
    ops: List[torch.distributed.P2POp],
    operations: List[_BridgeBatchOperation],
    launch_gate: Any,
):
    if launch_gate is None:
        return torch.distributed.batch_isend_irecv(ops)
    launch_scope = open_trace_scope(
        launch_gate,
        "bridge-p2p-launch",
        ctx=_bridge_batch_launch_context(communicator, operations),
    )
    with launch_scope:
        return torch.distributed.batch_isend_irecv(ops)


def _bridge_batch_wait_context(
    communicator: "BridgeCommunicator",
    operation: _BridgeBatchOperation,
) -> dict[str, Any]:
    return {
        **operation.trace_fields(),
        "batch_id": operation.batch_id,
        "communicator_kind": "bridge",
        "completion_guarantee": "current_stream_after_wait",
        "completion_included": True,
        "completion_kind": "work_wait",
        "completion_site": "bridge_internal_wait",
        "dest_module": communicator.dest_module_name,
        "duration_attribution": "per_request",
        "host_blocking_guaranteed": False,
        "op": "wait",
        "operation_count": 1,
        "operation_ids": [operation.operation_id],
        "operation_id_scope": "rank_local",
        "request_pairing": "position",
        "src_module": communicator.src_module_name,
        "timing_phase": "stream_dependency",
    }


def _wait_bridge_batch_operation(
    communicator: "BridgeCommunicator",
    request: Any,
    operation: _BridgeBatchOperation,
):
    completion_gate = prepare_trace_scope(operation.event_name)
    if completion_gate is None:
        return request.wait()
    completion_scope = open_trace_scope(
        completion_gate,
        operation.event_name,
        ctx=_bridge_batch_wait_context(communicator, operation),
        slots=("completed", "error_type"),
    )
    with completion_scope as completion:
        try:
            result = request.wait()
        except BaseException as wait_error:
            completion.set("completed", False)
            completion.set("error_type", type(wait_error).__name__)
            raise
        completion.set("completed", result is not False)
        return result


def _run_bridge_batch(
    communicator: "BridgeCommunicator",
    ops: List[torch.distributed.P2POp],
    specs: Optional[List[_BridgeBatchSpec]],
    launch_gate: Any,
) -> None:
    if specs is None:
        requests = torch.distributed.batch_isend_irecv(ops)
        for request in requests:
            request.wait()
        return

    operations = _build_bridge_batch_operations(specs)
    requests = _launch_bridge_batch(communicator, ops, operations, launch_gate)
    if len(requests) != len(operations):
        for request in requests:
            request.wait()
        return
    for request, operation in zip(requests, operations):
        _wait_bridge_batch_operation(communicator, request, operation)


def _bridge_grid_broadcast_context(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    *,
    source_rank: int,
    pipeline_direction: str,
    message_kind: str,
    grid_side: str,
    collective_role: str,
) -> dict[str, Any]:
    context = {
        "backend": "nccl",
        "collective_role": collective_role,
        "comm_type": "collective",
        "communicator_kind": "bridge",
        "completion_guarantee": "api_return_observed",
        "completion_included": True,
        "completion_kind": "inline_api_return",
        "data_bytes": int(tensor.numel() * tensor.element_size()),
        "dest_module": communicator.dest_module_name,
        "device_completion_guaranteed": False,
        "grid_side": grid_side,
        "message_kind": message_kind,
        "pipeline_direction": pipeline_direction,
        "source_rank": source_rank,
        "src_module": communicator.src_module_name,
        "timing_phase": "inline_api_call",
        "transport_api": "broadcast",
    }
    semantic_role = "activation" if pipeline_direction == "forward" else "gradient"
    if message_kind == "shape":
        context["shape_of"] = semantic_role
    else:
        context["payload_role"] = semantic_role
    return context


def _bridge_grid_broadcast(
    communicator: "BridgeCommunicator",
    tensor: torch.Tensor,
    *,
    source_rank: int,
    group: Any,
    pipeline_direction: str,
    message_kind: str,
    grid_side: str,
    collective_role: str,
):
    gate = prepare_trace_scope("bridge-grid-broadcast")
    if gate is None:
        return dist.broadcast(tensor, src=source_rank, group=group)
    scope = open_trace_scope(
        gate,
        "bridge-grid-broadcast",
        ctx=_bridge_grid_broadcast_context(
            communicator,
            tensor,
            source_rank=source_rank,
            pipeline_direction=pipeline_direction,
            message_kind=message_kind,
            grid_side=grid_side,
            collective_role=collective_role,
        ),
        slots=("completed", "error_type"),
    )
    with scope as observation:
        try:
            result = dist.broadcast(tensor, src=source_rank, group=group)
        except BaseException as broadcast_error:
            observation.set("completed", False)
            observation.set("error_type", type(broadcast_error).__name__)
            raise
        observation.set("completed", True)
        return result


class CommRole(Enum):
    """Communication role for ranks in bridge communication.

    SENDER: Leader tp-cp rank within each DP replica of source grid.
            Sends data to destination grid receivers.
    RECEIVER: Leader tp-cp rank within each DP replica of destination grid.
              Receives data from source grid senders.
    MEMBER: Non-leader ranks within DP replicas.
            Participate in broadcasts from their local leader.
    """

    SENDER = "SENDER"
    RECEIVER = "RECEIVER"
    MEMBER = "MEMBER"


@dataclass
class RankCommInfo:
    """Explicit communication plan for a single rank."""

    role: CommRole = CommRole.MEMBER
    send_to_ranks: List[int] = field(default_factory=list)
    recv_from_ranks: List[int] = field(default_factory=list)


class BridgeCommunicator:
    """Pipeline Communicator between two modules with different(TP/DP/PP/CP).

    BridgeCommunicator:
    - Initialize the communicator between a pair of source and destination grids
    - Build a communication schedule for each rank
    - Provide public methods: send_forward, recv_forward, send_forward_recv_backward,
      send_backward_recv_forward to be used by the pipeline schedule.
    """

    # Cache broadcast PGs to avoid creating duplicate NCCL communicators for identical rank sets.
    _broadcast_pg_cache: Dict[str, "torch.distributed.ProcessGroup"] = {}

    @classmethod
    def destroy_broadcast_pgs(cls):
        """Destroy all cached broadcast process groups."""
        for pg in cls._broadcast_pg_cache.values():
            if pg is not None:
                dist.destroy_process_group(pg)
        cls._broadcast_pg_cache.clear()

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        dim_mapping: Optional[Dict[str, int]] = None,
        comm_dtype: Optional[torch.dtype] = None,
        src_module_name: Optional[str] = None,
        dest_module_name: Optional[str] = None,
        tensor_ndim: int = 3,
    ):
        """Initialize the bridge communicator between source and destination grids.

        CP is not supported yet. Will be added in follow up PR.

        Args:
            src_grid: Source HyperCommGrid
            dest_grid: Destination HyperCommGrid
            dim_mapping: Dictionary mapping logical dimensions to tensor axes.
                        Expected keys: 's' (sequence), 'b' (batch), 'h' (hidden).
                        Defaults to {'s': 1, 'b': 0, 'h': 2} if None.
            tensor_ndim: Number of dimensions in tensors communicated through this
                        bridge. For 3D tensors (e.g. [S, B, H]), fan-in/fan-out
                        operates on dim_mapping['b']. For 2D tensors (e.g. [B*S, H]
                        where batch is folded into dim 0), fan-in/fan-out operates
                        on dim 0. Default: 3.
        """
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.comm_dtype = comm_dtype

        assert tensor_ndim in (2, 3), f"tensor_ndim must be 2 or 3, got {tensor_ndim}"
        self.tensor_ndim = tensor_ndim

        # TODO (ykarnati, pthombre) - CP support will be added in follow up PR.
        if 'cp' in self.src_grid.dim_names:
            assert self.src_grid.shape[self.src_grid.dim_names.index('cp')] == 1, (
                f"Source grid CP size must be 1, got "
                f"{self.src_grid.shape[self.src_grid.dim_names.index('cp')]}"
            )

        if 'cp' in self.dest_grid.dim_names:
            assert self.dest_grid.shape[self.dest_grid.dim_names.index('cp')] == 1, (
                f"Destination grid CP size must be 1, got "
                f"{self.dest_grid.shape[self.dest_grid.dim_names.index('cp')]}"
            )

        self.current_rank = dist.get_rank()
        self.comm_map: Dict[int, RankCommInfo] = {}
        if dim_mapping is None:
            self.dim_mapping = {'s': 1, 'b': 0, 'h': 2}
        else:
            assert set(dim_mapping.keys()) == {
                's',
                'b',
                'h',
            }, f"dim_mapping must have keys 's', 'b', 'h', got {set(dim_mapping.keys())}"
            assert all(
                v in {0, 1, 2} for v in dim_mapping.values()
            ), f"dim_mapping values must be 0, 1, or 2, got {list(dim_mapping.values())}"
            self.dim_mapping = dim_mapping

        self.src_grid_broadcast_pg = None
        self.dest_grid_broadcast_pg = None

        src_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(self.src_grid, is_src=True)
        dest_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(
            self.dest_grid, is_src=False
        )

        self.src_grid_broadcast_ranks = []
        if src_grid_broadcast_ranks_list:
            self.src_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                src_grid_broadcast_ranks_list
            )
            self.src_grid_broadcast_ranks = next(
                (ranks for ranks in src_grid_broadcast_ranks_list if self.current_rank in ranks), []
            )

        self.dest_grid_broadcast_ranks = []
        if dest_grid_broadcast_ranks_list:
            self.dest_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                dest_grid_broadcast_ranks_list
            )
            self.dest_grid_broadcast_ranks = next(
                (ranks for ranks in dest_grid_broadcast_ranks_list if self.current_rank in ranks),
                [],
            )

        self.src_tp_leaders, self.src_local_leader_rank = self.get_leader_rank(
            self.src_grid, is_src=True
        )
        self.dest_tp_leaders, self.dest_local_leader_rank = self.get_leader_rank(
            self.dest_grid, is_src=False
        )

        log_msg = (
            f"[Rank {self.current_rank}] "
            f"srcLeader={self.src_local_leader_rank} "
            f"destLeader={self.dest_local_leader_rank} "
            f"srcBroadcastGrpRanks={self.src_grid_broadcast_ranks} "
            f"destBroadcastGrpRanks={self.dest_grid_broadcast_ranks}"
        )
        logging.info(log_msg)

        self.build_comm_map(self.src_tp_leaders, self.dest_tp_leaders)
        dist.barrier()

    @property
    def _batch_dim(self) -> int:
        """Get the tensor dimension used for fan-in/fan-out (cat/split).

        For 3D tensors (e.g. [S, B, H]), this is dim_mapping['b'].
        For 2D tensors (e.g. [B*S, H] where batch is folded into the first
        dimension), this is 0.
        """
        if self.tensor_ndim == 2:
            return 0
        return self.dim_mapping['b']

    @classmethod
    def _get_or_create_broadcast_pg(cls, ranks_list: List[List[int]]):
        """Get or create a broadcast PG, caching to avoid duplicate NCCL communicators."""
        cache_key = str(sorted([tuple(r) for r in ranks_list]))
        if cache_key not in cls._broadcast_pg_cache:
            pg, _ = dist.new_subgroups_by_enumeration(ranks_list, backend='nccl')
            cls._broadcast_pg_cache[cache_key] = pg
        return cls._broadcast_pg_cache[cache_key]

    def get_leader_rank(self, grid: HyperCommGrid, is_src: bool) -> List[int]:
        """Get the leader rank for a given grid and direction.

        We elect leader rank for each dp replica, the first tp-cp rank in the group
        in the last pp stage (for src grid) or first pp stage (for dest grid) is the leader.
        """
        leader_ranks = []
        local_leader_rank = None
        # grid.gen_rank_enum(["tp", "cp", "pp"]) # vary tp & cp, but same dp
        # returns a list of sublists, each sublist is a group of ranks
        # that have different tp & cp & pp, same dp
        per_dp_replica_ranks = grid._gen_rank_enum([x for x in grid.dim_names if x != "dp"])
        if is_src:
            # Add rank from last pp stage
            ranks = []
            for group in per_dp_replica_ranks:
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[-1]
                ranks.append(group[-1])
            leader_ranks.extend(ranks)
        else:
            # Add rank from first pp stage
            ranks = []
            for group in per_dp_replica_ranks:
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[0]
                ranks.append(group[0])
            leader_ranks.extend(ranks)
        return leader_ranks, local_leader_rank

    def get_boundary_pp_stage_ranks(self, grid: HyperCommGrid, is_src: bool):
        """Get TP-CP ranks at boundary PP stage for each DP replica.

        Returns ranks at the last PP stage (if src) or first PP stage (if dest)
        for each DP dimension, ordered by DP dimension.
        """

        # Get tp-cp rank enumeration (each list has same dp and pp, different tp and cp)
        tpcp_rank_lists = grid._gen_rank_enum(['tp', 'cp'])
        pp_size = grid.shape[grid.dim_names.index('pp')]

        # Determine boundary pp stage
        boundary_pp_stage = pp_size - 1 if is_src else 0

        boundary_pp_stage_ranks = []

        for rank_list in tpcp_rank_lists:
            # We can check any rank in the list since they all have the same pp coordinate
            if not rank_list:
                continue
            sample_rank = rank_list[0]
            # Calculate rank coordinates
            rank_coords = []
            temp_rank = sample_rank - grid.rank_offset

            # Extract coordinates in the original dimension order
            for dim_size in grid.shape:
                rank_coords.append(temp_rank % dim_size)
                temp_rank //= dim_size

            pp_coord = rank_coords[grid.dim_names.index('pp')]

            if pp_coord == boundary_pp_stage:
                # This rank list is at the boundary pp stage, add all ranks from this list
                boundary_pp_stage_ranks.append(rank_list)

        return boundary_pp_stage_ranks

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)

    def build_comm_map(self, src_tp_leaders: List[int], dest_tp_leaders: List[int]):
        """Get src/dest tp leaders and populate comm_map for each rank.

        This method analyzes the source and destination grids to determine
        which ranks need to send/receive data and builds the communication
        schedule accordingly.
        """
        # Ensure that the number of leaders can be evenly divided
        src_count = len(src_tp_leaders)
        dest_count = len(dest_tp_leaders)

        if src_count % dest_count != 0 and dest_count % src_count != 0:
            raise ValueError(
                f"Source TP leaders count ({src_count}) and destination TP leaders count "
                f"({dest_count}) must be evenly divisible. One must be a multiple of the other."
            )
        # Get all ranks in source and destination grids
        src_all_ranks = list(
            range(self.src_grid.rank_offset, self.src_grid.rank_offset + self.src_grid.size)
        )
        dest_all_ranks = list(
            range(self.dest_grid.rank_offset, self.dest_grid.rank_offset + self.dest_grid.size)
        )

        all_ranks = src_all_ranks + dest_all_ranks

        # Initialize all ranks as MEMBER by default
        for rank in all_ranks:
            self.comm_map[rank] = RankCommInfo(role=CommRole.MEMBER)

        scale_factor = int(src_count / dest_count)
        if scale_factor > 1:
            # Fan-in: multiple source leaders send to fewer destination leaders
            for i, dest_rank in enumerate(dest_tp_leaders):
                # Each destination rank receives from scale_factor source ranks
                src_ranks = src_tp_leaders[i * scale_factor : (i + 1) * scale_factor]

                # Set up senders
                for src_rank in src_ranks:
                    self.comm_map[src_rank] = RankCommInfo(
                        role=CommRole.SENDER, send_to_ranks=[dest_rank]
                    )

                # Set up receiver
                self.comm_map[dest_rank] = RankCommInfo(
                    role=CommRole.RECEIVER, recv_from_ranks=src_ranks
                )
        else:
            # Fan-out: fewer source leaders send to more destination leaders
            scale_factor = int(dest_count / src_count)
            for i, src_rank in enumerate(src_tp_leaders):
                # Each source rank sends to scale_factor destination ranks
                dest_ranks = dest_tp_leaders[i * scale_factor : (i + 1) * scale_factor]

                # Set up sender
                self.comm_map[src_rank] = RankCommInfo(
                    role=CommRole.SENDER, send_to_ranks=dest_ranks
                )

                # Set up receivers
                for dest_rank in dest_ranks:
                    self.comm_map[dest_rank] = RankCommInfo(
                        role=CommRole.RECEIVER, recv_from_ranks=[src_rank]
                    )

    def send_forward(self, tensor_to_send: torch.Tensor):
        """Send forward activation tensor.

        Args:
            tensor_to_send: The tensor to send to the destination grid
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge Communicator] [send_forward] Rank {self.current_rank} "
                "is not in the source grid."
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.SENDER:
            # Send splits to destination ranks
            num_sends = len(rank_info.send_to_ranks)
            if num_sends > 0:
                tensor_splits = self._split_tensor_at_batch_dim(tensor_to_send, num_sends)
                self._communicate_shapes(tensor_to_send_next=tensor_splits[0])
                for dest_rank, tensor_split in zip(rank_info.send_to_ranks, tensor_splits):
                    logging.debug(
                        f"[Bridge Comunicator] [send_forward] Rank {self.current_rank} "
                        f"send to rank {dest_rank}"
                    )
                    _bridge_send_forward(self, tensor_split, dest_rank)

    def recv_forward(self) -> torch.Tensor:
        """Receive forward activation tensor.

        Args:
            tensor_shape: Expected tensor shape (None if using shape communication)

        Returns:
            torch.Tensor: The received activation tensor
        """
        # receive forward only gets called on the dest grid
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                "is not in the destination grid."
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"
        logging.debug(
            f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
            f"[src - {self.src_module_name}] [dest - {self.dest_module_name}] "
            f"rank_info: {rank_info}"
        )
        if rank_info.role == CommRole.RECEIVER:
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            # p2p call to receive the tensor
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_prev=True)
            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            received_tensors_list = []
            for src_rank, shape in zip(rank_info.recv_from_ranks, recv_forward_shapes):
                tensor_to_recv = torch.empty(
                    shape,
                    device=cur_platform.current_device(),  # FlagScale Add
                    dtype=self.comm_dtype,
                    requires_grad=True,
                )
                _bridge_recv_forward(self, tensor_to_recv, src_rank)
                logging.debug(
                    f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                    f"received tensor from src rank {src_rank} "
                    f"shape {tensor_to_recv.shape} sum {tensor_to_recv.sum()}"
                )
                received_tensors_list.append(tensor_to_recv)
            aggregated_tensor = torch.cat(received_tensors_list, dim=self._batch_dim)
            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"broadcasting tensor {aggregated_tensor.shape} sum {aggregated_tensor.sum()}"
            )

            # Step 1: broadcast its shape so receivers can allocate
            shape_tensor = torch.tensor(
                aggregated_tensor.shape, device=aggregated_tensor.device, dtype=torch.int64
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.current_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="shape",
                grid_side="dest",
                collective_role="source",
            )

            # Step 2: broadcast the actual tensor
            _bridge_grid_broadcast(
                self,
                aggregated_tensor,
                source_rank=self.current_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="payload",
                grid_side="dest",
                collective_role="source",
            )

            return aggregated_tensor

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ):
            # Non-leader rank - participate in broadcast
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.dest_local_leader_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="shape",
                grid_side="dest",
                collective_role="participant",
            )

            received_shape = tuple(shape_tensor.tolist())
            received_tensor = torch.empty(
                received_shape,
                device=cur_platform.current_device(),  # FlagScale Add
                dtype=self.comm_dtype,
                requires_grad=True,
            )

            # Receive the full tensor via broadcast
            _bridge_grid_broadcast(
                self,
                received_tensor,
                source_rank=self.dest_local_leader_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="payload",
                grid_side="dest",
                collective_role="participant",
            )

            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"received tensor via broadcast, shape {received_tensor.shape}"
            )
            return received_tensor

    def send_backward(self, grad_tensor: torch.Tensor):
        """Send backward gradient tensor.

        Note: Gradient senders are activation 'RECEIVERS'

        Args:
            grad_tensor: The gradient tensor to send back
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"[Bridge Communicator] [send_backward] Rank {self.current_rank} "
                "is not in the destination grid."
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.RECEIVER:
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            # Send gradients back to source ranks
            num_receives = len(rank_info.recv_from_ranks)
            tensor_splits = self._split_tensor_at_batch_dim(grad_tensor, num_receives)
            self._communicate_shapes(tensor_to_send_prev=tensor_splits[0])
            if num_receives > 0:
                for src_rank, tensor_split in zip(rank_info.recv_from_ranks, tensor_splits):
                    # Send the gradient split back to the source rank
                    logging.debug(
                        f"[Bridge Communicator] [send_backward] Rank {self.current_rank} "
                        f"sending gradient to src rank {src_rank} "
                        f"shape {tensor_split.shape} sum {tensor_split.sum()}"
                    )
                    _bridge_send_backward(self, tensor_split, src_rank)

    def recv_backward(self) -> torch.Tensor:
        """Receive backward gradient tensor.

        Note: Gradient receivers are activation 'SENDERS'

        Args:
            tensor_shape: Expected gradient tensor shape

        Returns:
            torch.Tensor: The received gradient tensor
        """
        # receive backward only gets called on the src grid
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                "is not in the source grid."
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.SENDER:
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_next=True)
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            # Receive gradient tensors from destination ranks
            received_gradients_list = []
            for dest_rank, grad_shape in zip(rank_info.send_to_ranks, recv_grad_shapes):
                # The destination rank that we sent to will send us gradients back
                grad_tensor = torch.empty(
                    grad_shape, device=cur_platform.current_device(), dtype=self.comm_dtype  # FlagScale Add
                )
                _bridge_recv_backward(self, grad_tensor, dest_rank)
                logging.debug(
                    f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                    f"received gradient from dest rank {dest_rank} "
                    f"shape {grad_tensor.shape} sum {grad_tensor.sum()}"
                )
                received_gradients_list.append(grad_tensor)

            # Concatenate received gradients
            aggregated_gradient = torch.cat(received_gradients_list, dim=self._batch_dim)
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"agg grad shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
            )

            shape_tensor = torch.tensor(
                aggregated_gradient.shape, device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.current_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="shape",
                grid_side="src",
                collective_role="source",
            )

            # Scatter the tensors to all ranks in the group
            _bridge_grid_broadcast(
                self,
                aggregated_gradient,
                source_rank=self.current_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="payload",
                grid_side="src",
                collective_role="source",
            )
            return aggregated_gradient

        elif (
            rank_info.role == CommRole.MEMBER and self.current_rank in self.src_grid_broadcast_ranks
        ):
            # Non-leader rank - participate in gather for gradients
            # Receive broadcasted tensor shape from leader rank
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.src_local_leader_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="shape",
                grid_side="src",
                collective_role="participant",
            )

            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received shape tensor {shape_tensor}"
            )
            received_shape = tuple(shape_tensor.tolist())
            received_gradient = torch.empty(
                received_shape, device=cur_platform.current_device(), dtype=self.comm_dtype  # FlagScale Add
            )

            _bridge_grid_broadcast(
                self,
                received_gradient,
                source_rank=self.src_local_leader_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="payload",
                grid_side="src",
                collective_role="participant",
            )
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received gradient from scatter operation, shape {received_gradient.shape}"
            )
            return received_gradient

    def send_forward_recv_backward(
        self, input_tensor: torch.Tensor, grad_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Combined operation: send forward activation and receive backward gradient.

        Args:
            input_tensor: The tensor to send forward
            grad_shape: Expected gradient tensor shape

        Returns:
            torch.Tensor: The received gradient tensor
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. "
                "send_forward_recv_backward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"
        logging.debug(
            f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
            f"[src - {self.src_module_name}] [dest - {self.dest_module_name}] "
            f"rank_info: {rank_info}"
        )
        if rank_info.role == CommRole.SENDER:
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"

            num_sends = len(rank_info.send_to_ranks)
            activation_splits = self._split_tensor_at_batch_dim(input_tensor, num_sends)
            # Communicate shapes for both directions (send forward, receive backward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_next=activation_splits[0], recv_next=True
            )
            logging.debug(
                f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            if num_sends > 0:
                # Prepare gradient receive tensors
                received_gradients_list = []
                for i, recv_grad_shape in enumerate(recv_grad_shapes):
                    grad_tensor = torch.empty(
                        recv_grad_shape, device=cur_platform.current_device(), dtype=self.comm_dtype  # FlagScale Add
                    )
                    received_gradients_list.append(grad_tensor)

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                launch_gate = prepare_trace_scope("bridge-p2p-launch")
                batch_specs = (
                    []
                    if _bridge_batch_observation_requested(launch_gate)
                    else None
                )
                for dest_rank, activation_split, grad_tensor in zip(
                    rank_info.send_to_ranks, activation_splits, received_gradients_list
                ):
                    # Send activation
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, activation_split, dest_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-send-forward",
                                "send",
                                "forward",
                                dest_rank,
                                int(activation_split.numel() * activation_split.element_size()),
                                "payload",
                            )
                        )
                    # Receive gradient
                    ops.append(
                        torch.distributed.P2POp(torch.distributed.irecv, grad_tensor, dest_rank)
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-recv-backward",
                                "recv",
                                "backward",
                                dest_rank,
                                int(grad_tensor.numel() * grad_tensor.element_size()),
                                "payload",
                            )
                        )

                logging.debug(
                    f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                    f"executing {len(ops)} simultaneous P2P operations"
                )
                _run_bridge_batch(self, ops, batch_specs, launch_gate)

                # Concatenate received gradients
                aggregated_gradient = torch.cat(received_gradients_list, dim=self._batch_dim)
                logging.debug(
                    f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                    f"agg grad shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
                )
                # Broadcast tensor shape to all ranks in scatter_pg
                tensor_shape_to_broadcast = aggregated_gradient.shape
                shape_tensor = torch.tensor(
                    # FlagScale Begin
                    tensor_shape_to_broadcast,
                    device=cur_platform.current_device(),
                    dtype=torch.int64,
                    # FlagScale End
                )
                _bridge_grid_broadcast(
                    self,
                    shape_tensor,
                    source_rank=self.current_rank,
                    group=self.src_grid_broadcast_pg,
                    pipeline_direction="backward",
                    message_kind="shape",
                    grid_side="src",
                    collective_role="source",
                )

                # Broadcast the tensors to all ranks in the group
                _bridge_grid_broadcast(
                    self,
                    aggregated_gradient,
                    source_rank=self.current_rank,
                    group=self.src_grid_broadcast_pg,
                    pipeline_direction="backward",
                    message_kind="payload",
                    grid_side="src",
                    collective_role="source",
                )

                return aggregated_gradient

        elif (
            rank_info.role == CommRole.MEMBER and self.current_rank in self.src_grid_broadcast_ranks
        ):
            # participate in both gather for gradients
            # Receive gradient from leader using broadcast
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.src_local_leader_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="shape",
                grid_side="src",
                collective_role="participant",
            )

            # Use the received shape to create tensor for broadcast
            received_shape = tuple(shape_tensor.tolist())
            received_gradient = torch.empty(
                received_shape, device=cur_platform.current_device(), dtype=self.comm_dtype  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                received_gradient,
                source_rank=self.src_local_leader_rank,
                group=self.src_grid_broadcast_pg,
                pipeline_direction="backward",
                message_kind="payload",
                grid_side="src",
                collective_role="participant",
            )
            logging.debug(
                f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                f"received gradient from broadcast, shape {received_gradient.shape}"
            )
            return received_gradient

    def send_backward_recv_forward(
        self, grad_tensor: torch.Tensor, forward_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Combined operation: send backward gradient and receive forward activation.

        Args:
            grad_tensor: The gradient tensor to send backward
            forward_shape: Expected forward tensor shape

        Returns:
            torch.Tensor: The received activation tensor
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. "
                "send_backward_recv_forward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.RECEIVER:
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"

            num_receives = len(rank_info.recv_from_ranks)
            gradient_splits = self._split_tensor_at_batch_dim(grad_tensor, num_receives)
            # Communicate shapes for both directions (send backward, receive forward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_prev=gradient_splits[0], recv_prev=True
            )
            logging.debug(
                f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            if num_receives > 0:
                # Prepare activation receive tensors
                received_activations_list = []
                for i, recv_forward_shape in enumerate(recv_forward_shapes):
                    activation_tensor = torch.empty(
                        recv_forward_shape,
                        device=cur_platform.current_device(),  # FlagScale Add
                        dtype=self.comm_dtype,
                        requires_grad=True,
                    )
                    received_activations_list.append(activation_tensor)

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                launch_gate = prepare_trace_scope("bridge-p2p-launch")
                batch_specs = (
                    []
                    if _bridge_batch_observation_requested(launch_gate)
                    else None
                )
                for src_rank, gradient_split, activation_tensor in zip(
                    rank_info.recv_from_ranks, gradient_splits, received_activations_list
                ):
                    # Send gradient
                    ops.append(
                        torch.distributed.P2POp(torch.distributed.isend, gradient_split, src_rank)
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-send-backward",
                                "send",
                                "backward",
                                src_rank,
                                int(gradient_split.numel() * gradient_split.element_size()),
                                "payload",
                            )
                        )

                    # Receive activation
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv, activation_tensor, src_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-recv-forward",
                                "recv",
                                "forward",
                                src_rank,
                                int(activation_tensor.numel() * activation_tensor.element_size()),
                                "payload",
                            )
                        )

                # Execute all operations simultaneously
                logging.debug(
                    f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank} "
                    f"executing {len(ops)} simultaneous P2P operations"
                )
                _run_bridge_batch(self, ops, batch_specs, launch_gate)

                # Concatenate received activations
                aggregated_activation = torch.cat(received_activations_list, dim=self._batch_dim)
                logging.debug(
                    f"[Bridge Communicator] [send_backward_recv_forward] Rank {self.current_rank} "
                    f"agg act shape {aggregated_activation.shape} sum {aggregated_activation.sum()}"
                )

                # Broadcast tensor shape to all ranks in scatter_pg
                tensor_shape_to_scatter = aggregated_activation.shape
                shape_tensor = torch.tensor(
                    tensor_shape_to_scatter, device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
                )
                _bridge_grid_broadcast(
                    self,
                    shape_tensor,
                    source_rank=self.current_rank,
                    group=self.dest_grid_broadcast_pg,
                    pipeline_direction="forward",
                    message_kind="shape",
                    grid_side="dest",
                    collective_role="source",
                )

                # Scatter the tensors to all ranks in the group
                _bridge_grid_broadcast(
                    self,
                    aggregated_activation,
                    source_rank=self.current_rank,
                    group=self.dest_grid_broadcast_pg,
                    pipeline_direction="forward",
                    message_kind="payload",
                    grid_side="dest",
                    collective_role="source",
                )
                return aggregated_activation

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
            )
            _bridge_grid_broadcast(
                self,
                shape_tensor,
                source_rank=self.dest_local_leader_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="shape",
                grid_side="dest",
                collective_role="participant",
            )

            # Use the received shape to create tensor for scatter operation
            received_shape = tuple(shape_tensor.tolist())
            received_activation = torch.empty(
                received_shape,
                device=cur_platform.current_device(),  # FlagScale Add
                dtype=self.comm_dtype,
                requires_grad=True,
            )
            _bridge_grid_broadcast(
                self,
                received_activation,
                source_rank=self.dest_local_leader_rank,
                group=self.dest_grid_broadcast_pg,
                pipeline_direction="forward",
                message_kind="payload",
                grid_side="dest",
                collective_role="participant",
            )
            logging.debug(
                f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank}  "
                f"received activation from scatter operation, shape {received_activation.shape}"
            )
            return received_activation

    def _communicate_shapes(
        self,
        tensor_to_send_next: Optional[torch.Tensor] = None,
        recv_next: bool = False,
        recv_prev: bool = False,
        tensor_to_send_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        """Communicate tensor shapes between sender and receiver ranks in the bridge.

        This is used to communicate tensor shapes before actual tensor communication
        when dealing with variable sequence lengths or dynamic shapes.

        Args:
            tensor_to_send_next: The tensor to send to the next rank (None if not sending)
            tensor_to_send_prev: The tensor to send to the previous rank (None if not sending)
            recv_next: Whether to receive from the next rank (None if not receiving)
            recv_prev: Whether to receive from the previous rank (None if not receiving)

        Returns:
            Tuple containing:
            - List of forward shapes that will be received (empty if not a receiver)
            - List of gradient shapes that will be received (empty if not expecting gradients)
        """
        rank_info = self.comm_map.get(self.current_rank)
        if not rank_info or rank_info.role == CommRole.MEMBER:
            return [], []

        recv_forward_shapes = []
        recv_grad_shapes = []
        logging.debug(
            f"[Bridge Communicator] [communicate_shapes] Rank {self.current_rank} "
            f"is a {rank_info.role} and is running the shape communication"
        )
        # Collect all P2P operations for batch execution
        ops = []
        launch_gate = prepare_trace_scope("bridge-p2p-launch")
        batch_specs = (
            []
            if _bridge_batch_observation_requested(launch_gate)
            else None
        )
        recv_forward_shape_tensors = []
        recv_grad_shape_tensors = []

        if rank_info.role == CommRole.SENDER:
            # Prepare send operations for forward shapes
            if tensor_to_send_next is not None:
                send_shape = tensor_to_send_next.shape
                send_shape_tensor = torch.tensor(
                    send_shape, device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
                )
                # Add send operations for each destination
                for dest_rank in rank_info.send_to_ranks:
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, send_shape_tensor, dest_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-send-forward",
                                "send",
                                "forward",
                                dest_rank,
                                int(send_shape_tensor.numel() * send_shape_tensor.element_size()),
                                "shape",
                            )
                        )

            # If expecting gradients back, prepare receive operations
            if recv_next:
                for dest_rank in rank_info.send_to_ranks:
                    grad_shape_tensor = torch.empty(
                        (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
                    )
                    recv_grad_shape_tensors.append(grad_shape_tensor)
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv, grad_shape_tensor, dest_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-recv-backward",
                                "recv",
                                "backward",
                                dest_rank,
                                int(grad_shape_tensor.numel() * grad_shape_tensor.element_size()),
                                "shape",
                            )
                        )

        elif rank_info.role == CommRole.RECEIVER:
            # Prepare receive operations for forward shapes
            if recv_prev:
                for src_rank in rank_info.recv_from_ranks:
                    forward_shape_tensor = torch.empty(
                        (self.tensor_ndim,), device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
                    )
                    recv_forward_shape_tensors.append(forward_shape_tensor)
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv, forward_shape_tensor, src_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-recv-forward",
                                "recv",
                                "forward",
                                src_rank,
                                int(
                                    forward_shape_tensor.numel()
                                    * forward_shape_tensor.element_size()
                                ),
                                "shape",
                            )
                        )

            # If we need to send gradient shapes back, prepare send operations
            if tensor_to_send_prev is not None:

                grad_shape = tensor_to_send_prev.shape
                grad_shape_tensor = torch.tensor(
                    grad_shape, device=cur_platform.current_device(), dtype=torch.int64  # FlagScale Add
                )

                for src_rank in rank_info.recv_from_ranks:
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, grad_shape_tensor, src_rank
                        )
                    )
                    if batch_specs is not None:
                        batch_specs.append(
                            (
                                "bridge-send-backward",
                                "send",
                                "backward",
                                src_rank,
                                int(grad_shape_tensor.numel() * grad_shape_tensor.element_size()),
                                "shape",
                            )
                        )

        # Execute all operations in a single batch
        if ops:
            _run_bridge_batch(self, ops, batch_specs, launch_gate)

        # Extract shapes from received tensors
        for forward_shape_tensor in recv_forward_shape_tensors:
            shape = forward_shape_tensor.tolist()
            recv_forward_shapes.append(tuple(shape))

        for grad_shape_tensor in recv_grad_shape_tensors:
            shape = grad_shape_tensor.tolist()
            recv_grad_shapes.append(tuple(shape))

        return recv_forward_shapes, recv_grad_shapes

    def _split_tensor_at_batch_dim(
        self, aggregated_tensor: torch.Tensor, num_splits: int
    ) -> List[torch.Tensor]:
        """Split an aggregated tensor into multiple tensors at the batch dimension.

        Args:
            aggregated_tensor: The tensor to split
            num_splits: The number of splits to create

        Returns:
            List of tensors split at the batch dimension
        """
        if num_splits <= 0:
            raise ValueError(f"num_splits must be positive, got {num_splits}")

        splits = torch.tensor_split(aggregated_tensor, num_splits, dim=self._batch_dim)
        # PyTorch p2p requires the tensors to be contiguous
        return [split.contiguous() for split in splits]
