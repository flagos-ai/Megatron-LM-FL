# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils, get_current_device, get_device_str


@pytest.mark.parametrize("foreach", [None, True])
def test_mixed_mesh_adam_updates_and_state_restore(foreach):
    """Uneven TP shards must not break Adam's DP-only parameters or checkpoint groups."""
    Utils.initialize_distributed()
    if torch.distributed.get_world_size() != 8:
        pytest.skip("Requires eight ranks for uneven DP=4, TP=2 shards")
    mesh = init_device_mesh(get_device_str(), (4, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh["dp"]
    device = get_current_device()
    dp_rank = dp_mesh.get_local_rank()
    groups = [mesh.get_group("dp"), mesh.get_group("tp")]

    def parameter(local_size, global_size, parameter_mesh, placements):
        value = DTensor.from_local(
            torch.ones(local_size, device=device),
            device_mesh=parameter_mesh,
            placements=placements,
            run_check=False,
            shape=torch.Size([global_size]),
            stride=(1,),
        )
        param = torch.nn.Parameter(value)
        # Exercise optimizer integration independently of model gradient buffers.
        param._megatron_fsdp_model = model_reference
        return param

    model_reference = object()
    try:
        params = [
            parameter(0 if dp_rank == 3 else 1, 6, mesh, (Shard(0), Shard(0))),
            parameter(2, 8, dp_mesh, (Shard(0),)),
            parameter(2, 16, mesh, (Shard(0), Shard(0))),
            parameter(2, 8, dp_mesh, (Shard(0),)),
        ]
        reference_params = [torch.nn.Parameter(p.to_local().detach().clone()) for p in params]
        optimizer = torch.optim.Adam(
            [{"params": params[:3], "lr": 0.01}, {"params": params[3:], "lr": 0.025}],
            foreach=foreach,
        )
        reference = torch.optim.Adam(
            [
                {"params": reference_params[:3], "lr": 0.01},
                {"params": reference_params[3:], "lr": 0.025},
            ],
            foreach=False,
        )
        original_group_ids = [id(group) for group in optimizer.param_groups]
        original_param_ids = [[id(p) for p in group["params"]] for group in optimizer.param_groups]
        original_specs = [(p.device_mesh, p.placements) for p in params]
        fully_shard_optimizer(optimizer, preproc_state_dict_for_dcp_ckpt=False)
        assert [id(group) for group in optimizer.param_groups] == original_group_ids
        assert optimizer.param_groups[1]["foreach"] is foreach
        expected_foreach = foreach if is_torch_min_version("2.7.0") else False
        assert optimizer.param_groups[0]["foreach"] is expected_foreach

        # Match FSDP's zero-gradient step that initializes optimizer state.
        for param in reference_params:
            if param.numel():
                param.grad = torch.zeros_like(param)
        reference.step()
        reference.zero_grad()

        for step in range(3):
            if step == 2:
                checkpoint = deepcopy(optimizer.state_dict())
                # Simulate a checkpoint written before the compatibility fallback.
                checkpoint["param_groups"][0]["foreach"] = foreach
                optimizer.state.clear()
                optimizer.load_state_dict(checkpoint)
                assert optimizer.param_groups[0]["foreach"] is expected_foreach
            for param, reference_param in zip(params, reference_params):
                if reference_param.numel():
                    param.grad = torch.full_like(param, float(step + 1))
                    reference_param.grad = torch.full_like(reference_param, float(step + 1))
            optimizer.step(
                sync_grad_before_optimizer_step=False, install_optimized_model_weights=False
            )
            reference.step()
            optimizer.zero_grad(zero_grad_buffer=False)
            reference.zero_grad()
            for param, reference_param in zip(params, reference_params):
                torch.testing.assert_close(param.to_local(), reference_param)
            assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.025]
            assert [
                [id(p) for p in group["params"]] for group in optimizer.param_groups
            ] == original_param_ids
            assert [(p.device_mesh, p.placements) for p in params] == original_specs
    finally:
        for group in groups:
            torch.distributed.destroy_process_group(group)
