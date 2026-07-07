# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.distributed as dist
from types import SimpleNamespace

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from tests.unit_tests.test_utilities import Utils


class TestProcessGroupsConfig:
    """Simple tests for process group dataclasses."""

    def test_transformer_process_groups(self, mocker):
        """Test basic functionality of TransformerProcessGroups."""
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        model_pgs.tp = mock_pg1
        model_pgs.pp = mock_pg2

        # Test accessing attributes
        assert model_pgs.tp == mock_pg1
        assert model_pgs.pp == mock_pg2

        # Test attribute existence
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert not hasattr(model_pgs, 'cp')  # Not set yet

    def test_grad_comm_process_groups(self, mocker):
        """Test basic functionality of ProcessGroupCollection."""
        # Create mock process groups
        mock_pg = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        grad_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        grad_pgs.dp = mock_pg

        # Test accessing attributes
        assert grad_pgs.dp == mock_pg

        # Test attribute existence
        assert hasattr(grad_pgs, 'dp')
        assert not hasattr(grad_pgs, 'dp_cp')  # Not set yet

    def test_hierarchical_context_parallel_groups(self, mocker):
        """Test setting and accessing the hierarchical context parallel list."""
        # Create mock process groups
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Set the hierarchical context parallel groups
        model_pgs.hcp = [mock_pg1, mock_pg2]

        # Test list access
        assert isinstance(model_pgs.hcp, list)
        assert len(model_pgs.hcp) == 2
        assert model_pgs.hcp[0] == mock_pg1
        assert model_pgs.hcp[1] == mock_pg2

    def test_repr(self, mocker):
        """Test __repr__ shows active process groups and their sizes."""
        tp_size = 4
        pp_size = 2
        mock_tp = mocker.Mock(spec=dist.ProcessGroup)
        mock_tp.size.return_value = tp_size
        mock_pp = mocker.Mock(spec=dist.ProcessGroup)
        mock_pp.size.return_value = pp_size

        # Test empty collection
        empty_pgs = ProcessGroupCollection()
        assert repr(empty_pgs) == "ProcessGroupCollection(empty)"

        # Test collection with process groups
        model_pgs = ProcessGroupCollection()
        model_pgs.tp = mock_tp
        model_pgs.pp = mock_pp

        repr_str = repr(model_pgs)
        assert "ProcessGroupCollection(" in repr_str
        assert f"tp({tp_size})" in repr_str
        assert f"pp({pp_size})" in repr_str


class TestPGConfigDefaultInitialization:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_initialization(self):
        """Test default initialization of ProcessGroupCollection."""
        # Create instance
        model_pgs = ProcessGroupCollection.use_mpu_process_groups()

        # Test that instance was created successfully
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'dp')
        assert hasattr(model_pgs, 'dp_cp')

        # Test that only required process groups were initialized
        model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'cp'])
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'cp')
        assert not hasattr(model_pgs, 'dp')

        # Test that an error is raised if an invalid process group is requested
        with pytest.raises(ValueError, match=r"Invalid process groups requested"):
            model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'foo'])


def test_process_group_collection_setup_optimizer_and_ddp_custom_paths(monkeypatch):
    class _Group:
        def __init__(self, name, size=1):
            self.name = name
            self._size = size

        def size(self):
            return self._size

        def __repr__(self):
            return f"Group({self.name})"

    created_groups = []
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(
        dist,
        "new_group",
        lambda ranks: created_groups.append(tuple(ranks)) or _Group(f"new-{ranks[0]}"),
    )

    model = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1),
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=1,
            use_distributed_optimizer=False,
        )
    )
    pg_collection = ProcessGroupCollection(
        dp=_Group("dp"),
        expt_dp=None,
        mp=_Group("mp"),
        tp=_Group("tp"),
        pp=_Group("pp"),
        ep=_Group("ep"),
        tp_ep_pp=_Group("tp_ep_pp"),
    )

    optimizer_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection,
        [model],
        use_gloo_process_groups=False,
    )
    assert optimizer_groups["dp_cp_group"] is pg_collection.dp
    assert optimizer_groups["intra_dp_cp_group"] is pg_collection.dp
    assert optimizer_groups["inter_dist_opt_group"] is None
    assert optimizer_groups["engram_dp_group"] is None

    ddp_groups = ProcessGroupCollection.setup_process_groups_for_ddp(
        pg_collection,
        SimpleNamespace(context_parallel_size=1),
        model.ddp_config,
    )
    assert ddp_groups["dp_cp_group"] is pg_collection.dp
    assert ddp_groups["expt_dp_group"].name == "new-3"
    assert created_groups == [(3,)]
    assert ddp_groups["tp_group"] is pg_collection.tp

    with pytest.raises(ValueError, match="Gloo process groups"):
        ProcessGroupCollection.setup_process_groups_for_optimizer(pg_collection, [model])

    with pytest.raises(ValueError, match="dp process group"):
        ProcessGroupCollection.setup_process_groups_for_optimizer(
            ProcessGroupCollection(expt_dp=None, mp=None, tp_ep_pp=None),
            [model],
            use_gloo_process_groups=False,
        )

    with pytest.raises(ValueError, match="dp_cp process group"):
        ProcessGroupCollection.setup_process_groups_for_ddp(
            ProcessGroupCollection(dp=_Group("dp"), expt_dp=None, tp=_Group("tp"), pp=_Group("pp"), ep=_Group("ep")),
            SimpleNamespace(context_parallel_size=2),
            model.ddp_config,
        )

    multi_instance_model = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1),
        ddp_config=SimpleNamespace(
            num_distributed_optimizer_instances=2,
            use_distributed_optimizer=True,
        )
    )
    complete_pg = ProcessGroupCollection(
        dp=_Group("dp"),
        dp_cp=_Group("dp_cp"),
        expt_dp=_Group("expt_dp"),
        intra_dp_cp=_Group("intra_dp_cp"),
        intra_expt_dp=_Group("intra_expt_dp"),
        inter_dist_opt=_Group("inter"),
        intra_dist_opt=_Group("intra"),
        mp=_Group("mp"),
        tp=_Group("tp"),
        pp=_Group("pp"),
        ep=_Group("ep"),
        tp_ep_pp=_Group("tp_ep_pp"),
        engram_dp=_Group("engram_dp"),
        engram_embed=_Group("engram_embed"),
        engram_mp=_Group("engram_mp"),
    )
    multi_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        complete_pg,
        [multi_instance_model],
        use_gloo_process_groups=False,
    )
    assert multi_groups["intra_dist_opt_group"] is complete_pg.intra_dist_opt
    assert multi_groups["inter_dist_opt_group"] is complete_pg.inter_dist_opt


def test_multi_module_process_group_collection_paths():
    class _Group:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    encoder = ProcessGroupCollection(cp=_Group(1))
    llm = ProcessGroupCollection(cp=_Group(4))
    collection = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": encoder, "llm": llm},
        language_model_module_name="llm",
    )

    assert collection.has_language_model() is True
    assert collection.get_language_model_collection() is llm
    assert collection.get_language_model_cp_size() == 4
    assert collection.get_module_collection("encoder") is encoder
    assert collection["llm"] is llm
    assert len(collection) == 2
    assert list(collection.keys()) == ["encoder", "llm"]
    assert list(collection.values()) == [encoder, llm]
    assert list(collection.items()) == [("encoder", encoder), ("llm", llm)]
    assert list(iter(collection)) == [encoder, llm]
    assert "language_model_module_name='llm'" in repr(collection)

    with pytest.raises(ValueError, match="cannot be empty"):
        MultiModuleProcessGroupCollection(module_pgs={})
    with pytest.raises(ValueError, match="not found"):
        MultiModuleProcessGroupCollection(module_pgs={"encoder": encoder}, language_model_module_name="llm")
    no_llm = MultiModuleProcessGroupCollection(module_pgs={"encoder": encoder})
    assert no_llm.has_language_model() is False
    with pytest.raises(ValueError, match="No language model"):
        no_llm.get_language_model_collection()
    with pytest.raises(ValueError, match="Module 'missing'"):
        collection.get_module_collection("missing")
