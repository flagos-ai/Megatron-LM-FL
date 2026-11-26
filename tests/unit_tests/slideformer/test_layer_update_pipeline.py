"""Scheduling and error boundaries of the bounded native update adapter."""
from types import SimpleNamespace
import threading

import pytest
import torch

from megatron.plugin.slideformer.layer_update_pipeline import GradientLease, OrderedLayerWork, communication_device


def test_gradient_lease_cannot_release_before_copy_completion():
    copied, released, entered = threading.Event(), [], threading.Event()
    def wait():
        entered.set()
        assert copied.wait(5)
    lease = GradientLease({}, SimpleNamespace(synchronize=wait),
                          SimpleNamespace(release=lambda slot: released.append(slot)), 7)
    thread = threading.Thread(target=lease.release)
    thread.start()
    assert entered.wait(5)
    assert not released
    copied.set()
    thread.join(5)
    assert not thread.is_alive() and released == [7]
    lease.release()
    assert released == [7]


def test_gradient_lease_holds_source_storage_until_consumed():
    original = torch.tensor([1., 2.])
    sources = {"p": original}
    lease = GradientLease(sources.copy(), None, SimpleNamespace(release=lambda slot: original.fill_(99)), 0)
    sources.clear()
    result = lease.gradients["p"].float().clone()
    lease.release()
    assert result.tolist() == [1., 2.]
    assert original.tolist() == [99., 99.]


@pytest.mark.parametrize("parallel", ["tensor_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"])
def test_native_model_parallelism_cannot_issue_background_cuda_collectives(parallel):
    config = SimpleNamespace(**{parallel: 2})
    assert communication_device(torch.device("cuda", 1), config) == torch.device("cpu")


def test_pure_dp_keeps_gpu_communication_and_cpu_runs_remain_cpu():
    config = SimpleNamespace()
    assert communication_device(torch.device("cuda", 1), config) == torch.device("cuda", 1)
    assert communication_device(torch.device("cpu"), config) == torch.device("cpu")


def test_out_of_order_callbacks_keep_collective_order_and_reset():
    owners = [SimpleNamespace(layer_idx=i) for i in (2, 1, 0)]
    seen = []
    work = OrderedLayerWork(owners, lambda owner: seen.append(owner.layer_idx), 2)
    try:
        work.submit_ready(owners[2])
        assert not seen
        work.submit_ready(owners[0])
        work.submit_ready(owners[1])
        work.finish()
        assert seen == [2, 1, 0]
        with pytest.raises(RuntimeError, match="Duplicate"):
            work.submit_ready(owners[0])
        work.reset()
        for owner in owners:
            work.submit_ready(owner)
        work.finish()
        assert seen == [2, 1, 0, 2, 1, 0]
    finally:
        work.close()


def test_backpressure_does_not_recycle_an_active_slot():
    owners = [SimpleNamespace(layer_idx=i) for i in (1, 0)]
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
    def operation(owner):
        if owner.layer_idx == 1:
            entered.set()
            assert release.wait(5)
    work = OrderedLayerWork(owners, operation, 1)
    try:
        work.submit_ready(owners[0])
        assert entered.wait(5)
        def submit_second():
            work.submit_ready(owners[1])
            submitted.set()
        thread = threading.Thread(target=submit_second)
        thread.start()
        assert not submitted.wait(.05)
        release.set()
        thread.join(5)
        assert not thread.is_alive() and submitted.is_set()
        work.finish()
    finally:
        release.set()
        work.close()


def test_worker_failure_is_visible_and_does_not_strand_shutdown():
    def failed(owner):
        raise RuntimeError("injected reduction failure")
    work = OrderedLayerWork([SimpleNamespace(layer_idx=0)], failed, 1)
    work.submit_ready(work.owners[0])
    with pytest.raises(RuntimeError, match="injected reduction failure"):
        work.finish()
    with pytest.raises(RuntimeError, match="injected reduction failure"):
        work.close()


def test_missing_backward_is_not_a_successful_step():
    work = OrderedLayerWork([SimpleNamespace(layer_idx=0)], lambda owner: None, 1)
    try:
        with pytest.raises(RuntimeError, match="Missing final backward"):
            work.finish()
    finally:
        work.close()


def test_queued_updates_do_not_execute_after_a_prior_failure():
    entered, release = threading.Event(), threading.Event()
    seen = []
    def operation(owner):
        seen.append(owner.layer_idx)
        if owner.layer_idx == 1:
            entered.set()
            assert release.wait(5)
            raise RuntimeError("first update failed")
    owners = [SimpleNamespace(layer_idx=i) for i in (1, 0)]
    work = OrderedLayerWork(owners, operation, 2)
    work.submit_ready(owners[0])
    assert entered.wait(5)
    work.submit_ready(owners[1])
    release.set()
    with pytest.raises(RuntimeError, match="first update failed"):
        work.close()
    assert seen == [1]
