"""Gradient leases must rotate past released slots with unfinished DMA."""
import torch

from megatron.plugin.slideformer.engine import _SharedGPUBufferPool


class PendingCopy:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


def test_two_slots_do_not_wait_on_the_just_released_copy():
    pool = _SharedGPUBufferPool(numel=4, dtype=torch.float32, count=2,
                                device=torch.device("cpu"))
    first, _ = pool.acquire()
    pending_first = PendingCopy()
    pool.release(first, event=pending_first)
    second, _ = pool.acquire()
    assert second != first
    assert pending_first.waits == 0
    pending_second = PendingCopy()
    pool.release(second, event=pending_second)
    third, _ = pool.acquire()
    assert third == first
    assert pending_first.waits == 1
    assert pending_second.waits == 0


def test_one_slot_still_waits_before_reusing_dma_storage():
    pool = _SharedGPUBufferPool(numel=4, dtype=torch.float32, count=1,
                                device=torch.device("cpu"))
    first, _ = pool.acquire()
    pending = PendingCopy()
    pool.release(first, event=pending)
    second, _ = pool.acquire()
    assert second == first
    assert pending.waits == 1


def test_gpu_only_reuse_orders_the_consumer_without_host_wait():
    pool = _SharedGPUBufferPool(numel=4, dtype=torch.float32, count=1,
                               device=torch.device("cpu"))
    first, tensor = pool.acquire()
    pending = PendingCopy()
    pool.release(first, event=pending)
    dependencies = []

    class ConsumerStream:
        def wait_event(self, event):
            dependencies.append(event)

    second, reused = pool.acquire(stream=ConsumerStream())
    assert second == first and reused.data_ptr() == tensor.data_ptr()
    assert dependencies == [pending]  # The old reader still guards the new writer.
    assert pending.waits == 0
