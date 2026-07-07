import torch

from megatron.core.tensor_parallel.data import broadcast_data
from tests.unit_tests.test_utilities import Utils


def test_broadcast_data():
    Utils.initialize_model_parallel(2, 4)
    try:
        dtype = torch.float32
        input_data = {
            0: torch.zeros((8, 8), dtype=dtype),
            1: torch.ones((8, 8), dtype=dtype),
        }
        actual_output = broadcast_data([0, 1], input_data, dtype)
        assert torch.equal(actual_output[0].cpu(), input_data[0])
        assert torch.equal(actual_output[1].cpu(), input_data[1])
    finally:
        Utils.destroy_model_parallel()
