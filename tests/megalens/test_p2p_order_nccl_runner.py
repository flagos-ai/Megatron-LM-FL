from __future__ import annotations

import pytest

from tests.test_utils.runners import run_p2p_order_nccl


def test_p2p_order_nccl_arguments_and_rank_plans() -> None:
    assert run_p2p_order_nccl._parser().parse_args(()).tensor_elements == 8
    assert (
        run_p2p_order_nccl._parser().parse_args(("--tensor-elements", "16")).tensor_elements == 16
    )
    assert run_p2p_order_nccl._expected_api_calls(0) == (
        ("send_next", "isend", "primary"),
        ("recv_next", "irecv", "world"),
    )
    assert run_p2p_order_nccl._expected_api_calls(1) == (
        ("recv_prev", "irecv", "primary"),
        ("send_prev", "isend", "world"),
    )


def test_p2p_order_nccl_rejects_an_invalid_element_count_or_rank() -> None:
    with pytest.raises(SystemExit):
        run_p2p_order_nccl._parser().parse_args(("--tensor-elements", "0"))
    with pytest.raises(ValueError, match="rank 0 or rank 1"):
        run_p2p_order_nccl._expected_api_calls(2)
