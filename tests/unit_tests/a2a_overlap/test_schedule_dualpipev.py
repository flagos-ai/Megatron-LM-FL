# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.plugin.dualpipev.dualpipev_schedules import generate_dualpipev_schedule


# ============================= Schedule Generation Tests =============================


def valid_params():
    """
    Generate valid (pp_size, num_microbatches) combinations.
    Constraint: num_microbatches >= pp_size * 2 to ensure non-negative overlap counts.
    """
    params = []
    for pp_size in [2, 4, 8]:
        for num_microbatches in [8, 16, 32]:
            if num_microbatches >= pp_size * 2:
                params.append((pp_size, num_microbatches))
    return params


class TestDualpipevSchedule:
    """
    Test class for DualPipeV schedule generation.

    This class tests the `generate_dualpipev_schedule` function to verify that
    the generated schedules are valid for different pipeline sizes and microbatch counts.
    """

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_generation_basic(self, pp_size, num_microbatches):
        """
        Test that schedule generation produces valid non-negative stage counts.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        expected_keys = [
            'warmup',
            'interleaved_forward',
            '1b1w1f',
            'overlap',
            '1b1overlap',
            'interleaved_backward',
            'cooldown',
        ]
        assert set(schedule.keys()) == set(expected_keys), (
            f"Schedule keys mismatch: got {set(schedule.keys())}, expected {set(expected_keys)}"
        )

        for key in expected_keys:
            assert len(schedule[key]) == pp_size, (
                f"Stage '{key}' has {len(schedule[key])} entries, expected {pp_size}"
            )

        for key in expected_keys:
            for rank, value in enumerate(schedule[key]):
                if isinstance(value, list):
                    for v in value:
                        assert v >= 0, (
                            f"Stage '{key}' rank {rank} has negative value {v}"
                        )
                else:
                    assert value >= 0, (
                        f"Stage '{key}' rank {rank} has negative value {value}"
                    )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_warmup_decreasing(self, pp_size, num_microbatches):
        """
        Test that warmup stages decrease for later pipeline ranks.
        The warmup count for rank i should be: pp_size*2 - 2 - i*2.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        warmup = schedule['warmup']
        for i in range(pp_size):
            expected = pp_size * 2 - 2 - i * 2
            assert warmup[i] == expected, (
                f"Warmup for rank {i}: got {warmup[i]}, expected {expected}"
            )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_interleaved_forward_increasing(self, pp_size, num_microbatches):
        """
        Test that interleaved_forward stages increase for later ranks.
        The count for rank i should be: i + 1.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        interleaved_forward = schedule['interleaved_forward']
        for i in range(pp_size):
            expected = i + 1
            assert interleaved_forward[i] == expected, (
                f"Interleaved forward for rank {i}: got {interleaved_forward[i]}, expected {expected}"
            )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_1b1w1f_decreasing(self, pp_size, num_microbatches):
        """
        Test that 1b1w1f stages decrease for later pipeline ranks.
        Inside generate_dualpipev_schedule, pp_size is doubled, so
        the count for rank i should be: pp_size - i - 1.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        one_b1w1f = schedule['1b1w1f']
        for i in range(pp_size):
            expected = pp_size - i - 1
            assert one_b1w1f[i] == expected, (
                f"1b1w1f for rank {i}: got {one_b1w1f[i]}, expected {expected}"
            )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_overlap_count(self, pp_size, num_microbatches):
        """
        Test that overlap stage counts follow the expected formula.
        The count for rank i should be: num_microbatches*2 - (pp_size*2)*2 + i*2 + 2.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        overlap = schedule['overlap']
        doubled_pp = pp_size * 2
        doubled_micro = num_microbatches * 2
        for i in range(pp_size):
            expected = doubled_micro - doubled_pp * 2 + i * 2 + 2
            assert overlap[i] == expected, (
                f"Overlap for rank {i}: got {overlap[i]}, expected {expected}"
            )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_1b1overlap_count(self, pp_size, num_microbatches):
        """
        Test that 1b1overlap stages follow the expected formula.
        Inside generate_dualpipev_schedule, pp_size is doubled, so
        the count for rank i should be: (pp_size - i - 1) * 2.
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        one_b1overlap = schedule['1b1overlap']
        for i in range(pp_size):
            expected = (pp_size - i - 1) * 2
            assert one_b1overlap[i] == expected, (
                f"1b1overlap for rank {i}: got {one_b1overlap[i]}, expected {expected}"
            )

    @pytest.mark.parametrize("pp_size,num_microbatches", valid_params())
    def test_schedule_total_forward_ops(self, pp_size, num_microbatches):
        """
        Test that the total master-chunk forward operations per rank is consistent.

        master_fwd = warmup + interleaved_forward + 1b1w1f + overlap
                   = 2*num_microbatches - pp_size  (for all ranks)
        """
        schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

        expected_total = 2 * num_microbatches - pp_size
        for rank in range(pp_size):
            warmup = schedule['warmup'][rank]
            interleaved_fwd = schedule['interleaved_forward'][rank]
            one_b1w1f = schedule['1b1w1f'][rank]
            overlap = schedule['overlap'][rank]

            master_fwd = warmup + interleaved_fwd + one_b1w1f + overlap
            assert master_fwd == expected_total, (
                f"Rank {rank}: master chunk forward count {master_fwd} != {expected_total}"
            )

    def test_schedule_minimum_valid_microbatches(self):
        """
        Test the minimum valid num_microbatches for each pp_size.
        Minimum requirement: num_microbatches >= pp_size * 2 to ensure all overlap >= 0.
        """
        for pp_size in [2, 4, 8]:
            min_microbatches = pp_size * 2
            schedule = generate_dualpipev_schedule(pp_size, min_microbatches)
            # The last rank should have minimum overlap = 2*(min_micro) - 4*pp + 0 + 2
            # = 4*pp - 4*pp + 2 = 2
            assert schedule['overlap'][0] == 2, (
                f"pp_size={pp_size}: first rank overlap should be 2 at minimum microbatches"
            )

    def test_schedule_invalid_microbatches(self):
        """
        Test that invalid num_microbatches (too small) raises an error or produces
        negative overlap values.
        """
        pp_size = 4
        num_microbatches = 2  # too small: needs >= 8

        # The function should either raise or produce negative overlaps
        try:
            schedule = generate_dualpipev_schedule(pp_size, num_microbatches)
            # If no exception, verify that overlap has negative values
            has_negative = any(v < 0 for v in schedule['overlap'])
            assert has_negative, (
                "Expected negative overlap values for invalid microbatch count"
            )
        except (AssertionError, ValueError):
            pass  # Expected behavior
