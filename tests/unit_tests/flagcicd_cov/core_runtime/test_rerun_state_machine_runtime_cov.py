# Copyright (c) 2026, BAAI. All rights reserved.
"""RerunStateMachine driven through its full state graph on a single process."""

import os
import re
from types import SimpleNamespace

import pytest
import torch

from megatron.core import rerun_state_machine
from megatron.core.rerun_state_machine import (
    EXIT_CODE_FAILED_ON_RESULT_VALIDATION,
    EXIT_CODE_RESUME_TO_DISAMBIGUATE,
    Caller,
    RerunDataIterator,
    RerunMode,
    RerunState,
    RerunStateMachine,
)


@pytest.fixture
def gpu_id(monkeypatch):
    """Expose a mutable current device so a resumed machine can land on another GPU."""
    holder = SimpleNamespace(current='gpu0')
    monkeypatch.setattr(
        rerun_state_machine,
        'cur_platform',
        SimpleNamespace(
            device=lambda value: torch.device('cpu'),
            current_device=lambda: holder.current,
            get_rng_state=torch.get_rng_state,
            set_rng_state=lambda state: None,
        ),
    )
    return holder


@pytest.fixture
def single_rank(monkeypatch):
    """Run as a plain single process: no collectives and no torchrun/SLURM rank environment."""
    distributed = rerun_state_machine.torch.distributed
    monkeypatch.setattr(distributed, 'all_reduce', lambda tensor: None)
    monkeypatch.setattr(distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(distributed, 'gather_object', lambda obj, dst=None: None)
    for name in ('RANK', 'SLURM_PROCID', 'SLURM_NTASKS'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_machine(gpu_id, single_rank):
    """Build a machine through the module's singleton API so data iterators observe it."""

    def build(**kwargs):
        rerun_state_machine.destroy_rerun_state_machine()
        rerun_state_machine.initialize_rerun_state_machine(**kwargs)
        return rerun_state_machine.get_rerun_state_machine()

    yield build
    rerun_state_machine.destroy_rerun_state_machine()


def _run_step(machine, iterator, outcomes, message='loss', **validate_kwargs):
    """Run one training step: one (result, rejected) pair per forward-backward pass."""
    pending = list(outcomes)
    batches = []
    while machine.should_run_forward_backward(iterator):
        result, rejected = pending.pop(0)
        batches.append(next(iterator))
        machine.validate_result(result, lambda value: rejected, message, **validate_kwargs)
    return machine.should_checkpoint_and_exit(), batches


def _run_warmup_step(machine, iterator):
    """Run the first iteration, which is never validated because no checkpoint exists yet."""
    exit_flags, batches = _run_step(machine, iterator, [(torch.tensor(1.0), False)])
    assert exit_flags == (False, False, 0)
    assert batches == [0]


def _checkpoint_after_first_rerun(make_machine, tmp_path, iterator):
    """Drive a machine to WILL_RERUN_FROM_CHECKPOINT and return its unwrapped checkpoint."""
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS,
        result_rejected_tracker_filename=str(tmp_path / 'tracker.log'),
    )
    _run_warmup_step(machine, iterator)
    exit_flags, _ = _run_step(
        machine, iterator, [(torch.tensor(2.0), True), (torch.tensor(2.0), False)]
    )
    assert exit_flags == (True, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE)
    checkpoint = machine.state_dict(iterator, 'torch_dist')
    checkpoint['sharded'] = checkpoint['sharded'].data
    return checkpoint


def test_disabled_machine_runs_one_pass_per_iteration(make_machine):
    machine = make_machine()
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.state == RerunState.INITIAL_RUN
    assert machine.current_iteration == 0
    assert machine.should_run_forward_backward(iterator) is False
    assert machine.state == RerunState.NOT_RUNNING_YET


def test_enabled_machine_without_rejection_moves_to_the_next_iteration(make_machine):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.state == RerunState.INITIAL_RUN
    assert machine.injected_result is None
    assert machine.should_run_forward_backward(iterator) is False
    assert machine.state == RerunState.NOT_RUNNING_YET


def test_validate_result_raises_when_reruns_are_disabled_and_result_is_fatal(
    make_machine, tmp_path
):
    machine = make_machine(result_rejected_tracker_filename=str(tmp_path / 'tracker.log'))
    with pytest.raises(RuntimeError, match="Unexpected result .+ \\(message='loss is NaN'\\)"):
        machine.validate_result(torch.tensor(float('nan')), torch.isnan, 'loss is NaN', fatal=True)


def test_validate_result_only_logs_when_reruns_are_disabled_and_result_is_not_fatal(
    make_machine, tmp_path
):
    tracker = tmp_path / 'tracker.log'
    machine = make_machine(result_rejected_tracker_filename=str(tracker))
    machine.validate_result(torch.tensor(float('nan')), torch.isnan, 'loss is NaN', fatal=False)
    machine.validate_result(torch.tensor(1.0), torch.isnan, 'loss is NaN', fatal=True)
    contents = tracker.read_text()
    assert contents.count('status=RerunValidationStatus.RERUN_DISABLED') == 1


def test_non_reproducible_rerun_requests_exit_when_validation_is_fatal(make_machine, tmp_path):
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS,
        result_rejected_tracker_filename=str(tmp_path / 'tracker.log'),
    )
    iterator = RerunDataIterator(iter(range(8)))
    _run_warmup_step(machine, iterator)
    exit_flags, batches = _run_step(
        machine, iterator, [(torch.tensor(1.0), True), (torch.tensor(2.0), False)]
    )
    assert batches == [1, 1]
    assert exit_flags == (False, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION)
    assert machine.state == RerunState.RERUNNING_IN_PLACE
    tracker = (tmp_path / 'tracker.log').read_text()
    assert 'status=RerunValidationStatus.FIRST_RERUN_NOT_REPRODUCIBLE' in tracker


def test_non_reproducible_rerun_continues_when_validation_is_not_fatal(make_machine):
    restored = []
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS,
        state_save_func=lambda: {'extra': 7},
        state_restore_func=restored.append,
    )
    iterator = RerunDataIterator(iter(range(8)))
    _run_warmup_step(machine, iterator)
    exit_flags, _ = _run_step(
        machine, iterator, [(torch.tensor(1.0), True), (torch.tensor(2.0), False)], fatal=False
    )
    assert restored == [{'extra': 7}]
    assert exit_flags == (False, False, 0)
    assert machine.state == RerunState.NOT_RUNNING_YET
    assert machine.continue_requested


def test_reproducible_rerun_requests_a_checkpoint_and_exits(make_machine, gpu_id, tmp_path):
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS,
        result_rejected_tracker_filename=str(tmp_path / 'tracker.log'),
    )
    iterator = RerunDataIterator(iter(range(8)))
    _run_warmup_step(machine, iterator)
    exit_flags, _ = _run_step(
        machine, iterator, [(torch.tensor(2.0), True), (torch.tensor(2.0), False)]
    )
    assert exit_flags == (True, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE)
    assert machine.state == RerunState.WILL_RERUN_FROM_CHECKPOINT
    assert machine.suspicious_node == os.uname()[1]
    assert machine.suspicious_device == 'gpu0'
    tracker = (tmp_path / 'tracker.log').read_text()
    assert 'status=RerunValidationStatus.INITIAL_RUN' in tracker
    assert 'status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE' in tracker


def test_reproducible_rerun_continues_when_validation_is_not_fatal(make_machine, gpu_id):
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    iterator = RerunDataIterator(iter(range(8)))
    _run_warmup_step(machine, iterator)
    exit_flags, _ = _run_step(
        machine, iterator, [(torch.tensor(2.0), True), (torch.tensor(2.0), False)], fatal=False
    )
    assert exit_flags == (False, False, 0)
    assert machine.state == RerunState.NOT_RUNNING_YET
    assert machine.continue_requested
    assert machine.suspicious_node == os.uname()[1]
    assert machine.suspicious_device == 'gpu0'


def test_resuming_on_the_same_gpu_requests_another_restart(make_machine, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    resumed = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    resumed.load_state_dict(checkpoint)
    assert resumed.state == RerunState.WILL_RERUN_FROM_CHECKPOINT
    assert resumed.checkpoint_requested
    assert resumed.initial_result == torch.tensor(2.0)
    assert resumed.data_iterator_checkpoints == [
        {'saved_microbatches': [1], 'replaying': True, 'replay_pos': 0}
    ]
    exit_flags, batches = _run_step(resumed, iterator, [(torch.tensor(2.0), False)])
    assert batches == [1]
    assert exit_flags == (False, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE)
    assert resumed.state == RerunState.RERUNNING_AGAIN_FROM_CHECKPOINT


def test_resuming_on_another_gpu_with_a_mismatch_continues(make_machine, gpu_id, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    tracker = tmp_path / 'tracker.log'
    gpu_id.current = 'gpu7'
    resumed = make_machine(
        mode=RerunMode.VALIDATE_RESULTS, result_rejected_tracker_filename=str(tracker)
    )
    resumed.load_state_dict(checkpoint)
    exit_flags, batches = _run_step(resumed, iterator, [(torch.tensor(4.0), False)], fatal=False)
    assert batches == [1]
    assert exit_flags == (False, False, 0)
    assert resumed.state == RerunState.NOT_RUNNING_YET
    assert 'status=RerunValidationStatus.SECOND_RERUN_NOT_REPRODUCIBLE' in tracker.read_text()
    exit_flags, batches = _run_step(resumed, iterator, [(torch.tensor(5.0), False)])
    assert batches == [2]
    assert exit_flags == (False, False, 0)
    assert resumed.data_iterator_checkpoints is None


def test_resuming_on_another_gpu_with_a_mismatch_exits_when_fatal(make_machine, gpu_id, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    gpu_id.current = 'gpu7'
    resumed = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    resumed.load_state_dict(checkpoint)
    exit_flags, _ = _run_step(resumed, iterator, [(torch.tensor(4.0), False)])
    assert exit_flags == (False, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION)
    assert resumed.state == RerunState.RERUNNING_FROM_CHECKPOINT


def test_resuming_on_another_gpu_with_a_match_continues(make_machine, gpu_id, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    tracker = tmp_path / 'tracker.log'
    gpu_id.current = 'gpu7'
    resumed = make_machine(
        mode=RerunMode.VALIDATE_RESULTS, result_rejected_tracker_filename=str(tracker)
    )
    resumed.load_state_dict(checkpoint)
    exit_flags, _ = _run_step(resumed, iterator, [(torch.tensor(2.0), False)], fatal=False)
    assert exit_flags == (False, False, 0)
    assert resumed.state == RerunState.NOT_RUNNING_YET
    assert 'status=RerunValidationStatus.SECOND_RERUN_REPRODUCIBLE' in tracker.read_text()


def test_report_mode_records_relative_differences_between_reruns(make_machine):
    machine = make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
    iterator = RerunDataIterator(iter(range(4)))
    assert machine.should_run_forward_backward(iterator) is True
    assert machine.should_run_forward_backward(iterator) is False
    assert machine.should_checkpoint_and_exit() == (False, False, 0)
    exit_flags, _ = _run_step(
        machine, iterator, [(torch.tensor(3.0), False), (torch.tensor(6.0), False)], message='loss'
    )
    assert exit_flags == (False, False, 0)
    assert machine.state == RerunState.NOT_RUNNING_YET
    stats = machine.stats[Caller('loss', 0)]
    assert stats.samples == [pytest.approx(2 / 3)]
    assert stats.zero_cnt == 0


def test_validate_state_dict_accepts_only_pending_reruns(make_machine, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    machine = make_machine(mode=RerunMode.VALIDATE_RESULTS)
    finished = dict(checkpoint, state=RerunState.NOT_RUNNING_YET)
    without_state = {name: value for name, value in checkpoint.items() if name != 'state'}
    assert machine.validate_state_dict(checkpoint) is True
    assert machine.validate_state_dict(finished) is False
    assert machine.validate_state_dict(without_state) is False
    assert machine.validate_state_dict(None) is False


def test_load_state_dict_is_ignored_when_reruns_are_disabled(make_machine, tmp_path):
    iterator = RerunDataIterator(iter(range(8)))
    checkpoint = _checkpoint_after_first_rerun(make_machine, tmp_path, iterator)
    machine = make_machine()
    machine.load_state_dict(checkpoint)
    assert machine.mode == RerunMode.DISABLED
    assert machine.state == RerunState.NOT_RUNNING_YET
    assert machine.checkpoint_requested is False


def test_validation_errors_are_appended_to_the_tracker_file(make_machine, tmp_path):
    tracker = tmp_path / 'tracker.log'
    machine = make_machine(
        mode=RerunMode.VALIDATE_RESULTS, result_rejected_tracker_filename=str(tracker)
    )
    iterator = RerunDataIterator(iter(range(8)))
    _run_warmup_step(machine, iterator)
    _run_step(machine, iterator, [(torch.tensor(2.0), True), (torch.tensor(2.0), False)])
    record = re.compile(
        r"ts=\S+ \S+ node=\S+ device=gpu0 jobID=\S+ rank=0 iteration=2 "
        r"status=RerunValidationStatus\.(\w+) result=2\.0 message='loss'"
    )
    lines = tracker.read_text().splitlines()
    assert len(lines) == 2
    matches = [record.fullmatch(line) for line in lines]
    assert [match.group(1) if match else '' for match in matches] == [
        'INITIAL_RUN',
        'FIRST_RERUN_REPRODUCIBLE',
    ]
