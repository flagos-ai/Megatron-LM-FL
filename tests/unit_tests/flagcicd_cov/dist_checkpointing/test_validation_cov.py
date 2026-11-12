# Copyright (c) 2026, BAAI. All rights reserved.

"""Checkpoint validation: strict-flag parsing, sharding-integrity errors for overlapping or missing shards, common-state consistency across ranks, and integrity manifest verification."""

import json
import logging
import shutil

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.validation import (
    StrictHandling,
    _validate_common_state_dict,
    parse_strict_flag,
    save_integrity_manifest,
    validate_integrity_and_strict_load,
    validate_sharding_integrity,
    verify_integrity_manifest,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def shared_ckpt_root(tmp_path_factory):
    root = tmp_path_factory.getbasetemp().parent / 'flagcicd_dist_ckpt'
    if Utils.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    yield root
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if Utils.rank == 0:
        shutil.rmtree(root, ignore_errors=True)


def test_parse_strict_flag_raises_value_error_for_unknown_string():
    with pytest.raises(ValueError, match='Invalid strict flag'):
        parse_strict_flag('bogus')


def test_parse_strict_flag_accepts_enum_instance_and_valid_string():
    assert parse_strict_flag(StrictHandling.LOG_ALL) is StrictHandling.LOG_ALL
    assert parse_strict_flag('log_all') is StrictHandling.LOG_ALL
    assert parse_strict_flag('return_all') is StrictHandling.RETURN_ALL


def test_parse_strict_flag_error_message_contains_original_value():
    with pytest.raises(ValueError, match="'bogus' is not a valid StrictHandling"):
        parse_strict_flag('bogus')


def test_validate_integrity_and_strict_load_requires_ckpt_metadata(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    with pytest.raises(CheckpointingException, match='ckpt_sharded_metadata=None'):
        validate_integrity_and_strict_load(
            {}, StrictHandling.LOG_ALL, validate_access_integrity=False
        )


def test_validate_integrity_and_strict_load_requires_global_metadata(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    with pytest.raises(CheckpointingException, match='without global_metadata'):
        validate_integrity_and_strict_load(
            {}, StrictHandling.ASSUME_OK_UNEXPECTED, validate_access_integrity=True
        )


def test_validate_common_state_dict_skips_check_without_distributed(model_parallel, monkeypatch):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    broadcast_calls = []
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        'broadcast_object_list',
        lambda *args, **kwargs: broadcast_calls.append(args),
    )

    assert _validate_common_state_dict({'iteration': 3}) is None
    assert broadcast_calls == []


def test_validate_sharding_integrity_raises_on_overlapping_main_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    sh_ten = ShardedTensor.from_rank_offsets('overlapping', torch.ones(2), replica_id=0)
    global_metadata = [[sh_ten] for _ in range(world_size)]

    if torch.distributed.get_rank() == 0:
        with pytest.raises(CheckpointingException, match='Invalid sharding pattern'):
            validate_sharding_integrity(global_metadata)
    else:
        validate_sharding_integrity(global_metadata)


def test_validate_objects_rejects_duplicate_main_shard_keys(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    first = ShardedObject('dup_obj', None, (1,), (0,), replica_id=0)
    second = ShardedObject('dup_obj', None, (1,), (0,), replica_id=0)
    global_metadata = [[first], [second]]

    if torch.distributed.get_rank() == 0:
        with pytest.raises(CheckpointingException, match='Duplicate ShardedObject keys'):
            validate_sharding_integrity(global_metadata)
    else:
        validate_sharding_integrity(global_metadata)


def test_validate_objects_reports_missing_shard_objects(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    partial_obj = ShardedObject('partial_obj', None, (2,), (0,), replica_id=0)
    global_metadata = [[partial_obj]]

    if torch.distributed.get_rank() == 0:
        with pytest.raises(CheckpointingException, match='1 ShardedObject are missing'):
            validate_sharding_integrity(global_metadata)
    else:
        validate_sharding_integrity(global_metadata)


def test_verify_integrity_manifest_raises_when_manifest_missing(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    ckpt_dir = shared_ckpt_root / 'missing_manifest'
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    with pytest.raises(CheckpointingException, match='Integrity manifest not found at'):
        verify_integrity_manifest(str(ckpt_dir))


def test_verify_integrity_manifest_reports_missing_checkpoint_file(
    model_parallel, shared_ckpt_root
):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    ckpt_dir = shared_ckpt_root / 'missing_file'
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manifest = {'algorithm': 'sha256', 'files': {'gone_file.distcp': '0' * 64}}
        with open(ckpt_dir / 'integrity.json', 'w') as f:
            json.dump(manifest, f)
    torch.distributed.barrier()

    with pytest.raises(CheckpointingException, match='gone_file.distcp: file missing'):
        verify_integrity_manifest(str(ckpt_dir))


def test_verify_integrity_manifest_rejects_unknown_hash_algorithm(model_parallel, shared_ckpt_root):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    ckpt_dir = shared_ckpt_root / 'bad_algorithm'
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manifest = {'algorithm': 'md5', 'files': {}}
        with open(ckpt_dir / 'integrity.json', 'w') as f:
            json.dump(manifest, f)
    torch.distributed.barrier()

    with pytest.raises(CheckpointingException, match='Unsupported hash algorithm'):
        verify_integrity_manifest(str(ckpt_dir))


def test_verify_integrity_manifest_runs_impl_directly_without_distributed(
    model_parallel, shared_ckpt_root, monkeypatch, caplog
):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    ckpt_dir = shared_ckpt_root / 'no_dist_verify'
    if torch.distributed.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / 'metadata.json', 'w') as f:
            json.dump({'sharded_backend': 'torch_dist'}, f)
        save_integrity_manifest(str(ckpt_dir))
    torch.distributed.barrier()

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    with caplog.at_level(logging.INFO):
        assert verify_integrity_manifest(str(ckpt_dir)) is None

    if torch.distributed.get_rank() == 0:
        assert 'Checkpoint integrity verified: 1 file(s) OK' in caplog.text
