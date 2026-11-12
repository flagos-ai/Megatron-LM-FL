# Copyright (c) 2026, BAAI. All rights reserved.
"""utils_s3 re-exports the S3 path prefix and the S3Client protocol from object_storage_utils."""

from megatron.core.datasets import utils_s3
from megatron.core.datasets.object_storage_utils import S3_PREFIX, S3Client


def test_utils_s3_reexports_the_s3_prefix_and_client_protocol():
    assert utils_s3.S3_PREFIX == S3_PREFIX == "s3://"
    assert utils_s3.S3Client is S3Client
    assert utils_s3.S3Client.__name__ == "S3Client"
