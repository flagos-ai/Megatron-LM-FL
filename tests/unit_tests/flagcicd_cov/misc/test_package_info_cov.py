# Copyright (c) 2026, BAAI. All rights reserved.
"""Package version string built from the release tuple plus the git short hash."""

import re

import megatron.core.package_info as package_info


def test_version_carries_git_short_hash_suffix_in_a_git_checkout():
    base = f"{package_info.MAJOR}.{package_info.MINOR}.{package_info.PATCH}"

    assert package_info.__shortversion__ == base
    assert re.fullmatch(rf"{re.escape(base)}\+[0-9a-f]{{7,}}", package_info.__version__)
