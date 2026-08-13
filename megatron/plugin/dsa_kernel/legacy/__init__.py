# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Legacy PyTorch BMM implementations for DSA sparse attention.

Provides fallback forward/backward for non-HP scenarios (decoding, testing).
Training always uses the HP WGMMA kernel in ``triton_sparse_attn.py``.
"""
