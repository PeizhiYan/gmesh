#
# Peizhi Yan
# Copyright (C) 2026
#

"""
JIT-compiles and loads the gmesh CUDA extension (./csrc) on first use.
"""

import glob
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def load_extension():
    from torch.utils.cpp_extension import load

    csrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    sources = [os.path.join(csrc, "ext.cpp")] + sorted(glob.glob(os.path.join(csrc, "*.cu")))
    return load(
        name="gmesh_cuda",
        sources=sources,
        extra_cflags=["-O3"],
        # same flags as gsplat, so alpha values match gsplat's kernels
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
