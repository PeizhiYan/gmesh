#
# Peizhi Yan
# Copyright (C) 2026
#

"""
JIT-compiles and loads the gmesh CUDA extension (./csrc) on first use.

./csrc/gsplat contains kernels adapted from gsplat (Apache-2.0, see the README
and LICENSE there); ./csrc/third_party/glm is the GLM math library (MIT).
"""

import glob
import os
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def load_extension():
    from torch.utils.cpp_extension import load

    csrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    sources = (
        [os.path.join(csrc, "ext.cpp")]
        + sorted(glob.glob(os.path.join(csrc, "*.cu")))
        + sorted(glob.glob(os.path.join(csrc, "gsplat", "*.cu")))
        + sorted(glob.glob(os.path.join(csrc, "gsplat", "*.cpp")))
    )

    # same compiler flags as gsplat
    extra_cflags = ["-O3"]
    if os.name != "nt":
        extra_cflags += ["-Wno-sign-compare"]
    extra_cuda_cflags = [
        "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
        "-diag-suppress", "20012,186",  # verbose GLM / Torch warnings
    ]
    if sys.platform == "win32":
        extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN"]

    return load(
        name="gmesh_cuda",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[
            os.path.join(csrc, "gsplat"),
            os.path.join(csrc, "third_party", "glm"),
        ],
        verbose=False,
    )
