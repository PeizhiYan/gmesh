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
import re
import subprocess
import sys

# functions the extension must provide (an older cached build may lack some)
_FUNCTIONS = (
    "projection_fwd", "projection_bwd", "isect_tiles", "isect_offset_encode", "sh_fwd", "sh_bwd",
    "rasterize_mesh_fwd", "rasterize_hybrid_fwd", "rasterize_hybrid_bwd",
)

_module = None
_build_error = None


def _nvcc_version():
    """(major, minor) of the nvcc that builds the extension, or None if unknown."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        return None
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_extension():
    global _module, _build_error
    if _module is not None:
        return _module
    if _build_error is not None:
        # do not retry: torch would then load a stale binary from an older build
        raise _build_error

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
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"]
    nvcc_version = _nvcc_version()
    if nvcc_version is not None and nvcc_version >= (11, 2):  # older nvcc has no -diag-suppress
        extra_cuda_cflags += ["-diag-suppress", "20012,186"]  # verbose GLM / Torch warnings
    if sys.platform == "win32":
        extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN"]

    try:
        module = load(
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
    except Exception as e:
        _build_error = e
        raise

    missing = [f for f in _FUNCTIONS if not hasattr(module, f)]
    if missing:
        _build_error = RuntimeError(
            f"The loaded gmesh_cuda extension ({getattr(module, '__file__', '?')}) is an old build "
            f"without {', '.join(missing)}. Delete its folder (torch_extensions cache) and run again."
        )
        raise _build_error
    _module = module
    return module
