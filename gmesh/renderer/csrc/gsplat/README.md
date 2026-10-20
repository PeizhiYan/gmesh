# Code adapted from gsplat

The files in this directory are adapted from **gsplat v1.5.2**
(https://github.com/nerfstudio-project/gsplat, tag `v1.5.2`, commit
`32f2a54d21c7ecb135320bb02b136b7407ae5712`), Copyright the gsplat authors,
licensed under the **Apache License, Version 2.0** (see [LICENSE](./LICENSE)).

| File in Gmesh | Source in gsplat | Changes |
|---|---|---|
| `Common.h` | `gsplat/cuda/include/Common.h` | namespace `gsplat` -> `gmesh_gsplat` |
| `Utils.cuh` | `gsplat/cuda/include/Utils.cuh` | namespace |
| `Projection.h` | `gsplat/cuda/csrc/Projection.h` | namespace; only the fused (non-packed) 3DGS projection |
| `ProjectionEWA3DGSFused.cu` | `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` | namespace; backward without `cg::labeled_partition` on compute capability < 7.0 |
| `Intersect.h`, `IntersectTile.cu` | `gsplat/cuda/csrc/Intersect.h`, `IntersectTile.cu` | namespace |
| `SphericalHarmonics.h`, `SphericalHarmonicsCUDA.cu` | `gsplat/cuda/csrc/SphericalHarmonics.h`, `SphericalHarmonicsCUDA.cu` | namespace |
| `GsplatOps.h`, `GsplatOps.cpp` | `gsplat/cuda/csrc/Projection.cpp`, `Intersect.cpp`, `SphericalHarmonics.cpp` | namespace; only the host functions above, in one file; pinhole-camera entry points for Gmesh |

Also adapted from gsplat (with the same license):

- `gmesh/renderer/gsplat_ops.py`, from `gsplat/cuda/_wrapper.py` (the Python
  autograd wrappers of the functions above);
- parts of `../rasterize_hybrid.cu` (Gmesh's hybrid compositing kernels), from
  `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` and `RasterizeToPixels3DGSBwd.cu`.

The kernels use [GLM](https://github.com/g-truc/glm) (commit `33b4a621`, the one
gsplat v1.5.2 uses), included in `../third_party/glm` with its license
(`copying.txt`, MIT / Happy Bunny License).

If you use Gmesh, please also cite gsplat:

```bibtex
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting},
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765},
}
```
