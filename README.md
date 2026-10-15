<p align="center">
    <img src="./assets/GMesh-logo.png" width="500px" />
</p>

<h1 align="center">Differentiable Hybrid Renderer for 3D Gaussians & Mesh</h1>


<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>

**Current Version**: v0.1 (July 02, 2025)

## TL;DR

- Combining Gaussians and meshes lets you leverage the strengths of both for more flexible and realistic 3D scene rendering.
- Gmesh supports differentiable rendering of mixed 3D Gaussians and meshes within a single scene.
- Built on gsplat and its own differentiable mesh rasterizer (no PyTorch3D needed), Gmesh is fully PyTorch-native.


## Examples

Please see ```./examples```:

1. 3D Mesh Rendering via Pytorch3D: [Link](./examples/example-1%20-%20mesh%20rendering.ipynb)
2. 3DGS via gsplat: [Link](./examples/example-2%20-%20gaussian%20splatting.ipynb)
3. Hybrid Gaussians and Mesh Rendering via Gmesh: [Link](./examples/example-3%20-%20gmesh%20hybrid%20rendering.ipynb)



## Setup


```bash
# create conda virtual environment
conda create --name gmesh -y python=3.10
conda activate gmesh

# install cuda toolkit
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja
conda install -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7

# install PyTorch
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117

# install other dependencies
pip install --no-build-isolation -r requirements.txt

# install Gmesh
pip install --no-build-isolation -e .
```

> [!TIP]
> I have only tested the following library versions:
> ```CUDA 11.7```, ```PyTorch 2.0.1```, and ```gsplat v1.5.2```.
> PyTorch3D is no longer required (the former PyTorch3D-based mesh renderer is kept in
> `gmesh/renderer/renderer_pytorch3d.py` for comparison only).
> If you encounter any issues or unexpected results, please double-check that you have installed the correct versions.


## Design Details

<p align="center">
    <b>Pipeline</b><br>
    <img src="./assets/pipeline.png" width="100%" />
</p>

### Mesh rasterizer (no PyTorch3D)

Meshes are rasterized by Gmesh's own differentiable rasterizer (`gmesh/renderer/rasterizer_mesh.py`), using exactly the
same pinhole camera and pixel convention as gsplat:

1. Triangles are binned into 16x16 tiles, and a CUDA kernel finds the nearest triangle at each pixel center (z-buffer).
   This visibility decision is the only non-differentiable step.
2. Barycentric coordinates, depth and vertex attributes (colors) are recomputed in PyTorch from the face ids, with
   perspective-correct interpolation, so autograd provides gradients w.r.t. vertex positions and vertex colors.

3. Edges are antialiased analytically (as in nvdiffrast): for every pair of neighboring pixels that sees a geometric
   discontinuity (a silhouette), the position where the front triangle's edge crosses between the two pixel centers
   gives a fractional coverage (and blends colors over a farther surface). This yields smooth silhouettes and
   gradients w.r.t. the silhouette vertex positions. Internal edges of a connected surface are left untouched.

Triangles with a vertex in front of the near plane are skipped (no near-plane clipping).

### Hybrid rendering (default: `mode='hybrid'`)

At every pixel, each Gaussian is split at the mesh surface into the part in front of it and the part behind it.
The Gaussian's depth at pixel `p` uses the same local affine (EWA) approximation as its 2D splat: the camera-space
depth conditioned on the projected position is Gaussian with mean `z*(p) = z_c + g . (p - mean2d)` and standard
deviation `sigma`, where `g = (Sigma_uv + eps I)^-1 Sigma_uv,z` and `sigma^2 = Sigma_zz - Sigma_z,uv g` come from the
projected 3D covariance (so a flat Gaussian lying on a tilted surface has a depth that tilts with it). The fraction
in front of the mesh depth `D` is

```
f = Phi((D - z*(p)) / (sigma * softness))
```

The front parts (`alpha * f`) and back parts (`alpha * (1 - f)`) are alpha-composited into two layers, in the usual
sorted 3DGS order, and combined with the opaque mesh (coverage `a_m`, color `c_m`):

```
C = C_front + T_front * (a_m * c_m + (1 - a_m) * (C_back + T_back * bg))
```

- A Gaussian crossing the mesh is cut smoothly and *along the surface* (a flat Gaussian lying on a tilted surface
  tilts with it), instead of a per-pixel staircase. No `depth_bias` is needed for Gaussians on the surface.
- Everything is differentiable: Gaussian means, rotations, scales (also through the split), opacities and colors,
  and the mesh vertices (through the mesh depth and the interpolated colors).
- `softness` scales the width of the transition (1.0 = the Gaussian's own depth extent; smaller is sharper).
- With antialiased mesh edges, the coverage `a_m` is fractional at silhouettes, so Gaussians behind the mesh fade
  out smoothly at its outline instead of being cut along it.
- `cull_occluded=True` removes Gaussians that are certainly hidden behind the mesh in fully covered 16x16 tiles
  before compositing. This does not change the result (they would be skipped anyway) and saves time in scenes
  where the mesh hides many Gaussians.

The compositing is a custom CUDA kernel (forward and backward, `gmesh/renderer/csrc/rasterize_hybrid.cu`); the
projection, spherical harmonics and tile sorting stages are gsplat's, and the depth model is computed in PyTorch. The CUDA extension is JIT-compiled the first
time it is used (this needs `nvcc` and `ninja`, see Setup, and takes about a minute).

```python
gmesh_renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0), softness=1.0,
                               antialias=True, cull_occluded=True)
rgb, depth, alpha = gmesh_renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose)
```

Other modes, for comparison:
- `mode='zbuffer'`: hard per-pixel test of each Gaussian's *center* depth against the mesh z-buffer
  (`depth_bias` moves the test slightly behind the surface).
- `mode='depth_blend'`: the original behavior, which renders mesh and Gaussians separately and blends the two
  images by comparing their depth maps.

<p align="center">
    <b>Camera Coordinates</b><br>
    <img src="./assets/cam_coordinates.png" width="150px" />
</p>





## Tests

The tests need a CUDA GPU (the renderer tests also need open3d to read the example mesh):

```bash
python -m pytest tests -v
```

They check the hybrid and z-buffer rasterizers against gsplat (no mesh) and against brute-force PyTorch references
(with a mesh), for both the forward pass and the gradients; the mesh rasterizer against a brute-force rasterizer and
finite-difference gradients; and that the mesh is pixel- and depth-aligned with gsplat.


## Benchmark

```bash
python benchmarks/benchmark_renderer.py --width 1024 --height 768 --extra-gaussians 200000 --backward
```

Times (per frame, forward and forward + backward) the Gmesh mesh rasterizer, gsplat alone, the hybrid modes (with
and without antialiasing / occlusion culling), and, if PyTorch3D is installed, the former PyTorch3D mesh renderer
and the original v0.1 pipeline (PyTorch3D + gsplat + depth blending).


## Demo

You can play with our interactive demonstration to see the rendering results: 

```bash
python interactive_demo.py
```

<p align="center">
    <a href="https://www.youtube.com/watch?v=iVwSZtwr9GI">
        <img src="https://img.youtube.com/vi/iVwSZtwr9GI/0.jpg" alt="GMesh Demo Video" width="300px" height="auto"/>
    </a>
    <br>
    <a href="https://www.youtube.com/watch?v=iVwSZtwr9GI">Watch the demo video on YouTube</a>
</p>




## Limitations and Todos

### The current version has the following main limitations:
- Gaussians are still sorted by their center depth (as in 3DGS); only the split against the mesh is per pixel.
- One opaque mesh layer (no transparent meshes).
- Edge antialiasing is a per-axis 1D approximation (exact for straight edges away from corners).
- No near-plane clipping of triangles.
- 3D Gaussians' are not relightable.
- Does not support mesh opacity.
- Does not support mesh texture map.
- Does not support batch rendering.

### Todos:
- [x] Support z-buffering.
- [x] Own differentiable mesh rasterizer (no PyTorch3D) and soft per-ray hybrid compositing.
- [x] Anti-aliased mesh edges (fractional coverage and silhouette gradients).
- [x] Skip Gaussians hidden behind fully covered tiles (speed).
- [x] Benchmark against PyTorch3D + gsplat.
- [ ] Support mesh texture map.
- [ ] Support batch (batch of camera poses) rendering.
- [ ] Add two additional mesh to Gaussians conversion methods.
- [ ] Add some training examples.



## Acknowledgement and Disclaimer

The code can be used for research purposes, **provided that the terms of the licenses of any third-party libraries, data or dependencies are followed**. For commercial use, the parts of code we wrote are for free, but please be aware to get permissions from any third-party to use their code or dependencies. We do not assume any responsibility for any issues, damages, or liabilities that may arise from the use of this code. Users are responsible for ensuring compliance with any legal requirements, including licensing terms and conditions, and for verifying that the code is suitable for their intended purposes.

> [!NOTE]
> The example 3D mesh used in ./assets/data/spot.obj is sourced from: https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot. Please ensure you adhere to the license agreement provided on that site. We do not claim ownership of this mesh file.



## 🧸 Citation

Please cite this work if you find it useful in your projects. 

```bibtex
@software{Yan_GMesh_Differentiable_Hybrid_2025,
    author = {Yan, Peizhi},
    month = jul,
    title = {{GMesh: Differentiable Hybrid Renderer for 3D Gaussians and Meshes}},
    url = {https://github.com/PeizhiYan/gmesh},
    version = {0.1},
    year = {2025}
}
```

