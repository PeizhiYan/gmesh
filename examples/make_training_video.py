#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Short videos (MP4 and GIF) of the hybrid training, for presentations.

Both demos learn the object from 24 images. Every frame shows, from a slowly
orbiting camera:

    target | the model, rendered by Gmesh | its 3D shape (white mesh, Gaussians in orange)

with a caption for the current step and a progress bar. The clips start with the
training images and end with a full turn around the result.

  demo 1 (example 6): a sphere is deformed and painted to match the images,
                      then 3D Gaussians are added and trained together with the mesh
  demo 2 (example 7): 3D Gaussians are trained from random positions, a mesh is
                      extracted from them and refined, then fewer Gaussians are
                      added and trained together with the mesh

Usage (from the repository root; needs ffmpeg on the PATH, and scikit-image for demo 2):
    python examples/make_training_video.py --demo 1
    python examples/make_training_video.py --demo 2 --size 512 --fps 30

It saves to --out (default: output/training_video/demo<N>): training.mp4, training.gif and the frames (PNG).
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import matplotlib.font_manager as font_manager
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.cameras import OrbitCamera
from train_textured_toy import SH_C0, load_scene, orbit_poses, make_gaussians
from train_hybrid_toy import initial_sphere, make_mesh, make_renderers, train_mesh_stage, train_hybrid_stage

ORANGE = (1.0, 0.55, 0.1)   # color of the Gaussians in the 3D shape panel
MARGIN, GAP = 24, 16        # pixels around and between the panels
HEADER, LABEL, BAR = 96, 44, 28
TEXT, MUTED = (30, 30, 30), (110, 110, 110)
INTRO = ("Learning a 3D object from 24 pictures",
         "GMesh: a differentiable renderer for triangle meshes and 3D Gaussians, together")


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def to_image(img):
    """[H, W, 3] tensor in [0, 1] -> PIL image"""
    return Image.fromarray((img.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


def capture_iterations(iters, frames):
    """`frames` iterations in 1..iters, denser at the start (where the model changes fastest)"""
    its = np.unique(np.round(np.geomspace(1, iters, frames)).astype(int))
    return set(its.tolist())


class Video:
    def __init__(self, out_dir, size):
        self.size = size
        self.width = 2 * MARGIN + 3 * size + 2 * GAP
        self.height = HEADER + size + LABEL + BAR
        self.frames_dir = os.path.join(out_dir, "frames")
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        os.makedirs(self.frames_dir)
        self.n = 0
        self.fonts = {"title": font(30, bold=True), "sub": font(20), "label": font(20)}

    def save(self, canvas, repeat=1):
        for _ in range(repeat):
            canvas.save(os.path.join(self.frames_dir, f"{self.n:05d}.png"))
            self.n += 1

    def _header(self, draw, title, subtitle):
        draw.text((MARGIN, 18), title, fill=TEXT, font=self.fonts["title"])
        draw.text((MARGIN, 58), subtitle, fill=MUTED, font=self.fonts["sub"])

    def frame(self, panels, labels, title, subtitle, progress, repeat=1):
        canvas = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(canvas)
        self._header(draw, title, subtitle)
        for k, (panel, label) in enumerate(zip(panels, labels)):
            x = MARGIN + k * (self.size + GAP)
            canvas.paste(to_image(panel), (x, HEADER))
            w = draw.textlength(label, font=self.fonts["label"])
            draw.text((x + (self.size - w) / 2, HEADER + self.size + 10), label, fill=TEXT, font=self.fonts["label"])
        # progress bar over the whole training
        y = self.height - BAR + 8
        draw.rounded_rectangle((MARGIN, y, self.width - MARGIN, y + 10), radius=5, fill=(225, 225, 225))
        if progress > 0:
            x1 = MARGIN + progress * (self.width - 2 * MARGIN)
            draw.rounded_rectangle((MARGIN, y, max(x1, MARGIN + 10), y + 10), radius=5, fill=(240, 140, 25))
        self.save(canvas, repeat)

    def intro(self, images, title, subtitle, repeat):
        """title card with the training images"""
        canvas = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(canvas)
        self._header(draw, title, subtitle)
        cols, rows = 8, (len(images) + 7) // 8
        cell = min((self.width - 2 * MARGIN) // cols, (self.height - HEADER - MARGIN) // rows)
        x0 = (self.width - cols * cell) // 2
        for k, img in enumerate(images):
            thumb = to_image(img).resize((cell - 6, cell - 6), Image.LANCZOS)
            canvas.paste(thumb, (x0 + (k % cols) * cell + 3, HEADER + (k // cols) * cell + 3))
        self.save(canvas, repeat)

    def encode(self, out_dir, fps, gif_width):
        mp4, gif = os.path.join(out_dir, "training.mp4"), os.path.join(out_dir, "training.gif")
        frames = os.path.join(self.frames_dir, "%05d.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", frames,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
        palette = (f"fps={min(fps, 15)},scale={gif_width}:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", palette, gif], check=True)
        return mp4, gif


class Scene:
    """the training setup (as in examples 6 and 7) and the display: a larger camera, orbiting slowly"""

    def __init__(self, args, video):
        device = args.device
        self.args, self.video, self.device = args, video, device
        self.gt_mesh, self.light = load_scene(device)
        self.camera = OrbitCamera(image_width=args.train_size, image_height=args.train_size, fov_x=40, fov_y=40,
                                  z_near=0.1, z_far=100, device=device)
        self.renderers = make_renderers(self.camera, self.light)
        self.train_poses, self.test_poses = orbit_poses(device)
        with torch.no_grad():
            render = self.renderers[0].render
            self.gt_train = [(r[0], a[0]) for r, _, a in (render(self.gt_mesh, p) for p in self.train_poses)]
            self.gt_test = [(r[0], a[0]) for r, _, a in (render(self.gt_mesh, p) for p in self.test_poses)]
        display_camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                                     z_near=0.1, z_far=100, device=device)
        self.mesh_renderer, self.gaussian_rasterizer, self.gmesh_renderer = make_renderers(display_camera, self.light)
        self.orange_sh = (torch.tensor(ORANGE, device=device) - 0.5) / SH_C0

    def data(self):
        """the training and test data, as the training functions take them"""
        return self.gt_train, self.gt_test, self.train_poses, self.test_poses

    def pose(self, azimuth=None):
        if azimuth is None:
            azimuth = 2 * math.pi * self.video.n / self.args.orbit_frames
        return torch.tensor([[0.15, azimuth, self.args.distance]], device=self.device)

    def orange(self, g_params):
        return make_gaussians({**g_params, "sh": self.orange_sh.expand_as(g_params["sh"]).contiguous()})

    @torch.no_grad()
    def panels(self, mesh=None, g_params=None, azimuth=None):
        """[target, model, 3D shape] for a mesh, Gaussians (raw parameters), or both"""
        p = self.pose(azimuth)
        target = self.mesh_renderer.render(self.gt_mesh, p)[0][0]
        if mesh is None:
            return [target, self.gaussian_rasterizer.rasterize(make_gaussians(g_params), p)[0][0],
                    self.gaussian_rasterizer.rasterize(self.orange(g_params), p)[0][0]]
        shape = Mesh(verts=mesh.verts, faces=mesh.faces, colors=torch.ones_like(mesh.verts))
        if g_params is None:
            return [target, self.mesh_renderer.render(mesh, p)[0][0], self.mesh_renderer.render(shape, p)[0][0]]
        return [target, self.gmesh_renderer.render(mesh, make_gaussians(g_params), p)[0][0],
                self.gmesh_renderer.render(shape, self.orange(g_params), p)[0][0]]

    def recorder(self, iters, offset, total, title, subtitle, labels, panels_of, hold_first=0, hold_last=0):
        """a fit() callback recording frames of a stage (iterations offset + 1 .. offset + iters of `total`)"""
        capture = capture_iterations(iters, self.args.frames_per_stage)

        def record(it, params):
            if it in capture:
                repeat = hold_first if it == 1 else hold_last if it == iters else 1
                self.video.frame(panels_of(params), labels, title, f"{subtitle}  |  iteration {it} / {iters}",
                                 (offset + it) / total, repeat=max(repeat, 1))
        return record

    def turntable(self, mesh, g_params, labels, title, subtitle):
        """one full turn around the result, from where the camera is"""
        start = 2 * math.pi * self.video.n / self.args.orbit_frames
        for k in range(self.args.turntable_frames):
            azimuth = start + 2 * math.pi * k / self.args.turntable_frames
            self.video.frame(self.panels(mesh, g_params, azimuth), labels, title, subtitle, 1.0)


def demo1(scene, args):
    """example 6: sphere -> textured mesh -> mesh + Gaussians"""
    fps, total = args.fps, args.mesh_iters + args.hybrid_iters
    sphere = initial_sphere(scene.gt_mesh.verts)
    labels_mesh = ["Target", "Our model", "Its 3D shape"]
    labels_hybrid = ["Target", "Our model", "Its 3D shape + Gaussians (orange)"]

    print("stage 1: mesh")
    record = scene.recorder(args.mesh_iters, 0, total, "Step 1: a sphere is reshaped and painted",
                            "to match the pictures", labels_mesh,
                            lambda p: scene.panels(make_mesh(sphere, p)), hold_last=fps)
    _, m_params = train_mesh_stage(scene.renderers, sphere, *scene.data(), args.mesh_iters,
                                   log_every=args.mesh_iters // 4, callback=record)

    print("stage 2: mesh + Gaussians")
    record = scene.recorder(args.hybrid_iters, args.mesh_iters, total,
                            f"Step 2: {args.n_gaussians} 3D Gaussians are added",
                            "and trained together with the mesh", labels_hybrid,
                            lambda p: scene.panels(make_mesh(sphere, p), p), hold_first=fps // 2)
    _, m_params2, g_params = train_hybrid_stage(scene.renderers, sphere, m_params, args.n_gaussians, *scene.data(),
                                                args.hybrid_iters, log_every=args.hybrid_iters // 4, callback=record)

    print("turntable")
    scene.turntable(make_mesh(sphere, m_params2), g_params, labels_hybrid,
                    "Result: a mesh and Gaussians, rendered together",
                    "every part is learned through the differentiable renderer")


def demo2(scene, args):
    """example 7: Gaussians -> extracted mesh -> trained mesh -> mesh + fewer Gaussians"""
    import train_gs2mesh_toy as gs2mesh

    fps, n, n_small = args.fps, args.n_gaussians_demo2, args.n_gaussians_demo2 // 4
    total = args.gaussian_iters + args.mesh_iters_demo2 + args.hybrid_iters
    labels_gaussians = ["Target", "Our model", "Its Gaussians (orange)"]
    labels_mesh = ["Target", "Our model", "Its 3D shape"]
    labels_hybrid = ["Target", "Our model", "Its 3D shape + Gaussians (orange)"]

    print(f"stage 1: {n} Gaussians")
    g1 = gs2mesh.random_gaussian_params(n, device=args.device)
    record = scene.recorder(args.gaussian_iters, 0, total, f"Step 1: {n} 3D Gaussians learn the object",
                            "starting from random positions", labels_gaussians,
                            lambda p: scene.panels(g_params=p), hold_first=fps // 2, hold_last=fps // 2)
    hist1 = gs2mesh.train_gaussians(scene.renderers, g1, *scene.data(), args.gaussian_iters,
                                    log_every=args.gaussian_iters // 4, callback=record)

    print("stage 2: mesh extraction")
    verts, faces = gs2mesh.extract_mesh(scene.renderers[1], scene.camera, g1, gs2mesh.extraction_poses(args.device),
                                        grid=args.grid)
    mesh = gs2mesh.initial_mesh(verts, faces, gs2mesh.colors_from_gaussians(verts, faces, g1, scene.light))
    progress = args.gaussian_iters / total
    for _ in range(int(1.5 * fps)):
        scene.video.frame(scene.panels(gs2mesh.make_mesh(mesh)), labels_mesh,
                          "Step 2: a mesh is extracted from the Gaussians",
                          f"by fusing their depth maps  |  {faces.shape[0]} triangles", progress)

    print("stage 3: mesh")
    record = scene.recorder(args.mesh_iters_demo2, args.gaussian_iters, total, "Step 3: the mesh is refined",
                            "its shape and colors, to match the pictures", labels_mesh,
                            lambda p: scene.panels(gs2mesh.make_mesh(mesh, p)), hold_last=fps // 2)
    gs2mesh.train_mesh(scene.renderers, mesh, *scene.data(), args.mesh_iters_demo2,
                       log_every=args.mesh_iters_demo2 // 4, callback=record)

    print(f"stage 4: mesh + {n_small} Gaussians")
    record = scene.recorder(args.hybrid_iters, args.gaussian_iters + args.mesh_iters_demo2, total,
                            f"Step 4: {n_small} 3D Gaussians are added to the mesh",
                            "and trained together with it", labels_hybrid,
                            lambda p: scene.panels(gs2mesh.make_mesh(mesh, p), p), hold_first=fps // 2)
    hist4, m_params, g4 = gs2mesh.train_hybrid(scene.renderers, mesh, n_small, *scene.data(), args.hybrid_iters,
                                               log_every=args.hybrid_iters // 4, callback=record)

    print("turntable")
    psnr1, psnr4 = hist1["psnr"][-1], hist4["psnr"][-1]
    print(f"test PSNR: {n} Gaussians {psnr1:.2f} dB, mesh + {n_small} Gaussians {psnr4:.2f} dB")
    comparison = "a closer match to new views" if psnr4 > psnr1 else "compared on new views"
    scene.turntable(gs2mesh.make_mesh(mesh, m_params), g4, labels_hybrid,
                    f"Result: a mesh and {n_small} Gaussians, rendered together",
                    f"vs the {n} Gaussians of step 1: {comparison} ({psnr4:.1f} vs {psnr1:.1f} dB PSNR)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mesh-iters", type=int, default=2000, help="demo 1: stage 1 (mesh) iterations")
    parser.add_argument("--n-gaussians", type=int, default=2000, help="demo 1: Gaussians added to the mesh")
    parser.add_argument("--gaussian-iters", type=int, default=3000, help="demo 2: stage 1 (Gaussians) iterations")
    parser.add_argument("--mesh-iters-demo2", type=int, default=1500, help="demo 2: stage 3 (mesh) iterations")
    parser.add_argument("--n-gaussians-demo2", type=int, default=20000,
                        help="demo 2: Gaussians of stage 1 (stage 4 adds a quarter of them)")
    parser.add_argument("--grid", type=int, default=48, help="demo 2: resolution of the mesh extraction volume")
    parser.add_argument("--hybrid-iters", type=int, default=2000, help="iterations of the last (hybrid) stage")
    parser.add_argument("--train-size", type=int, default=256, help="resolution of the training images")
    parser.add_argument("--size", type=int, default=384, help="resolution of each panel of the video")
    parser.add_argument("--frames-per-stage", type=int, default=150)
    parser.add_argument("--turntable-frames", type=int, default=96, help="frames of the final full turn around the result")
    parser.add_argument("--orbit-frames", type=int, default=240, help="frames per revolution of the camera")
    parser.add_argument("--distance", type=float, default=3.6, help="distance of the display camera (training: 3.0)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: output/training_video/demo<N>")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "output", "training_video", f"demo{args.demo}")
    os.makedirs(out, exist_ok=True)

    video = Video(out, args.size)
    scene = Scene(args, video)
    print("intro")
    video.intro([rgb for rgb, _ in scene.gt_train], *INTRO, repeat=2 * args.fps)
    (demo1 if args.demo == 1 else demo2)(scene, args)

    mp4, gif = video.encode(out, args.fps, args.gif_width)
    print(f"{video.n} frames ({video.n / args.fps:.1f}s) -> {mp4}, {gif}")


if __name__ == "__main__":
    main()
