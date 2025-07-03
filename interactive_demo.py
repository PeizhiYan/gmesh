#
# Peizhi Yan
# Copyright (C) 2025
#

# ## Enviroment Setup
# import os, sys
# WORKING_DIR = '/home/peizhi/Documents/gmesh'
# os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
# print("Current Working Directory: ", os.getcwd())

## Computing Device
device = 'cuda:0'
import torch
torch.cuda.set_device(device)

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import dearpygui.dearpygui as dpg

import gmesh.utils.o3d_utils as o3d_utils
from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians
from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera, OrbitCamera
from gmesh.renderer.gmesh_renderer import GmeshRenderer
from gmesh.renderer.shader import simple_phone_shading




# Parameters
RENDER_W = 1024
RENDER_H = 768
WINDOW_W = RENDER_W
WINDOW_H = RENDER_H
IMAGE_DISPLAY_X = 0
IMAGE_DISPLAY_Y = (WINDOW_H - RENDER_H) // 2
FPS_UPDATE_INTERVAL = 0.5

display_buffer = np.ones((WINDOW_H, WINDOW_W, 3), dtype=np.float32)
fps_last_time = time.time()
fps_last_val = 0.0




# Load mesh and gaussians
vertices, faces = o3d_utils._read_obj_file('./assets/data/spot.obj', uv=False)
theta = np.pi * 0.8
rotation_matrix = torch.tensor([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
], dtype=torch.float32, device=device)
vertices = torch.from_numpy(vertices).float().to(device) @ rotation_matrix.T
faces = torch.from_numpy(faces).to(device)





# Create Orbit Camera
camera = OrbitCamera(
    image_width=RENDER_W, image_height=RENDER_H,
    fov_x=40, fov_y=((1.0*RENDER_H)/RENDER_W)*40, z_near=0.1, z_far=100, device=device
)

# Create Gmesh renderer
gmesh_renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0))





@torch.no_grad()
def update_image():
    global display_buffer, fps_last_time, fps_last_val

    # Get slider values
    elevation = dpg.get_value("elevation_slider")
    azimuth   = dpg.get_value("azimuth_slider")
    distance  = dpg.get_value("distance_slider")

    elevation = np.deg2rad(elevation)
    azimuth   = np.deg2rad(azimuth)
    camera_pose = torch.tensor([[elevation, azimuth, distance]], dtype=torch.float32, device=device)

    # Get render mode and channel to display
    render_mode = dpg.get_value("render_mode_selector")  # "Mesh", "3DGS", "GMesh"
    channel     = dpg.get_value("channel_selector")      # "RGB", "Depth", "Alpha"

    # Get Gaussian scale and opacity
    scale = dpg.get_value("scale_slider")
    opacity = dpg.get_value("opacity_slider")

    # Create Mesh and Gaussians
    mesh = Mesh(verts=torch.clone(vertices), faces=faces)
    mesh2 = Mesh(verts=torch.clone(vertices), faces=faces)
    gaussians = mesh_to_gaussians(mesh=mesh2, method='vertex', trainable=False)
    gaussians.colors[:,:,0] = 1.0
    gaussians.colors[:,:,1] = 1.0
    gaussians.colors[:,:,2] = 0.2
    gaussians.opacities = opacity * torch.ones([gaussians.num_gaussians], dtype=torch.float32).to(device)
    gaussians.scales = scale * torch.ones([gaussians.num_gaussians, 3], dtype=torch.float32).to(device)

    # Add x,y,z offsets to mesh and Gaussians
    mesh.verts[:,0] += dpg.get_value("mesh_offset_x_slider")
    mesh.verts[:,1] += dpg.get_value("mesh_offset_y_slider")
    mesh.verts[:,2] += dpg.get_value("mesh_offset_z_slider")
    gaussians.means[:,0] += dpg.get_value("3dgs_offset_x_slider")
    gaussians.means[:,1] += dpg.get_value("3dgs_offset_y_slider")
    gaussians.means[:,2] += dpg.get_value("3dgs_offset_z_slider")

    # Shading for visualizing mesh 
    mesh.colors = simple_phone_shading(vertices=vertices, faces=faces)

    # Render
    ret_dict = gmesh_renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, return_all=True)

    # Retrieve results
    if render_mode == "Mesh":
        [rgb, depth, alpha] = ret_dict['mesh']
    elif render_mode == "3DGS":
        [rgb, depth, alpha] = ret_dict['3dgs']
    else:  # "GMesh" or default
        [rgb, depth, alpha] = ret_dict['blend']

    # Prepare image to display
    if channel == "RGB":
        rgb_np = rgb.detach().cpu().numpy()[0]
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        display_np = rgb_np
    elif channel == "Depth":
        depth_np = depth.detach().cpu().numpy()[0,...,0]   # [H,W]
        # Normalize to 0-1, then 3 channels
        depth_pos = depth_np[depth_np > 0.1]
        if depth_pos.size > 0:
            dmin = np.percentile(depth_pos, 5)
            dmax = np.percentile(depth_pos, 95)
        else:
            dmin, dmax = 0.0, 1.0  # fallback if all are zero
        # dmin, dmax = np.percentile(depth_np, 1), np.percentile(depth_np, 99)
        dnorm = (depth_np - dmin) / max(1e-6, dmax - dmin)
        dnorm = 1 - np.clip(dnorm, 0, 1) # revsese color
        display_np = np.repeat(dnorm[...,None], 3, axis=-1)
    elif channel == "Alpha":
        alpha_np = alpha.detach().cpu().numpy()[0,...,0]   # [H,W]
        anorm = np.clip(alpha_np, 0, 1)
        display_np = np.repeat(anorm[...,None], 3, axis=-1)
    else:
        # fallback
        display_np = np.ones((RENDER_H, RENDER_W, 3), dtype=np.float32)

    # Set display
    dpg.set_value("_scene_texture", display_np)

    # Calculate FPS (smoothed)
    t = time.time()
    fps = 1.0 / max(1e-5, (t - fps_last_time))
    fps_val = 0.7*fps + 0.3*fps_last_val if fps_last_val != 0 else fps
    fps_last_val = fps_val
    fps_last_time = t
    dpg.set_value("fps_text", f"FPS: {fps_val:.2f}")




def reset_camera_sliders():
    dpg.set_value("elevation_slider", 0.0)
    dpg.set_value("azimuth_slider", 0.0)
    dpg.set_value("distance_slider", 7.0)
    update_image()





# --- DearPyGui GUI ---
dpg.create_context()
dpg.create_viewport(title='GMesh (Gaussian + Mesh) Demo Viewer', width=WINDOW_W+360, height=WINDOW_H+40)

# Texture registry
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(WINDOW_W, WINDOW_H, display_buffer, format=dpg.mvFormat_Float_rgb, tag="_scene_texture")

# Main canvas window
with dpg.window(label="canvas", tag="_canvas_window", width=WINDOW_W+360, height=WINDOW_H+20, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
    dpg.add_image("_scene_texture", width=WINDOW_W, height=WINDOW_H, tag="_image")

# Camera controls
with dpg.window(label="Camera Controls", width=330, height=180, pos=(WINDOW_W+5, 5)):
    dpg.add_text("Orbit Camera")
    dpg.add_slider_float(tag="elevation_slider", label="Elevation (deg)", default_value=0.0, min_value=-89.0, max_value=89.0, callback=update_image)
    dpg.add_slider_float(tag="azimuth_slider", label="Azimuth (deg)", default_value=0.0, min_value=-180.0, max_value=180.0, callback=update_image)
    dpg.add_slider_float(tag="distance_slider", label="Distance", default_value=7.0, min_value=1.0, max_value=20.0, callback=update_image)
    dpg.add_button(label="Reset Camera", callback=reset_camera_sliders)
    dpg.add_text("FPS: 0.00", tag="fps_text")

# Content controls
with dpg.window(label="Content Controls", width=330, height=130, pos=(WINDOW_W+5, 180+10)):
    dpg.add_text("Rendering")
    dpg.add_combo(["Mesh", "3DGS", "GMesh"], tag="render_mode_selector", default_value="GMesh", callback=lambda: update_image())
    dpg.add_text("Display Channel")
    dpg.add_combo(["RGB", "Depth", "Alpha"], tag="channel_selector", default_value="RGB", callback=lambda: update_image())

# Mesh/Gaussian translation window
with dpg.window(label="Mesh & 3DGS Transform", width=330, height=240, pos=(WINDOW_W+5, 180+130+15)):
    dpg.add_text("Translate Mesh:")
    dpg.add_slider_float(tag="mesh_offset_x_slider", label="dx", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_slider_float(tag="mesh_offset_y_slider", label="dy", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_slider_float(tag="mesh_offset_z_slider", label="dz", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_text("Translate Gaussians:")
    dpg.add_slider_float(tag="3dgs_offset_x_slider", label="dx", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_slider_float(tag="3dgs_offset_y_slider", label="dy", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_slider_float(tag="3dgs_offset_z_slider", label="dz", default_value=0.0, min_value=-2.0, max_value=2.0, callback=update_image)
    dpg.add_button(label="Reset Offset", callback=lambda: [dpg.set_value("mesh_offset_x_slider", 0.0), 
                                                           dpg.set_value("mesh_offset_y_slider", 0.0), 
                                                           dpg.set_value("mesh_offset_z_slider", 0.0),
                                                           dpg.set_value("3dgs_offset_x_slider", 0.0),
                                                           dpg.set_value("3dgs_offset_y_slider", 0.0),
                                                           dpg.set_value("3dgs_offset_z_slider", 0.0), update_image()])

# 3DGS Attributes controls
with dpg.window(label="Gaussian Attributes Controls", width=330, height=180, pos=(WINDOW_W+5, 180+130+240+20)):
    dpg.add_slider_float(tag="opacity_slider", label="Opacity", default_value=1.0, min_value=0, max_value=1.0, callback=update_image)
    dpg.add_slider_float(tag="scale_slider", label="Scale", default_value=0.02, min_value=0.001, max_value=0.1, callback=update_image)
    dpg.add_button(label="Reset Gaussians", callback=lambda: [dpg.set_value("opacity_slider", 1.0), 
                                                              dpg.set_value("scale_slider", 0.02), update_image()])


"""
Start Demo
"""

# Set up the initial loop callback
dpg.set_frame_callback(1, update_image)  # Start the update loop

# Launch the app
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

