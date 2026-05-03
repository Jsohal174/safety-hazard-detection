"""
HAWKEYE Demo Drone Shots
========================
Renders 4 test images from drone perspective — one per aisle.
Run this to verify camera positions before the full batch render.

Usage in Blender: Scripting tab → Open → Run Script
Usage headless:   blender -b warehouse_base.blend --python demo_drone_shots.py
"""

import bpy
import math
import os

# ── Output folder ──
OUTPUT_DIR = os.path.join(os.path.dirname(bpy.data.filepath), "..", "..", "..", "..", "outputs", "renders", "demo")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*50}")
print(f"  HAWKEYE Demo Drone Shots")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*50}\n")

# ── Aisle positions from scene_layout.json ──
# Your 4 aisles (center X), Y range from scene bounds
AISLES = [
    {"name": "aisle_1", "x": -29.59, "y_start": -34.0, "y_end": 20.0},
    {"name": "aisle_2", "x": -20.54, "y_start": -34.0, "y_end": 20.0},
    {"name": "aisle_3", "x": -11.39, "y_start": -34.0, "y_end": 20.0},
    {"name": "aisle_4", "x": -2.26,  "y_start": -34.0, "y_end": 20.0},
]

# Drone settings
DRONE_HEIGHT = 3.5      # meters above floor
DRONE_PITCH_DEG = 55    # degrees below horizontal (looking down)
DRONE_LENS = 28         # mm focal length (wider = more visible)

# ── Setup camera ──
cam = bpy.data.cameras.get("DroneDemo_Cam")
if not cam:
    cam = bpy.data.cameras.new("DroneDemo_Cam")
cam.lens = DRONE_LENS
cam.clip_start = 0.1
cam.clip_end = 200

cam_obj = bpy.data.objects.get("DroneDemo_Cam")
if not cam_obj:
    cam_obj = bpy.data.objects.new("DroneDemo_Cam", cam)
    bpy.context.scene.collection.objects.link(cam_obj)

bpy.context.scene.camera = cam_obj

# ── Render settings (FAST - low res, minimal samples) ──
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 360
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

# Minimal quality for speed
eevee = getattr(scene, "eevee", None)
if eevee and hasattr(eevee, "taa_render_samples"):
    eevee.taa_render_samples = 1

# ── Render just 1 test shot ──
aisle = AISLES[1]  # aisle 2 (middle-ish)
x = aisle["x"]
y = (aisle["y_start"] + aisle["y_end"]) / 2
cam_obj.location = (x, y, DRONE_HEIGHT)

pitch_rad = math.radians(90 - DRONE_PITCH_DEG)
cam_obj.rotation_euler = (pitch_rad, 0, 0)

filepath = os.path.join(OUTPUT_DIR, f"demo_test.png")
scene.render.filepath = filepath
bpy.ops.render.render(write_still=True)
print(f"  Rendered: {filepath}")

print(f"\n{'='*50}")
print(f"  Done! Check {OUTPUT_DIR}")
print(f"{'='*50}\n")
