"""
HAWKEYE Drone Flight Renderer
==============================
Moves camera along the current aisle and renders viewport images.

*** RUN INSIDE BLENDER (Scripting tab) ***
*** First: position camera in an aisle, enter camera view ***

Steps before running:
1. Fly to aisle position (Shift+`)
2. Align camera to view (View → Align View → Align Active Camera to View)
3. Enter camera view (View → Cameras → Active Camera)
4. Run this script
"""

import bpy
import math
import os

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

STEP_SIZE = 2.0             # meters between frames
AISLE_LENGTH = 38.0         # 20 frames * 2m = 38m
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
START_FRAME = 191           # numbering continues from last batch (LAST AISLE)
OUTPUT_DIR = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/outputs/renders/base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# READ CURRENT CAMERA POSITION AND DIRECTION
# ═══════════════════════════════════════════════════════════

cam_obj = bpy.context.scene.camera
if not cam_obj:
    print("ERROR: No active camera! Set one first.")
else:
    start_x = cam_obj.location.x
    start_y = cam_obj.location.y
    start_z = cam_obj.location.z
    rot_x = cam_obj.rotation_euler.x
    rot_y = cam_obj.rotation_euler.y
    rot_z = cam_obj.rotation_euler.z

    # Figure out movement direction from camera's Z rotation
    # Z=0° → moving along +Y, Z=90° → moving along -X, Z=180° → moving along -Y
    move_dir_x = -math.sin(rot_z)
    move_dir_y = math.cos(rot_z)

    print(f"\n{'='*50}")
    print(f"  Camera start: ({start_x:.1f}, {start_y:.1f}, {start_z:.1f})")
    print(f"  Rotation: ({math.degrees(rot_x):.1f}, {math.degrees(rot_y):.1f}, {math.degrees(rot_z):.1f})")
    print(f"  Move direction: ({move_dir_x:.2f}, {move_dir_y:.2f})")
    print(f"  Steps: {int(AISLE_LENGTH / STEP_SIZE)} frames")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*50}\n")

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # Lock viewport to camera
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
            break

    # Render loop
    num_frames = int(AISLE_LENGTH / STEP_SIZE)
    for i in range(num_frames + 1):
        # Move camera along aisle
        cam_obj.location.x = start_x + move_dir_x * STEP_SIZE * i
        cam_obj.location.y = start_y + move_dir_y * STEP_SIZE * i
        # Keep same Z and rotation

        bpy.context.view_layer.update()

        filepath = os.path.join(OUTPUT_DIR, f"frame_{START_FRAME + i:04d}.png")
        scene.render.filepath = filepath
        bpy.ops.render.opengl(write_still=True)

        if i % 5 == 0:
            print(f"  [{i+1}/{num_frames+1}] pos=({cam_obj.location.x:.1f}, {cam_obj.location.y:.1f})")

    # Save blend file
    bpy.ops.wm.save_mainfile()

    print(f"\n{'='*50}")
    print(f"  DONE! {num_frames+1} frames rendered")
    print(f"  Scene saved.")
    print(f"{'='*50}\n")
