"""
HAWKEYE Warehouse Scene Setup Script
Run this script in Blender to automatically set up the warehouse scene.

How to run:
1. Open Blender
2. Go to Scripting workspace (top menu bar)
3. Click "Open" and select this file
4. Click "Run Script" (play button)
"""

import bpy
import os
import math
import random

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")

print(f"Assets directory: {ASSETS_DIR}")


def clear_scene():
    """Remove default objects but keep the warehouse if it exists."""
    # Delete default cube if it exists
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)


def setup_lighting():
    """Set up industrial warehouse lighting."""

    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add Sun light for overall illumination
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Sun_Main"
    sun.data.energy = 5
    sun.data.color = (1.0, 0.95, 0.9)  # Slightly warm
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Add Area lights for industrial ceiling lights
    light_positions = [
        (-10, -5, 7),
        (0, -5, 7),
        (10, -5, 7),
        (-10, 5, 7),
        (0, 5, 7),
        (10, 5, 7),
    ]

    for i, pos in enumerate(light_positions):
        bpy.ops.object.light_add(type='AREA', location=pos)
        light = bpy.context.active_object
        light.name = f"Ceiling_Light_{i}"
        light.data.energy = 1000
        light.data.size = 2
        light.data.color = (1.0, 0.98, 0.95)  # Warm white
        light.rotation_euler = (0, 0, 0)  # Point straight down

    print("✓ Lighting setup complete")


def setup_camera_drone():
    """Set up camera at drone perspective."""

    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add drone camera
    bpy.ops.object.camera_add(location=(0, -15, 4))
    camera = bpy.context.active_object
    camera.name = "Drone_Camera"

    # Set rotation (looking down at ~50 degrees)
    camera.rotation_euler = (math.radians(70), 0, 0)

    # Set as active camera
    bpy.context.scene.camera = camera

    # Camera settings
    camera.data.lens = 35  # Wide angle for drone
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100

    print("✓ Drone camera setup complete")


def import_fbx(filepath, name_prefix=""):
    """Import an FBX file and return the imported objects."""
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return []

    # Get objects before import
    before = set(bpy.data.objects.keys())

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=filepath)

    # Get newly imported objects
    after = set(bpy.data.objects.keys())
    new_objects = list(after - before)

    print(f"✓ Imported: {filepath}")
    return new_objects


def import_glb(filepath):
    """Import a GLB/GLTF file."""
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return []

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = set(bpy.data.objects.keys())
    new_objects = list(after - before)

    print(f"✓ Imported: {filepath}")
    return new_objects


def append_from_blend(filepath, object_name=None):
    """Append objects from a .blend file."""
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return []

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        if object_name:
            if object_name in data_from.objects:
                data_to.objects = [object_name]
        else:
            data_to.objects = data_from.objects

    # Link to scene
    imported = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported.append(obj.name)

    print(f"✓ Appended from: {filepath}")
    return imported


def add_shelving_row(start_pos, num_shelves=5, spacing=3.0, scale=1.0):
    """Create a row of shelving units."""

    shelf_blend = os.path.join(ASSETS_DIR, "source", "Metalshelves.blend")

    if not os.path.exists(shelf_blend):
        print(f"✗ Shelf file not found: {shelf_blend}")
        # Create placeholder cubes instead
        for i in range(num_shelves):
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(start_pos[0] + i * spacing, start_pos[1], start_pos[2] + 1.5)
            )
            cube = bpy.context.active_object
            cube.scale = (1.2, 0.5, 3)
            cube.name = f"Shelf_Placeholder_{i}"
        return

    for i in range(num_shelves):
        # Append shelf
        with bpy.data.libraries.load(shelf_blend, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                obj.location = (start_pos[0] + i * spacing, start_pos[1], start_pos[2])
                obj.scale = (scale, scale, scale)


def add_boxes(num_boxes=10):
    """Add cardboard boxes scattered around."""

    box_fbx = os.path.join(ASSETS_DIR, "cardboard-boxes", "source", "Boxes_group_Y.fbx")

    if not os.path.exists(box_fbx):
        print(f"✗ Box file not found: {box_fbx}")
        # Create placeholder boxes
        for i in range(num_boxes):
            x = random.uniform(-8, 8)
            y = random.uniform(-8, 8)
            bpy.ops.mesh.primitive_cube_add(
                size=0.5,
                location=(x, y, 0.25)
            )
            box = bpy.context.active_object
            box.name = f"Box_Placeholder_{i}"
            box.scale = (
                random.uniform(0.3, 0.6),
                random.uniform(0.3, 0.6),
                random.uniform(0.3, 0.6)
            )
        return

    # Import box group once
    imported = import_fbx(box_fbx)

    if imported:
        # Position the boxes
        for obj_name in imported:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.location = (random.uniform(-5, 5), random.uniform(-5, 5), 0)
                obj.scale = (0.01, 0.01, 0.01)  # FBX often imports too large


def add_forklift():
    """Add a forklift to the scene."""

    # Check for the forklift - need to unzip first
    forklift_dir = os.path.join(ASSETS_DIR, "forklift-low-poly", "source")

    # Look for FBX file
    forklift_fbx = None
    if os.path.exists(forklift_dir):
        for f in os.listdir(forklift_dir):
            if f.endswith('.fbx'):
                forklift_fbx = os.path.join(forklift_dir, f)
                break

    if forklift_fbx and os.path.exists(forklift_fbx):
        imported = import_fbx(forklift_fbx)
        for obj_name in imported:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.location = (5, 0, 0)
                obj.rotation_euler = (0, 0, math.radians(45))
        print("✓ Forklift added")
    else:
        print("✗ Forklift FBX not found (may need to unzip Loader.zip)")


def setup_render_settings():
    """Configure render settings for quality output."""

    scene = bpy.context.scene

    # Use Eevee for fast rendering (good for synthetic data)
    scene.render.engine = 'BLENDER_EEVEE'

    # Resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100

    # Output settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # Eevee settings for quality
    if hasattr(scene.eevee, 'taa_render_samples'):
        scene.eevee.taa_render_samples = 64

    # Enable ambient occlusion
    if hasattr(scene.eevee, 'use_gtao'):
        scene.eevee.use_gtao = True

    # Color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'

    print("✓ Render settings configured")


def setup_world():
    """Set up world/environment lighting."""

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes

    # Get or create background node
    bg_node = nodes.get("Background")
    if bg_node is None:
        bg_node = nodes.new("ShaderNodeBackground")

    # Set to dark gray (indoor warehouse)
    bg_node.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)  # Dark gray
    bg_node.inputs[1].default_value = 0.5  # Low strength

    print("✓ World environment setup complete")


def main():
    """Main function to set up the warehouse scene."""

    print("\n" + "="*50)
    print("HAWKEYE Warehouse Scene Setup")
    print("="*50 + "\n")

    # Clear default objects
    clear_scene()

    # Set up environment
    setup_world()

    # Set up lighting
    setup_lighting()

    # Set up drone camera
    setup_camera_drone()

    # Configure render settings
    setup_render_settings()

    # Note: The warehouse model should already be imported
    # If not, uncomment below:
    # warehouse_fbx = os.path.join(ASSETS_DIR, "warehouse-fbx-model-free", "source", "Warehouse.fbx")
    # import_fbx(warehouse_fbx)

    print("\n" + "="*50)
    print("Scene setup complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Press Numpad 0 to view through drone camera")
    print("2. Press F12 (or fn+F12) to render")
    print("3. Adjust camera position if needed")
    print("4. Add more assets (shelves, boxes) as needed")


# Run the script
if __name__ == "__main__":
    main()
