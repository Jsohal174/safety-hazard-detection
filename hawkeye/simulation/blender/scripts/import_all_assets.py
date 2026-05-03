"""
HAWKEYE - Import All Assets (Spaced Out)
Imports all available assets and places them spaced apart so you can move them.

How to run:
1. Open Blender with your warehouse scene
2. Go to Scripting workspace
3. Open this file
4. Click Run Script
"""

import bpy
import os

# Base path
ASSETS_DIR = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/hawkeye/simulation/blender/assets"

# All assets to import with their positions (spaced out)
ASSETS = [
    # (file_path, name, position_x, position_y, position_z, scale)

    # Cardboard Boxes - Option 1
    ("cardboard-boxes/source/Boxes_group_Y.fbx", "Boxes_Group", -15, -15, 0, 0.01),

    # Cardboard Boxes - Option 2
    ("cardboard-boxes-2/source/Cardboard.fbx", "Cardboard_Box", -10, -15, 0, 0.01),

    # PSX Style Boxes
    ("psx-style-cardboard-boxes/source/Cardboard Boxes.fbx", "PSX_Boxes", -5, -15, 0, 0.01),

    # Pallet with crates
    ("pallet/source/Crates_Pallet_Unwrapped.fbx", "Pallet_Crates", 0, -15, 0, 0.01),

    # Pallets
    ("pallets/source/pallets.fbx", "Pallets", 5, -15, 0, 0.01),

    # Pallet Jack
    ("pallet-jack/source/pallet jack.fbx", "Pallet_Jack", 10, -15, 0, 0.01),

    # Low Poly Pallet Jack
    ("low-poly-pallet-jack/source/jackylow.fbx", "Pallet_Jack_LowPoly", 15, -15, 0, 0.01),

    # Hard Hat (GLB)
    ("hard_hat.glb", "Hard_Hat", -15, -20, 0, 1.0),

    # Safety Vest (GLB)
    ("safety-vest-pose/source/model.glb", "Safety_Vest", -10, -20, 0, 1.0),

    # Human Figure (GLB)
    ("big-brother-ggbintm-lower-poly/source/111e.glb", "Human_Figure", -5, -20, 0, 1.0),
]

print(f"\n{'='*50}")
print("HAWKEYE - Importing All Assets")
print(f"{'='*50}\n")


def import_fbx(filepath, name, location, scale):
    """Import FBX file."""
    if not os.path.exists(filepath):
        print(f"  ✗ Not found: {filepath}")
        return None

    # Get objects before import
    before = set(bpy.data.objects.keys())

    try:
        bpy.ops.import_scene.fbx(filepath=filepath)

        # Get newly imported objects
        after = set(bpy.data.objects.keys())
        new_objects = list(after - before)

        # Move and scale all imported objects
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.location = location
                obj.scale = (scale, scale, scale)

        print(f"  ✓ Imported: {name} ({len(new_objects)} objects)")
        return new_objects

    except Exception as e:
        print(f"  ✗ Error importing {name}: {e}")
        return None


def import_glb(filepath, name, location, scale):
    """Import GLB/GLTF file."""
    if not os.path.exists(filepath):
        print(f"  ✗ Not found: {filepath}")
        return None

    before = set(bpy.data.objects.keys())

    try:
        bpy.ops.import_scene.gltf(filepath=filepath)

        after = set(bpy.data.objects.keys())
        new_objects = list(after - before)

        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.location = location
                obj.scale = (scale, scale, scale)

        print(f"  ✓ Imported: {name} ({len(new_objects)} objects)")
        return new_objects

    except Exception as e:
        print(f"  ✗ Error importing {name}: {e}")
        return None


def main():
    """Import all assets."""

    print("Importing assets...\n")

    for asset in ASSETS:
        rel_path, name, x, y, z, scale = asset
        filepath = os.path.join(ASSETS_DIR, rel_path)
        location = (x, y, z)

        if filepath.endswith('.fbx'):
            import_fbx(filepath, name, location, scale)
        elif filepath.endswith('.glb') or filepath.endswith('.gltf'):
            import_glb(filepath, name, location, scale)

    print(f"\n{'='*50}")
    print("DONE! Assets imported and spaced out.")
    print(f"{'='*50}")
    print("\nAssets are placed OUTSIDE the warehouse at Y = -15 to -20")
    print("Use G to move them, S to scale, Shift+D to duplicate")
    print("\nTip: Press Numpad 7 for top view to see all assets")


if __name__ == "__main__":
    main()
