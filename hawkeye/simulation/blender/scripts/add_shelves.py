"""
HAWKEYE - Add Parallel Shelf Rows
Run this in Blender to add shelving rows to your warehouse.

How to run:
1. Open Blender with your warehouse scene
2. Go to Scripting workspace
3. Open this file
4. Click Run Script
"""

import bpy
import os
import math

# Asset path - using absolute path
SHELF_FILE = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/hawkeye/simulation/blender/assets/source/Metalshelves.blend"

print(f"\n{'='*50}")
print("HAWKEYE - Adding Shelf Rows")
print(f"{'='*50}\n")


def import_shelf():
    """Import shelf from blend file and return the object."""

    if not os.path.exists(SHELF_FILE):
        print(f"ERROR: Shelf file not found: {SHELF_FILE}")
        print("Creating placeholder shelf instead...")

        # Create a placeholder shelf (simple box)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1.5))
        shelf = bpy.context.active_object
        shelf.scale = (2, 0.5, 3)
        shelf.name = "Shelf_Placeholder"
        return shelf

    # Load objects from blend file
    with bpy.data.libraries.load(SHELF_FILE, link=False) as (data_from, data_to):
        # Import all objects from the file
        data_to.objects = data_from.objects
        print(f"Found objects: {data_from.objects}")

    # Link imported objects to scene and find the main shelf
    imported_shelf = None
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported_shelf = obj
            print(f"Imported: {obj.name}")

    return imported_shelf


def create_shelf_rows():
    """Create two parallel rows of shelves."""

    # Configuration - ADJUST THESE VALUES if shelves are in wrong place
    # ================================================================

    # Where to start placing shelves (X, Y, Z)
    # Adjust these based on your warehouse interior
    START_X = -8          # Starting X position
    START_Y_ROW1 = -3     # Y position for first row
    START_Y_ROW2 = 3      # Y position for second row (parallel)
    START_Z = 0           # Floor level

    # Shelf arrangement
    NUM_SHELVES_PER_ROW = 4    # How many shelves in each row
    SHELF_SPACING = 4          # Distance between shelves in a row
    SHELF_SCALE = 1.0          # Scale factor (adjust if too big/small)
    SHELF_ROTATION_Z = 0       # Rotation in degrees (0 = facing Y axis)

    # ================================================================

    print(f"\nConfiguration:")
    print(f"  Start position: ({START_X}, {START_Y_ROW1}, {START_Z})")
    print(f"  Shelves per row: {NUM_SHELVES_PER_ROW}")
    print(f"  Spacing: {SHELF_SPACING}m")
    print(f"  Scale: {SHELF_SCALE}")

    # Import the first shelf
    print("\nImporting shelf template...")
    original_shelf = import_shelf()

    if original_shelf is None:
        print("ERROR: Could not import shelf!")
        return

    # Set initial scale
    original_shelf.scale = (SHELF_SCALE, SHELF_SCALE, SHELF_SCALE)

    # Create Row 1
    print(f"\nCreating Row 1 ({NUM_SHELVES_PER_ROW} shelves)...")
    row1_shelves = []

    for i in range(NUM_SHELVES_PER_ROW):
        if i == 0:
            # Use the original for first position
            shelf = original_shelf
        else:
            # Duplicate for rest
            bpy.ops.object.select_all(action='DESELECT')
            original_shelf.select_set(True)
            bpy.context.view_layer.objects.active = original_shelf
            bpy.ops.object.duplicate()
            shelf = bpy.context.active_object

        # Position
        shelf.location = (
            START_X + (i * SHELF_SPACING),
            START_Y_ROW1,
            START_Z
        )
        shelf.rotation_euler = (0, 0, math.radians(SHELF_ROTATION_Z))
        shelf.name = f"Shelf_Row1_{i+1}"
        row1_shelves.append(shelf)
        print(f"  Placed {shelf.name} at ({shelf.location.x:.1f}, {shelf.location.y:.1f}, {shelf.location.z:.1f})")

    # Create Row 2 (parallel to Row 1)
    print(f"\nCreating Row 2 ({NUM_SHELVES_PER_ROW} shelves)...")
    row2_shelves = []

    for i in range(NUM_SHELVES_PER_ROW):
        # Duplicate from original
        bpy.ops.object.select_all(action='DESELECT')
        original_shelf.select_set(True)
        bpy.context.view_layer.objects.active = original_shelf
        bpy.ops.object.duplicate()
        shelf = bpy.context.active_object

        # Position (same X, different Y)
        shelf.location = (
            START_X + (i * SHELF_SPACING),
            START_Y_ROW2,
            START_Z
        )
        # Rotate 180 degrees so shelves face the aisle
        shelf.rotation_euler = (0, 0, math.radians(SHELF_ROTATION_Z + 180))
        shelf.name = f"Shelf_Row2_{i+1}"
        row2_shelves.append(shelf)
        print(f"  Placed {shelf.name} at ({shelf.location.x:.1f}, {shelf.location.y:.1f}, {shelf.location.z:.1f})")

    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    print(f"\n{'='*50}")
    print(f"DONE! Added {NUM_SHELVES_PER_ROW * 2} shelves in 2 rows")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("1. Press Numpad 0 to check camera view")
    print("2. If shelves are in wrong position, edit the script values:")
    print("   - START_X, START_Y_ROW1, START_Y_ROW2")
    print("   - SHELF_SPACING, SHELF_SCALE")
    print("3. Delete shelves and run again if needed")


# Run
if __name__ == "__main__":
    create_shelf_rows()
