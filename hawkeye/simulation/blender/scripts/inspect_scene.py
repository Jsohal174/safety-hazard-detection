"""
HAWKEYE Scene Inspector
=======================
Run this inside Blender on your warehouse scene.
It scans all objects and outputs the layout info needed
for the drone render script.

Usage: Open in Blender Scripting tab → Run Script
Output: Prints to Blender's System Console (Window → Toggle System Console on Windows,
        or check the terminal if you launched Blender from command line)
        Also saves to a text file next to the .blend file.
"""

import bpy
import os
import json
from mathutils import Vector
from collections import defaultdict

print("\n" + "=" * 60)
print("  HAWKEYE Scene Inspector")
print("=" * 60)

objects = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.visible_get():
        bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_x = min(v.x for v in bbox_world)
        max_x = max(v.x for v in bbox_world)
        min_y = min(v.y for v in bbox_world)
        max_y = max(v.y for v in bbox_world)
        min_z = min(v.z for v in bbox_world)
        max_z = max(v.z for v in bbox_world)

        objects.append({
            "name": obj.name,
            "loc": [round(obj.location.x, 2), round(obj.location.y, 2), round(obj.location.z, 2)],
            "min": [round(min_x, 2), round(min_y, 2), round(min_z, 2)],
            "max": [round(max_x, 2), round(max_y, 2), round(max_z, 2)],
            "height": round(max_z - min_z, 2),
        })

# Find scene bounds
if objects:
    scene_min_x = min(o["min"][0] for o in objects)
    scene_max_x = max(o["max"][0] for o in objects)
    scene_min_y = min(o["min"][1] for o in objects)
    scene_max_y = max(o["max"][1] for o in objects)
    scene_max_z = max(o["max"][2] for o in objects)

    print(f"\nScene bounds:")
    print(f"  X: {scene_min_x:.1f} to {scene_max_x:.1f} (width: {scene_max_x - scene_min_x:.1f}m)")
    print(f"  Y: {scene_min_y:.1f} to {scene_max_y:.1f} (depth: {scene_max_y - scene_min_y:.1f}m)")
    print(f"  Z: 0 to {scene_max_z:.1f} (height: {scene_max_z:.1f}m)")
    print(f"  Total objects: {len(objects)}")

# Group objects by approximate X position to find rack rows
# (racks in the same row will have similar X values)
# Find tall objects (likely racks/shelves, height > 1m)
racks = [o for o in objects if o["height"] > 1.0]
other = [o for o in objects if o["height"] <= 1.0]

print(f"\n  Rack-like objects (height > 1m): {len(racks)}")
print(f"  Other objects: {len(other)}")

# Cluster racks by X position (within 0.5m = same row)
if racks:
    racks_sorted = sorted(racks, key=lambda o: o["loc"][0])

    rows = []
    current_row = [racks_sorted[0]]

    for rack in racks_sorted[1:]:
        if abs(rack["loc"][0] - current_row[-1]["loc"][0]) < 1.5:
            current_row.append(rack)
        else:
            rows.append(current_row)
            current_row = [rack]
    rows.append(current_row)

    print(f"\n  Detected {len(rows)} rack rows (by X position):")

    row_centers_x = []
    for i, row in enumerate(rows):
        xs = [o["loc"][0] for o in row]
        ys = [o["loc"][1] for o in row]
        min_y = min(o["min"][1] for o in row)
        max_y = max(o["max"][1] for o in row)
        avg_x = sum(xs) / len(xs)
        row_centers_x.append(avg_x)
        print(f"    Row {i+1}: X={avg_x:.1f}, Y range=[{min_y:.1f} to {max_y:.1f}], {len(row)} objects")

    # Find aisles (gaps between rows)
    print(f"\n  Aisle centers (between rack rows):")
    aisle_centers = []
    for i in range(len(row_centers_x) - 1):
        aisle_x = (row_centers_x[i] + row_centers_x[i+1]) / 2
        gap = row_centers_x[i+1] - row_centers_x[i]
        aisle_type = "WIDE (forklift)" if gap > 3.0 else "narrow"
        aisle_centers.append({
            "x": round(aisle_x, 2),
            "gap": round(gap, 2),
            "type": aisle_type,
        })
        print(f"    Aisle {i+1}: X={aisle_x:.1f}, gap={gap:.1f}m ({aisle_type})")

# Save results to JSON
output = {
    "scene_bounds": {
        "x": [round(scene_min_x, 2), round(scene_max_x, 2)],
        "y": [round(scene_min_y, 2), round(scene_max_y, 2)],
        "z_max": round(scene_max_z, 2),
    },
    "total_objects": len(objects),
    "rack_rows": len(rows) if racks else 0,
    "row_centers_x": [round(x, 2) for x in row_centers_x] if racks else [],
    "aisles": aisle_centers if racks else [],
    "all_objects": objects,
}

# Save next to blend file or to project
save_path = os.path.join(
    os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "/tmp",
    "scene_layout.json"
)
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Layout saved to: {save_path}")
print("=" * 60)

# Also create a Blender text block so you can see it without console
text_name = "scene_layout_output"
if text_name in bpy.data.texts:
    bpy.data.texts.remove(bpy.data.texts[text_name])
txt = bpy.data.texts.new(text_name)
txt.write(json.dumps(output, indent=2))
print(f"  Also saved as Blender text block: '{text_name}'")
print("  (Check Scripting tab → text dropdown → scene_layout_output)")
