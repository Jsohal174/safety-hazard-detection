"""
Quick single drone shot from middle of an aisle.
Inspects scene first, then renders 1 image.
"""
import bpy
import math
import os
import json
from mathutils import Vector

SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes")
OUTPUT_DIR = os.path.normpath(os.path.join(SCENES_DIR, "..", "..", "..", "outputs", "renders", "demo"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 50)
print("  Inspecting scene...")
print("=" * 50)

# ── Inspect: find rack rows and aisles ──
objects = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.visible_get():
        try:
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            min_x = min(v.x for v in bbox_world)
            max_x = max(v.x for v in bbox_world)
            min_y = min(v.y for v in bbox_world)
            max_y = max(v.y for v in bbox_world)
            min_z = min(v.z for v in bbox_world)
            max_z = max(v.z for v in bbox_world)
            objects.append({
                "name": obj.name,
                "x": round(obj.location.x, 2),
                "y": round(obj.location.y, 2),
                "min_x": round(min_x, 2), "max_x": round(max_x, 2),
                "min_y": round(min_y, 2), "max_y": round(max_y, 2),
                "height": round(max_z - min_z, 2),
            })
        except:
            pass

print(f"  Total visible mesh objects: {len(objects)}")

# Find scene bounds
if objects:
    scene_min_x = min(o["min_x"] for o in objects)
    scene_max_x = max(o["max_x"] for o in objects)
    scene_min_y = min(o["min_y"] for o in objects)
    scene_max_y = max(o["max_y"] for o in objects)
    print(f"  X: {scene_min_x:.1f} to {scene_max_x:.1f}")
    print(f"  Y: {scene_min_y:.1f} to {scene_max_y:.1f}")

# Find racks (tall objects) and cluster by X
racks = [o for o in objects if o["height"] > 1.0]
print(f"  Rack-like objects: {len(racks)}")

if racks:
    racks_sorted = sorted(racks, key=lambda o: o["x"])
    rows = []
    current_row = [racks_sorted[0]]
    for rack in racks_sorted[1:]:
        if abs(rack["x"] - current_row[-1]["x"]) < 2.0:
            current_row.append(rack)
        else:
            rows.append(current_row)
            current_row = [rack]
    rows.append(current_row)

    row_centers = []
    for i, row in enumerate(rows):
        avg_x = sum(o["x"] for o in row) / len(row)
        min_y = min(o["min_y"] for o in row)
        max_y = max(o["max_y"] for o in row)
        row_centers.append({"x": avg_x, "min_y": min_y, "max_y": max_y, "count": len(row)})
        print(f"  Row {i+1}: X={avg_x:.1f}, Y=[{min_y:.1f}, {max_y:.1f}], {len(row)} objs")

    # Find aisles
    aisles = []
    for i in range(len(row_centers) - 1):
        ax = (row_centers[i]["x"] + row_centers[i+1]["x"]) / 2
        gap = row_centers[i+1]["x"] - row_centers[i]["x"]
        min_y = max(row_centers[i]["min_y"], row_centers[i+1]["min_y"])
        max_y = min(row_centers[i]["max_y"], row_centers[i+1]["max_y"])
        if gap > 1.5:  # only real aisles
            aisles.append({"x": ax, "gap": gap, "min_y": min_y, "max_y": max_y})
            print(f"  Aisle {len(aisles)}: X={ax:.1f}, gap={gap:.1f}m, Y=[{min_y:.1f}, {max_y:.1f}]")

    # Save layout
    layout = {
        "scene_bounds": {"x": [scene_min_x, scene_max_x], "y": [scene_min_y, scene_max_y]},
        "total_objects": len(objects),
        "rows": [{"x": r["x"], "min_y": r["min_y"], "max_y": r["max_y"]} for r in row_centers],
        "aisles": aisles,
    }
    layout_path = os.path.join(SCENES_DIR, "scene_layout.json")
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"  Layout saved: {layout_path}")

    # ── Render one drone shot from the widest aisle ──
    if aisles:
        # Pick the widest aisle
        best = max(aisles, key=lambda a: a["gap"])
        cam_x = best["x"]
        cam_y = (best["min_y"] + best["max_y"]) / 2  # middle of aisle
        cam_z = 3.5  # drone height
        pitch = 55   # degrees looking down

        print(f"\n  Rendering from aisle at X={cam_x:.1f}, Y={cam_y:.1f}, Z={cam_z}")

        # Create/get camera
        cam = bpy.data.cameras.get("DroneCam")
        if not cam:
            cam = bpy.data.cameras.new("DroneCam")
        cam.lens = 28
        cam.clip_start = 0.1
        cam.clip_end = 200

        cam_obj = bpy.data.objects.get("DroneCam")
        if not cam_obj:
            cam_obj = bpy.data.objects.new("DroneCam", cam)
            bpy.context.scene.collection.objects.link(cam_obj)

        cam_obj.location = (cam_x, cam_y, cam_z)
        cam_obj.rotation_euler = (math.radians(90 - pitch), 0, 0)
        bpy.context.scene.camera = cam_obj

        # Fast render settings
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = 960
        scene.render.resolution_y = 540
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"

        eevee = getattr(scene, "eevee", None)
        if eevee and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 16

        filepath = os.path.join(OUTPUT_DIR, "drone_test.png")
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        print(f"\n  DONE! Saved: {filepath}")
    else:
        print("  No aisles found!")
else:
    print("  No rack objects found!")

print("=" * 50)
