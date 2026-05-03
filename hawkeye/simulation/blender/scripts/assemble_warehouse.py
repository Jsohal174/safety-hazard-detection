"""
HAWKEYE Warehouse Assembly Script
==================================
Creates the complete warehouse scene for synthetic data generation.
Builds structure, racks, shelves, exits, spawn zones, flight paths,
lighting, and drone camera — all from scratch.

Usage (headless):
    blender --background --python assemble_warehouse.py

Usage (GUI):
    Open Blender > Scripting tab > Open this file > Run Script

Output:
    hawkeye/simulation/blender/scenes/warehouse_assembled.blend
"""

import bpy
import math
import os
import sys

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
SCENES_DIR = os.path.join(SCRIPT_DIR, "..", "scenes")
OUTPUT_FILE = os.path.join(SCENES_DIR, "warehouse_assembled.blend")

# Asset paths
SHELF_BLEND = os.path.join(ASSETS_DIR, "source", "Metalshelves.blend")
WAREHOUSE_FBX = os.path.join(ASSETS_DIR, "warehouse-fbx-model-free", "source", "Warehouse.fbx")
FORKLIFT_FBX = os.path.join(ASSETS_DIR, "forklift-low-poly", "source", "Loader_car.fbx")
BOXES_FBX = os.path.join(ASSETS_DIR, "cardboard-boxes", "source", "Boxes_group_Y.fbx")
PALLET_FBX = os.path.join(ASSETS_DIR, "pallet", "source", "Crates_Pallet_Unwrapped.fbx")
HUMAN_GLB = os.path.join(ASSETS_DIR, "big-brother-ggbintm-lower-poly", "source", "111e.glb")
HARD_HAT_GLB = os.path.join(ASSETS_DIR, "hard_hat.glb")

# ═══════════════════════════════════════════════════════════════════════
# WAREHOUSE LAYOUT CONSTANTS (meters)
# ═══════════════════════════════════════════════════════════════════════

# Overall dimensions
WH_WIDTH = 50.0    # X axis
WH_DEPTH = 30.0    # Y axis
WH_HEIGHT = 8.0    # Z axis
WALL_THICK = 0.3

# Y layout (north = +Y, south = -Y)
LOADING_DOCK_Y = (11.0, 14.5)        # Open loading area (north)
TALL_RACK_A_Y = 9.5                   # North tall rack row center
TALL_RACK_B_Y = 5.0                   # South tall rack row center
WIDE_AISLE_Y = (5.6, 8.9)            # Forklift aisle (~3.3m)
CROSS_AISLE_Y = (2.5, 4.4)           # Pedestrian cross aisle (~1.9m)
SOUTH_EXIT_Y = -12.0                  # South exit corridor

# Forklift zone tall racks (6 per row)
TALL_RACK_X = [-15, -9, -3, 3, 9, 15]
TALL_RACK_DIMS = (2.4, 1.2, 5.0)     # width, depth, height

# Human zone shelves: 5 columns with 4 narrow aisles between
SHELF_COLS_X = [-7.0, -3.5, 0.0, 3.5, 7.0]
SHELF_ROWS_Y = [0.5, -2.5, -5.5, -8.5]   # 4 shelf units per column
SHELF_DIMS = (1.2, 1.8, 2.2)              # width, depth, height

# Narrow aisle centers (between shelf columns)
NARROW_AISLE_X = [-5.25, -1.75, 1.75, 5.25]

# Exit doors (on east/west walls)
EXITS = [
    ("Exit_MidLeft",      (-24.85, 3.5, 0)),
    ("Exit_MidRight",     (24.85, 3.5, 0)),
    ("Exit_SouthLeft",    (-24.85, -11.0, 0)),
    ("Exit_SouthRight",   (24.85, -11.0, 0)),
]

# Drone camera
DRONE_HEIGHT = 2.5
DRONE_PITCH_DEG = 55


# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def clear_scene():
    """Remove all objects, collections, and orphan data."""
    # Select and delete all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Remove non-default collections
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)

    # Purge orphan data
    for attr in ("meshes", "materials", "lights", "cameras", "textures", "images"):
        data_block = getattr(bpy.data, attr)
        for block in list(data_block):
            if block.users == 0:
                data_block.remove(block)

    print("[OK] Scene cleared")


def get_or_create_collection(name):
    """Get or create a collection linked to the scene root."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def make_material(name, color, roughness=0.8, metallic=0.0):
    """Create a Principled BSDF material."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat


def add_box(name, location, size, material=None, collection=None):
    """Create a box mesh. size = (width, depth, height) in meters."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    # Build a unit cube mesh using bmesh
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()

    obj.scale = size
    obj.location = (location[0], location[1], location[2] + size[2] / 2.0)

    if material:
        obj.data.materials.append(material)

    target = collection if collection else bpy.context.scene.collection
    target.objects.link(obj)
    return obj


def add_empty(name, location, display_type="CUBE", display_size=1.0,
              scale=None, props=None, collection=None):
    """Create an empty with optional custom properties."""
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = display_size
    empty.location = location
    if scale:
        empty.scale = scale
    if props:
        for k, v in props.items():
            empty[k] = v

    target = collection if collection else bpy.context.scene.collection
    target.objects.link(empty)
    return empty


def import_fbx(filepath, location=(0, 0, 0), scale=1.0, rotation_z=0):
    """Import FBX, reposition, and return new object names."""
    if not os.path.exists(filepath):
        print(f"  [SKIP] Not found: {filepath}")
        return []
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=filepath)
    after = set(bpy.data.objects.keys())
    new_names = list(after - before)
    for n in new_names:
        obj = bpy.data.objects[n]
        obj.location = location
        obj.scale = (scale, scale, scale)
        obj.rotation_euler.z = math.radians(rotation_z)
    return new_names


def import_glb(filepath, location=(0, 0, 0), scale=1.0, rotation_z=0):
    """Import GLB/glTF, reposition, and return new object names."""
    if not os.path.exists(filepath):
        print(f"  [SKIP] Not found: {filepath}")
        return []
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = set(bpy.data.objects.keys())
    new_names = list(after - before)
    for n in new_names:
        obj = bpy.data.objects[n]
        obj.location = location
        obj.scale = (scale, scale, scale)
        obj.rotation_euler.z = math.radians(rotation_z)
    return new_names


# ═══════════════════════════════════════════════════════════════════════
# 1. WAREHOUSE STRUCTURE (floor, walls, ceiling)
# ═══════════════════════════════════════════════════════════════════════

def build_structure():
    col = get_or_create_collection("Structure")
    mat_floor = make_material("Concrete_Floor", (0.50, 0.50, 0.47), 0.95)
    mat_wall = make_material("Wall_Metal", (0.65, 0.65, 0.62), 0.7, metallic=0.3)
    mat_ceiling = make_material("Ceiling_Panel", (0.48, 0.48, 0.46), 0.85)

    # Floor
    add_box("Floor", (0, 0, -0.1), (WH_WIDTH, WH_DEPTH, 0.1), mat_floor, col)

    # Ceiling
    add_box("Ceiling", (0, 0, WH_HEIGHT), (WH_WIDTH, WH_DEPTH, 0.1), mat_ceiling, col)

    hw = WH_WIDTH / 2
    hd = WH_DEPTH / 2
    t = WALL_THICK

    # Walls
    add_box("Wall_North", (0, hd, 0), (WH_WIDTH, t, WH_HEIGHT), mat_wall, col)
    add_box("Wall_South", (0, -hd, 0), (WH_WIDTH, t, WH_HEIGHT), mat_wall, col)
    add_box("Wall_East", (hw, 0, 0), (t, WH_DEPTH, WH_HEIGHT), mat_wall, col)
    add_box("Wall_West", (-hw, 0, 0), (t, WH_DEPTH, WH_HEIGHT), mat_wall, col)

    # Loading dock platform (slightly raised area at north end)
    mat_dock = make_material("Dock_Concrete", (0.4, 0.4, 0.38), 0.9)
    dock_cy = (LOADING_DOCK_Y[0] + LOADING_DOCK_Y[1]) / 2
    dock_depth = LOADING_DOCK_Y[1] - LOADING_DOCK_Y[0]
    add_box("LoadingDock_Platform", (0, dock_cy, 0), (WH_WIDTH * 0.8, dock_depth, 0.15), mat_dock, col)

    print(f"[OK] Structure: {WH_WIDTH}m x {WH_DEPTH}m x {WH_HEIGHT}m")


# ═══════════════════════════════════════════════════════════════════════
# 2. TALL RACKS (forklift zone)
# ═══════════════════════════════════════════════════════════════════════

def build_tall_racks():
    col = get_or_create_collection("Racks_Forklift")
    mat_frame = make_material("Rack_Orange", (0.85, 0.45, 0.08), 0.5, metallic=0.6)
    mat_beam = make_material("Rack_Blue", (0.15, 0.25, 0.55), 0.5, metallic=0.6)

    w, d, h = TALL_RACK_DIMS
    count = 0

    for i, x in enumerate(TALL_RACK_X):
        # Row A (north)
        rack = add_box(f"Rack_A_{i+1}", (x, TALL_RACK_A_Y, 0), (w, d, h), mat_frame, col)

        # Horizontal beams (3 levels)
        for level in range(1, 4):
            beam_z = level * (h / 4)
            add_box(f"Rack_A_{i+1}_Beam_{level}", (x, TALL_RACK_A_Y, beam_z),
                    (w * 0.95, d * 0.8, 0.08), mat_beam, col)

        # Row B (south)
        add_box(f"Rack_B_{i+1}", (x, TALL_RACK_B_Y, 0), (w, d, h), mat_frame, col)
        for level in range(1, 4):
            beam_z = level * (h / 4)
            add_box(f"Rack_B_{i+1}_Beam_{level}", (x, TALL_RACK_B_Y, beam_z),
                    (w * 0.95, d * 0.8, 0.08), mat_beam, col)

        count += 2

    print(f"[OK] Tall racks: {count} units (forklift zone)")


# ═══════════════════════════════════════════════════════════════════════
# 3. HUMAN-ZONE SHELVES (5 columns x 4 rows)
# ═══════════════════════════════════════════════════════════════════════

def build_human_shelves():
    col = get_or_create_collection("Shelves_Human")
    mat_shelf = make_material("Shelf_Steel", (0.5, 0.5, 0.52), 0.4, metallic=0.7)
    mat_shelf_board = make_material("Shelf_Board", (0.6, 0.55, 0.45), 0.8)

    w, d, h = SHELF_DIMS
    count = 0

    for ci, cx in enumerate(SHELF_COLS_X):
        for ri, ry in enumerate(SHELF_ROWS_Y):
            name = f"Shelf_C{ci+1}_R{ri+1}"

            # Main frame
            add_box(name, (cx, ry, 0), (w, d, h), mat_shelf, col)

            # Shelf boards (4 levels)
            for level in range(1, 5):
                board_z = level * (h / 5)
                add_box(f"{name}_Board_{level}", (cx, ry, board_z),
                        (w * 0.9, d * 0.85, 0.03), mat_shelf_board, col)

            count += 1

    print(f"[OK] Human shelves: {count} units (5 cols x 4 rows)")


# ═══════════════════════════════════════════════════════════════════════
# 4. EXIT DOORS
# ═══════════════════════════════════════════════════════════════════════

def build_exits():
    col = get_or_create_collection("Exits")
    mat_door = make_material("Exit_Door_Red", (0.75, 0.12, 0.1), 0.4)
    mat_frame = make_material("Door_Frame", (0.3, 0.3, 0.3), 0.6, metallic=0.5)
    mat_sign = make_material("Exit_Sign_Green", (0.1, 0.8, 0.2), 0.3)

    door_w = 2.0
    door_h = 2.8
    door_d = 0.12

    for name, pos in EXITS:
        # Door frame
        add_box(f"{name}_Frame", pos, (door_d + 0.1, door_w + 0.2, door_h + 0.1), mat_frame, col)
        # Door panel
        add_box(name, pos, (door_d, door_w, door_h), mat_door, col)
        # EXIT sign above
        add_box(f"{name}_Sign", (pos[0], pos[1], door_h + 0.3), (0.05, 0.6, 0.2), mat_sign, col)

    print(f"[OK] Exit doors: {len(EXITS)}")


# ═══════════════════════════════════════════════════════════════════════
# 5. FLOOR MARKINGS (safety lines)
# ═══════════════════════════════════════════════════════════════════════

def build_floor_markings():
    col = get_or_create_collection("FloorMarkings")
    mat_yellow = make_material("Safety_Yellow", (0.95, 0.85, 0.05), 0.5)
    mat_white = make_material("Safety_White", (0.95, 0.95, 0.95), 0.5)

    line_h = 0.005  # Painted on floor

    # Cross aisle boundary lines
    for i, y in enumerate([CROSS_AISLE_Y[0], CROSS_AISLE_Y[1]]):
        add_box(f"Line_CrossAisle_{i}", (0, y, 0), (30, 0.1, line_h), mat_yellow, col)

    # Forklift zone boundary
    add_box("Line_ForkliftZone_South", (0, WIDE_AISLE_Y[0], 0), (30, 0.1, line_h), mat_yellow, col)
    add_box("Line_ForkliftZone_North", (0, WIDE_AISLE_Y[1], 0), (30, 0.1, line_h), mat_yellow, col)

    # Loading dock edge
    add_box("Line_LoadingDock", (0, LOADING_DOCK_Y[0], 0), (30, 0.15, line_h), mat_yellow, col)

    # Exit door markings (hatched area around each exit)
    for name, pos in EXITS:
        add_box(f"Marking_{name}", (pos[0], pos[1], 0), (1.0, 3.0, line_h), mat_white, col)

    # Narrow aisle center lines
    for i, ax in enumerate(NARROW_AISLE_X):
        add_box(f"Line_NarrowAisle_{i}", (ax, -4.0, 0), (0.08, 12, line_h), mat_white, col)

    print("[OK] Floor markings")


# ═══════════════════════════════════════════════════════════════════════
# 6. SPAWN ZONES (empties with custom properties)
# ═══════════════════════════════════════════════════════════════════════

def build_spawn_zones():
    col = get_or_create_collection("SpawnZones")
    zones = []

    # --- Floor zones (spills, obstacles) ---
    for i, ax in enumerate(NARROW_AISLE_X):
        zones.append((
            f"SpawnZone_Floor_NarrowAisle{i+1}",
            (ax, -4.0, 0.01),
            (1.1, 5.5, 0.05),
            {"zone_type": "floor", "zone_width": 2.2, "zone_depth": 11.0,
             "hazard_classes": "spill,obstacle"}
        ))

    zones.append((
        "SpawnZone_Floor_WideAisle",
        (0, (WIDE_AISLE_Y[0] + WIDE_AISLE_Y[1]) / 2, 0.01),
        (15.0, 1.6, 0.05),
        {"zone_type": "floor", "zone_width": 30.0, "zone_depth": 3.3,
         "hazard_classes": "spill,obstacle"}
    ))

    zones.append((
        "SpawnZone_Floor_CrossAisle",
        (0, (CROSS_AISLE_Y[0] + CROSS_AISLE_Y[1]) / 2, 0.01),
        (15.0, 0.95, 0.05),
        {"zone_type": "floor", "zone_width": 30.0, "zone_depth": 1.9,
         "hazard_classes": "spill,obstacle"}
    ))

    zones.append((
        "SpawnZone_Floor_LoadingDock",
        (0, (LOADING_DOCK_Y[0] + LOADING_DOCK_Y[1]) / 2, 0.01),
        (15.0, 1.75, 0.05),
        {"zone_type": "floor", "zone_width": 30.0, "zone_depth": 3.5,
         "hazard_classes": "spill,obstacle"}
    ))

    # --- Human walkable zones (missing_ppe) ---
    for i, ax in enumerate(NARROW_AISLE_X):
        zones.append((
            f"SpawnZone_Human_NarrowAisle{i+1}",
            (ax, -4.0, 0.01),
            (1.1, 5.5, 0.05),
            {"zone_type": "human", "zone_width": 2.2, "zone_depth": 11.0,
             "hazard_classes": "missing_ppe"}
        ))

    zones.append((
        "SpawnZone_Human_CrossAisle",
        (0, (CROSS_AISLE_Y[0] + CROSS_AISLE_Y[1]) / 2, 0.01),
        (15.0, 0.95, 0.05),
        {"zone_type": "human", "zone_width": 30.0, "zone_depth": 1.9,
         "hazard_classes": "missing_ppe"}
    ))

    # --- Forklift zones ---
    zones.append((
        "SpawnZone_Forklift_WideAisle",
        (0, (WIDE_AISLE_Y[0] + WIDE_AISLE_Y[1]) / 2, 0.01),
        (15.0, 1.6, 0.05),
        {"zone_type": "forklift", "zone_width": 30.0, "zone_depth": 3.3,
         "hazard_classes": "forklift_violation"}
    ))

    zones.append((
        "SpawnZone_Forklift_LoadingDock",
        (0, (LOADING_DOCK_Y[0] + LOADING_DOCK_Y[1]) / 2, 0.01),
        (15.0, 1.75, 0.05),
        {"zone_type": "forklift", "zone_width": 30.0, "zone_depth": 3.5,
         "hazard_classes": "forklift_violation"}
    ))

    # --- Exit zones (blocked_exit) ---
    for name, pos in EXITS:
        zones.append((
            f"SpawnZone_{name}",
            (pos[0], pos[1], 0.01),
            (1.0, 1.5, 0.05),
            {"zone_type": "exit", "zone_width": 2.0, "zone_depth": 3.0,
             "hazard_classes": "blocked_exit"}
        ))

    # --- Rack zones (damaged_rack) ---
    for i, x in enumerate(TALL_RACK_X):
        zones.append((
            f"SpawnZone_Rack_A{i+1}",
            (x, TALL_RACK_A_Y, 0.01),
            (1.2, 0.6, 0.05),
            {"zone_type": "rack", "zone_width": 2.4, "zone_depth": 1.2,
             "hazard_classes": "damaged_rack"}
        ))
        zones.append((
            f"SpawnZone_Rack_B{i+1}",
            (x, TALL_RACK_B_Y, 0.01),
            (1.2, 0.6, 0.05),
            {"zone_type": "rack", "zone_width": 2.4, "zone_depth": 1.2,
             "hazard_classes": "damaged_rack"}
        ))

    for name, loc, scale, props in zones:
        add_empty(name, loc, display_type="CUBE", display_size=1.0,
                  scale=scale, props=props, collection=col)

    print(f"[OK] Spawn zones: {len(zones)}")


# ═══════════════════════════════════════════════════════════════════════
# 7. FLIGHT PATHS (waypoint empties)
# ═══════════════════════════════════════════════════════════════════════

def build_flight_paths():
    col = get_or_create_collection("FlightPaths")
    total = 0

    # Narrow aisle paths (fly down each aisle in Y direction)
    for i, ax in enumerate(NARROW_AISLE_X):
        waypoints_y = [y * 0.5 for y in range(-20, 4, 4)]  # Y from -10 to 1.5, step 2m
        for wi, wy in enumerate(waypoints_y):
            add_empty(
                f"Path_NarrowAisle{i+1}_WP{wi:03d}",
                (ax, wy, DRONE_HEIGHT),
                display_type="ARROWS", display_size=0.3,
                props={
                    "waypoint_index": wi,
                    "path_name": f"Path_NarrowAisle{i+1}",
                    "height": DRONE_HEIGHT,
                    "pitch": DRONE_PITCH_DEG,
                    "yaw_deg": 0.0,
                },
                collection=col,
            )
            total += 1

    # Wide aisle path (forklift zone, fly across in X direction)
    wide_y = (WIDE_AISLE_Y[0] + WIDE_AISLE_Y[1]) / 2
    for wi, x in enumerate(range(-18, 19, 3)):
        add_empty(
            f"Path_WideAisle_WP{wi:03d}",
            (x, wide_y, 3.0),
            display_type="ARROWS", display_size=0.3,
            props={
                "waypoint_index": wi,
                "path_name": "Path_WideAisle",
                "height": 3.0,
                "pitch": 50.0,
                "yaw_deg": 90.0,
            },
            collection=col,
        )
        total += 1

    # Cross aisle path (fly across in X direction)
    cross_y = (CROSS_AISLE_Y[0] + CROSS_AISLE_Y[1]) / 2
    for wi, x in enumerate(range(-22, 23, 3)):
        add_empty(
            f"Path_CrossAisle_WP{wi:03d}",
            (x, cross_y, DRONE_HEIGHT),
            display_type="ARROWS", display_size=0.3,
            props={
                "waypoint_index": wi,
                "path_name": "Path_CrossAisle",
                "height": DRONE_HEIGHT,
                "pitch": float(DRONE_PITCH_DEG),
                "yaw_deg": 90.0,
            },
            collection=col,
        )
        total += 1

    # Loading dock sweep (fly across in X direction, higher altitude)
    dock_y = (LOADING_DOCK_Y[0] + LOADING_DOCK_Y[1]) / 2
    for wi, x in enumerate(range(-18, 19, 4)):
        add_empty(
            f"Path_LoadingDock_WP{wi:03d}",
            (x, dock_y, 3.5),
            display_type="ARROWS", display_size=0.3,
            props={
                "waypoint_index": wi,
                "path_name": "Path_LoadingDock",
                "height": 3.5,
                "pitch": 45.0,
                "yaw_deg": 90.0,
            },
            collection=col,
        )
        total += 1

    print(f"[OK] Flight path waypoints: {total}")


# ═══════════════════════════════════════════════════════════════════════
# 8. PROPS (forklift, boxes, pallets)
# ═══════════════════════════════════════════════════════════════════════

def build_props():
    col = get_or_create_collection("Props")

    # Forklift in loading dock area
    names = import_fbx(FORKLIFT_FBX, location=(5, 12, 0), scale=0.01, rotation_z=180)
    if names:
        for n in names:
            obj = bpy.data.objects.get(n)
            if obj:
                for c in obj.users_collection:
                    c.objects.unlink(obj)
                col.objects.link(obj)
        print(f"  [OK] Forklift imported")
    else:
        # Placeholder forklift
        mat = make_material("Forklift_Yellow", (0.9, 0.7, 0.1), 0.6)
        add_box("Forklift_Placeholder", (5, 12, 0), (1.2, 2.5, 1.8), mat, col)
        print("  [OK] Forklift placeholder created")

    # Some pallets/boxes in the loading dock
    mat_wood = make_material("Pallet_Wood", (0.55, 0.4, 0.25), 0.9)
    mat_box = make_material("Cardboard", (0.65, 0.55, 0.35), 0.85)

    pallet_positions = [(-8, 13, 0), (-4, 12.5, 0), (0, 13.5, 0), (10, 12, 0)]
    for i, pos in enumerate(pallet_positions):
        add_box(f"Pallet_{i+1}", pos, (1.2, 1.0, 0.15), mat_wood, col)
        add_box(f"PalletBox_{i+1}", (pos[0], pos[1], 0.15), (1.0, 0.8, 0.6), mat_box, col)

    # A few scattered boxes on some shelves
    box_positions = [
        (-7.0, 0.5, 2.2), (-3.5, -2.5, 2.2), (0.0, 0.5, 2.2),
        (3.5, -5.5, 2.2), (7.0, -8.5, 2.2),
    ]
    for i, pos in enumerate(box_positions):
        add_box(f"ShelfBox_{i+1}", pos, (0.4, 0.3, 0.3), mat_box, col)

    print("[OK] Props placed")


# ═══════════════════════════════════════════════════════════════════════
# 9. LIGHTING
# ═══════════════════════════════════════════════════════════════════════

def build_lighting():
    col = get_or_create_collection("Lighting")

    # Sun (simulates skylights / high clerestory windows)
    sun_data = bpy.data.lights.new("Sun_Main", "SUN")
    sun_data.energy = 5.0
    sun_data.color = (1.0, 0.95, 0.9)
    sun_data.angle = math.radians(1)
    sun_obj = bpy.data.objects.new("Sun_Main", sun_data)
    sun_obj.location = (0, 0, WH_HEIGHT + 5)
    sun_obj.rotation_euler = (math.radians(50), math.radians(15), 0)
    col.objects.link(sun_obj)

    # Ceiling area lights (industrial high-bay fixtures)
    positions = [
        # Loading dock
        (-10, 12, 7.5), (0, 12, 7.5), (10, 12, 7.5),
        # Forklift zone
        (-10, 7, 7.5), (0, 7, 7.5), (10, 7, 7.5),
        # Cross aisle
        (-10, 3, 7.5), (0, 3, 7.5), (10, 3, 7.5),
        # Human zone north
        (-5, -1, 7.5), (5, -1, 7.5),
        # Human zone south
        (-5, -6, 7.5), (5, -6, 7.5),
        (-5, -10, 7.5), (5, -10, 7.5),
    ]

    for i, pos in enumerate(positions):
        light_data = bpy.data.lights.new(f"HighBay_{i}", "AREA")
        light_data.energy = 800
        light_data.size = 2.0
        light_data.color = (1.0, 0.97, 0.92)
        light_obj = bpy.data.objects.new(f"HighBay_{i}", light_data)
        light_obj.location = pos
        light_obj.rotation_euler = (0, 0, 0)  # Area lights emit along -Z by default (downward)
        col.objects.link(light_obj)

    print(f"[OK] Lighting: 1 sun + {len(positions)} area lights")


# ═══════════════════════════════════════════════════════════════════════
# 10. DRONE CAMERA
# ═══════════════════════════════════════════════════════════════════════

def build_camera():
    cam_data = bpy.data.cameras.new("Drone_Camera")
    cam_data.lens = 35
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100

    cam_obj = bpy.data.objects.new("Drone_Camera", cam_data)
    # Default position: hovering over a narrow aisle looking down at DRONE_PITCH_DEG
    cam_obj.location = (0, -4, DRONE_HEIGHT)

    # Blender camera default: looks down -Z (straight down).
    # rotation_euler.x = 0  -> straight down
    # rotation_euler.x = 90 -> horizontal (looking toward +Y)
    # For DRONE_PITCH_DEG below horizontal: x = 90 - DRONE_PITCH_DEG
    cam_obj.rotation_euler = (math.radians(90 - DRONE_PITCH_DEG), 0, 0)

    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    print(f"[OK] Drone camera: height={DRONE_HEIGHT}m, pitch={DRONE_PITCH_DEG} deg")


# ═══════════════════════════════════════════════════════════════════════
# 11. RENDER / WORLD SETTINGS
# ═══════════════════════════════════════════════════════════════════════

def configure_render():
    scene = bpy.context.scene

    # Engine — Blender 4+/5+ uses BLENDER_EEVEE_NEXT
    if "BLENDER_EEVEE_NEXT" in dir(bpy.types):
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Color management
    scene.view_settings.view_transform = "Filmic"

    # Eevee quality
    eevee = getattr(scene, "eevee", None)
    if eevee:
        for attr, val in [
            ("taa_render_samples", 64),
            ("use_gtao", True),
            ("use_ssr", True),
        ]:
            if hasattr(eevee, attr):
                setattr(eevee, attr, val)

    print("[OK] Render settings: 1920x1080, Eevee, Filmic")


def configure_world():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.04, 0.04, 0.04, 1.0)
        bg.inputs["Strength"].default_value = 0.3

    print("[OK] World: dark indoor environment")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  HAWKEYE — Warehouse Scene Assembly")
    print("=" * 60)

    clear_scene()

    print("\n--- Structure ---")
    build_structure()

    print("\n--- Racks (Forklift Zone) ---")
    build_tall_racks()

    print("\n--- Shelves (Human Zone) ---")
    build_human_shelves()

    print("\n--- Exit Doors ---")
    build_exits()

    print("\n--- Floor Markings ---")
    build_floor_markings()

    print("\n--- Spawn Zones ---")
    build_spawn_zones()

    print("\n--- Flight Paths ---")
    build_flight_paths()

    print("\n--- Props ---")
    build_props()

    print("\n--- Lighting ---")
    build_lighting()

    print("\n--- Drone Camera ---")
    build_camera()

    print("\n--- Render/World ---")
    configure_render()
    configure_world()

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_FILE)

    # Summary
    obj_count = len(bpy.data.objects)
    collections = [c.name for c in bpy.data.collections]
    print("\n" + "=" * 60)
    print(f"  Assembly complete!")
    print(f"  Objects: {obj_count}")
    print(f"  Collections: {', '.join(collections)}")
    print(f"  Saved: {OUTPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
