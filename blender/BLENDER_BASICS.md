# Blender Navigation Basics

Quick reference for building warehouse scenes.

---

## 1. Viewport Navigation (Moving Your View)

| Action | Trackpad (MacBook) | Mouse |
|--------|-------------------|-------|
| **Orbit** (rotate view) | Two-finger drag | Middle mouse drag |
| **Pan** (slide view) | Shift + two-finger drag | Shift + middle mouse |
| **Zoom** | Pinch or two-finger scroll | Scroll wheel |
| **Reset view** | `Home` key or `View` → `Frame All` | Same |
| **Focus on selected** | Numpad `.` or `View` → `Frame Selected` | Same |

**Tip:** If you don't have a numpad, go to `Blender` → `Preferences` → `Input` → check **"Emulate Numpad"**

---

## 2. Selecting Objects

| Action | Shortcut |
|--------|----------|
| **Select** | Left click |
| **Add to selection** | Shift + left click |
| **Select all** | `A` |
| **Deselect all** | `Alt + A` |
| **Box select** | `B`, then drag |

---

## 3. Transforming Objects (Move, Rotate, Scale)

| Action | Shortcut | Then... |
|--------|----------|---------|
| **Move** | `G` | Move mouse, click to confirm |
| **Rotate** | `R` | Move mouse, click to confirm |
| **Scale** | `S` | Move mouse, click to confirm |
| **Cancel** | `Esc` or right-click | Cancels transform |

**Constrain to axis:**
- After pressing `G`, `R`, or `S`, press `X`, `Y`, or `Z` to lock to that axis
- Example: `G` → `Z` = move only up/down

**Precise values:**
- Type a number after the shortcut
- Example: `S` → `2` → `Enter` = scale 2x

---

## 4. Adding Objects

| Action | Shortcut |
|--------|----------|
| **Add menu** | `Shift + A` |
| **Add mesh** | `Shift + A` → `Mesh` → (Cube, Plane, etc.) |
| **Add camera** | `Shift + A` → `Camera` |
| **Add light** | `Shift + A` → `Light` |

**For BlenderKit assets:** Click and drag from the asset panel into your scene.

---

## 5. Deleting Objects

| Action | Shortcut |
|--------|----------|
| **Delete** | `X` → confirm |
| **Quick delete** | `Delete` key |

---

## 6. Camera Controls

| Action | Shortcut |
|--------|----------|
| **View through camera** | Numpad `0` or `View` → `Cameras` → `Active Camera` |
| **Align camera to current view** | `Ctrl + Alt + Numpad 0` |
| **Move camera** | Select camera → `G` |
| **Rotate camera** | Select camera → `R` |

**Quick camera setup:**
1. Navigate to the view you want
2. Press `Ctrl + Alt + Numpad 0` (camera snaps to your view)

---

## 7. Render & Save

| Action | How |
|--------|-----|
| **Quick render preview** | `F12` |
| **Render animation** | `Ctrl + F12` |
| **Save rendered image** | After render: `Image` → `Save As` |
| **Save project** | `⌘S` |
| **Save as new file** | `⌘ Shift S` |

---

## 8. Useful Panels

| Panel | Toggle | Purpose |
|-------|--------|---------|
| **Side panel** | `N` | Properties, BlenderKit, transforms |
| **Tool panel** | `T` | Tools (move, rotate, scale, etc.) |
| **Outliner** | Top right | Shows all objects in scene |
| **Properties** | Bottom right | Object settings, materials, render |

---

## 9. View Modes

| View | Shortcut |
|------|----------|
| **Front** | Numpad `1` |
| **Back** | `Ctrl + Numpad 1` |
| **Right** | Numpad `3` |
| **Left** | `Ctrl + Numpad 3` |
| **Top** | Numpad `7` |
| **Bottom** | `Ctrl + Numpad 7` |
| **Toggle perspective/orthographic** | Numpad `5` |

---

## 10. Viewport Shading (How Objects Look)

Press `Z` to open shading pie menu, or use icons in top-right of viewport:

| Mode | Use |
|------|-----|
| **Wireframe** | See through objects |
| **Solid** | Default, fast |
| **Material Preview** | See textures/colors |
| **Rendered** | Final look (slower) |

---

## BlenderKit Quick Start

1. **Open side panel:** Press `N`
2. **Find BlenderKit tab:** Click "BlenderKit" in side panel
3. **Login:** Click login button, sign in at blenderkit.com
4. **Search assets:** Use search bar at top of viewport
5. **Download:** Click and drag free assets into scene
6. **Filter free only:** Look for assets WITHOUT lock icon

### Useful BlenderKit Searches for Warehouse Project
- `warehouse`
- `industrial shelf`
- `pallet`
- `forklift`
- `cardboard box`
- `concrete floor`
- `industrial light`
- `safety cone`
- `hard hat`
- `worker` or `human`

---

## Practice Exercise

1. **Delete default cube:** Click it → `X` → Delete
2. **Add floor:** `Shift + A` → `Mesh` → `Plane`
3. **Scale floor:** `S` → `20` → `Enter`
4. **Search BlenderKit:** Type "industrial shelf"
5. **Add shelf:** Drag free shelf into scene
6. **Position it:** `G` → move mouse → click
7. **Duplicate:** `Shift + D` → move → click
8. **Add camera:** `Shift + A` → `Camera`
9. **Position camera:** Navigate to view → `Ctrl + Alt + Numpad 0`
10. **Render:** `F12`
11. **Save:** `⌘S` → `blender/scenes/warehouse_test.blend`

---

## Keyboard Shortcut Cheat Sheet

```
NAVIGATION
  Orbit .............. Two-finger drag / MMB
  Pan ................ Shift + two-finger / Shift + MMB
  Zoom ............... Pinch / Scroll
  Focus selected ..... Numpad . / View → Frame Selected

SELECTION
  Select ............. Left click
  Add to selection ... Shift + click
  Select all ......... A
  Deselect all ....... Alt + A
  Box select ......... B

TRANSFORM
  Move ............... G
  Rotate ............. R
  Scale .............. S
  Lock to X axis ..... (after G/R/S) X
  Lock to Y axis ..... (after G/R/S) Y
  Lock to Z axis ..... (after G/R/S) Z
  Cancel ............. Esc / Right-click
  Confirm ............ Left-click / Enter

OBJECTS
  Add object ......... Shift + A
  Delete ............. X
  Duplicate .......... Shift + D
  Hide ............... H
  Unhide all ......... Alt + H

CAMERA
  Camera view ........ Numpad 0
  Align camera ....... Ctrl + Alt + Numpad 0

RENDER
  Render image ....... F12

SAVE
  Save ............... ⌘S
  Save as ............ ⌘ Shift S

PANELS
  Side panel ......... N
  Tool panel ......... T
  Shading menu ....... Z
```
