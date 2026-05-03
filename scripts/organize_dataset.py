#!/usr/bin/env python3
"""
Reorganize the HAWKEYE dataset into a clean, flat structure.

Copies all 999 images into dataset/images/ with category-prefixed names,
creates unified metadata.jsonl, and saves a reversible mapping file.

Does NOT modify original files.
"""

import json
import shutil
from pathlib import Path

PROJECT = Path("/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection")
OUT_DIR = PROJECT / "dataset"
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Source directories and label files
CATEGORIES = {
    "spill": {
        "image_dir": PROJECT / "outputs/datasets/images/spill",
        "label_file": PROJECT / "outputs/datasets/spill_labels.jsonl",
        "subtype_field": "spill_type",
    },
    "forklift": {
        "image_dir": PROJECT / "outputs/datasets/images/forklift_violation",
        "label_file": PROJECT / "outputs/datasets/forklift_labels.jsonl",
        "subtype_field": "violation_type",
    },
    "stacking": {
        "image_dir": PROJECT / "outputs/datasets/images/improper_stacking",
        "label_file": PROJECT / "outputs/datasets/stacking_labels.jsonl",
        "subtype_field": "stacking_type",
    },
    "obstacle": {
        "image_dir": PROJECT / "outputs/datasets/images/obstacle",
        "label_file": PROJECT / "outputs/datasets/obstacle_labels.jsonl",
        "subtype_field": "obstacle_type",
    },
    "safe": {
        "image_dir": PROJECT / "realistic",
        "label_file": PROJECT / "outputs/datasets/safe_labels.jsonl",
        "subtype_field": None,
    },
}

# Original category names (for old_path traceability)
OLD_CATEGORY_DIRS = {
    "spill": "spill",
    "forklift": "forklift_violation",
    "stacking": "improper_stacking",
    "obstacle": "obstacle",
    "safe": "realistic",
}

all_metadata = []
image_mapping = {}  # old_path -> new_filename
errors = []

for cat_prefix, info in CATEGORIES.items():
    print(f"\n=== Processing {cat_prefix} ===")

    # Load labels into a dict keyed by filename
    labels_by_name = {}
    with open(info["label_file"]) as f:
        for line in f:
            entry = json.loads(line)
            labels_by_name[entry["filename"]] = entry

    # Get all image files sorted
    image_files = sorted(info["image_dir"].glob("*.png"))

    # Filter to only files that have labels
    labeled_files = []
    unlabeled = []
    for img in image_files:
        if img.name in labels_by_name:
            labeled_files.append(img)
        else:
            unlabeled.append(img.name)

    if unlabeled:
        print(f"  WARNING: {len(unlabeled)} images without labels: {unlabeled[:5]}...")
        errors.append(f"{cat_prefix}: {len(unlabeled)} unlabeled images")

    print(f"  {len(labeled_files)} labeled images to copy")

    # Copy and rename sequentially
    for idx, img_path in enumerate(labeled_files, start=1):
        new_name = f"{cat_prefix}_{idx:04d}.png"
        new_path = IMG_DIR / new_name

        # Copy image
        shutil.copy2(img_path, new_path)

        # Get label
        label = labels_by_name[img_path.name]

        # Extract subtype
        subtype = None
        if info["subtype_field"] and info["subtype_field"] in label:
            subtype = label[info["subtype_field"]]

        # Build unified metadata entry
        meta = {
            "image": new_name,
            "category": label.get("category", cat_prefix),
            "quality": label.get("quality", "pass"),
            "description": label.get("description", ""),
            "subtype": subtype,
            "severity": label.get("severity", "none"),
            "location": label.get("location", ""),
            "source_frame": None,
            "variant": None,
            "old_filename": img_path.name,
            "old_path": f"{OLD_CATEGORY_DIRS[cat_prefix]}/{img_path.name}",
        }

        # Parse source_frame and variant from old filename
        # Pattern: frame_XXXX.png or frame_XXXX_vN.png
        stem = img_path.stem  # e.g. "frame_0042_v1" or "frame_0042"
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                meta["source_frame"] = int(parts[1])
            except ValueError:
                pass
        if len(parts) >= 3 and parts[2].startswith("v"):
            try:
                meta["variant"] = int(parts[2][1:])
            except ValueError:
                pass

        all_metadata.append(meta)
        image_mapping[f"{OLD_CATEGORY_DIRS[cat_prefix]}/{img_path.name}"] = new_name

    print(f"  Copied {len(labeled_files)} → {cat_prefix}_0001.png to {cat_prefix}_{len(labeled_files):04d}.png")

# Write metadata.jsonl
metadata_path = OUT_DIR / "metadata.jsonl"
with open(metadata_path, "w") as f:
    for m in all_metadata:
        f.write(json.dumps(m) + "\n")
print(f"\nWrote {len(all_metadata)} entries to {metadata_path}")

# Write image_mapping.json
mapping_path = OUT_DIR / "image_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(image_mapping, f, indent=2)
print(f"Wrote mapping ({len(image_mapping)} entries) to {mapping_path}")

# Summary
print("\n" + "=" * 60)
print("DATASET REORGANIZATION COMPLETE")
print("=" * 60)
cats = {}
for m in all_metadata:
    c = m["category"]
    cats[c] = cats.get(c, 0) + 1
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")
print(f"  TOTAL: {len(all_metadata)}")

if errors:
    print(f"\nWARNINGS: {len(errors)}")
    for e in errors:
        print(f"  {e}")

# Verify file count matches
actual_files = list(IMG_DIR.glob("*.png"))
print(f"\nVerification: {len(actual_files)} files in dataset/images/")
assert len(actual_files) == len(all_metadata), f"MISMATCH: {len(actual_files)} files vs {len(all_metadata)} metadata entries"
print("✓ File count matches metadata count")
