# qwen3-vl:8b — simple (nothink)

Date: 2026-03-12 14:28
Test set: 99 images

## Results Summary

**Overall accuracy: 67.7%** (99/99 valid)
**Average time: 27.3s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 97.0% | 0.970 | 0.970 | 0.970 | 32 | 1 | 1 |
| improper_stacking | 6.1% | 1.000 | 0.061 | 0.114 | 2 | 0 | 31 |
| safe | 100.0% | 0.516 | 1.000 | 0.680 | 33 | 31 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 32 | 0 | 1 | 0 |
| improper_stacking | 1 | 2 | 30 | 0 |
| safe | 0 | 0 | 33 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 16.5s | TTFT: 15.19s | 277 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: wet patch on concrete floor near the forklift
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 21.1s | TTFT: 18.94s | 374 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry; all shelves have neatly stacked, stable boxes without tilting, overhang, or uneven alignment
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 24.0s | TTFT: 22.63s | 428 tokens @ 20.1 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 100%  
location: wet reflective patch on concrete floor in the center aisle
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 20.8s | TTFT: 19.15s | 361 tokens @ 20.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is dry and clean, all shelves have neatly stacked and stable boxes
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.0s | TTFT: 11.99s | 226 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor is clean and dry with no spills; all shelves have boxes neatly aligned and stable with no stacking issues
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 41.1s | TTFT: 38.99s | 738 tokens @ 19.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: warehouse floor is dry and clear, all shelves have neatly stacked boxes without tilting, overhang, or damage
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 11.5s | TTFT: 10.02s | 172 tokens @ 19.7 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: center of the warehouse aisle, on the concrete floor
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 37.1s | TTFT: 34.5s | 668 tokens @ 19.6 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Warehouse floor shows tire marks but no liquid spills; shelves on both sides have boxes neatly stacked vertically with no tilted, overhanging, crushed, or unevenly arranged items
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 13.1s | TTFT: 11.15s | 209 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: The warehouse floor is clean and dry with no visible spills, and all shelves have neatly stacked, stable boxes with no signs of improper stacking.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.0s | TTFT: 9.67s | 170 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: large liquid puddle on concrete floor in the center aisle
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 19.0s | TTFT: 17.73s | 328 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: liquid puddle on the concrete floor in the central aisle
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.7s | TTFT: 10.35s | 183 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle of the warehouse, on the concrete floor
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 19.4s | TTFT: 17.2s | 336 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor appears dry and clean; all shelving units have neatly stacked, stable boxes without tilting, overhang, or crushed items
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 22.5s | TTFT: 20.6s | 398 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: no safety violations observed on floor or shelves; floor is clean and dry, shelves are neatly organized with stable stacking
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 18.0s | TTFT: 16.54s | 310 tokens @ 20.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: central aisle, concrete floor with a visible liquid puddle
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 20.7s | TTFT: 18.97s | 349 tokens @ 19.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: no hazards identified; floor is clean/dry, shelves are neatly organized with stable stacking
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 11.5s | TTFT: 9.95s | 177 tokens @ 20.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: liquid spill on the concrete floor near the boxes in the foreground
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 13.2s | TTFT: 11.07s | 214 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry with no visible spills; shelves are neatly stacked with stable, aligned boxes showing no improper stacking or hazards
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 10.1s | TTFT: 8.65s | 151 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: liquid puddle on the warehouse floor between the shelving units
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 9.9s | TTFT: 8.59s | 148 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: throughout the warehouse floor and all shelving units
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 18.6s | TTFT: 16.76s | 319 tokens @ 20.1 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: warehouse floor and all shelves appear clean, dry, and properly organized with stable stacking of boxes
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 25.7s | TTFT: 22.87s | 461 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse scene shows clean, dry concrete floor with no fluid hazards; all shelving units have boxes neatly stacked in organized rows with no tilted, overhanging, crushed, or unstable loads
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 24.1s | TTFT: 21.59s | 430 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Warehouse floor is dry and clean with no fluids, and all shelves have neatly stacked, stable boxes without any tilting, overhanging, or unstable arrangements.
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 14.7s | TTFT: 13.3s | 245 tokens @ 20.3 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: large liquid puddle on concrete floor in the middle aisle between shelves
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.5s | TTFT: 9.97s | 181 tokens @ 20.3 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: large liquid puddle on concrete floor in the center aisle between shelves
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 12.1s | TTFT: 10.54s | 191 tokens @ 20.3 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: large liquid puddle in the center aisle of the warehouse floor
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 21.3s | TTFT: 19.92s | 375 tokens @ 20.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: wet puddle on concrete floor in the central aisle
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 12.8s | TTFT: 11.04s | 207 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry, shelves are neatly stacked with no unstable or improperly arranged boxes
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 16.3s | TTFT: 15.02s | 276 tokens @ 20.1 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: central aisle floor with visible liquid puddle
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 17.8s | TTFT: 15.87s | 305 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have neatly stacked and stable boxes with no visible spills or improper stacking
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 77.7s | TTFT: 75.2s | 1469 tokens @ 19.8 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: warehouse floor is dry with no visible spills; all shelves have boxes neatly stacked and stable without tilting, overhang, or debris
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 17.1s | TTFT: 14.7s | 292 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Entire warehouse floor is clean/dry with no fluid hazards; all shelves have neatly stacked, stable boxes with no tilted, overhanging, or crushed items.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.2s | TTFT: 9.92s | 176 tokens @ 20.3 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: middle of the aisle, concrete floor showing a wet reflective patch
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.6s | TTFT: 12.59s | 242 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Floor is dry with no visible spills; shelves are neatly stacked with boxes aligned in rows and no unstable or tilted items
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 18.5s | TTFT: 16.66s | 317 tokens @ 20.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor is clean and dry with no visible spills; all shelves have neatly stacked boxes without unstable or tilted items
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 17.3s | TTFT: 14.77s | 296 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry with no visible spills; all shelves have boxes neatly stacked, aligned, and stable with no tilting, overhang, or uneven stacking
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 24.2s | TTFT: 21.19s | 432 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is dry and clean with no visible fluid spills; all shelves are neatly organized with boxes stacked flat, aligned in rows, and stable with no overhanging, crushed, or misaligned items.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.7s | TTFT: 12.63s | 245 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is dry and clean, all shelving units have neatly stacked, stable boxes with no signs of spills or improper stacking
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.2s | TTFT: 14.35s | 256 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: none
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 127.4s | TTFT: 125.47s | 2405 tokens @ 19.5 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: right side shelves with unevenly stacked, jumbled boxes that appear unstable and could potentially fall
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 128.1s | TTFT: 126.27s | 2414 tokens @ 19.5 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: center of the warehouse floor, near the aisle, showing a visible liquid puddle
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.3s | TTFT: 13.13s | 250 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have neatly aligned and stable box stacks with no tilting, overhang, or uneven loading
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 19.5s | TTFT: 16.85s | 340 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry with no visible fluids, and all shelves have neatly organized, stable box stacks with no tilting, overhang, crushing, or uneven alignment.
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 26.2s | TTFT: 24.32s | 471 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: No visible spills on the floor or improper stacking on shelves; all shelves are neatly organized with stable box arrangements.
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 67.0s | TTFT: 64.8s | 1263 tokens @ 19.8 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean/dry with no visible fluid, shelves are neatly stacked with no tilted/overhanging/crushed boxes
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.0s | TTFT: 12.07s | 226 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor is clean and dry; all shelves have neatly aligned, stable boxes with no unstable stacking or spills
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 11.9s | TTFT: 9.92s | 185 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: floor is clean and dry with no visible spills; shelves are neatly organized with boxes stacked flat and stable on both sides
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 21.9s | TTFT: 19.65s | 387 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: no visible spills or improper stacking; floor appears dry with minor surface marks (not fluid-related), shelves are neatly organized with stable, aligned box stacks
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 10.4s | TTFT: 8.97s | 158 tokens @ 20.3 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: central area of the warehouse floor with a visible liquid puddle
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 16.3s | TTFT: 14.57s | 276 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have boxes neatly stacked and stable with no visible hazards
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 18.7s | TTFT: 17.2s | 324 tokens @ 20.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: concrete floor near the forklift, with visible liquid puddle
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.2s | TTFT: 11.93s | 235 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry with no visible fluid; all shelves have neatly aligned, stable box stacks with no tilting, overhang, or damage
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 14.6s | TTFT: 13.41s | 243 tokens @ 20.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: No safety issues observed on floor or shelves
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 322.8s | TTFT: 310.83s | 5973 tokens @ 18.9 tok/s

**Model response:**
```
To determine the appropriate classification for the given scene, we analyze the options provided:

- **spill**: This refers to a situation where liquids or solids have spilled, creating a hazardous condition. However, the description does not indicate any visible spillage.
  
- **improper_stacking**: This refers to the unsafe arrangement of items, such as stacking materials in a way that creates instability, poses a risk of collapse, or violates safety standards. Proper stacking is crucial to prevent accidents in environments like warehouses or storage areas.

- **minor_hazard**: This would generally refer to a less severe safety issue, but the term is vague and less commonly used in this context compared to "improper_stacking," which is a specific and standard category for unsafe material arrangement.

Based on the description provided (which implies an unsafe stacking situation without any mention of spills), the most accurate and standard classification is **improper_stacking**. It directly addresses the potential hazard of unstable or incorrectly arranged items, which is a well-established safety concern in industrial or storage settings.

---

$$
\boxed{improper\_stacking}
$$
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 85%
- **Time:** 101.5s | TTFT: 99.64s | 1908 tokens @ 19.6 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 85%
location: right side shelves with torn shrink wrap on top boxes and some boxes rotated/tilted
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.0s | TTFT: 9.42s | 170 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: large liquid spill on concrete floor near center of image, between shelving units
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 10.9s | TTFT: 9.12s | 169 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have neatly organized and stable box stacks with no visible hazards
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.0s | TTFT: 12.87s | 248 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry, all shelves are neatly stacked with stable, aligned boxes showing no signs of instability or spills
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 16.1s | TTFT: 14.35s | 271 tokens @ 20.1 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all shelves are clean, dry, and properly stacked with no visible hazards
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.4s | TTFT: 13.52s | 258 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Warehouse floor is clean and dry; all shelves have neatly stacked, stable boxes with no improper stacking or spills visible
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 13.1s | TTFT: 11.28s | 211 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor is clean and dry, shelves have neatly organized and stable box stacks with no unstable or messy arrangements
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 11.0s | TTFT: 9.48s | 169 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: middle of the warehouse floor, visible wet patches near the person walking
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 22.9s | TTFT: 20.68s | 402 tokens @ 20.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry with no spills; all shelves are neatly organized with boxes stacked uniformly and stable, showing no stacking violations
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 13.0s | TTFT: 11.66s | 209 tokens @ 20.0 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: floor between shelving units, visible large liquid puddle
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 85%
- **Time:** 11.4s | TTFT: 9.93s | 177 tokens @ 20.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 85%  
location: liquid puddle on the concrete floor in the central aisle
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 19.4s | TTFT: 17.67s | 336 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: warehouse floor has a dark, wet reflective patch (liquid spill) near the center of the aisle
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.0s | TTFT: 9.46s | 166 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: liquid puddle on concrete floor in the central area of the warehouse
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 10.7s | TTFT: 8.38s | 163 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry with no visible spills; all shelves are neatly organized with boxes stacked flat, aligned, and stable without overhang or damage
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 56.3s | TTFT: 55.07s | 1051 tokens @ 19.8 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: no issues observed on floor or shelves
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.7s | TTFT: 12.75s | 241 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and shelving units show no spills or improper stacking; all boxes are neatly aligned and stable on shelves
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.2s | TTFT: 13.89s | 252 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: no issues observed in floor or shelving areas
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 13.1s | TTFT: 11.06s | 211 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor is clean and dry with no visible fluids, and all shelves are neatly stacked with stable, properly aligned boxes
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.4s | TTFT: 13.34s | 257 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor is clean and dry with no fluid issues; all shelves have boxes neatly stacked and stable with no visible stacking hazards
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.4s | TTFT: 13.08s | 256 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have boxes neatly stacked and stable with no visible tilting, overhanging, or deformed boxes
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 17.9s | TTFT: 15.62s | 306 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and shelving units show no visible spills or improper stacking; boxes are neatly aligned and stable on shelves, floor is clean and dry
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 13.6s | TTFT: 11.75s | 219 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and shelving units show no spills or improper stacking; items are neatly organized and stable
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 28.0s | TTFT: 25.32s | 503 tokens @ 20.0 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: Warehouse floor is dry with no visible liquid spills; all shelving units have boxes stacked neatly and stably with no signs of tilting, overhanging, or uneven loading.
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 10.5s | TTFT: 8.7s | 158 tokens @ 20.2 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: floor near center aisle, between two shelves with a visible liquid puddle and white container nearby
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 15.8s | TTFT: 14.48s | 264 tokens @ 20.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: liquid puddle on the concrete floor between the shelving units
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 11.0s | TTFT: 9.12s | 170 tokens @ 20.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse floor and shelves are clean, dry, and all boxes are neatly organized and stable with no visible hazards
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 31.6s | TTFT: 30.3s | 571 tokens @ 20.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: central aisle floor with visible wet/liquid spill
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 13.1s | TTFT: 10.61s | 204 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Floor is clean and dry with no visible spills; all shelves have neatly organized, stable stacks of boxes with no tilting, overhang, or uneven stacking.
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 14.9s | TTFT: 12.76s | 245 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 98%
location: Warehouse floor is clean and dry with no spills; all shelves are neatly stacked with stable, aligned boxes showing no improper stacking or hazards
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.9s | TTFT: 13.0s | 244 tokens @ 20.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is clean and dry, all shelves have neatly stacked boxes with no unstable or improper stacking issues
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 50.3s | TTFT: 47.52s | 929 tokens @ 19.8 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: floor is clean and dry with no fluid traces; all shelves have neatly stacked, stable boxes with no tilted, overhanging, crushed, or jumbled loads
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.0s | TTFT: 11.98s | 225 tokens @ 20.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is dry and clean; all shelves have neatly stacked, stable boxes with no visible instability or spills
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.2s | TTFT: 13.2s | 251 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse floor and shelves show clean, dry conditions with neatly organized and stable box stacking on all visible shelves
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 14.4s | TTFT: 13.21s | 237 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: N/A (no hazards identified)
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.6s | TTFT: 13.6s | 261 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Entire warehouse floor is dry and clean; all shelves have neatly aligned, stable box stacks with no visible instability or spills
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 12.7s | TTFT: 11.21s | 204 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: liquid puddle on concrete floor in central aisle between shelving units
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 26.9s | TTFT: 25.19s | 482 tokens @ 20.0 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: Wet reflective patches on the concrete floor, particularly in the central aisle of the warehouse
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 112.6s | TTFT: 109.92s | 2105 tokens @ 19.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor is clean and dry with no visible spills; shelves on both sides have boxes stacked neatly in rows with no tilting, overhang, crushing, or uneven alignment
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 98.2s | TTFT: 96.99s | 1839 tokens @ 19.5 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: no issues identified
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 16.8s | TTFT: 14.46s | 284 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse floor is clean and dry; all shelves have neatly organized, stable box stacks with no tilting, overhanging, or crushing issues
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 11.7s | TTFT: 10.65s | 180 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: central aisle floor with visible liquid puddle
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 14.7s | TTFT: 13.08s | 239 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: liquid puddle on concrete floor in the middle of the warehouse aisle
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 16.1s | TTFT: 14.48s | 268 tokens @ 20.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: warehouse floor near center with a visible liquid puddle and footprints around it
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 105.1s | TTFT: 102.81s | 1973 tokens @ 19.5 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor is dry and clear; all shelves have neatly stacked boxes with no visible tilting, overhanging, or uneven arrangement
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 13.7s | TTFT: 12.5s | 221 tokens @ 20.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: no hazards identified on floor or shelves
```

---

