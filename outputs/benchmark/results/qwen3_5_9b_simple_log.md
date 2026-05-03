# qwen3.5:9b — simple

Date: 2026-03-13 02:17
Test set: 99 images

## Results Summary

**Overall accuracy: 85.6%** (90/99 valid)
**Average time: 115.9s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 96.9% | 1.000 | 0.969 | 0.984 | 31 | 0 | 1 |
| improper_stacking | 60.0% | 1.000 | 0.600 | 0.750 | 18 | 0 | 12 |
| safe | 100.0% | 0.683 | 1.000 | 0.812 | 28 | 13 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 31 | 0 | 1 | 0 |
| improper_stacking | 0 | 18 | 12 | 0 |
| safe | 0 | 0 | 28 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** N/A
- **Confidence:** N/A%
- **Time:** N/As | TTFT: N/As | N/A tokens @ N/A tok/s

**Error:** timed out

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 166.3s | TTFT: 164.85s | 2117 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 65.4s | TTFT: 63.35s | 818 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: central floor aisle (foreground to mid-ground)
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 99.8s | TTFT: 97.46s | 1258 tokens @ 13.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: Foreground shelving units on the left and right sides
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 51.9s | TTFT: 50.21s | 628 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor and shelving units
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 157.7s | TTFT: 156.18s | 1975 tokens @ 13.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 36.8s | TTFT: 34.92s | 431 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: large brown puddle in the center aisle
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 260.9s | TTFT: 259.3s | 3271 tokens @ 12.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: N/A
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 79.0s | TTFT: 77.47s | 969 tokens @ 13.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: N/A
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 36.5s | TTFT: 34.9s | 419 tokens @ 12.9 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 55.9s | TTFT: 54.29s | 673 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 149.1s | TTFT: 146.95s | 1874 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor (dark liquid patch/streak)
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 358.6s | TTFT: 335.4s | 4383 tokens @ 12.5 tok/s

**Model response:**
```
Based on the visual evidence, I would classify this scene as **safe**.

Here is my reasoning:
1.  **Floor Condition:** The concrete floor shows dark tire tracks and general dust/wear, but there are no visible puddles, oily sheens, or active leaks that would indicate a "spill_hazard" or "floor_wetness". The area looks dry enough to walk safely (no slip hazard from liquid).
2.  **Stacking Integrity:** The boxes on the shelves and wire racks appear neatly aligned and stable. There are no boxes visibly tilting, overhanging, or crushed that would suggest an "improper_stacking" scenario involving instability. While there might be some loose tape or minor wear on plastic wrapping (which could arguably fall under a strict interpretation of improper stacking), the overall arrangement is orderly and secure, not "messy" or "jumbled."
3.  **Overall Organization:** The items are stored within the designated racking areas without obvious encroachment into walkways.

Therefore, the scene does not exhibit the critical safety violations defined for the other categories.

**Category:** safe
**Severity:** none
**Confidence Score:** 95%
**Description:** A dry, clean warehouse environment with boxes neatly stacked on wire racking. There are no significant spills, wet floors, or unstable loads visible. Minor cosmetic issues like tape placement do not constitute a safety hazard in this context.
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 681.4s | TTFT: 667.93s | 8170 tokens @ 12.2 tok/s

**Model response:**
```
The image shows a warehouse aisle with shelving units and boxes.
- **Damaged packaging**: No obvious tears or crushes are visible on the boxes.
- **Wet floor**: The floor appears to be concrete/polished with some shadows, but no distinct wet patches or high-gloss reflections typical of spills.
- **Improper stacking / Damaged goods**: While loose boxes are present (standard for wire racks), they don't appear jumbled or precariously unstable in a way that suggests a violation requiring a specific hazard category unless the dataset considers ANY loose items as improper. However, looking at standard safety definitions, this looks like organized storage.
- **Minor hazard / Unsafe environment**: No small spills or obvious hazards detected.

The scene appears to be a normal, well-maintained warehouse aisle with no active violations.

safe
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 482.5s | TTFT: 473.27s | 5889 tokens @ 12.5 tok/s

**Model response:**
```
safe

The image shows a warehouse shelf with boxes that appear neatly organized and stacked on both the left and right sides. There are no visible signs of spills, wet patches, or leaks on the floor. The shelves themselves do not show any evidence of toppling or misalignment. While there is a robot figure in the aisle, based on the specific inspection criteria provided (looking for spills and stacking issues), the scene presents no violations related to those specific hazards, leading to a classification of safe.

**safe**: floor is clean and shelves are properly stacked with no signs of leaks or falling boxes.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 73.7s | TTFT: 71.44s | 922 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A - No hazards observed in the aisle or on shelves
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 59.5s | TTFT: 57.67s | 738 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: foreground floor, near the bottom right boxes
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 247.4s | TTFT: 244.61s | 3175 tokens @ 13.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: top boxes on the nearest shelving units in the foreground are tilted forward and appear unstable/loose
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 98%
- **Time:** 39.2s | TTFT: 37.49s | 469 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 98%
location: Center of the main aisle
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 102.9s | TTFT: 101.59s | 1296 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 167.1s | TTFT: 164.54s | 2120 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and all visible shelving units appear organized, dry, and stable
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 139.3s | TTFT: 137.72s | 1742 tokens @ 13.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire scene
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 647.5s | TTFT: 628.62s | 7756 tokens @ 12.2 tok/s

**Model response:**
```
Based on the visual evidence, here is the classification:

The warehouse appears to be well-organized and well-maintained. The shelves are fully stocked but not overfilled or leaning, and the boxes appear stable. There are no visible liquid spills or significant debris that would constitute a spill or improper stacking hazard in the strict sense (though there is some minor debris like wood planks, which might be construction materials).

**safe**
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 98%
- **Time:** 67.6s | TTFT: 65.37s | 832 tokens @ 13.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 98%
location: center of the main walking aisle between the rows of shelving
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 76.5s | TTFT: 74.56s | 960 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center of the aisle on the concrete floor
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 40.1s | TTFT: 37.88s | 480 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: center of the main walkway between shelving units
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 36.2s | TTFT: 34.62s | 422 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 85%
- **Time:** 82.3s | TTFT: 78.53s | 1012 tokens @ 13.0 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 85%
location: Left side shelving units, particularly the foreground and mid-ground shelves where boxes are tilted, leaning, and unevenly stacked.
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 98%
- **Time:** 49.0s | TTFT: 45.75s | 586 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 98%
location: Center aisle floor, specifically a large wet patch/puddle visible in the foreground and mid-ground area of the walkway.
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 90%
- **Time:** 66.3s | TTFT: 65.01s | 823 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 90%
location: None
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 89.0s | TTFT: 86.5s | 1121 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A - No violations found; floor is dry and shelves are neatly stacked.
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 94.3s | TTFT: 92.84s | 1190 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 54.5s | TTFT: 52.18s | 667 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Middle of the main aisle, extending slightly to the left
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 128.7s | TTFT: 124.98s | 1630 tokens @ 13.2 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: Upper left shelving units and the upper shelf of the foreground rack on the left side, where boxes are visibly tilted and leaning over the edge.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 96.2s | TTFT: 92.01s | 1199 tokens @ 13.1 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: Right-side shelving unit, specifically on the second and third shelves from the top where boxes are tilted, leaning, and stacked in a jumbled manner rather than neatly aligned.
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 451.9s | TTFT: 446.7s | 5576 tokens @ 12.6 tok/s

**Model response:**
```
```json
{
    "category": "improper_stacking",
    "severity": "high",
    "confidence": 0.95,
    "location": "Boxes on the shelving units in the foreground and the standalone rack in the middle-left aisle are tilted and unstable."
}
```
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 183.0s | TTFT: 178.96s | 2337 tokens @ 13.2 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: left side shelving units (blue racks), specifically on the upper and middle levels where shrink wrap is torn/hanging loosely and boxes appear unevenly stacked or leaning.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 68.7s | TTFT: 66.57s | 850 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire visible warehouse area including floor and all shelving units
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 71.5s | TTFT: 69.45s | 862 tokens @ 12.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: All visible shelving units and floor areas
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 69.7s | TTFT: 66.23s | 859 tokens @ 13.2 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: Upper right shelf rack, where a pallet of boxes wrapped in plastic is tilted and hanging precariously over the edge, unstable.
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 91.8s | TTFT: 89.51s | 1163 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Center aisle floor, visible in the foreground and extending slightly into the mid-ground
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 90.0s | TTFT: 88.45s | 1128 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 71.3s | TTFT: 69.75s | 891 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** None%
- **Time:** 1476.6s | TTFT: 1466.49s | 17872 tokens @ 12.3 tok/s

**Model response:**
```
Based on the visual analysis of the provided images and the context from the user's message:

**Reasoning:**
The images depict a distribution center aisle with shelves stocked with boxes. In Image 1 (or one of the images), there is visible debris (such as a box) on the floor near the end of an aisle. While this creates a potential tripping hazard or minor obstruction for forklifts/pedestrians, it does not block the pathway entirely or pose a critical safety risk like fire. Therefore, this situation fits the criteria for a "minor_hazard".

**Classification:** `**minor_hazard**`
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 93.5s | TTFT: 90.6s | 1182 tokens @ 13.3 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: Foreground right-hand shelving unit, specifically the lower shelves where boxes are leaning and stacked unevenly/jumbled.
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 120.5s | TTFT: 119.2s | 1533 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 53.7s | TTFT: 52.13s | 662 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: N/A
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 75.6s | TTFT: 72.1s | 950 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: Right-hand shelving units, particularly the upper levels where boxes are tilted, jumbled, and not aligned flat on the pallets.
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 47.4s | TTFT: 45.99s | 578 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: central aisle
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 77.0s | TTFT: 74.63s | 954 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A - No hazards detected in the visible floor or shelving units
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 97.3s | TTFT: 95.46s | 1230 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, large reflective wet patch
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** None%
- **Time:** 540.2s | TTFT: 522.54s | 6627 tokens @ 12.5 tok/s

**Model response:**
```
Based on the visual inspection of the image:

1.  **Spills/Liquids**: I have examined the floor area under the shelves and in the aisles. The yellow safety lines are visible. There are no signs of liquid spills (oil slicks, puddles) or chemical leaks. The dark patches on the floor appear to be shadows or dirt, not hazardous fluids.
2.  **Stacking**: I have checked every shelf.
    *   **Alignment**: Boxes are arranged neatly and uniformly.
    *   **Stability**: There are no boxes propped up by other items. No unstable stacking configurations (like boxes leaning) are observed.
3.  **Debris/Other Hazards**: While there might be a tiny indistinct object in the bottom right corner, it is too small to classify as a significant hazard like "minor_hazard" in this context, especially given the overall cleanliness and likely synthetic nature of the image (perfect geometry).

**Conclusion**:
No hazards related to spills or improper stacking are observed. The environment appears controlled and orderly.

Category: **safe**
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 64.4s | TTFT: 62.07s | 803 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Entire aisle and shelving units show no signs of spills or improper stacking
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 50.7s | TTFT: 47.0s | 621 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 100%
location: Right-hand shelving unit, specifically on the upper racks where a large cardboard box is tilted at a dangerous angle and appears unstable.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 83.4s | TTFT: 79.68s | 1051 tokens @ 13.3 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: Top shelves on the right side (crushed boxes, loose shrink wrap) and various levels on the left side (unstable stacking, compressed boxes).
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 42.5s | TTFT: 39.93s | 514 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: central aisle floor, large brown liquid puddle visible between the shelving rows
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 44.7s | TTFT: 43.16s | 545 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: No issues detected
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 90%
- **Time:** 101.2s | TTFT: 97.4s | 1280 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 90%
location: N/A - The floor is dry and clear of spills; all shelved loads are neatly stacked, stable, and aligned with no overhangs or crushing visible.
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 115.8s | TTFT: 113.68s | 1396 tokens @ 12.7 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: None - floor is dry and shelves are neatly organized
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 90%
- **Time:** 116.1s | TTFT: 114.12s | 1407 tokens @ 12.7 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 90%
location: Warehouse aisles and shelving units
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 89.4s | TTFT: 87.11s | 1080 tokens @ 12.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Entire floor area and all visible shelving units are compliant
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 92.3s | TTFT: 90.45s | 1136 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle on the concrete floor
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 51.9s | TTFT: 50.39s | 617 tokens @ 12.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 90.5s | TTFT: 88.59s | 1109 tokens @ 12.9 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Center aisle floor between the foreground shelving units
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 52.1s | TTFT: 49.99s | 621 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 90%
location: foreground, left-center of the main aisle
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 71.9s | TTFT: 70.28s | 874 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 81.1s | TTFT: 79.5s | 1000 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: Center aisle floor
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 77.0s | TTFT: 75.15s | 942 tokens @ 13.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: general warehouse floor and shelving units
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 112.3s | TTFT: 109.02s | 1401 tokens @ 13.1 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: Foreground right shelf stack (crushed/leaning box), lower shelves on the left side (tilted boxes)
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 83.4s | TTFT: 79.7s | 1030 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Warehouse floor is dry and clean; all shelving units on both sides contain neatly stacked, stable boxes with no signs of improper loading or damage.
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 52.7s | TTFT: 50.29s | 634 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor, roughly halfway down the image towards the back shelves
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 121.7s | TTFT: 118.14s | 1517 tokens @ 13.0 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: Left side shelving units (foreground and mid-ground), where boxes are rotated at angles and loose white bags/sacks are hanging from shelves
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** N/A
- **Confidence:** N/A%
- **Time:** N/As | TTFT: N/As | N/A tokens @ N/A tok/s

**Error:** [Errno 54] Connection reset by peer

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 96.1s | TTFT: 94.7s | 1196 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** N/A
- **Confidence:** N/A%
- **Time:** N/As | TTFT: N/As | N/A tokens @ N/A tok/s

**Error:** timed out

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** N/A
- **Confidence:** N/A%
- **Time:** N/As | TTFT: N/As | N/A tokens @ N/A tok/s

**Error:** timed out

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 79.1s | TTFT: 76.92s | 974 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A - no hazards detected in visible floors or shelves
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 33.8s | TTFT: 31.84s | 389 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: center aisle floor near the white bucket
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 78.5s | TTFT: 76.73s | 958 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: center aisle floor, foreground
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 62.5s | TTFT: 58.91s | 756 tokens @ 13.0 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: Right side shelving units (middle section), where a large group of boxes is piled haphazardly, leaning and unstable.
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 62.9s | TTFT: 60.57s | 770 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Center aisle floor, stretching from the foreground cart area towards the background worker
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** N/A
- **Confidence:** N/A%
- **Time:** N/As | TTFT: N/As | N/A tokens @ N/A tok/s

**Error:** timed out

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 85.6s | TTFT: 84.14s | 1054 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire scene
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1007.9s | TTFT: 1006.87s | 12057 tokens @ 12.2 tok/s

**Model response:**
```
Based on the visual evidence, there are no safety hazards.
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** None%
- **Time:** 471.7s | TTFT: 434.37s | 5711 tokens @ 12.4 tok/s

**Model response:**
```
Based on a detailed visual inspection of the warehouse storage area shown in the image:

**Observations:**
*   **Left Side (Top Shelf):** The boxes stacked on the top shelf on the far left appear jumbled and uneven. Unlike the shelves on the right or the back shelves which show neat, aligned rows, these boxes are tilted, rotated, and appear to be resting at angles rather than flat. Some in the very corner look potentially crushed or deformed and sticking out, indicating a lack of secure stacking.
*   **Right Side & Back Shelves:** The storage on the right side features a large, shrink-wrapped load which is stable. The back shelves show aligned boxes.
*   **Floor:** The floor appears dry with some dark stains (likely tire marks or dirt) but no obvious slippery spills that would cause slipping.

**Analysis:**
The prompt defines "improper_stacking" as when "boxes are stacked unevenly or jumbled instead of neatly aligned in rows". The condition on the top left shelf clearly fits this description. Loose boxes stacked without shrink wrap on upper shelves, especially if not palletized or perfectly flat, create a risk of falling objects (medium to high severity depending on height and weight). Given they are relatively low shelves (maybe 2-3 boxes high), the severity is likely **medium**. It's not an active collapse (critical) but a clear violation of orderly stacking which can lead to instability.

**Conclusion:**
There is a visible hazard related to storage stability.

**improper_stacking**, medium severity with confidence
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 90%
- **Time:** 201.0s | TTFT: 199.42s | 2554 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 90%
location: entire warehouse scene
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 103.3s | TTFT: 101.39s | 1298 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Entire warehouse aisle and shelving units
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 102.4s | TTFT: 100.45s | 1281 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: Entire visible warehouse floor and shelving units
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 62.8s | TTFT: 61.29s | 773 tokens @ 13.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: N/A
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 61.9s | TTFT: 59.8s | 757 tokens @ 13.1 tok/s

**Model response:**
```
category: spill
severity: low
confidence: 100%
location: center of the main aisle in the foreground
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 65.9s | TTFT: 63.71s | 812 tokens @ 13.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor showing wet, reflective patches indicating liquid spill
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 90%
- **Time:** 156.6s | TTFT: 155.08s | 1976 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 90%
location: entire scene
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 131.0s | TTFT: 127.05s | 1637 tokens @ 13.0 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: Right-hand shelving unit, upper levels where boxes are jumbled, tilted, and piled unevenly; loose plastic wrap hanging from shelves on both sides.
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 121.9s | TTFT: 118.44s | 1526 tokens @ 13.1 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: foreground right shelving unit where a large pile of cardboard boxes is leaning precariously to the left and stacked haphazardly.
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 64.7s | TTFT: 63.13s | 782 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Center aisle floor
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 116.9s | TTFT: 114.92s | 1458 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Foreground right side of the main aisle
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 79.8s | TTFT: 77.08s | 982 tokens @ 13.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: Center of the main walkway, between the shelving units on the left and right sides
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 90.1s | TTFT: 86.14s | 1112 tokens @ 13.0 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: On the shelving units to the right side of the aisle, specifically a large box is tilted significantly at an angle and appears unstable, creating a falling hazard.
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 60.9s | TTFT: 59.33s | 739 tokens @ 13.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: N/A
```

---

