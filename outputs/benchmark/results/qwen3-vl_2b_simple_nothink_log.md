# qwen3-vl:2b — simple (nothink)

Date: 2026-03-12 17:10
Test set: 99 images

## Results Summary

**Overall accuracy: 63.8%** (94/99 valid)
**Average time: 31.2s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 90.9% | 0.698 | 0.909 | 0.789 | 30 | 13 | 3 |
| improper_stacking | 41.9% | 0.591 | 0.419 | 0.491 | 13 | 9 | 18 |
| safe | 56.7% | 0.586 | 0.567 | 0.576 | 17 | 12 | 13 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 30 | 0 | 3 | 0 |
| improper_stacking | 9 | 13 | 9 | 0 |
| safe | 4 | 9 | 17 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 19.3s | TTFT: 18.56s | 1260 tokens @ 72.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: the entire warehouse (no visible spills; shelves appear neatly stacked with boxes in aligned rows, no tilted, overhanging, or unstable items)
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 19.8s | TTFT: 19.17s | 1317 tokens @ 72.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 100%
location: Shelves on both sides show boxes tilted or rotated, stacked unevenly, and some appearing unstable due to misalignment.
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 30.2s | TTFT: 29.71s | 1951 tokens @ 69.7 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: the central aisle with visible liquid puddles on the concrete floor
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 10.6s | TTFT: 10.35s | 680 tokens @ 72.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 9.8s | 661 tokens @ 72.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: Floor is clean and dry; all shelves have neatly organized boxes without tilting, hanging, or unstable items.
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 38.3s | TTFT: 38.27s | 2460 tokens @ 68.5 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: warehouse interior (general area)
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 10.0s | TTFT: 9.36s | 642 tokens @ 72.6 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 95%  
location: Central area of the warehouse, where a large brown liquid spill is visible on the concrete floor, centered in the aisle between the rows of shelves.
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 30.0s | TTFT: 29.45s | 1960 tokens @ 70.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: middle of the warehouse (floor has visible liquid puddles and wet reflective patches)
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 77.4s | TTFT: 74.58s | 4894 tokens @ 66.8 tok/s

**Model response:**
```
To classify the scene, we analyze key elements based on the problem's criteria:

1. **Spill Check**: The floor is concrete, and there are no visible puddles, oil, or other signs of spills. Thus, **no spill**.
   
2. **Improper Stacking Check**: 
   - Boxes are arranged in neat rows with no apparent tilting, hanging, or uneven stacking.
   - There are no signs of crushed or deformed boxes supporting weight.
   - No torn or missing shrink wrap is evident.
   - Loads are aligned with the shelf structure, with no shifting to one side.
   - Overall arrangement is orderly and stable.

Given these observations, the scene meets all criteria for safety. There are no signs of improper staking, spills, or instability.

**Final Answer**:  
The scene is classified as **safe**.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 7.9s | TTFT: 7.53s | 505 tokens @ 73.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: center of the warehouse floor between the two rows of shelves
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 43.8s | TTFT: 43.45s | 2847 tokens @ 69.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 90%  
location: Center of the warehouse floor
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 100%
- **Time:** 31.9s | TTFT: 31.45s | 2105 tokens @ 70.7 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 100%
location: a liquid puddle on the concrete floor near the center of the warehouse aisle
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 16.9s | TTFT: 16.35s | 1113 tokens @ 72.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Left shelves (some boxes tilted) and Right shelves (boxes slightly tilted/unevenly stacked)
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 68.5s | TTFT: 65.39s | 4406 tokens @ 67.8 tok/s

**Model response:**
```
**Analysis of the Scene**

- **Spill Check**: The floor is concrete and appears clean without any visible puddles, stains, or oily residues. There are no signs of spills.
- **Stacking Condition**: The boxes are neatly arranged in rows on the shelves. There are no tilted boxes, boxes hanging over the shelf edge, crushed or deformed boxes, or boxes stacked unevenly. The shelves themselves seem level and stable.
- **Stability Check**: The boxes are securely placed on the shelves, with no indication of instability (e.g., loose boxes, leaning items, or shelves that could topple).
- **Overall Clarity**: The scene is orderly and well-organized. There are no minor hazards or violations (such as incomplete or improper stacking, or spills) that would be considered a "minor hazard."

**Classification**  
The scene meets all criteria for the "safe" classification. There are no spills, improper stacking, or unstable elements.

**Answer**  
safe
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 41.5s | TTFT: 40.55s | 2713 tokens @ 69.7 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor appears clean and dry with no visible spills or fluid on the concrete. All shelves are filled with neatly arranged, stable boxes in orderly rows without signs of tilting, hanging, crushing, or uneven stacking.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 75%
- **Time:** 14.9s | TTFT: 14.21s | 982 tokens @ 73.0 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 75%
location: Several dark patches on the concrete floor that appear to be oil spills or wet reflective patches, possibly indicating a spill hazard.
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 12.1s | TTFT: 11.61s | 799 tokens @ 73.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: a brownish liquid puddle on the concrete floor near the bottom right corner of the warehouse scene
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 19.8s | TTFT: 19.36s | 1320 tokens @ 72.3 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 80%
location: left side shelves (notable tilt of boxes on left shelves)
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 55.1s | TTFT: 54.56s | 3547 tokens @ 68.1 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 100%
location: central area near the image's background (water puddle on concrete floor)
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 17.6s | TTFT: 16.84s | 1163 tokens @ 72.6 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry, with all shelves holding boxes in neat, stable rows without visible tilt, overhang, or deformity.
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 85%
- **Time:** 50.6s | TTFT: 50.07s | 3247 tokens @ 68.0 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 85%
location: center of the warehouse floor, with visible liquid puddles and wet reflective patches on concrete
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 70%
- **Time:** 75.0s | TTFT: 72.85s | 4791 tokens @ 67.5 tok/s

**Model response:**
```
The scene is classified as **improper_stacking**.  
- **Reasoning**: The shelves display boxes that appear to be tilted or misaligned, indicating improper stacking. While the floor has no evident fluid spills (as the dark patches are likely due to lighting or reflection rather than liquid), the primary issue observed is the misalignment of boxes on the shelves.  
- **Severity**: Medium (a moderate issue affecting storage integrity but not critical).  
- **Confidence**: 70% (based on visual analysis of box alignment).  
- **Location**: Shelves on both the left and right sides of the warehouse.  

**Answer**: improper_stacking
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 95%
- **Time:** 41.2s | TTFT: 40.67s | 2681 tokens @ 69.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 95%
location: both sides of the warehouse (left and right aisles)
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 8.6s | TTFT: 8.03s | 528 tokens @ 73.2 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 95%  
location: central area of the warehouse floor, visible large liquid puddle
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 34.4s | TTFT: 33.65s | 2259 tokens @ 70.3 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 90%  
location: large brown liquid spill on the concrete floor in the central area of the warehouse, near the intersection of the two main aisles.
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 7.7s | TTFT: 7.34s | 489 tokens @ 74.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: large puddle in the central aisle between the metal shelving units on both sides
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 35.6s | TTFT: 35.1s | 2343 tokens @ 70.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: A large liquid puddle is visible on the concrete floor in the center of the warehouse.
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 14.3s | TTFT: 13.6s | 944 tokens @ 73.1 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: aisles between the left and right shelves, particularly on the left side where boxes are tilted, overhanging, and unevenly stacked
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 69.4s | TTFT: 68.9s | 4445 tokens @ 67.6 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: wet puddle on the concrete floor in the center of the warehouse
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 70.9s | TTFT: 68.16s | 4535 tokens @ 67.5 tok/s

**Model response:**
```
- **category**: spill  
- **severity**: critical  
- **confidence**: 100%  
- **location**: floor with liquid puddles and wet reflective patches  

**Explanation**:  
- The floor exhibits liquid puddles and a sheen, indicating fluid on the warehouse floor.  
- While improper stacking issues are present on the shelves, the **spill** category is directly and prominently confirmed by the presence of liquid on the floor.  
- The **severity** is critical due to the direct hazard of spills (e.g., potential fire, contamination).  
- The **confidence** is 100% because the image clearly shows the liquid puddles on the floor.  
- The **location** refers to the floor where the spill is observed.  

This classification aligns with the criteria provided in the problem statement.
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 43.6s | TTFT: 43.14s | 2832 tokens @ 69.0 tok/s

**Model response:**
```
category: spill
severity: low
confidence: 90%
location: floor has visible liquid puddles and wet reflective patches
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 21.9s | TTFT: 21.17s | 1441 tokens @ 71.8 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry; all items on shelves are neatly organized and stable in rows without visible tilt, hanging, or instability.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 69.6s | TTFT: 66.59s | 4448 tokens @ 67.5 tok/s

**Model response:**
```
The scene is described as a large warehouse with many shelves filled with boxes, a person standing in the middle, and a concrete floor with a few boxes on the floor. 

- **Spill**: There are no clear indicators of liquid or fluid on the floor. The reflective patches are not consistent with a spill (e.g., they are not a large puddle or liquid-covered area). 
- **Improper Stacking**: The boxes on the shelves are neatly arranged in rows, and there are no visible issues like tilted boxes, overhanging boxes, or unstable items.
- **Minor Hazard**: There are no significant deviations from the normal warehouse environment.
- **Safe**: The overall arrangement is clean and orderly, with no immediate signs of a hazard. The boxes are neatly stacked, and the floor does not show any liquid or unstable elements.

Thus, the most accurate classification for the scene is **safe**.

**Answer**: safe
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 55.7s | TTFT: 54.75s | 3537 tokens @ 67.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry, with no visible spills. All items on the shelves are neatly organized and stable, with boxes stacked in rows without tilting or overhanging.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 658.0s | TTFT: 0s | 40960 tokens @ 64.8 tok/s

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 14.2s | TTFT: 13.63s | 902 tokens @ 72.5 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: central warehouse area with shelves showing tilted boxes, overhanging items, and uneven stacking patterns
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 103.1s | TTFT: 97.58s | 6499 tokens @ 66.3 tok/s

**Model response:**
```
To determine the issue with the warehouse setup described in the question, we analyze the key points from the user's description:

- The warehouse has shelves, boxes, and a forklift.
- The shelves are "in good shape" but not explicitly described as level or tilted.
- There is no mention of a wet floor or excessive water-related hazards.
- The main concern is the arrangement of the boxes on the shelves.

### Key Observations:
- The image shows boxes stacked on shelves, which is a common warehouse configuration.
- The user specifies "the shelves are in good shape" but does not provide details about the arrangement of the boxes.
- The term "improper_stacking" is used in the context of the user's question to describe the problem.

### Reasoning:
1. **Box Arrangement**: The user's description emphasizes that the shelves are "in good shape" and the boxes are "stacked." However, the term "improper_stacking" is directly tied to the problem described.
2. **Problem Identification**: The problem is not about the shelves themselves (e.g., they are not level or tilted), but about how the boxes are arranged. If the boxes are stacked unevenly or not aligned neatly, this would be a violation of standard warehouse safety and operational standards.
3. **Logical Conclusion**: The phrase "improper_stacking" is the most direct and consistent interpretation of the user's question about the issue with the warehouse setup. It aligns with the description of boxes being stacked in a way that may cause instability or inefficiency, without any explicit mention of the shelves being level or tilted.

### Final Answer:
The issue is: **improper_stacking**.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.4s | TTFT: 14.38s | 1003 tokens @ 72.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor appears clean and dry, with no visible liquid spills or reflective patches. All shelves have boxes neatly aligned in rows with no tilted, rotated, or unstable items. No spilled fluids or improperly stacked boxes are observed.
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 11.9s | TTFT: 11.34s | 771 tokens @ 72.7 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: near the central aisle between the two main shelves (around 10-20 feet from the image center)
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 47.8s | TTFT: 47.35s | 3071 tokens @ 68.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: floor (presence of liquid puddles and wet reflective patches)
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 35.2s | TTFT: 34.73s | 2285 tokens @ 69.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: central area of the warehouse floor, near the aisle between the two sets of shelves
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 25.7s | TTFT: 25.01s | 1669 tokens @ 70.7 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse floor and shelves, all boxes neatly stacked in organized rows without tilting, overhanging, or instability issues
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 80%
- **Time:** 64.5s | TTFT: 61.24s | 4099 tokens @ 67.1 tok/s

**Model response:**
```
**Category:** improper_stacking  
**Severity:** low  
**Confidence:** 80%  
**Location:** right side of the warehouse, shelves with boxes stacked unevenly  

**Explanation:**  
- **Spill:** No visible fluid or hazardous material on the concrete floor, so this is not a spill.  
- **Improper_stacking:** Multiple instances of boxes being tilted, stacked unevenly, or jumbled (not aligned in neat rows) are observed. These issues are minor but pose a potential risk of products falling or causing tripping hazards.  
- **Severity:** The problem is "low" because the observed issues are not critical (e.g., no crushing of boxes or significant instability of the shelves).  
- **Location:** The issue is most evident on the right side of the warehouse, where the arrangement of boxes appears uneven.  

This classification aligns with safety standards for warehouse management, where minor but observable stacking irregularities are categorized as "improper_stacking" with a low severity level.
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 57.5s | TTFT: 56.94s | 3646 tokens @ 67.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse (central aisle between the two main shelves)
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 75%
- **Time:** 8.9s | TTFT: 8.39s | 561 tokens @ 72.6 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 75%
location: multiple shelves with tilted boxes, some boxes rotated at an angle, and loose items on shelves showing instability
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 16.4s | TTFT: 15.73s | 1072 tokens @ 71.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Left and right shelves show tilted boxes, uneven stacking, and some boxes at an angle; no visible spills or critical instability issues.
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 16.6s | TTFT: 16.26s | 1080 tokens @ 71.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: central aisle of the warehouse
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 22.8s | TTFT: 22.22s | 1486 tokens @ 70.8 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: middle of the warehouse, between the two rows of shelves with visible liquid puddles on the concrete floor
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 26.4s | TTFT: 25.9s | 1732 tokens @ 70.6 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: central area with visible wet reflective patches and liquid puddles on the concrete floor
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 49.7s | TTFT: 48.95s | 3187 tokens @ 68.0 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: shelves on both sides of the warehouse floor, with boxes neatly arranged and stable; floor is clean and dry; no visible spills or improper stacking issues.
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** None%
- **Time:** 69.2s | TTFT: 66.02s | 4408 tokens @ 67.2 tok/s

**Model response:**
```
**category**: spill  
**severity**: critical  
**confidence**: high  

**Explanation**:  
- The floor exhibits a wet reflective patch, which indicates the presence of liquid (e.g., oil slicks or spills on concrete), clearly matching the description for a "spill."  
- The shelves show improper stacking (e.g., tilted boxes), but this is a secondary issue. The primary hazard is the spill on the floor, which is a critical safety risk.  
- While improper stacking is present, the question explicitly instructs to classify the scene into **one** category. The spill on the floor is the most prominent and severe hazard, thus the "spill" category is the most accurate classification.  

This classification is based on the clear evidence of a liquid spill on the floor, which is a critical hazard. The other issues (improper stacking) are secondary and do not override the primary hazard.  

**Final Answer**:  
category: spill  
severity: critical  
confidence: high
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 70%
- **Time:** 66.4s | TTFT: 63.64s | 4207 tokens @ 67.1 tok/s

**Model response:**
```
**Answer:**  
**Category:** improper_stacking  
**Severity:** low  
**Confidence:** 70%  

**Explanation:**  
- **Spill Check:** The floor is concrete with no visible liquid or spills, so it does not qualify as a "spill".  
- **Improper Stacking Check:** The boxes on the shelves show minor inconsistencies in arrangement (e.g., some boxes appear tilted or unevenly stacked), indicating a "improper_stacking" issue. This is a non-critical issue but requires attention.  
- **Severity:** The problem is minor (not a critical hazard), so the severity is "low".  
- **Confidence:** The image is clear enough to observe the boxes' alignment, so confidence is 70%.  

**Location:** The shelves on the left side of the warehouse.
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 58.4s | TTFT: 0s | 0 tokens @ 0 tok/s

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 100%
- **Time:** 22.2s | TTFT: 21.48s | 1459 tokens @ 71.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 100%
location: on the shelves, particularly visible on the right side where boxes are tilted and overhanging, causing instability and potential for falling.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 70%
- **Time:** 54.7s | TTFT: 54.36s | 3484 tokens @ 67.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 70%
location: floor
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 85%
- **Time:** 14.7s | TTFT: 14.22s | 951 tokens @ 72.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 85%
location: the central floor area with a large brown liquid puddle visible on the concrete ground
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 44.2s | TTFT: 43.64s | 2857 tokens @ 68.6 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry, with all items on shelves neatly organized and stable.
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 47.0s | TTFT: 46.02s | 3012 tokens @ 68.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor appears clean and dry with no visible liquid spills. All shelves have boxes neatly stacked in aligned rows, with no signs of tilting, overhanging, crushed boxes, or unstable load placement.
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 39.6s | TTFT: 39.18s | 2556 tokens @ 69.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: the entire warehouse interior space, including all shelves and floor area
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 28.2s | TTFT: 27.29s | 1829 tokens @ 70.5 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: The warehouse floor is clean and dry with no visible spills. All shelves appear to have boxes neatly arranged in orderly rows, with no signs of tilted, overhanging, crushed, or unstable items.
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 41.2s | TTFT: 40.73s | 2657 tokens @ 69.0 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 70%
location: left side shelves (noted tilted boxes, uneven stacking, and loose alignment)
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 66.8s | TTFT: 66.5s | 4230 tokens @ 67.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: floor
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 7.5s | TTFT: 6.77s | 455 tokens @ 73.2 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry with no visible spills. All shelves hold neatly arranged, stable boxes in orderly rows without tilted, overhanging, or unstable elements.
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 80%
- **Time:** 11.3s | TTFT: 10.82s | 735 tokens @ 72.8 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 80%
location: central area of the warehouse with a large, wet puddle on the concrete floor
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 6.1s | TTFT: 5.76s | 380 tokens @ 73.7 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: central aisle, near the middle of the warehouse floor
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 80%
- **Time:** 26.3s | TTFT: 25.85s | 1730 tokens @ 70.7 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 80%  
location: central floor near the middle of the warehouse between the two rows of shelves
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 9.57s | 659 tokens @ 73.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: A large liquid puddle is visible on the concrete floor in the middle of the warehouse, near the shelves with stacked boxes.
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 27.3s | TTFT: 26.59s | 1782 tokens @ 70.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: warehouse floor is clean; shelves are neatly stacked with no tilted boxes, overhanging items, or unstable loads
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 7.0s | TTFT: 6.03s | 441 tokens @ 73.3 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: multiple areas on both sides of the warehouse, including left-side shelves with tilted/rotated boxes, overhanging items, crushed/deformed boxes, and unevenly stacked items; right-side shelves show boxes tilted, torn/missing shrink wrap, and shifted loads
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.3s | TTFT: 14.64s | 1001 tokens @ 72.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: the warehouse floor has visible cracks but no liquid puddles, wet reflective patches, oil slicks, or chemical leaks; all shelves hold boxes in neat, stable rows without tilting, overhanging, or uneven stacking.
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 34.5s | TTFT: 33.76s | 2249 tokens @ 69.6 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The floor appears dry and clean; all shelves have boxes neatly organized and stable with no visible tilting, overhanging, crushed, or uneven stacking issues.
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 30.3s | TTFT: 29.12s | 1982 tokens @ 70.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: The warehouse floor appears clean and dry, with no visible liquid puddles, wet patches, oil slicks, or chemical leaks. All shelves have boxes neatly stacked in rows, with no tilted boxes, hanging over shelf edges, crushed/deformed boxes, torn/shrink wrap issues, or shifted loads.
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 36.2s | TTFT: 35.31s | 2313 tokens @ 68.2 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: The warehouse floor appears clean and dry with no visible liquid spills or wet patches. All shelves hold boxes neatly aligned in rows without tilting, overhanging, or unstable conditions.
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 85%
- **Time:** 16.3s | TTFT: 15.58s | 1037 tokens @ 70.6 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 85%
location: The middle aisle between the left and right stacks of shelves, where several boxes appear tilted, unevenly stacked, and some may be overhanging or deformed.
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 4.0s | 281 tokens @ 73.0 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor appears clean and dry with no visible liquid spills. All shelves have neatly organized and stable stacked boxes with no tilted, overhanging, crushed, or unstable items.
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 4.32s | 281 tokens @ 72.6 tok/s

**Model response:**
```
category: safe  
severity: low  
confidence: 100%  
location: entire warehouse interior with evenly stacked boxes on both sides of the aisle
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 58.6s | TTFT: 0s | 0 tokens @ 0 tok/s

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 28.9s | TTFT: 28.35s | 1889 tokens @ 70.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: a liquid spill and broken container on the concrete floor between the two main rows of shelves
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 40.9s | TTFT: 39.89s | 2621 tokens @ 68.9 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: center of the warehouse, near the mid-point between the two main shelves with visible liquid puddle on the concrete floor
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 100%
- **Time:** 14.5s | TTFT: 14.15s | 937 tokens @ 72.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 100%
location: right side shelves with tilted boxes indicating unstable stacking conditions
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 18.1s | TTFT: 17.79s | 1192 tokens @ 71.8 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: floor (wet reflective patches and liquid puddles)
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 8.8s | TTFT: 8.24s | 545 tokens @ 73.2 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: Entire warehouse (all shelves and floor appear clean and orderly with no visible spills or unstable stacking)
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 100%
- **Time:** 21.3s | TTFT: 20.78s | 1402 tokens @ 71.3 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 100%
location: left side of the warehouse, with some boxes visibly tilted on lower shelves (unstable stacking)
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 14.6s | TTFT: 13.95s | 954 tokens @ 72.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: left side shelves (boxes tilted/rotated) and right side shelves (unevenly stacked, unstable boxes)
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 85%
- **Time:** 20.8s | TTFT: 20.13s | 1345 tokens @ 71.4 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 85%
location: Left side shelves have visible tilted boxes, and right side shelves have unevenly stacked boxes with some tilting.
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 90%
- **Time:** 40.2s | TTFT: 39.9s | 2596 tokens @ 69.0 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 90%
location: left side shelves
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 660.0s | TTFT: 0s | 40960 tokens @ 64.8 tok/s

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 86.0s | TTFT: 83.6s | 5402 tokens @ 66.5 tok/s

**Model response:**
```
The scene shows a warehouse with shelves holding boxes. The floor has dark patches that appear to be oil slicks or chemical leaks. These patches indicate a spill on the concrete surface. 

Additionally, while the shelves may have minor issues (e.g., some boxes appearing slightly tilted), these are not severe enough to classify the entire scene as "improper_stacking." The presence of fluid on the floor is the primary factor for classification, as the problem explicitly instructs to "look at the FLOOR for spills."

Thus, the scene is classified under the category of **spill**, as the floor contains visible fluid (oil slicks or chemical leaks), which qualifies as a spill.

**Answer:** spill
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 45.0s | TTFT: 44.09s | 2887 tokens @ 68.6 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: Warehouse floor is clean and dry; all items on shelves are neatly organized and stable (no liquid spills, boxes are stacked in neat rows without tilting, crushing, or instability).
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 5.6s | TTFT: 5.24s | 335 tokens @ 73.6 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: the center of the warehouse floor has a liquid puddle
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 31.9s | TTFT: 31.44s | 2078 tokens @ 69.8 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: floor with visible liquid puddles and wet reflective patches
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 75.8s | TTFT: 73.72s | 4804 tokens @ 66.8 tok/s

**Model response:**
```
**Answer:**  
- **Category:** spill  
- **Severity:** medium  
- **Confidence:** 100%  
- **Location:** floor  

**Explanation:**  
The warehouse scene exhibits liquid puddles and wet reflective patches on the floor, indicating a spill hazard. The shelves have improperly arranged and unstable boxes, but the primary safety concern is the spill on the floor, which is classified under "spill." The severity of the spill is considered medium (not a critical chemical leak or oil slick, but a liquid hazard). The image is clear, so confidence is 100%. The location of the issue is the floor.
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 100%
- **Time:** 15.0s | TTFT: 14.34s | 968 tokens @ 72.2 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 100%
location: Left and right shelves throughout the warehouse, with numerous boxes tilted/rotated, hanging over shelf edges, and unevenly stacked boxes causing instability.
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 8.8s | TTFT: 8.6s | 566 tokens @ 73.1 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 100%  
location: none
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 8.7s | TTFT: 8.16s | 554 tokens @ 73.2 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 80%
location: central concrete floor with visible wet patch (liquid puddle) between two aisles of shelving
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 12.2s | TTFT: 11.68s | 797 tokens @ 72.6 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: Central aisle near the middle of the warehouse, marked by a visible brown liquid puddle on the concrete floor
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 20.5s | TTFT: 20.02s | 1339 tokens @ 71.3 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: central concrete floor with visible liquid spill patch
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 60.9s | TTFT: 60.61s | 3866 tokens @ 67.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 100%  
location: floor of the warehouse
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 36.2s | TTFT: 35.07s | 2350 tokens @ 69.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: The warehouse floor is clean and dry with no visible liquid puddles, wet patches, or oil slicks. Shelves hold boxes neatly aligned in rows, with no tilted, overhanging, crushed, or jumbled boxes, and no unstable items.
```

---

