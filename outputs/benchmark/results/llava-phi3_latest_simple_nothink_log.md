# llava-phi3:latest — simple (nothink)

Date: 2026-03-12 18:06
Test set: 99 images

## Results Summary

**Overall accuracy: 52.0%** (98/99 valid)
**Average time: 5.3s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 78.8% | 0.531 | 0.788 | 0.634 | 26 | 23 | 7 |
| improper_stacking | 57.6% | 0.487 | 0.576 | 0.528 | 19 | 20 | 14 |
| safe | 18.8% | 0.600 | 0.188 | 0.286 | 6 | 4 | 26 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 26 | 7 | 0 | 0 |
| improper_stacking | 10 | 19 | 4 | 0 |
| safe | 13 | 13 | 6 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 6.1s | TTFT: 3.08s | 117 tokens @ 38.6 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: shelves in the middle of the warehouse with boxes not neatly aligned or stable

category: spill
severity: medium
confidence: 50%
location: small puddles of water on the floor near the aisle

category: unable_to_determine
severity: none
confidence: 0%
location: no specific issue found in the image
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 8.3s | TTFT: 2.62s | 217 tokens @ 38.1 tok/s

**Model response:**
```

category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: high / critical / medium / low / unable_to_determine
confidence: 90%
location: middle of warehouse on left side

The image shows a warehouse with many shelves filled with cardboard boxes. There is a large spill on the floor, covering a significant portion of the ground. The spill is causing a high risk of slipping and falling accidents. There is also improper stacking of boxes on the shelves, with some boxes tilted at an angle and others hanging over the edge. This poses a critical risk of the boxes falling and causing injury or damage. Some boxes are crushed and torn, indicating a lack of proper organization and handling. The overall safety condition of the warehouse is poor, and immediate action is needed to address the spill and the improper stacking of boxes.
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.0s | TTFT: 2.75s | 51 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: On shelves, boxes are tilted and stacked unevenly, some are hanging over the edge.
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 3.7s | TTFT: 2.57s | 46 tokens @ 39.2 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 100%
location: The warehouse appears to be well-organized with no visible spills or improper stacking issues.
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 6.2s | TTFT: 2.58s | 139 tokens @ 38.3 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: shelves on both sides of the warehouse

Explanation: The image shows several warehouse shelves filled with stacks of boxes. The boxes are not neatly organized, some are tilted, and a few are stacked unevenly or jumbled. This poses a potential hazard due to the instability of the stacked items, which could lead to accidents or damage to the goods. The warehouse floor appears to be clean and dry, with no visible spills or fluid issues.
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.0s | TTFT: 2.64s | 52 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: All the shelves in the warehouse are filled with crates and boxes that are poorly arranged and unstable.
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 60%
- **Time:** 6.5s | TTFT: 2.63s | 146 tokens @ 38.1 tok/s

**Model response:**
```

category: spill
severity: medium
confidence: 60%
location: puddle of liquid on the floor

category: improper_stacking
severity: high
confidence: 80%
location: stacks of boxes leaning to one side, some boxes tilted at an angle

category: minor_hazard
severity: medium
confidence: 20%
location: boxes hanging over the shelf edge, some crushed or deformed

category: safe
severity: none
confidence: 10%
location: no clear violations of safety standards
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 3.5s | TTFT: 2.66s | 33 tokens @ 39.4 tok/s

**Model response:**
```

* The image shows a warehouse full of stacked boxes on shelves, with a concrete floor and metal beams on the ceiling.
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.0s | TTFT: 2.8s | 50 tokens @ 38.9 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: shelves on both sides of the aisle are filled with boxes that are poorly arranged and unstable.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 6.0s | TTFT: 2.89s | 124 tokens @ 38.2 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: liquid puddles on the floor

category: improper_stacking
severity: medium
confidence: 80%
location: stacks of boxes on the shelves are unevenly arranged, tilted at an angle, and some are crushed

category: safe
severity: none
confidence: 10%
location: no spills on the floor and all items on the shelves are neatly organized
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.3s | TTFT: 2.76s | 62 tokens @ 38.7 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Shelves on both sides of the aisle have several boxes stacked improperly, with boxes tilting at an angle and some boxes hanging over the edge.
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 9.6s | TTFT: 2.83s | 257 tokens @ 37.6 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 70%
location: liquid puddles on the concrete floor

category: improper_stacking
severity: high
confidence: 80%
location: boxes tilted or overhanging on shelves

category: spill
severity: medium
confidence: 50%
location: wet reflective patches on the floor

category: improper_stacking
severity: medium
confidence: 60%
location: boxes hanging over the shelf edge

category: spill
severity: low
confidence: 30%
location: no visible spills on the floor

category: improper_stacking
severity: high
confidence: 70%
location: boxes stacked unevenly on shelves

category: safe
severity: high
confidence: 90%
location: clean and dry floor, neatly organized and stable boxes on shelves

Note: The confidence ratings are subjective and based on the image quality and clarity.
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 8.6s | TTFT: 2.74s | 219 tokens @ 37.6 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: boxes on shelves are tilted, overhanging, and crushed, suggesting a risk of falling

In the image, the warehouse appears to be clean and dry, with no visible spills or hazards on the floor. However, upon closer inspection, there are several issues with the stacking of the boxes on the shelves. Some boxes are tilted or rotated at an angle instead of being flat, while others are hanging over the edge of the shelf, posing a risk of falling. There are also boxes that are crushed or deformed, which could potentially be damaged or difficult to move. The boxes are not neatly aligned in rows, and it appears that loads have been shifted to one side of the shelf, creating an uneven distribution of weight. These issues are serious and could lead to accidents or injuries, so they must be addressed immediately.
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 7.0s | TTFT: 2.86s | 161 tokens @ 37.7 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: floor near the center of the warehouse

category: improper_stacking
severity: medium
confidence: 50%
location: shelves near the entrance of the warehouse

category: minor_hazard
severity: low
confidence: 10%
location: shelves near the back of the warehouse

category: safe
severity: critical
confidence: 0%
location: entire warehouse

category: unable_to_determine
severity: unknown
confidence: 0%
location: entire warehouse
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 75%
- **Time:** 6.6s | TTFT: 2.79s | 143 tokens @ 37.8 tok/s

**Model response:**
```

* The floor has a puddle of liquid on the left side.
* Several shelves have boxes stacked neatly and securely, with no visible issues.

Category: spill
Severity: medium
Confidence: 75%
Location: left side of the warehouse

The image shows a warehouse with a puddle of liquid on the left side of the floor, which can pose a risk of slipping or falling hazards. While the shelves are mostly organized, the presence of the puddle suggests a need for better floor maintenance and cleaning to ensure the safety of the workers.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 7.5s | TTFT: 2.84s | 179 tokens @ 37.7 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Multiple shelves with boxes tilted at an angle, some boxes stacked unevenly or jumbled instead of neatly aligned in rows.

explanation: The image shows a warehouse with rows of shelves filled with boxes and packages. Upon closer inspection, it becomes apparent that the boxes and packages are not neatly aligned on the shelves, and some boxes are tilted at an angle. This could pose a risk of instability and accidents. The severity of this issue is high, as it could result in injuries to workers or damage to the goods being stored. The confidence level is 90% because the image provides a clear view of the warehouse and its contents.
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.8s | TTFT: 3.15s | 103 tokens @ 37.8 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 80%
location: liquid puddles and wet reflective patches on the floor

category: improper_stacking
severity: medium
confidence: 50%
location: boxes tilted or rotated at an angle on shelves, boxes hanging over the shelf edge

category: safe
severity: none
confidence: 0%
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 5.3s | TTFT: 2.95s | 95 tokens @ 37.9 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: warehouse

The image shows a large warehouse with rows of shelves filled with stacks of boxes. The floor appears clean and dry, and there are no visible spills or hazards. The boxes are neatly organized and appear to be stable. There is no visible work being done by a person in the warehouse.
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 80%
- **Time:** 11.5s | TTFT: 2.77s | 327 tokens @ 37.3 tok/s

**Model response:**
```

| Category | Severity | Confidence | Location |
| --- | --- | --- | --- |
| Improper_stacking | High | 80% | On all shelves with a mix of tilted, hanging, and crushed boxes |
| Spill | Critical | 60% | A wet reflective patch on the floor near the entrance |
| Improper_stacking | High | 80% | Shelves on the left side of the warehouse |
| Improper_stacking | High | 70% | Shelves on the right side of the warehouse |
| Improper_stacking | High | 80% | Shelves in the center of the warehouse |
| Spill | High | 60% | A small puddle of liquid near the entrance |
| Improper_stacking | High | 80% | Shelves in the back of the warehouse |
| Spill | High | 60% | A large spill on the floor near the entrance |
| Improper_stacking | High | 70% | Shelves on the left side of the warehouse |
| Improper_stacking | High | 80% | Shelves on the right side of the warehouse |
| Spill | High | 60% | A small spill on the floor near the entrance |
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.0s | TTFT: 2.92s | 47 tokens @ 38.2 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: Warehouse full of cardboard boxes with varying heights and widths on shelves and floor.
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.0s | TTFT: 2.84s | 46 tokens @ 38.1 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: Several boxes are tilted and not stacked properly on the shelves, causing instability.
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 2.82s | 33 tokens @ 39.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 100%
location: concrete floor with visible wet spots and stains
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 90%
- **Time:** 3.9s | TTFT: 2.92s | 43 tokens @ 38.9 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 90%
location: warehouse floor is clean and dry, all items on shelves are neatly organized and stable
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.8s | TTFT: 2.79s | 38 tokens @ 38.8 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: Floor near center of warehouse, black spot indicating liquid or fluid
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 6.9s | TTFT: 2.78s | 154 tokens @ 37.4 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: both sides of warehouse, multiple shelves

During my inspection, I observed that the warehouse floor has a liquid puddle, and there are wet reflective patches in the area. Additionally, there is a brown substance on the floor near the shelves. The shelves are filled with stacks of boxes that are tilted or rotated at an angle, causing instability. There are also some crushed boxes and torn shrink wrap visible. The overall scene appears to be in high risk, and I am 90% confident in my assessment.
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 5.3s | TTFT: 2.78s | 97 tokens @ 38.1 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 100%
location: liquid puddles and wet reflective patches on the concrete floor

category: improper_stacking
severity: high
confidence: 90%
location: Boxes hanging over the shelf edge, crushed boxes, and boxes stacked unevenly or jumbled instead of neatly aligned in rows.
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 5.0s | TTFT: 2.76s | 86 tokens @ 38.4 tok/s

**Model response:**
```

category: spill
severity: medium
confidence: 70%
location: A large puddle of liquid on the floor in the warehouse.

category: improper_stacking
severity: high
confidence: 80%
location: Several boxes on the shelves are stacked unevenly, tilting to the side.
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 3.9s | TTFT: 2.79s | 46 tokens @ 38.8 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: multiple shelves throughout the warehouse with boxes tilted and stacked unevenly
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.7s | TTFT: 3.05s | 35 tokens @ 39.0 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: A liquid spill is on the warehouse floor.
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 8.3s | TTFT: 3.59s | 193 tokens @ 37.8 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 100%
location: liquid puddles and wet reflective patches on the floor

The image shows a large warehouse with rows of shelves stacked high with boxes. The floor appears to be wet, with puddles and wet patches scattered across the concrete. The warehouse is poorly lit, casting shadows on the shelves and making it difficult to see the contents of the boxes. The overall safety condition of the warehouse is critical due to the spills on the floor and the poorly organized shelves. The spills pose a risk of slips and falls, while the poorly organized shelves increase the risk of accidents and injuries caused by falling objects. The warehouse management should take immediate action to address these issues and ensure the safety of their employees.
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.7s | TTFT: 2.83s | 36 tokens @ 38.8 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: The floor is wet and there are puddles of water.
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 4.7s | TTFT: 3.1s | 69 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 100%
location: corner of warehouse, near entrance

There is a significant amount of fluid on the floor in the warehouse, which poses a critical safety hazard due to the risk of slips and falls.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.8s | TTFT: 2.95s | 74 tokens @ 38.3 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: On both sides of the warehouse, there are stacks of boxes on the shelves that appear to be improperly stacked, tilted, or overhanging, and some boxes are crushed or deformed.
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 4.0s | TTFT: 2.86s | 49 tokens @ 38.3 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 100%
location: warehouse floor

Please note that the actual image is not provided, so this response is based on the description provided.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.1s | TTFT: 2.73s | 50 tokens @ 38.8 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: multiple areas of the warehouse with boxes stacked unevenly and in some cases tilted or deformed

```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 7.3s | TTFT: 2.77s | 173 tokens @ 38.0 tok/s

**Model response:**
```

* The warehouse floor appears to be clean and dry, with no visible spills or fluid on the concrete.
* There are multiple rows of shelves filled with brown cardboard boxes neatly aligned and stacked, indicating proper organization.
* There are no visible issues with the shelf arrangement, such as overhanging boxes, unstable or crushed boxes, or boxes stacked unevenly.
* There are no visible workers in the image, and the overall environment appears to be quiet and empty.

Based on the image, I would classify the safety condition of this warehouse scene as "safe" with a "low" severity level and a "high" confidence level. The image appears to show a well-organized and clean warehouse environment with no obvious safety concerns.
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.2s | TTFT: 2.79s | 57 tokens @ 38.3 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: Shelves on both sides of the aisle have boxes stacked unevenly and some are tilted or rotated at an angle.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.2s | TTFT: 2.77s | 94 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 80%
location: area near center of photo

category: improper_stacking
severity: high
confidence: 70%
location: several rows of shelves on left and right sides of photo

category: safe
severity: low
confidence: 20%
location: area near center of photo
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 2.73s | 78 tokens @ 38.5 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 100%
location: warehouse is clean and dry with neatly organized and stable boxes on shelves.

category: unable_to_determine
severity: cannot be determined from the image
confidence: 0%
location: image is unclear.
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.5s | TTFT: 2.84s | 69 tokens @ 38.8 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: "The boxes are not neatly aligned and are stacked unevenly. Some boxes are tilted or rotated at an angle instead of flat, and others are hanging over the shelf edge."
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 50%
- **Time:** 7.0s | TTFT: 2.83s | 163 tokens @ 38.1 tok/s

**Model response:**
```

category: spill
severity: medium
confidence: 50%
location: wet reflective patches on the floor near shelves

category: improper_stacking
severity: high
confidence: 70%
location: several stacks of boxes tilted on shelves and leaning against each other, some crushed, and boxes hanging over the edge of the shelf

category: safe
severity: none
confidence: 0%
location: none

Please note that the severity levels are based on the potential risk to workers and the likelihood of accidents occurring. The confidence level is a measure of the inspector's certainty in their assessment.
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 8.5s | TTFT: 2.72s | 220 tokens @ 38.1 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: warehouse floor near the entrance

The image shows a large warehouse with shelves filled with cardboard boxes. The floor of the warehouse has a few liquid puddles and wet reflective patches, indicating a spill. A worker is seen walking near the entrance of the warehouse. The boxes on the shelves are poorly arranged, with some leaning against each other and others stacked unevenly. This suggests improper stacking. The warehouse appears to be relatively clean and dry, except for the spill on the floor. The severity of the spill is critical as it could pose a slip hazard for workers. The improper stacking of boxes is a high-severity issue as it could lead to accidents or damage to the boxes. The worker is walking near the entrance, so the spill is located close to the entrance of the warehouse.
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.8s | TTFT: 2.71s | 81 tokens @ 38.7 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: Shelves on both sides with boxes tilted, hanging, or crushed

category: spill
severity: medium
confidence: 50%
location: Floor has liquid puddles and wet reflective patches
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.7s | TTFT: 2.9s | 76 tokens @ 38.8 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: Many shelves throughout the warehouse with boxes stacked in various ways. Some boxes are tilted, crushed, and stacked unevenly, which poses a risk of falling and causing damage to the boxes and surrounding area.
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.2s | TTFT: 2.71s | 94 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: shelves on both sides of the aisle

There are several boxes on the shelves that are tilted or rotated at an angle instead of being flat. This indicates that the warehouse staff may not be following proper stacking protocols, which could lead to accidents or damage to the goods.
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 2.82s | 110 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: boxes on shelves are tilted or rotated at an angle, boxes hanging over edge, boxes stacked unevenly, crushed boxes supporting weight above, torn shrink wrap, boxes jumbled instead of neatly aligned in rows.

The warehouse floor is clean and dry, but there are several instances of improperly stacked boxes and items on the shelves.
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.2s | TTFT: 2.75s | 56 tokens @ 38.8 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: multiple shelves with boxes tilted and rotated, stacked unevenly, and overhanging the edge of the shelf.
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.0s | TTFT: 2.71s | 50 tokens @ 38.9 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Shelves on both sides with boxes tilted, deformed, and overhanging, causing instability.
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.8s | TTFT: 2.85s | 41 tokens @ 38.4 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 90%
location: Several liquid puddles and wet reflective patches on the warehouse floor.
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 7.3s | TTFT: 2.88s | 172 tokens @ 38.2 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: multiple shelves throughout the warehouse

explanation: The image shows a warehouse filled with stacked boxes. However, upon closer inspection, there are several instances of improper stacking. Some boxes are tilted or rotated at an angle instead of being flat, while others appear crushed or deformed, supporting weight above them. There are also instances of boxes hanging over the shelf edge, and the load seems to be shifted to one side of the shelf. These issues pose a risk of the boxes falling, potentially causing injury or damage. The severity of the situation is high, indicating that the warehouse has safety concerns that need to be addressed.
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.0s | TTFT: 2.77s | 46 tokens @ 39.1 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 90%
location: Warehouse with many racks of boxes with the forklift in the center of the room.
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.9s | TTFT: 2.77s | 47 tokens @ 38.9 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: multiple shelves have boxes stacked unevenly or jumbled instead of neatly aligned in rows
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 75%
- **Time:** 6.2s | TTFT: 2.76s | 135 tokens @ 38.2 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 75%
location: center of the image, on the floor

The floor of the warehouse has a visible spill, which is a significant safety concern. The spill is located in the center of the image, and it's likely to pose a hazard to warehouse workers. The spill could cause slips and falls, and it could also contaminate the goods stored in the warehouse. It's important to clean up the spill as soon as possible to minimize the risk of accidents and contamination.
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.2s | TTFT: 2.87s | 98 tokens @ 38.3 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: Multiple shelves have boxes tilted or rotated at an angle instead of flat, boxes hanging over the shelf edge, crushed or deformed boxes supporting weight above, torn or missing shrink wrap, boxes stacked unevenly or jumbled instead of neatly aligned in rows, loads shifted to one side of the shelf.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 3.5s | TTFT: 2.86s | 34 tokens @ 39.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 80%
location: a puddle of fluid on the concrete floor.
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 6.3s | TTFT: 2.68s | 140 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: liquid on the concrete floor

category: improper_stacking
severity: high
confidence: 90%
location: boxes tilted on shelves

[Note: In this image, there is a large spill of oil on the floor, with a puddle next to a shelf. There are also several stacks of boxes on shelves in the background. The boxes are not neatly arranged and are leaning on their sides. Some boxes are also stacked unevenly or are deformed.]
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 7.8s | TTFT: 2.71s | 194 tokens @ 38.1 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: multiple areas on shelves with poorly arranged boxes and items

The image shows a warehouse with several rows of metal shelving units. The floor appears clean and dry, with no visible spills or hazards. However, upon closer inspection, there are several areas on the shelves where the boxes and items are poorly arranged, tilted, or stacked unevenly. There are also some boxes hanging over the shelf edge, which could pose a risk of falling. Some boxes are crushed or deformed, and others appear torn or missing their shrink wrap. These issues suggest that the warehouse may not be following proper stacking guidelines and safety protocols, which could lead to accidents or damage to the items stored in the warehouse.
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 3.8s | TTFT: 2.68s | 43 tokens @ 39.2 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: floor

category: safe
severity: none
confidence: 80%
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 4.5s | TTFT: 2.68s | 69 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: The concrete floor of the warehouse is covered in spills.

category: safe
severity: none
confidence: 80%
location: The warehouse is otherwise clean and dry.
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 5.5s | TTFT: 2.67s | 110 tokens @ 38.5 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: None

Explanation: After inspecting the image, it appears that the warehouse is clean and dry, with no visible spills, stacking problems, or safety hazards. The boxes are neatly organized and stable on the shelves, and there are no signs of improper stacking or instability. The overall condition of the warehouse appears to be safe and well-maintained.
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 10.9s | TTFT: 2.68s | 310 tokens @ 37.9 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 100%
location: floor near the entrance

In the image, the warehouse floor is covered with a dark gray concrete surface. The floor appears to be wet with several liquid puddles and wet reflective patches scattered across the area. The ceiling and walls are made of metal, creating a caged-in atmosphere.

In the center of the image, a worker dressed in a yellow vest is walking down the aisle, which is filled with rows of metal shelves. These shelves are stacked with cardboard boxes of various sizes and colors. However, upon closer inspection, some of these boxes are found to be tilted or rotated at an angle instead of being flat. In addition, some boxes are hanging over the shelf edge, which could cause them to fall and create a hazard. There are also boxes that appear to be crushed or deformed, possibly due to improper stacking or weight distribution.

There are no signs of a chemical leak or any other obvious hazards in the image. The overall scene suggests that there is a spill on the floor, but it's unclear what substance is causing the liquid puddles. The improperly stacked boxes and potential hazards pose a significant risk to the safety of the worker and the integrity of the goods stored in the warehouse.
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.3s | TTFT: 2.74s | 101 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 80%
location: the puddle in the warehouse

category: improper_stacking
severity: medium
confidence: 60%
location: the boxes stacked on the shelves

category: safe
severity: none
confidence: 70%
location: the warehouse floor is clean and dry


```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 4.8s | TTFT: 2.67s | 81 tokens @ 38.5 tok/s

**Model response:**
```

* category: safe
* severity: none
* confidence: 95%
* location: no issues found

* category: safe
* severity: none
* confidence: 95%
* location: no issues found

* category: safe
* severity: none
* confidence: 95%
* location: no issues found
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 2.69s | 92 tokens @ 38.2 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: large puddle on floor near boxes

category: improper_stacking
severity: high
confidence: 70%
location: Shelves on left side leaning and boxes tilted

Please note: The image is not clear, and some details may be inaccurate.
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.6s | TTFT: 2.65s | 111 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 80%
location: liquid on the floor in the center of the warehouse

category: improper_stacking
severity: medium
confidence: 70%
location: Several stacked boxes on the shelves are uneven and tilted

Note: This image was taken in a warehouse with a gray floor, many shelves filled with boxes, and a person walking in the background.
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 5.0s | TTFT: 2.92s | 82 tokens @ 38.3 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 70%
location: large puddle on the ground

category: improper_stacking
severity: high
confidence: 80%
location: several boxes hanging over the shelf edge and some boxes tilted at an angle instead of flat on the shelves
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 4.9s | TTFT: 2.71s | 85 tokens @ 38.6 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: center of warehouse floor

There is a visible puddle of liquid on the concrete floor, which poses a risk of slipping or falling. The spill is relatively small, but the presence of a puddle indicates a potential hazard that needs to be addressed.
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.9s | TTFT: 2.75s | 80 tokens @ 38.4 tok/s

**Model response:**
```
category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: low / high / medium / low / none
confidence: 80% / 50% / 60% / 90% / 10%
location: Warehouse, center of the image
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 6.8s | TTFT: 2.7s | 154 tokens @ 38.2 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: multiple shelves throughout the warehouse

Reasoning:
While the image does not show any spills, there are several instances of improper stacking throughout the warehouse. Boxes are tilted or hanging over the shelf edge, crushed or deformed boxes are supporting weight above, and boxes are stacked unevenly or jumbled instead of neatly aligned in rows. There are also instances where loads are shifted to one side of the shelf. These issues pose a high risk for accidents and injuries, making this a critical issue that needs to be addressed.
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 7.3s | TTFT: 2.74s | 177 tokens @ 38.2 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 100%
location: entire warehouse

The image shows a large warehouse filled with rows of shelves stacked high with boxes. The concrete floor appears to be clean and dry. There are no visible spills, liquid puddles, wet reflective patches, oil slicks, chemical leaks, or any fluid on the concrete. The boxes on the shelves appear to be neatly aligned and stable, with no signs of stacking problems such as boxes tilting or hanging over the shelf edge, crushed or deformed boxes supporting weight above, or torn or missing shrink wrap. The image is clear and easily discernible, allowing for a confident classification of the warehouse's safety condition as safe.
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.0s | TTFT: 2.66s | 53 tokens @ 38.8 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Shelves are not neatly organized and some boxes are tilted, hanging over the edge, and appear unstable.
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 5.0s | TTFT: 2.84s | 92 tokens @ 38.6 tok/s

**Model response:**
```

category: improper_stacking
severity: medium
confidence: 70%
location: All shelves have some boxes stacked improperly or in a messy manner

category: improper_stacking
severity: high
confidence: 90%
location: Several shelves have crushed boxes and boxes leaning over the edge, creating a risk of collapse.
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.7s | TTFT: 2.7s | 118 tokens @ 38.6 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 80%
location: floor near aisle 3

category: improper_stacking
severity: medium
confidence: 60%
location: aisle 5

category: spill
severity: critical
confidence: 90%
location: floor near aisle 1

category: safe
severity: none
confidence: 70%
location: aisle 8
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 75%
- **Time:** 4.1s | TTFT: 2.66s | 57 tokens @ 39.0 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 75%
location: multiple shelves with boxes tilted or hanging over the edge, some boxes supporting weight above, and a load shifted to one side.
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 2.66s | 38 tokens @ 39.2 tok/s

**Model response:**
```

category: safe
severity: none
confidence: 100%
location: Warehouse with neatly organized shelves and a clean floor.
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 2.66s | 92 tokens @ 38.4 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: The floor is covered with various pieces of cardboard boxes, and there are liquid puddles on the concrete floor.

category: improper_stacking
severity: high
confidence: 80%
location: The boxes on the shelves are poorly arranged and some are tilted.
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 6.1s | TTFT: 2.71s | 127 tokens @ 38.3 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 90%
location: visible liquid puddles on the concrete floor

category: spill
severity: critical
confidence: 90%
location: visible oil slicks on the concrete floor

category: spill
severity: critical
confidence: 90%
location: visible chemical leaks on the concrete floor

category: spill
severity: critical
confidence: 90%
location: visible fluid spots on the concrete floor
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.7s | TTFT: 2.66s | 40 tokens @ 38.6 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: floor in warehouse has liquid puddles and wet reflective patches
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.8s | TTFT: 2.66s | 81 tokens @ 38.4 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 90%
location: Floor has water puddles and oil slicks

category: improper_stacking
severity: high
confidence: 80%
location: Several boxes are stacked unevenly or jumbled instead of neatly aligned in rows.
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 90%
- **Time:** 5.7s | TTFT: 3.07s | 91 tokens @ 38.1 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 90%
location: front and center of warehouse

category: improper_stacking
severity: high
confidence: 80%
location: shelves on right side of warehouse

category: safe
severity: low
confidence: 10%
location: no significant issues detected
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 50%
- **Time:** 5.8s | TTFT: 2.69s | 109 tokens @ 38.2 tok/s

**Model response:**
```

category: spill
severity: medium
confidence: 50%
location: liquid puddles on floor near shelves

category: improper_stacking
severity: high
confidence: 70%
location: boxes tilted on shelves, some boxes hanging over edge

category: safe
severity: none
confidence: 30%
location: warehouse appears mostly clear but some potential issues are present
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 4.9s | TTFT: 2.67s | 87 tokens @ 38.4 tok/s

**Model response:**
```

category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: critical / high / medium / low / none
confidence: 95%
location: The image shows a large warehouse with numerous shelves filled with boxes. The floor appears to be wet and there are several cardboard boxes with visible wear.
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 2.83s | 34 tokens @ 39.1 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: all shelves and floor appear to be clean and organized
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** critical
- **Confidence:** 80%
- **Time:** 3.7s | TTFT: 2.68s | 40 tokens @ 39.0 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 80%
location: The floor has a few yellow reflective strips indicating that there might be spills.
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 3.9s | TTFT: 2.78s | 47 tokens @ 39.1 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Both sides of the warehouse, all the shelves, and the warehouse floor.
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 3.8s | TTFT: 2.7s | 46 tokens @ 38.6 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: Shelves on the left side of the warehouse with multiple boxes stacked unstable.
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 3.5s | TTFT: 2.75s | 34 tokens @ 39.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 80%
location: Warehouse floor appears to have fluid spillage.
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 4.8s | TTFT: 2.75s | 69 tokens @ 38.9 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 80%
location: The shelves on both sides of the warehouse are filled with boxes stacked in an uneven manner, some boxes are tilted, and a few boxes appear to be crushed or deformed.
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.5s | TTFT: 3.13s | 53 tokens @ 39.1 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: shelves on left and right are filled with boxes stacked unevenly and some appear to be crushed or deformed.
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 80%
- **Time:** 3.7s | TTFT: 2.68s | 43 tokens @ 39.2 tok/s

**Model response:**
```

category: spill
severity: critical
confidence: 80%
location: The floor has a noticeable brown puddle near the center of the warehouse.
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 2.66s | 105 tokens @ 38.2 tok/s

**Model response:**
```

* spill: there is a puddle of water on the warehouse floor.
* improper_stacking: many boxes are stacked haphazardly on the shelves, some are tilted or deformed, and others are crushed or missing shrink wrap.
* severity: high
* confidence: 90%
* location: Floor in the center and on the right side of the image, and on the shelves.
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 90%
- **Time:** 5.5s | TTFT: 2.65s | 111 tokens @ 38.3 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: On both shelves, there are boxes tilted or rotated at an angle instead of flat, boxes hanging over the shelf edge, crushed or deformed boxes supporting weight above, torn or missing shrink wrap, boxes stacked unevenly or jumbled instead of neatly aligned in rows, loads shifted to one side of the shelf, anything that looks unstable or could fall.
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.9s | TTFT: 2.71s | 126 tokens @ 38.5 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 80%
location: Floor in front of the forklift

category: improper_stacking
severity: high
confidence: 70%
location: Shelves on both sides of the warehouse

category: safe
severity: none
confidence: 60%
location: No issues found

category: unable_to_determine
severity: none
confidence: 10%
location: Insufficient information
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.9s | TTFT: 2.72s | 126 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: All sides of warehouse

This warehouse scene has a high level of safety concern as the boxes and items on the shelves are poorly arranged, unstable, and potentially hazardous. There are several boxes tilted or hanging over the shelf edge, and some boxes appear crushed or deformed. The overall organization of the items on the shelves is messy, and some loads are shifted to one side of the shelf.
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 100%
- **Time:** 4.4s | TTFT: 2.84s | 63 tokens @ 38.8 tok/s

**Model response:**
```

category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: critical / high / medium / low / none
confidence: 0-100%
location: brief description of where the issue is
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 75%
- **Time:** 5.9s | TTFT: 2.78s | 122 tokens @ 38.4 tok/s

**Model response:**
```

category: spill
severity: high
confidence: 75%
location: liquid puddle on floor near center of warehouse

category: improper_stacking
severity: high
confidence: 85%
location: Boxes leaning on shelves and hanging over edge on right side of warehouse

category: safe
severity: none
confidence: 90%
location: Warehouse is clean and dry, all items on shelves are organized and stable
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 3.6s | TTFT: 2.85s | 34 tokens @ 39.4 tok/s

**Model response:**
```
category: spill
severity: medium
confidence: 80%
location: floor has a wet area, possibly from a spilled liquid
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 80%
- **Time:** 5.8s | TTFT: 2.67s | 119 tokens @ 38.5 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 80%
location: Warehouse 3, shelves filled with boxes and cargo, some boxes tilted or rotated at an angle instead of flat, boxes hanging over the shelf edge, crushed or deformed boxes supporting weight above, torn or missing shrink wrap, boxes stacked unevenly or jumbled instead of neatly aligned in rows, loads shifted to one side of the shelf, anything that looks unstable or could fall.
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** high
- **Confidence:** 90%
- **Time:** 4.1s | TTFT: 2.67s | 54 tokens @ 38.9 tok/s

**Model response:**
```

category: improper_stacking
severity: high
confidence: 90%
location: Shelves on the right side of the image are filled with stacks of boxes that appear to be poorly arranged and unstable.
```

---

