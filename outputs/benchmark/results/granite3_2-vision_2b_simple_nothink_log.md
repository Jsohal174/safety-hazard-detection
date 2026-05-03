# granite3.2-vision:2b — simple (nothink)

Date: 2026-03-12 17:42
Test set: 99 images

## Results Summary

**Overall accuracy: 42.4%** (99/99 valid)
**Average time: 10.1s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 18.2% | 1.000 | 0.182 | 0.308 | 6 | 0 | 27 |
| improper_stacking | 9.1% | 0.188 | 0.091 | 0.122 | 3 | 13 | 30 |
| safe | 100.0% | 0.429 | 1.000 | 0.600 | 33 | 44 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 6 | 13 | 14 | 0 |
| improper_stacking | 0 | 3 | 30 | 0 |
| safe | 0 | 0 | 33 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.5s | TTFT: 6.83s | 181 tokens @ 48.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.8s | TTFT: 5.94s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.12s | 192 tokens @ 49.2 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.28s | 193 tokens @ 49.0 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.35s | 181 tokens @ 47.9 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.32s | 181 tokens @ 48.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 8.2s | TTFT: 6.33s | 90 tokens @ 48.4 tok/s

**Model response:**
```

Category: spill
Severity: medium
Confidence: 80%
Location: The warehouse floor shows a large liquid puddle, which is a spill. The puddle is located in the center of the warehouse, surrounded by empty shelves. The spill is not widespread, but it is a significant safety concern due to the potential for slipping and the risk of contaminating goods.
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.38s | 190 tokens @ 48.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The boxes do not seem to be stacked unevenly or jumbled.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.43s | 193 tokens @ 48.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 9.8s | TTFT: 6.53s | 166 tokens @ 48.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: There are boxes that appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

The classification for this scene would be:

category: minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and there are boxes that are not neatly stacked or aligned.
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.0s | TTFT: 6.45s | 176 tokens @ 48.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: There are boxes that appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall scene does not appear to be a clear violation of safety standards, but there are some issues with the stacking of boxes.

The classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and there are boxes that are tilted or rotated at an angle, indicating potential issues with stacking.
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.5s | TTFT: 6.38s | 200 tokens @ 48.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.36s | 180 tokens @ 48.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.33s | 192 tokens @ 48.8 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 11.1s | TTFT: 6.4s | 219 tokens @ 48.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is no visible liquid or fluid on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly organized and stable, with no obvious signs of tilting, overhanging, crushing, or torn or missing shrink wrap.
- Minor_hazard: There are no obvious signs of instability or potential hazards that would indicate a minor safety issue.
- Safe: The warehouse floor is clean and dry, and the items on the shelves are neatly aligned and stable.
- Unable_to_determine: The image is clear, and no additional safety concerns are apparent.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor is clean and dry, and the items on the shelves are neatly organized and stable.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 12.1s | TTFT: 6.65s | 280 tokens @ 48.9 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no obvious tilting, overhanging, crushing, or torn shrink wrap. However, the boxes are stacked in a way that some are not fully aligned, which could be considered a minor issue.
- Minor Hazard: The boxes are not perfectly aligned, but this does not appear to be a significant safety concern. The warehouse does not seem to be in a state of immediate danger.
- Safe: The warehouse floor is clean and dry, and the items on the shelves are neatly organized and stable.
- Unable to Determine: The image is clear, and the warehouse appears to be well-maintained. However, the minor issue with the stacking of the boxes could be considered a minor hazard.

Category: minor_hazard
Severity: low
Confidence: 100%
Location: The issue is with the stacking of the boxes on the shelves, which is a minor hazard in this context.
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 9.7s | TTFT: 6.41s | 166 tokens @ 49.2 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and potential spills.

The classification for this scene would be:

category: minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a spill, and the shelves show signs of improper stacking.
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.32s | 193 tokens @ 49.2 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.33s | 201 tokens @ 49.3 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.31s | 193 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.49s | 181 tokens @ 49.3 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.28s | 199 tokens @ 48.9 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.41s | 192 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.8s | TTFT: 6.33s | 222 tokens @ 49.4 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is no visible liquid or fluid on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly organized and stable, with no obvious signs of tilting, overhanging, crushing, or torn or missing shrink wrap.
- Minor_hazard: There are no obvious signs of instability or potential hazards that would require immediate attention.
- Safe: The warehouse floor is clean and dry, and the items on the shelves are neatly aligned and stable.
- Unable_to_determine: The image is clear, and no additional context is provided that would suggest a different classification.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor is clean and dry, and the items on the shelves are neatly organized and stable.
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 10.2s | TTFT: 6.37s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

Therefore, the classification for this warehouse scene would be:

category: spill / improper_stacking / minor_hazard
severity: medium
confidence: 70%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating issues with stacking and organization.
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 9.9s | TTFT: 6.26s | 182 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and potential spills.

The classification for this scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated, indicating potential issues with stacking and stability.
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.1s | TTFT: 6.3s | 187 tokens @ 49.0 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking of the boxes.

Therefore, the classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating issues with improper stacking.
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.26s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.35s | 186 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. There is no evidence of unstable or messy stacking.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.25s | 182 tokens @ 49.3 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.36s | 180 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.29s | 192 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.5s | TTFT: 6.29s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: There are no obvious signs of boxes being tilted or rotated at an angle, hanging over the shelf edge, crushed or deformed, or torn or missing shrink wrap. The boxes appear to be neatly aligned in rows.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves appear to be in good condition with no immediate safety hazards.
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.24s | 181 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.28s | 181 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.22s | 192 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.23s | 181 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.7s | TTFT: 6.23s | 222 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is no visible liquid or fluid on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly organized and stable, with no obvious signs of tilting, overhanging, crushing, or torn or missing shrink wrap.
- Minor_hazard: There are no obvious signs of instability or potential hazards that would require immediate attention.
- Safe: The warehouse floor is clean and dry, and the items on the shelves are neatly aligned and stable.
- Unable_to_determine: The image is clear, and no additional context is provided that would suggest a different classification.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor is clean and dry, and the items on the shelves are neatly organized and stable.
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.45s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 8.4s | TTFT: 6.33s | 92 tokens @ 49.9 tok/s

**Model response:**
```

Category: improper_stacking
Severity: medium
Confidence: 80%
Location: The boxes on the middle shelf are tilted at an angle, which indicates improper stacking. The boxes are not flat, and there is no evidence of crushed or deformed boxes supporting weight above. The boxes are not stacked unevenly or jumbled, but the tilted position suggests a lack of stability.
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 11.1s | TTFT: 6.3s | 236 tokens @ 49.4 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: There are no obvious signs of boxes being tilted or rotated at an angle, hanging over the shelf edge, crushed or deformed, or torn or missing shrink wrap. The boxes appear to be neatly aligned in rows.
- Minor Hazard: There is a small, dark spot on the floor that could be a minor spill or a stain, but it is not clear enough to be a significant safety hazard.
- Safe: The warehouse floor appears to be clean and dry, and the items on the shelves are neatly organized and stable.

The classification for this warehouse scene would be:

category: minor_hazard
severity: low
confidence: 100%
location: The dark spot on the floor is the only area that could be considered a minor hazard.
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.28s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.29s | 191 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be stacked unevenly or jumbled.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.31s | 189 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. There is no evidence of boxes stacked unevenly or jumbled.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.28s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.29s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.22s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.23s | 181 tokens @ 49.8 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 70%
- **Time:** 11.8s | TTFT: 6.27s | 271 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: There are boxes that appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall organization of the shelves seems to be neat, but there are a few boxes that are not perfectly aligned, which could be considered a minor hazard.
- Safe: The warehouse floor is not visibly wet or slick, and there are no obvious signs of chemical leaks or other hazards.
- Unable_to_determine: The image is clear, but the exact nature of the spill and the stability of the boxes cannot be fully determined without additional context or information.

The classification for this warehouse scene would be:

category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: medium
confidence: 70%
location: The spill is on the floor, the improper stacking is on the shelves, and the minor hazard is in the organization of the shelves.
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.36s | 180 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.34s | 180 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.31s | 200 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.26s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.49s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.32s | 180 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.34s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: There are boxes that appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor Hazard: The overall organization of the shelves seems to be neat, but there are a few boxes that are not aligned properly, which could be considered a minor hazard.

The classification for this warehouse scene would be:

- Spill: High
- Improper Stacking: Medium
- Minor Hazard: Medium
- Safe: Low
- Unable to Determine: 0-100%

The location of the issue is primarily on the floor, with some spills and improper stacking on the shelves.
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.23s | 180 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.27s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.39s | 193 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.28s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.8s | TTFT: 6.25s | 223 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly organized and stable, with no visible tilting, overhanging, crushing, or torn boxes.
- Minor Hazard: There are no obvious signs of unstable or falling items on the shelves.
- Safe: The warehouse floor is clean and dry, and the items on the shelves are neatly aligned in rows.
- Unable to Determine: The image is clear, and no significant issues are immediately apparent.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor is clean and dry, and the items on the shelves are neatly organized and stable.
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.24s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.26s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 8.1s | TTFT: 6.29s | 93 tokens @ 50.0 tok/s

**Model response:**
```

Category: spill
Severity: medium
Confidence: 80%
Location: The warehouse floor shows a large liquid puddle in the center, which is a clear indication of a spill. The puddle is not widespread but is noticeable and could potentially cause slipping hazards. The rest of the warehouse appears to be in good condition with no visible signs of improper stacking or other hazards.
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.0s | TTFT: 6.23s | 186 tokens @ 49.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

Therefore, the classification for this warehouse scene would be:

category: minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating some issues with stacking and organization.
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.0s | TTFT: 6.27s | 188 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

The classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating issues with stacking and organization.
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 100%
- **Time:** 8.2s | TTFT: 6.25s | 98 tokens @ 49.6 tok/s

**Model response:**
```

Category: spill
Severity: high
Confidence: 100%
Location: The warehouse floor shows a large liquid puddle, which is a clear indication of a spill. The puddle is large enough to cover a significant portion of the floor, and it appears to be a liquid that could potentially cause slipping hazards. The rest of the warehouse does not show any obvious signs of improper stacking or other hazards.
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.34s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.26s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.32s | 192 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.23s | 192 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.24s | 201 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.24s | 200 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.38s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.23s | 193 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.29s | 201 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.24s | 201 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.2s | TTFT: 6.48s | 180 tokens @ 49.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which indicates a spill.
- Improper Stacking: The boxes on the shelves are not neatly aligned and appear to be tilted or rotated at an angle. This suggests that the boxes are not stacked properly.
- Minor Hazard: The presence of the liquid puddle and the tilted boxes indicate that there are issues with the storage and organization of items, which could potentially lead to accidents or damage.

The classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor and the shelves where the boxes are stored.
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.33s | 181 tokens @ 49.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.23s | 193 tokens @ 49.7 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or jumbled arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.24s | 199 tokens @ 49.7 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.3s | 193 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.31s | 192 tokens @ 49.7 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.25s | 193 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 9.9s | TTFT: 6.24s | 180 tokens @ 49.7 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or missing shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.3s | 199 tokens @ 49.7 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper_stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no indication of any unstable or falling items.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.2s | TTFT: 6.48s | 192 tokens @ 49.7 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or deformed boxes. The stacks are uniform, and there is no evidence of torn or missing shrink wrap.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.5s | TTFT: 6.33s | 210 tokens @ 49.6 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no indication of any unstable or falling items.

Considering the lack of visible spills, improper stacking, and minor hazards, the scene can be classified as:

category: safe

Severity: low

Confidence: 100%

Location: The image does not provide a clear view of the entire warehouse, but from what is visible, the shelves are well-organized and the floor appears clean and dry.
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.24s | 201 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 9.6s | TTFT: 6.48s | 166 tokens @ 49.6 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and potential spills.

The classification for this scene would be:

category: minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a spill, and the shelves show signs of improper stacking.
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.34s | 189 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. There is no evidence of boxes stacked unevenly or jumbled.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.4s | TTFT: 6.25s | 201 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there is no evidence of loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.0s | TTFT: 6.29s | 181 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The stacks are uniform and stable.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.1s | TTFT: 6.35s | 192 tokens @ 49.5 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. The stacks are uniform, and there is no evidence of unstable or messy arrangements.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.1s | TTFT: 6.43s | 191 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

Therefore, the classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating issues with stacking and organization.
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.1s | TTFT: 6.46s | 173 tokens @ 49.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper Stacking: The boxes on the shelves are not neatly aligned and appear to be tilted or rotated at an angle.
- Minor Hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

The classification for this warehouse scene would be:

category: minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a spill, and the boxes on the shelves are not neatly organized, indicating a minor hazard.
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 80%
- **Time:** 10.2s | TTFT: 6.28s | 188 tokens @ 48.5 tok/s

**Model response:**
```

Based on the image provided, the warehouse scene can be classified as follows:

- Spill: There is a visible liquid puddle on the floor, which could be a spill.
- Improper_stacking: The boxes on the shelves appear to be tilted or rotated at an angle, which could indicate improper stacking.
- Minor_hazard: The overall condition of the warehouse does not appear to be a clear violation, but there are some issues with the stacking and organization of the boxes.

The classification for this warehouse scene would be:

category: improper_stacking / minor_hazard
severity: medium
confidence: 80%
location: The warehouse floor has a liquid puddle, and the boxes on the shelves are tilted or rotated at an angle, indicating issues with stacking and organization.
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 11.0s | TTFT: 6.53s | 223 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn boxes. However, the boxes on the lower shelves are not fully visible, so it is not possible to confirm if they are stacked unevenly or jumbled.
- Minor Hazard: The image is clear, and the warehouse appears to be well-organized. There are no obvious signs of instability or potential hazards.

Therefore, the classification for this warehouse scene is:

category: minor_hazard
severity: low
confidence: 100%
location: The warehouse appears to be well-organized with no visible spills or improper stacking.
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 100%
- **Time:** 10.3s | TTFT: 6.31s | 199 tokens @ 49.4 tok/s

**Model response:**
```

Based on the provided image, the warehouse scene can be classified as follows:

- Spill: There are no visible liquid puddles, wet reflective patches, oil slicks, or chemical leaks on the warehouse floor.
- Improper Stacking: The boxes on the shelves appear to be neatly aligned in rows, with no visible tilting, overhanging, crushing, or torn shrink wrap. The boxes do not seem to be supporting weight above, and there are no loads shifted to one side of the shelf.
- Safe: The floor is clean and dry, and the items on the shelves are neatly organized and stable.

Therefore, the classification for this warehouse scene is:

category: safe
severity: low
confidence: 100%
location: The warehouse floor and shelves are clean and organized, with no visible issues.
```

---

