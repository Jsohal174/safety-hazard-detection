# gemma3:4b — simple (nothink)

Date: 2026-03-12 15:03
Test set: 99 images

## Results Summary

**Overall accuracy: 33.3%** (99/99 valid)
**Average time: 5.5s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0 | 33 |
| improper_stacking | 100.0% | 0.333 | 1.000 | 0.500 | 33 | 66 | 0 |
| safe | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0 | 33 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 0 | 33 | 0 | 0 |
| improper_stacking | 0 | 33 | 0 | 0 |
| safe | 0 | 33 | 0 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 5.24s | 40 tokens @ 30.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted and stacked unevenly on the shelves on both sides of the image.
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 95%
- **Time:** 5.7s | TTFT: 4.69s | 40 tokens @ 35.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 95%
location: Several boxes on the shelves appear to be stacked somewhat unevenly and not neatly aligned in rows.
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.19s | 42 tokens @ 35.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and/or not neatly aligned, suggesting a potential for instability.
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.7s | TTFT: 4.15s | 51 tokens @ 35.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves on the right side of the image. Some boxes are tilted slightly, and the overall arrangement appears disorganized.
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 7.9s | TTFT: 6.81s | 34 tokens @ 33.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Boxes on shelves appear somewhat disorganized, with some boxes leaning slightly.
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 6.8s | TTFT: 5.77s | 43 tokens @ 32.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear slightly askew on the shelves, indicating potential instability and a need for review of stacking practices.
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 7.7s | TTFT: 6.61s | 40 tokens @ 35.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted or not neatly aligned, suggesting a potential instability issue.
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 6.0s | TTFT: 4.72s | 40 tokens @ 34.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves, with some boxes tilted and not aligned neatly.
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.32s | 34 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.24s | 41 tokens @ 35.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right shelves appear slightly tilted and not fully supported, presenting a potential instability hazard.
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.5s | TTFT: 4.39s | 40 tokens @ 35.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.3s | 42 tokens @ 35.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly and slightly tilted on the shelves on both sides of the image.
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 4.34s | 34 tokens @ 34.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and not neatly stacked.
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.7s | TTFT: 4.48s | 45 tokens @ 35.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right-hand shelves appear to be slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.7s | TTFT: 4.46s | 40 tokens @ 35.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.6s | TTFT: 4.41s | 40 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right side shelf appear to be leaning at an angle and not stacked neatly.
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 4.39s | 39 tokens @ 35.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves on both sides, creating an unstable appearance.
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 4.51s | 43 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear to be stacked unevenly and slightly tilted on the shelves on the right side of the image.
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.8s | TTFT: 4.43s | 46 tokens @ 35.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right-hand shelves appear to be stacked slightly askew, with some boxes tilted and not neatly aligned.
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.43s | 36 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly stacked in rows.
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.27s | 42 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be stacked unevenly and slightly tilted, suggesting a potential instability issue.
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.28s | 42 tokens @ 35.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear stacked unevenly and slightly askew on the shelves on the right side of the image.
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.25s | 34 tokens @ 34.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves on both sides.
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.22s | 43 tokens @ 35.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and out of alignment, suggesting a need for organization and stability checks.
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.24s | 36 tokens @ 35.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly stacked in rows.
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.25s | 41 tokens @ 35.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be slightly tilted and not neatly aligned, creating an unstable appearance.
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.23s | 40 tokens @ 36.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 4.27s | 40 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted or not perfectly aligned, suggesting a potential instability issue.
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.22s | 38 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly on shelves, with some boxes leaning slightly.
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.5s | TTFT: 4.32s | 42 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly and at an angle on the shelves on the right side of the image.
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.6s | TTFT: 4.23s | 48 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be slightly tilted or not neatly aligned, suggesting a need for organization and attention to proper stacking techniques.
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.33s | 35 tokens @ 36.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted on the shelves, suggesting a potential instability.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.8s | TTFT: 4.23s | 53 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly on the shelves on both sides of the image. Some boxes are slightly tilted, and the overall arrangement lacks neat rows.
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.21s | 36 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes are stacked on shelves with an uneven appearance, suggesting potential instability.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.24s | 40 tokens @ 35.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, indicating a potential instability issue.
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.25s | 34 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned.
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.21s | 34 tokens @ 34.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and out of alignment.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.2s | 40 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.2s | 37 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes are stacked unevenly on the shelves, with some boxes leaning slightly.
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 95%
- **Time:** 5.7s | TTFT: 4.21s | 42 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 95%
location: Several boxes appear to be stacked unevenly and slightly tilted on the shelves, creating an unstable visual impression.
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.19s | 40 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.19s | 41 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be slightly tilted and not neatly stacked, creating a potential instability.
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.15s | 40 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential for instability.
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.19s | 43 tokens @ 35.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted on the shelves on the right side of the image, indicating a potential instability issue.
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.7s | TTFT: 4.14s | 55 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be stacked unevenly and at slight angles.  Specifically, the boxes on the right side of the image seem to be leaning slightly.
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.11s | 46 tokens @ 36.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be slightly leaning or not perfectly aligned, suggesting a need for organization and potentially unstable stacking.
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.12s | 40 tokens @ 35.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked somewhat unevenly on the shelves, with some boxes leaning slightly.
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.18s | 38 tokens @ 35.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly askew on the shelves on the right side of the image.
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.9s | TTFT: 4.3s | 55 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly on the shelves on the right side of the image. Some boxes are leaning slightly, suggesting a lack of proper stability and alignment.
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 95%
- **Time:** 5.1s | TTFT: 4.19s | 35 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 95%
location: Several boxes are stacked unevenly and appear slightly disorganized on the shelves.
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 4.13s | 43 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear to be slightly askew on the shelves, suggesting a lack of proper organization and potential instability.
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.19s | 37 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right shelves appear to be slightly tilted and stacked unevenly.
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 4.13s | 43 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be stacked unevenly and at odd angles, presenting a potential instability hazard.
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.11s | 41 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and are not neatly aligned, suggesting a potential instability issue.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.1s | TTFT: 4.11s | 36 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly and slightly tilted on the shelves.
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.11s | 45 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be slightly tilted and not stacked neatly, particularly on the right side of the image.
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.1s | 40 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.1s | 41 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked slightly unevenly on the shelves on the right side of the image.
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.1s | TTFT: 4.09s | 37 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted on the shelves on the right side of the image.
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.1s | 40 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not stacked neatly, creating an unstable visual impression.
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.2s | TTFT: 4.12s | 38 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves on the right side of the image.
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.11s | 41 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be slightly tilted or not fully aligned, creating a potential instability.
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 4.65s | 45 tokens @ 35.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear to be stacked unevenly on the shelves, with some boxes tilted slightly and not aligned neatly in rows.
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 4.14s | 34 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear slightly tilted or unevenly stacked on the shelves.
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.6s | TTFT: 4.32s | 47 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly disorganized and not neatly stacked in rows. Some boxes are tilted slightly, though not dangerously so.
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.08s | 38 tokens @ 35.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes are stacked unevenly and slightly askew on the shelves on both sides.
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 95%
- **Time:** 5.3s | TTFT: 4.08s | 42 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 95%
location: Several boxes are stacked on shelves appear unstable and not neatly aligned. Some boxes are leaning at an angle.
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.2s | TTFT: 4.08s | 39 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked somewhat haphazardly on the shelves, with some boxes appearing slightly tilted.
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.1s | TTFT: 4.09s | 36 tokens @ 36.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not stacked neatly in rows.
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.12s | 42 tokens @ 35.9 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Boxes are stacked on the shelves, but there's no obvious instability or significant issues with the arrangement.
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.2s | TTFT: 4.09s | 37 tokens @ 36.0 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear slightly tilted on the shelves on the right side of the image.
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.14s | 38 tokens @ 36.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked unevenly on the shelves, with some leaning slightly.
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.19s | 39 tokens @ 35.8 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted on the shelves on both sides, indicating a potential instability issue.
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 4.39s | 40 tokens @ 35.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, indicating a potential instability issue.
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.13s | 40 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 4.15s | 35 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the right shelves appear slightly tilted and not neatly stacked.
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.13s | 43 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes appear to be stacked unevenly and slightly askew on the shelves, presenting a potential instability hazard.
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.5s | TTFT: 4.09s | 45 tokens @ 34.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be stacked unevenly and slightly tilted, particularly on the right side of the image.
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.4s | TTFT: 4.13s | 45 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be stacked at an angle and are not neatly aligned, creating an unstable visual impression.
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.18s | 46 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be stacked somewhat unevenly and slightly tilted, particularly on the right side of the image.
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.1s | TTFT: 4.09s | 36 tokens @ 36.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the right side shelf appear slightly tilted and not stacked neatly.
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.0s | TTFT: 4.15s | 33 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted or misaligned on the shelves.
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 4.09s | 36 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves, creating an unstable appearance.
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.09s | 40 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes are stacked unevenly on the shelves, some appear slightly tilted and not neatly aligned.
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.3s | TTFT: 4.1s | 43 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be stacked unevenly and at an angle, creating a potential instability hazard.
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.1s | 40 tokens @ 35.1 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned, suggesting a potential instability issue.
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 6.1s | TTFT: 4.56s | 43 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not stacked neatly, particularly on the right side of the image.
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.2s | 38 tokens @ 36.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes are stacked unevenly on the shelves, some are tilted and appear unstable.
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.4s | TTFT: 4.18s | 43 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly unevenly stacked and one box is leaning, suggesting a potential instability issue.
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.7s | TTFT: 4.49s | 43 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly off-kilter and not perfectly aligned, suggesting a need for organization.
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.3s | TTFT: 4.06s | 42 tokens @ 36.5 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear to be stacked unevenly and slightly tilted, suggesting a potential instability issue.
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.0s | TTFT: 4.07s | 34 tokens @ 36.7 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly aligned.
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.9s | TTFT: 4.71s | 43 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear to be stacked slightly unevenly on the shelves, with some boxes rotated and not neatly aligned.
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.2s | TTFT: 4.07s | 40 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly on the shelves, some appear slightly tilted and not aligned neatly.
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 4.08s | 40 tokens @ 36.3 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes on the shelves appear slightly tilted and not neatly stacked, suggesting a potential instability issue.
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 10.1s | TTFT: 8.9s | 42 tokens @ 36.2 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear slightly tilted and/or stacked unevenly, presenting a potential instability hazard.
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.2s | TTFT: 4.07s | 41 tokens @ 36.4 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes are stacked unevenly and slightly askew on the shelves, indicating a potential instability issue.
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** low
- **Confidence:** 90%
- **Time:** 5.1s | TTFT: 4.07s | 36 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 90%
location: Several boxes on the shelves appear to be slightly tilted and not stacked neatly.
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 85%
- **Time:** 5.1s | TTFT: 4.11s | 37 tokens @ 36.6 tok/s

**Model response:**
```
category: improper_stacking
severity: low
confidence: 85%
location: Several boxes appear slightly tilted on the shelves on the right side of the image.
```

---

