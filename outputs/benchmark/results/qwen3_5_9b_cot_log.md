# qwen3.5:9b — cot

Date: 2026-03-13 03:49
Test set: 99 images

## Results Summary

**Overall accuracy: 81.8%** (99/99 valid)
**Average time: 55.9s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 97.0% | 1.000 | 0.970 | 0.985 | 32 | 0 | 1 |
| improper_stacking | 48.5% | 1.000 | 0.485 | 0.653 | 16 | 0 | 17 |
| safe | 100.0% | 0.647 | 1.000 | 0.786 | 33 | 18 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 32 | 0 | 1 | 0 |
| improper_stacking | 0 | 16 | 17 | 0 |
| safe | 0 | 0 | 33 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 37.7s | TTFT: 31.74s | 459 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 94.4s | TTFT: 88.38s | 1180 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 90%
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 74.6s | TTFT: 68.17s | 926 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill
confidence: 95%
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 79.5s | TTFT: 77.94s | 989 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO
Answer: YES
category: improper_stacking
confidence: 95%
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 31.4s | TTFT: 29.65s | 369 tokens @ 12.9 tok/s

**Model response:**
```
Answer 1: NO
Answer 2: NO
category: safe
confidence: 95%
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 47.2s | TTFT: 45.15s | 571 tokens @ 12.9 tok/s

**Model response:**
```
Answer to QUESTION 1: NO

Answer to QUESTION 2: NO

category: safe
confidence: 90%
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 30.0s | TTFT: 28.26s | 351 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: YES

QUESTION 2: NO

category: spill
confidence: 95%
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 64.5s | TTFT: 58.59s | 793 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.9s | TTFT: 32.9s | 462 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 34.1s | TTFT: 32.07s | 407 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: YES
Answer to QUESTION 2: NO
category: spill
confidence: 95%
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 42.6s | TTFT: 40.63s | 516 tokens @ 13.0 tok/s

**Model response:**
```
Answer to Question 1: YES
Answer to Question 2: NO

category: spill
confidence: 95%
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 46.1s | TTFT: 40.16s | 559 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 46.9s | TTFT: 40.87s | 570 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 171.2s | TTFT: 165.24s | 2148 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 32.9s | TTFT: 31.51s | 391 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 33.4s | TTFT: 27.36s | 396 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 52.4s | TTFT: 46.47s | 630 tokens @ 12.8 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 66.1s | TTFT: 59.97s | 815 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 100%
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 39.5s | TTFT: 33.55s | 475 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 32.9s | TTFT: 30.88s | 392 tokens @ 13.1 tok/s

**Model response:**
```
Answer to Question 1: NO
Answer to Question 2: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 135.8s | TTFT: 129.81s | 1700 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.4s | TTFT: 36.7s | 461 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: NO
QUESTION 2: NO

category: safe
confidence: 95%
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 35.1s | TTFT: 33.05s | 419 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO
Answer to QUESTION 2: NO
category: safe
confidence: 95%
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 36.0s | TTFT: 30.01s | 431 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 38.1s | TTFT: 30.06s | 457 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
category: spill
confidence: 95%
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 46.5s | TTFT: 40.33s | 563 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 31.1s | TTFT: 29.36s | 368 tokens @ 13.0 tok/s

**Model response:**
```
Question 1: YES
Question 2: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 96.0s | TTFT: 93.79s | 1198 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO
Answer to QUESTION 2: YES
category: improper_stacking
confidence: 85%
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 64.3s | TTFT: 62.76s | 792 tokens @ 13.0 tok/s

**Model response:**
```
Answer: YES
Answer: NO
category: spill
confidence: 95%
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 32.1s | TTFT: 30.06s | 381 tokens @ 13.0 tok/s

**Model response:**
```
Answer for Question 1: NO
Answer for Question 2: NO
category: safe
confidence: 95%
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 37.4s | TTFT: 35.3s | 448 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO

Answer to QUESTION 2: NO

category: safe
confidence: 95%
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 36.2s | TTFT: 34.86s | 434 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO

Answer: NO

category: safe
confidence: 100%
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 37.0s | TTFT: 34.67s | 444 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: NO

category: spill / improper_stacking / safe -> spill
confidence: 100%
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 63.5s | TTFT: 61.19s | 784 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO
Answer to QUESTION 2: YES

category: improper_stacking / safe
confidence: 95%
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 73.7s | TTFT: 71.78s | 910 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: NO
QUESTION 2: YES
category: improper_stacking
confidence: 85%
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 71.8s | TTFT: 65.71s | 887 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 90%
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 85.9s | TTFT: 84.04s | 1066 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: NO
QUESTION 2: NO

category: safe
confidence: 90%
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 33.0s | TTFT: 31.03s | 393 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO

Answer to QUESTION 2: NO

category: safe
confidence: 95%
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 35.7s | TTFT: 29.41s | 406 tokens @ 12.4 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 77.8s | TTFT: 71.66s | 946 tokens @ 12.8 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 95%
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 57.7s | TTFT: 55.69s | 710 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: YES
Answer to QUESTION 2: NO

category: spill
confidence: 95%
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 37.5s | TTFT: 31.52s | 452 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 100%
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 34.8s | TTFT: 33.02s | 404 tokens @ 12.7 tok/s

**Model response:**
```
QUESTION 1: NO

QUESTION 2: NO

category: safe
confidence: 95%
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 99.2s | TTFT: 93.21s | 1216 tokens @ 12.8 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 59.7s | TTFT: 53.48s | 736 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 95%
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 57.6s | TTFT: 51.84s | 717 tokens @ 13.2 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.9s | TTFT: 33.12s | 481 tokens @ 13.4 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 61.5s | TTFT: 60.13s | 772 tokens @ 13.3 tok/s

**Model response:**
```
Answer: NO
Answer: NO
category: safe
confidence: 95%
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 74.8s | TTFT: 68.89s | 932 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 36.8s | TTFT: 34.71s | 443 tokens @ 13.1 tok/s

**Model response:**
```
Answer to QUESTION 1: NO
Answer to QUESTION 2: NO

category: safe
confidence: 100%
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 44.3s | TTFT: 42.49s | 537 tokens @ 13.0 tok/s

**Model response:**
```
Answer 1: YES
Answer 2: NO

category: spill
confidence: 95%
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 50.3s | TTFT: 48.26s | 614 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: NO
Answer to QUESTION 2: NO

category: safe
confidence: 95%
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 51.1s | TTFT: 49.36s | 620 tokens @ 12.9 tok/s

**Model response:**
```
Answer for Question 1: NO

Answer for Question 2: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 44.6s | TTFT: 43.08s | 540 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO
Answer: YES
category: improper_stacking / confidence: 95%
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 95.6s | TTFT: 93.77s | 1188 tokens @ 13.0 tok/s

**Model response:**
```
Question 1: NO
Question 2: YES
category: improper_stacking
confidence: 95%
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 39.9s | TTFT: 38.49s | 481 tokens @ 13.0 tok/s

**Model response:**
```
Answer: YES
Answer: NO
category: spill
confidence: 95%
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 31.6s | TTFT: 30.33s | 375 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO
Answer: NO
category: safe
confidence: 95%
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 32.4s | TTFT: 31.1s | 383 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO

Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 28.2s | TTFT: 26.83s | 331 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO
Answer: NO
category: safe
confidence: 95%
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 34.0s | TTFT: 28.04s | 405 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 36.2s | TTFT: 34.17s | 435 tokens @ 13.1 tok/s

**Model response:**
```
Answer to QUESTION 1: NO

Answer to QUESTION 2: NO

category: safe
confidence: 95%
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 35.2s | TTFT: 28.85s | 423 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill / improper_stacking / safe
confidence: 95%
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 36.2s | TTFT: 30.33s | 436 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 29.3s | TTFT: 27.57s | 346 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: NO

category: spill
confidence: 95%
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 41.8s | TTFT: 39.72s | 505 tokens @ 13.0 tok/s

**Model response:**
```
Answer to QUESTION 1: YES
Answer to QUESTION 2: NO
category: spill
confidence: 95%
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 66.3s | TTFT: 60.33s | 820 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 114.3s | TTFT: 112.83s | 1428 tokens @ 13.0 tok/s

**Model response:**
```
Answer: YES

Answer: NO

category: spill
confidence: 95%
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 25.2s | TTFT: 23.22s | 294 tokens @ 13.1 tok/s

**Model response:**
```
Answer to QUESTION 1: NO

Answer to QUESTION 2: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 49.9s | TTFT: 48.34s | 615 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO

Answer: YES

category: improper_stacking
confidence: 95%
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.8s | TTFT: 32.76s | 465 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 45.1s | TTFT: 43.14s | 549 tokens @ 13.1 tok/s

**Model response:**
```
Answer to QUESTION 1: YES
Answer to QUESTION 2: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 87.9s | TTFT: 83.58s | 1096 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO

Question 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 55.9s | TTFT: 54.44s | 688 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO

Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 165.5s | TTFT: 159.55s | 2080 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 47.3s | TTFT: 41.4s | 577 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 48.7s | TTFT: 42.86s | 592 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 58.0s | TTFT: 52.07s | 713 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 100%
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 98%
- **Time:** 53.1s | TTFT: 51.41s | 652 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: NO

category: spill
confidence: 98%
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 43.9s | TTFT: 37.53s | 533 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill / improper_stacking / safe
confidence: 95%
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 58.6s | TTFT: 52.63s | 722 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking / confidence: 95%
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 57.9s | TTFT: 51.92s | 712 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 95.9s | TTFT: 89.63s | 1179 tokens @ 12.8 tok/s

**Model response:**
```
Question 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

Question 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 95%
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 35.7s | TTFT: 29.75s | 424 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 72.7s | TTFT: 66.57s | 901 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 90%
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 149.4s | TTFT: 143.22s | 1878 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 90%
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 31.3s | TTFT: 25.25s | 371 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 100%
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 32.3s | TTFT: 30.92s | 383 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO

Answer: NO

category: safe
confidence: 95%
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 32.2s | TTFT: 30.74s | 384 tokens @ 13.1 tok/s

**Model response:**
```
Answer: NO
Answer: NO
category: safe
confidence: 100%
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 37.5s | TTFT: 31.63s | 450 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 41.5s | TTFT: 33.4s | 501 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
**spill**
confidence: **100%**
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 44.0s | TTFT: 41.96s | 528 tokens @ 12.9 tok/s

**Model response:**
```
Answer to QUESTION 1: YES

Answer to QUESTION 2: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 95.8s | TTFT: 89.87s | 1195 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 201.9s | TTFT: 195.84s | 2542 tokens @ 12.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: safe
confidence: 95%
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.2s | TTFT: 32.08s | 459 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

category: improper_stacking
confidence: 95%
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 51.5s | TTFT: 45.09s | 631 tokens @ 13.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill / improper_stacking / safe
confidence: 100%
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 37.1s | TTFT: 31.15s | 446 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 90%
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 38.0s | TTFT: 32.04s | 457 tokens @ 13.1 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

category: spill
confidence: 95%
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 63.8s | TTFT: 61.91s | 786 tokens @ 13.0 tok/s

**Model response:**
```
Answer 1: NO
Answer 2: YES

category: improper_stacking
confidence: 90%
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 29.0s | TTFT: 27.55s | 340 tokens @ 13.0 tok/s

**Model response:**
```
Answer: NO

Answer: NO

category: safe
confidence: 95%
```

---

