# gemma3:4b — cot (nothink)

Date: 2026-03-12 15:14
Test set: 99 images

## Results Summary

**Overall accuracy: 36.4%** (99/99 valid)
**Average time: 6.8s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 9.1% | 1.000 | 0.091 | 0.167 | 3 | 0 | 30 |
| improper_stacking | 100.0% | 0.344 | 1.000 | 0.512 | 33 | 63 | 0 |
| safe | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0 | 33 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 3 | 30 | 0 | 0 |
| improper_stacking | 0 | 33 | 0 | 0 |
| safe | 0 | 33 | 0 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.65s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 7.0s | TTFT: 3.82s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 80%
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.68s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here’s the breakdown of the warehouse image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.73s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.6s | TTFT: 3.64s | 103 tokens @ 35.6 tok/s

**Model response:**
```
Okay, let's analyze the image.

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.6s | TTFT: 3.61s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.6s | TTFT: 3.66s | 103 tokens @ 35.8 tok/s

**Model response:**
```
Here’s the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.66s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.8s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 6.3s | TTFT: 3.62s | 93 tokens @ 35.9 tok/s

**Model response:**
```
Here’s the analysis based on the image:

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Final Classification:

category: improper_stacking
confidence: 75%
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 4.01s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.2s | TTFT: 4.16s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.6s | TTFT: 3.63s | 103 tokens @ 35.9 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.64s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.6s | TTFT: 3.64s | 103 tokens @ 35.6 tok/s

**Model response:**
```
Here's my assessment of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.65s | 103 tokens @ 35.5 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.77s | 103 tokens @ 35.8 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.68s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 6.7s | TTFT: 3.66s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 80%
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.73s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.3s | TTFT: 4.19s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.71s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.3s | TTFT: 3.69s | 104 tokens @ 31.4 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.1s | TTFT: 4.0s | 104 tokens @ 34.3 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.86s | 104 tokens @ 35.1 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.87s | 104 tokens @ 35.2 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.85s | 104 tokens @ 35.2 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.83s | 104 tokens @ 35.3 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.91s | 104 tokens @ 35.3 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.0s | TTFT: 3.84s | 104 tokens @ 34.5 tok/s

**Model response:**
```
Here’s the analysis based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 103 tokens @ 34.8 tok/s

**Model response:**
```
Here’s the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.0s | TTFT: 3.85s | 103 tokens @ 35.3 tok/s

**Model response:**
```
Here's my analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.74s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.7s | TTFT: 3.73s | 103 tokens @ 35.8 tok/s

**Model response:**
```
Here’s the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 104 tokens @ 35.0 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.74s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.83s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's my assessment based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 103 tokens @ 35.7 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.79s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's my analysis of the warehouse image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 103 tokens @ 35.4 tok/s

**Model response:**
```
Here's my analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.79s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 103 tokens @ 35.5 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.75s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.0s | TTFT: 3.76s | 104 tokens @ 35.9 tok/s

**Model response:**
```
Here's the analysis based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 4.18s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 75%
- **Time:** 7.0s | TTFT: 3.76s | 112 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**
category: spill
confidence: 75% (There's a visible liquid spill and boxes that appear disorganized/potentially unstable.)
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.9s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.77s | 104 tokens @ 34.9 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's my analysis based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 75%
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 75%
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.0s | TTFT: 3.78s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 4.18s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.5s | TTFT: 3.76s | 93 tokens @ 35.7 tok/s

**Model response:**
```
Here's the analysis based on the image:

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Final classification:
category: improper_stacking
confidence: 85%
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.89s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 6.8s | TTFT: 3.78s | 101 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** spill
*   **confidence:** 95%
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.79s | 101 tokens @ 35.5 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** spill
*   **confidence:** 85%
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.81s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.0s | TTFT: 3.86s | 104 tokens @ 34.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.3s | TTFT: 4.19s | 104 tokens @ 34.8 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 4.19s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.81s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 103 tokens @ 35.7 tok/s

**Model response:**
```
Here's my analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 103 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.79s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here’s the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 4.01s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.3 tok/s

**Model response:**
```
Okay, let’s analyze the warehouse image.

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 80%
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.77s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.76s | 103 tokens @ 35.5 tok/s

**Model response:**
```
Here's the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**
Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**
Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 4.12s | 103 tokens @ 35.5 tok/s

**Model response:**
```
Here's my analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 7.2s | TTFT: 4.23s | 103 tokens @ 35.5 tok/s

**Model response:**
```
Here’s the analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.9s | TTFT: 3.78s | 104 tokens @ 34.8 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 103 tokens @ 35.7 tok/s

**Model response:**
```
Here's my analysis of the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.78s | 104 tokens @ 35.5 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: YES

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 6.6s | TTFT: 3.78s | 93 tokens @ 35.6 tok/s

**Model response:**
```
Here’s the breakdown based on the image:

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Final Classification:
category: improper_stacking
confidence: 80%
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.79s | 104 tokens @ 35.7 tok/s

**Model response:**
```
Here's the breakdown of the image analysis:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 6.8s | TTFT: 3.86s | 104 tokens @ 35.6 tok/s

**Model response:**
```
Here's the breakdown based on the image:

**QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?**

Answer: NO

**QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?**

Answer: YES

**Final Classification:**

*   **category:** improper_stacking
*   **confidence:** 85%
```

---

