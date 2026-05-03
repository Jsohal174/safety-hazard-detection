# qwen3-vl:2b — cot (nothink)

Date: 2026-03-12 17:24
Test set: 99 images

## Results Summary

**Overall accuracy: 68.4%** (98/99 valid)
**Average time: 8.4s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 84.8% | 0.903 | 0.848 | 0.875 | 28 | 3 | 5 |
| improper_stacking | 21.2% | 1.000 | 0.212 | 0.350 | 7 | 0 | 26 |
| safe | 100.0% | 0.533 | 1.000 | 0.696 | 32 | 28 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 28 | 0 | 5 | 0 |
| improper_stacking | 3 | 7 | 23 | 0 |
| safe | 0 | 0 | 32 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 4.47s | 301 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.13s | 200 tokens @ 75.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 8.5s | TTFT: 7.3s | 545 tokens @ 72.9 tok/s

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

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 8.8s | TTFT: 8.45s | 578 tokens @ 73.8 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.2s | TTFT: 3.07s | 249 tokens @ 74.9 tok/s

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

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.8s | TTFT: 2.6s | 221 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: NO  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 5.2s | TTFT: 4.78s | 328 tokens @ 74.8 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

Final classification:  
category: spill  
confidence: 100%
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 30.0s | TTFT: 29.59s | 1989 tokens @ 70.8 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 3.07s | 189 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.6s | TTFT: 2.36s | 147 tokens @ 76.0 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 3.27s | 217 tokens @ 75.1 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 12.4s | TTFT: 11.16s | 826 tokens @ 73.4 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: YES  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: NO  

classification: spill / improper_stacking / safe  
confidence: 90%
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.5s | TTFT: 4.19s | 278 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.1s | TTFT: 2.9s | 249 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: NO  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.8s | TTFT: 6.82s | 436 tokens @ 74.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.78s | 181 tokens @ 75.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 7.0s | TTFT: 6.63s | 457 tokens @ 74.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

Final classification:  
category: spill  
confidence: 100%
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 5.4s | TTFT: 5.06s | 340 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 5.0s | TTFT: 4.69s | 310 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.9s | TTFT: 3.56s | 238 tokens @ 75.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.0s | TTFT: 5.7s | 381 tokens @ 74.6 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.7s | TTFT: 3.38s | 226 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.17s | 208 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 4.5s | TTFT: 4.18s | 277 tokens @ 74.8 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

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
- **Time:** 6.9s | TTFT: 6.58s | 450 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 10.6s | TTFT: 10.07s | 679 tokens @ 73.6 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 90%
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 3.22s | 216 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 25.9s | TTFT: 25.53s | 1728 tokens @ 71.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: YES  

category: improper_stacking  
confidence: 100%
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 12.4s | TTFT: 11.34s | 813 tokens @ 73.3 tok/s

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

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.8s | TTFT: 2.52s | 163 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.24s | 206 tokens @ 75.7 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.4s | TTFT: 3.04s | 196 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.5s | TTFT: 3.46s | 282 tokens @ 74.6 tok/s

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

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 26.6s | TTFT: 25.35s | 1774 tokens @ 71.2 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: NO  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: YES  

Final classification:  
category: improper_stacking  
confidence: 95%
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.2s | TTFT: 2.92s | 191 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.2s | TTFT: 2.85s | 188 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.9s | TTFT: 4.58s | 301 tokens @ 73.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.7s | TTFT: 3.34s | 215 tokens @ 74.8 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.0s | TTFT: 2.81s | 238 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: NO  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 17.1s | TTFT: 16.81s | 1146 tokens @ 72.5 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 80%
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 21.0s | TTFT: 20.63s | 1396 tokens @ 71.6 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.0s | TTFT: 3.67s | 231 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.16s | 210 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 13.7s | TTFT: 13.29s | 916 tokens @ 73.1 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

Final classification:  
category: safe  
confidence: 100%
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 17.5s | TTFT: 17.14s | 1162 tokens @ 72.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: YES  

category: improper_stacking  
confidence: 90%
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.6s | TTFT: 2.26s | 145 tokens @ 75.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.4s | TTFT: 3.83s | 239 tokens @ 74.6 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 14.0s | TTFT: 13.52s | 927 tokens @ 72.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: YES  

Final classification:  
category: improper_stacking  
confidence: 90%
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.5s | TTFT: 6.29s | 423 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.5s | TTFT: 3.43s | 282 tokens @ 74.6 tok/s

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

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 45.3s | TTFT: 44.94s | 2948 tokens @ 68.9 tok/s

**Model response:**
```
YES  
NO  

**Final Classification**  
category: spill  
confidence: 100%
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.9s | TTFT: 3.63s | 237 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.8s | TTFT: 2.72s | 222 tokens @ 74.6 tok/s

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

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 11.5s | TTFT: 10.33s | 767 tokens @ 73.4 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: NO  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: YES  

Final classification:  
category: improper_stacking  
confidence: 80%
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 34.9s | TTFT: 34.56s | 2304 tokens @ 70.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: YES  

category: spill  
confidence: 90%
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.9s | TTFT: 4.58s | 309 tokens @ 74.6 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 2.93s | 194 tokens @ 75.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.14s | 203 tokens @ 75.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.1s | TTFT: 2.98s | 247 tokens @ 75.0 tok/s

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

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.73s | 170 tokens @ 75.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.6s | TTFT: 3.24s | 216 tokens @ 75.8 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 32.5s | TTFT: 31.96s | 2143 tokens @ 70.6 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.62s | 164 tokens @ 75.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

final classification: category: safe, confidence: 100%
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 3.9s | TTFT: 3.5s | 236 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

Final classification:  
category: spill  
confidence: 90%
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 4.4s | TTFT: 4.05s | 263 tokens @ 74.9 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 90%
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 9.4s | TTFT: 9.06s | 623 tokens @ 74.0 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 7.9s | TTFT: 7.56s | 519 tokens @ 74.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 2.93s | 193 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.2s | TTFT: 2.95s | 189 tokens @ 75.6 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.9s | TTFT: 2.57s | 168 tokens @ 74.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.74s | 167 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.0s | TTFT: 3.67s | 242 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.7s | TTFT: 2.35s | 152 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.2s | TTFT: 2.86s | 184 tokens @ 73.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 2.93s | 191 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.67s | 168 tokens @ 72.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 9.6s | TTFT: 9.05s | 540 tokens @ 66.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.7s | TTFT: 3.34s | 219 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 10.8s | TTFT: 10.42s | 711 tokens @ 73.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 90%
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 20.3s | TTFT: 19.95s | 1356 tokens @ 72.1 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: YES  

category: improper_stacking  
confidence: 90%
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 3.8s | 291 tokens @ 74.9 tok/s

**Model response:**
```
QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?  
Answer: YES  

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?  
Answer: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.9s | TTFT: 3.66s | 237 tokens @ 75.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.4s | TTFT: 2.99s | 196 tokens @ 75.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.1s | TTFT: 2.66s | 171 tokens @ 76.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.21s | 209 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.8s | TTFT: 2.42s | 149 tokens @ 76.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.6s | TTFT: 6.28s | 425 tokens @ 74.5 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.16s | 200 tokens @ 75.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.7s | TTFT: 3.48s | 230 tokens @ 75.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.8s | TTFT: 4.5s | 301 tokens @ 74.8 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 25.5s | TTFT: 25.15s | 1708 tokens @ 71.6 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 51.9s | TTFT: 51.53s | 3365 tokens @ 68.5 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 95%
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.0s | TTFT: 5.62s | 367 tokens @ 74.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.18s | 213 tokens @ 75.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 6.2s | TTFT: 5.85s | 399 tokens @ 74.7 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 90%
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 9.6s | TTFT: 9.28s | 639 tokens @ 74.1 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 14.5s | TTFT: 14.11s | 971 tokens @ 73.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 31.6s | TTFT: 31.25s | 2106 tokens @ 70.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: YES  

category: improper_stacking  
confidence: 80%
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.5s | TTFT: 3.14s | 208 tokens @ 75.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

