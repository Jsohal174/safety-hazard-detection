# moondream:latest — simple (nothink)

Date: 2026-03-12 18:16
Test set: 99 images

## Results Summary

**Overall accuracy: 33.3%** (99/99 valid)
**Average time: 1.6s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 100.0% | 0.333 | 1.000 | 0.500 | 33 | 66 | 0 |
| improper_stacking | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0 | 33 |
| safe | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0 | 33 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 33 | 0 | 0 | 0 |
| improper_stacking | 33 | 0 | 0 | 0 |
| safe | 33 | 0 | 0 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.9s | TTFT: 1.94s | 15 tokens @ 106.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 106.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.36s | 15 tokens @ 103.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 99.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.28s | 15 tokens @ 106.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.39s | 15 tokens @ 108.5 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.32s | 15 tokens @ 105.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 106.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.38s | 15 tokens @ 108.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 104.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 102.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.38s | 15 tokens @ 108.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 105.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 104.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.32s | 15 tokens @ 107.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 107.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.29s | 15 tokens @ 106.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.36s | 15 tokens @ 108.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.32s | 15 tokens @ 100.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.39s | 15 tokens @ 97.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 107.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 105.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 106.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 106.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 103.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 104.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.34s | 15 tokens @ 107.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.4s | 15 tokens @ 90.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 103.5 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 104.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.3s | 15 tokens @ 106.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 104.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.32s | 15 tokens @ 107.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 106.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.35s | 15 tokens @ 105.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.41s | 15 tokens @ 104.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.31s | 15 tokens @ 107.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.7s | TTFT: 1.75s | 15 tokens @ 104.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.39s | 15 tokens @ 103.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.49s | 15 tokens @ 107.5 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 107.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.32s | 15 tokens @ 105.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.33s | 15 tokens @ 104.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.57s | 15 tokens @ 105.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 100.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.7s | TTFT: 1.66s | 15 tokens @ 105.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.48s | 15 tokens @ 102.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 107.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.7s | TTFT: 1.66s | 15 tokens @ 106.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.53s | 15 tokens @ 102.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.64s | 15 tokens @ 107.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.46s | 15 tokens @ 104.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.36s | 15 tokens @ 104.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 103.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.64s | 15 tokens @ 107.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.51s | 15 tokens @ 105.5 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.58s | 15 tokens @ 103.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.57s | 15 tokens @ 107.5 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.57s | 15 tokens @ 106.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.36s | 15 tokens @ 104.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.38s | 15 tokens @ 103.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.35s | 15 tokens @ 105.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.55s | 15 tokens @ 103.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.43s | 15 tokens @ 106.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.4s | 15 tokens @ 104.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.57s | 15 tokens @ 102.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.4s | 15 tokens @ 104.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 105.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.63s | 15 tokens @ 103.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.53s | 15 tokens @ 104.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.36s | 15 tokens @ 104.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.42s | 15 tokens @ 106.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.39s | 15 tokens @ 104.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.39s | 15 tokens @ 104.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.34s | 15 tokens @ 103.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 107.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.38s | 15 tokens @ 103.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.4s | 15 tokens @ 104.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.38s | 15 tokens @ 104.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 104.0 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.39s | 15 tokens @ 104.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.37s | 15 tokens @ 105.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.63s | 15 tokens @ 104.6 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.51s | 15 tokens @ 105.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.65s | 15 tokens @ 104.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.49s | 15 tokens @ 96.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.55s | 15 tokens @ 106.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.64s | 15 tokens @ 105.2 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.49s | 15 tokens @ 104.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.54s | 15 tokens @ 103.1 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.64s | 15 tokens @ 106.7 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.53s | 15 tokens @ 107.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.6s | 15 tokens @ 104.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.51s | 15 tokens @ 104.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.58s | 15 tokens @ 102.4 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.6s | TTFT: 1.64s | 15 tokens @ 104.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.49s | 15 tokens @ 104.9 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.52s | 15 tokens @ 104.8 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** high
- **Confidence:** None%
- **Time:** 1.7s | TTFT: 1.65s | 15 tokens @ 105.3 tok/s

**Model response:**
```

 category: spill 
 severity: high 
 confidence: 0
```

---

