# moondream:latest — cot (nothink)

Date: 2026-03-12 18:18
Test set: 99 images

## Results Summary

**Overall accuracy: 33.3%** (99/99 valid)
**Average time: 1.3s per image**

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
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.41s | 5 tokens @ 108.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 126.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.16s | 5 tokens @ 124.1 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.21s | 5 tokens @ 121.5 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 126.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.25s | 5 tokens @ 126.4 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.22s | 5 tokens @ 126.8 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 123.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.33s | 5 tokens @ 123.4 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.15s | 5 tokens @ 123.6 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.33s | 5 tokens @ 126.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.19s | 5 tokens @ 123.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.16s | 5 tokens @ 123.3 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 124.7 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 123.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.26s | 5 tokens @ 127.2 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 126.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 119.6 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.28s | 5 tokens @ 124.4 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 124.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 125.7 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 125.2 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.16s | 5 tokens @ 126.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.22s | 5 tokens @ 126.3 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 123.3 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.33s | 5 tokens @ 123.7 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 126.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.28s | 5 tokens @ 125.7 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.19s | 5 tokens @ 125.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.2s | 5 tokens @ 122.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.33s | 5 tokens @ 124.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 127.2 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.32s | 5 tokens @ 124.6 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 127.2 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 125.2 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 122.3 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.33s | 5 tokens @ 119.7 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 124.8 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 121.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.35s | 5 tokens @ 123.5 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.15s | 5 tokens @ 124.3 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.29s | 5 tokens @ 127.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 126.8 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.46s | 5 tokens @ 123.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 124.5 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.33s | 5 tokens @ 123.6 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 123.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 122.0 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 121.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.17s | 5 tokens @ 126.9 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.2s | 5 tokens @ 122.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.44s | 5 tokens @ 127.4 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.19s | 5 tokens @ 124.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.16s | 5 tokens @ 127.7 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 126.6 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 125.6 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 124.6 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 118.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 124.8 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.33s | 5 tokens @ 124.4 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 120.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 127.4 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 124.5 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.17s | 5 tokens @ 127.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.22s | 5 tokens @ 124.3 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 127.5 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.5s | TTFT: 1.54s | 5 tokens @ 127.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.25s | 5 tokens @ 127.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 126.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.16s | 5 tokens @ 124.3 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.26s | 5 tokens @ 125.6 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.2s | 5 tokens @ 122.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.21s | 5 tokens @ 124.8 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.35s | 5 tokens @ 127.4 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.17s | 5 tokens @ 127.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.25s | 5 tokens @ 121.3 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.28s | 5 tokens @ 121.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.18s | 5 tokens @ 124.4 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.19s | 5 tokens @ 124.6 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.21s | 5 tokens @ 123.4 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 123.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.43s | 5 tokens @ 127.7 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 125.9 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.16s | 5 tokens @ 127.4 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.19s | 5 tokens @ 125.0 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.23s | 5 tokens @ 127.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.32s | 5 tokens @ 126.1 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.2s | 5 tokens @ 123.8 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.26s | 5 tokens @ 124.5 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.29s | 5 tokens @ 124.8 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.44s | 5 tokens @ 127.5 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.18s | 5 tokens @ 124.6 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.28s | 5 tokens @ 125.2 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.3s | TTFT: 1.27s | 5 tokens @ 123.6 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.15s | 5 tokens @ 127.0 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.36s | 5 tokens @ 127.5 tok/s

**Model response:**
```

 category: spill
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.22s | 5 tokens @ 111.3 tok/s

**Model response:**
```

 category: spill
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.4s | TTFT: 1.31s | 5 tokens @ 126.7 tok/s

**Model response:**
```

 category: spill
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 1.2s | TTFT: 1.21s | 5 tokens @ 124.9 tok/s

**Model response:**
```

 category: spill
```

---

