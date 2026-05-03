# qwen3-vl:8b — cot (nothink)

Date: 2026-03-12 14:52
Test set: 99 images

## Results Summary

**Overall accuracy: 66.7%** (99/99 valid)
**Average time: 14.3s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 97.0% | 0.941 | 0.970 | 0.955 | 32 | 2 | 1 |
| improper_stacking | 3.0% | 1.000 | 0.030 | 0.059 | 1 | 0 | 32 |
| safe | 100.0% | 0.516 | 1.000 | 0.680 | 33 | 31 | 0 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 32 | 0 | 1 | 0 |
| improper_stacking | 2 | 1 | 30 | 0 |
| safe | 0 | 0 | 33 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 14.5s | TTFT: 13.4s | 255 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 9.5s | TTFT: 8.39s | 156 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 17.2s | TTFT: 16.12s | 309 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 14.0s | TTFT: 12.9s | 245 tokens @ 20.2 tok/s

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
- **Time:** 11.7s | TTFT: 10.47s | 198 tokens @ 20.1 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 12.0s | TTFT: 10.88s | 203 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 11.2s | TTFT: 10.08s | 190 tokens @ 20.3 tok/s

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
- **Confidence:** 100%
- **Time:** 14.8s | TTFT: 13.57s | 260 tokens @ 20.2 tok/s

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
- **Time:** 13.3s | TTFT: 12.11s | 231 tokens @ 20.3 tok/s

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
- **Confidence:** 90%
- **Time:** 13.0s | TTFT: 11.91s | 226 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.7s | TTFT: 11.56s | 220 tokens @ 20.3 tok/s

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
- **Confidence:** 100%
- **Time:** 32.5s | TTFT: 31.29s | 609 tokens @ 20.1 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 10.2s | TTFT: 9.08s | 170 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 14.1s | TTFT: 12.95s | 245 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 26.6s | TTFT: 25.57s | 492 tokens @ 20.1 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 12.1s | TTFT: 11.01s | 204 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 11.6s | TTFT: 10.48s | 197 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 100%
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 12.1s | TTFT: 10.96s | 207 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 14.6s | TTFT: 13.51s | 254 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.0s | TTFT: 10.85s | 206 tokens @ 20.3 tok/s

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
- **Time:** 25.4s | TTFT: 21.54s | 470 tokens @ 20.1 tok/s

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

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.7s | TTFT: 11.57s | 210 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 14.9s | TTFT: 13.82s | 264 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 12.9s | TTFT: 11.8s | 222 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 9.2s | TTFT: 8.08s | 150 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 95%
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 14.9s | TTFT: 13.88s | 264 tokens @ 20.2 tok/s

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
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 14.9s | TTFT: 13.33s | 254 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 95%
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 14.2s | TTFT: 13.1s | 245 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 19.4s | TTFT: 18.34s | 352 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 95%
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 25.8s | TTFT: 24.65s | 479 tokens @ 20.1 tok/s

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
- **Confidence:** 95%
- **Time:** 11.4s | TTFT: 10.28s | 194 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 13.2s | TTFT: 12.05s | 229 tokens @ 20.2 tok/s

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
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 14.3s | TTFT: 13.18s | 251 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 11.2s | TTFT: 10.07s | 188 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 10.8s | TTFT: 9.7s | 182 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 13.9s | TTFT: 12.89s | 239 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 95%
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 15.8s | TTFT: 14.71s | 280 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 16.7s | TTFT: 15.51s | 295 tokens @ 20.2 tok/s

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
- **Time:** 11.9s | TTFT: 10.83s | 204 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 63.8s | TTFT: 62.62s | 1216 tokens @ 19.8 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.9s | TTFT: 9.71s | 178 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 13.7s | TTFT: 12.57s | 239 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.7s | TTFT: 9.52s | 180 tokens @ 20.3 tok/s

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
- **Time:** 10.3s | TTFT: 9.18s | 173 tokens @ 20.4 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  

category: safe  
confidence: 100%
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 12.5s | TTFT: 11.39s | 215 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 9.5s | TTFT: 8.35s | 155 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 10.9s | TTFT: 9.84s | 182 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 19.2s | TTFT: 17.99s | 346 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 80%
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 13.1s | TTFT: 11.92s | 226 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

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
- **Time:** 12.2s | TTFT: 11.01s | 209 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 13.2s | TTFT: 12.14s | 223 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
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
- **Time:** 13.6s | TTFT: 12.46s | 238 tokens @ 20.2 tok/s

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
- **Time:** 13.1s | TTFT: 11.86s | 226 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 27.4s | TTFT: 23.5s | 509 tokens @ 20.1 tok/s

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

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 13.5s | TTFT: 12.28s | 231 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 11.5s | TTFT: 10.38s | 196 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 95%
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.7s | TTFT: 9.53s | 180 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 11.3s | TTFT: 10.19s | 192 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 12.9s | TTFT: 11.76s | 223 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 12.1s | TTFT: 10.94s | 204 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.1s | TTFT: 10.89s | 202 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 13.5s | TTFT: 12.44s | 234 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 95%
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.9s | TTFT: 9.71s | 180 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 14.5s | TTFT: 13.26s | 254 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 15.5s | TTFT: 14.42s | 275 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

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
- **Time:** 12.4s | TTFT: 11.29s | 210 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 95%
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 10.4s | TTFT: 9.31s | 174 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 11.7s | TTFT: 10.59s | 199 tokens @ 20.2 tok/s

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
- **Time:** 13.1s | TTFT: 11.88s | 221 tokens @ 19.7 tok/s

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
- **Confidence:** 95%
- **Time:** 13.8s | TTFT: 12.68s | 234 tokens @ 19.9 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 14.8s | TTFT: 13.6s | 251 tokens @ 20.0 tok/s

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
- **Confidence:** 95%
- **Time:** 12.9s | TTFT: 11.76s | 220 tokens @ 20.0 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 11.0s | TTFT: 9.84s | 186 tokens @ 20.3 tok/s

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
- **Time:** 13.0s | TTFT: 11.74s | 223 tokens @ 20.2 tok/s

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
- **Confidence:** 95%
- **Time:** 13.4s | TTFT: 12.21s | 231 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.8s | TTFT: 9.64s | 181 tokens @ 20.3 tok/s

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
- **Time:** 13.8s | TTFT: 12.65s | 242 tokens @ 20.2 tok/s

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
- **Time:** 10.2s | TTFT: 9.1s | 170 tokens @ 20.3 tok/s

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
- **Time:** 12.7s | TTFT: 11.61s | 220 tokens @ 20.3 tok/s

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
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 14.3s | TTFT: 13.15s | 251 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 10.7s | TTFT: 9.51s | 179 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 100%
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 95%
- **Time:** 11.5s | TTFT: 10.48s | 196 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 14.2s | TTFT: 13.13s | 250 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.9s | TTFT: 11.78s | 223 tokens @ 20.3 tok/s

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
- **Time:** 10.5s | TTFT: 9.39s | 177 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 11.0s | TTFT: 9.86s | 182 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 14.8s | TTFT: 13.74s | 262 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 11.1s | TTFT: 10.03s | 189 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 11.1s | TTFT: 9.94s | 187 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
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
- **Time:** 14.0s | TTFT: 12.78s | 243 tokens @ 20.2 tok/s

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
- **Confidence:** 90%
- **Time:** 14.4s | TTFT: 13.25s | 253 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  

category: spill  
confidence: 90%
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 26.5s | TTFT: 25.32s | 490 tokens @ 20.1 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 100%
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 11.1s | TTFT: 10.0s | 181 tokens @ 20.3 tok/s

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
- **Confidence:** 95%
- **Time:** 15.2s | TTFT: 11.09s | 264 tokens @ 20.3 tok/s

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

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.7s | TTFT: 11.55s | 217 tokens @ 20.1 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 100%
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 11.3s | TTFT: 10.19s | 192 tokens @ 20.3 tok/s

**Model response:**
```
QUESTION 1: YES  
QUESTION 2: NO  
category: spill  
confidence: 90%
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 12.9s | TTFT: 11.71s | 223 tokens @ 20.3 tok/s

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
- **Confidence:** 90%
- **Time:** 28.3s | TTFT: 24.37s | 528 tokens @ 20.1 tok/s

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

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 95%
- **Time:** 9.8s | TTFT: 8.71s | 159 tokens @ 20.2 tok/s

**Model response:**
```
QUESTION 1: NO  
QUESTION 2: NO  
category: safe  
confidence: 95%
```

---

