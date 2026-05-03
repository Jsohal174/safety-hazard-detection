# qwen3.5:9b — simple (nothink)

Date: 2026-03-12 13:18
Test set: 99 images

## Results Summary

**Overall accuracy: 78.8%** (99/99 valid)
**Average time: 9.9s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 90.9% | 1.000 | 0.909 | 0.952 | 30 | 0 | 3 |
| improper_stacking | 51.5% | 0.895 | 0.515 | 0.654 | 17 | 2 | 16 |
| safe | 93.9% | 0.620 | 0.939 | 0.747 | 31 | 19 | 2 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 30 | 0 | 3 | 0 |
| improper_stacking | 0 | 17 | 16 | 0 |
| safe | 0 | 2 | 31 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 98%
- **Time:** 11.5s | TTFT: 2.85s | 115 tokens @ 13.6 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire warehouse floor and all visible shelving units  

Rationale: The concrete floor is dry, clean, and free of spills, oil slicks, or wet patches. All boxes on shelves are neatly stacked, aligned horizontally, not overhanging, crushed, or jumbled — no signs of instability or poor arrangement. No torn shrink wrap or shifted loads observed. Forklift is parked safely in aisle center with no obstruction hazards. Overall environment complies with OSHA warehouse safety standards.
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 18.1s | TTFT: 3.15s | 198 tokens @ 13.5 tok/s

**Model response:**
```
Looking at the image:

- **Floor**: The concrete floor is dry and clean. No spills, wet patches, oil slicks, or fluid are visible. Yellow safety lines are intact. One worker in a high-vis vest stands on the aisle — no indication of slip hazard.

- **Shelves (both sides)**: 
  - Boxes are uniformly stacked, aligned vertically with flat bases.
  - No boxes are tilted, overhanging, crushed, or jumbled.
  - Shrink wrap appears intact where applied; loads are evenly distributed.
  - Shelving units are upright and stable — no signs of shifting or instability.
  - Even in background areas, stacking remains orderly.

- **Overall**: The environment appears well-maintained, compliant with standard warehouse safety practices. No obvious violations detected.

category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all shelving units shown
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 11.5s | TTFT: 3.61s | 110 tokens @ 13.5 tok/s

**Model response:**
```
Based on the visual evidence in the drone image, here is the classification:

**Observation:** There is a distinct wet patch and reflection on the concrete floor in the center of the aisle. This indicates a liquid spill or pooling fluid that creates a slip hazard for workers. The shelves appear to be stocked with boxes stacked neatly; there are no obvious tilted, crushed, or overhanging loads visible from this distance.

**Classification:**
category: spill
severity: high
confidence: 95%
location: center aisle floor
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 6.5s | TTFT: 2.98s | 47 tokens @ 13.6 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: warehouse floor and all visible shelves — no spills, tilted boxes, or unstable stacking observed; environment appears organized and hazard-free per OSHA standards.
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 7.3s | TTFT: 3.08s | 56 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire floor and all shelving units on both sides are clear, dry, properly stacked, and stable — no visible spills, tilts, overhangs, crushing, or misalignment.
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 7.4s | TTFT: 3.1s | 56 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and all shelving units on both sides show no visible spills, leaks, or improper stacking; boxes are neatly arranged, upright, and aligned without overhangs or instability.
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 8.7s | TTFT: 3.13s | 73 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor — large brown liquid spill with reflective surface, indicating hazardous fluid on concrete walkway

Note: No shelf issues detected — all boxes appear neatly stacked and stable. Only concern is the floor spill which poses slip/trip/fall hazard and potential chemical exposure risk.
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 16.1s | TTFT: 3.03s | 170 tokens @ 13.2 tok/s

**Model response:**
```
category: proper_stack
severity: none
confidence: 95%
location: all shelves on both sides have boxes neatly aligned, flat, and stable; no visible spills or fluid on floor. The only potential minor concern is the torn shrink wrap on a pallet of film rolls (right side), but it does not indicate instability or hazard under current conditions — thus classified as safe with low-severity observation if noted internally. However, per instructions to classify strictly based on defined categories and severity thresholds, this falls under “safe” since no active hazard exists.

Note: Torn shrink wrap alone without shift/collapse risk is not improper_stacking under given criteria. No fluid detected. Therefore:

category: safe
severity: none
confidence: 95%
location: entire visible warehouse area including all shelving units and floor
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 11.1s | TTFT: 3.31s | 101 tokens @ 13.2 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all visible shelving units  

Rationale: The concrete floor is clean, dry, and free of spills, puddles, or slicks. All boxes on shelves are neatly aligned, flat, and securely stacked with no signs of tilting, overhang, crushing, shifting, or missing packaging. Yellow aisle markings are intact and unobstructed. No visible hazards detected in drone imagery.
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 6.9s | TTFT: 3.1s | 50 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor — large, irregular brown puddle with visible drip trail leading toward foreground; wet reflective patches and debris around edges indicate active or recent fluid leakage.
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 10.5s | TTFT: 3.02s | 97 tokens @ 13.2 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 95%  
location: center aisle floor — large blue-colored liquid puddle with reflective sheen, clearly indicating fluid accumulation  

(rationale: OSHA 1910.22(b) requires walkways and working surfaces to be kept clean and dry; this visible spill creates a slip/fall hazard. The presence of a worker nearby also increases immediate risk. Severity is high due to potential for serious injury.)
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 5.6s | TTFT: 3.17s | 32 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center foreground floor, dark wet patch with reflective sheen indicating fluid accumulation
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 80%
- **Time:** 5.7s | TTFT: 3.12s | 35 tokens @ 13.4 tok/s

**Model response:**
```
safe
none
80%
warehouse floor and shelving units show no spills, wet spots, or unstable box stacking; all items appear properly aligned and secure.
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 9.2s | TTFT: 3.14s | 80 tokens @ 13.3 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: multiple shelving units on both sides (left and right), particularly in rows 3–6, where boxes are visibly tilted, unevenly aligned, overhanging shelf edges, or stacked without proper support — creating potential for collapse or falling items. Some loads appear shifted laterally, indicating instability.
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 90%
- **Time:** 22.0s | TTFT: 3.05s | 245 tokens @ 13.1 tok/s

**Model response:**
```
Based on the analysis of the drone camera image, here is the classification for this OSHA warehouse safety scenario.

### Analysis Report

**Visual Observation:**
*   **Floor Conditions:** The concrete floor shows minor wear and scuff marks, but there are no visible liquid puddles, wet reflective patches, or chemical leaks. It appears dry.
*   **Shelf Inspections:** Scanning the metal shelving units on both sides of the aisle reveals that all cardboard boxes are loaded uniformly. There are no signs of boxes hanging over the edges, leaning, crushed, or stacked in a jumbled/uneven manner. The shrink wrap appears intact where visible.
*   **Other Elements:** While there is a robot spider and a worker standing far away in the background which could represent other operational concerns, strictly adhering to the safety classification criteria for spills and stacking problems (OSHA standards), these items do not violate the specific conditions requested for this prompt.

**Classification Result:**
- **Category:** safe
- **Severity:** none
- **Confidence:** 90%
- **Location:** The entire visible aisle area including shelving units is safe regarding spills and stacking violations.
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** medium
- **Confidence:** 85%
- **Time:** 16.7s | TTFT: 3.53s | 174 tokens @ 13.2 tok/s

**Model response:**
```
category: unsafe  
severity: medium  
confidence: 85%  
location: left side aisle shelves – boxes appear jumbled, unevenly stacked, and some show torn shrink wrap or slight tilting; right-side shelves are relatively neat but a few loads show minor misalignment. Floor shows no clear spills, so “spill” is ruled out. The primary issue is improper stacking on the left shelves, creating a medium-level hazard for falling loads or collapse during racking operations.

Note: While the image resolution limits definitive judgment of every box, multiple observable irregularities on the left shelf (tilted boxes, non-aligned pallets, damaged wrapping) meet OSHA standards for “improper_stacking”. Right-side shelves are mostly compliant but not perfect — however, since one side has clear violations, the overall scene is classified as unsafe with medium severity.
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 6.2s | TTFT: 3.1s | 41 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: liquid spill on the warehouse floor near the center-right foreground, adjacent to stacked cardboard boxes and a metal shelving unit
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 6.9s | TTFT: 3.08s | 52 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse scene — floor is dry and clean; all shelving units have neatly stacked, aligned boxes with no visible tilting, overhangs, crushing, or instability
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 13.4s | TTFT: 3.01s | 136 tokens @ 13.2 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 95%  
location: center aisle floor, large wet reflective puddle visible between shelving units — likely liquid spill posing slip/fall and electrical hazard risk

—

Note: While shelves appear orderly with no obvious instability, the presence of a significant fluid spill on the main walkway constitutes an immediate safety violation under OSHA standards (1910.22 for housekeeping and 1910.147 for lockout/tagout if contaminants affect equipment). The high confidence stems from clear visual evidence of moisture pooling in a high-traffic area, which is universally hazardous in industrial settings.
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 10.8s | TTFT: 3.35s | 97 tokens @ 13.3 tok/s

**Model response:**
```
Based on the visual analysis of the provided image:

category: safe
severity: none
confidence: 98%
location: N/A (No issues found)

**Analysis:**
The floor is clean, dry, and shows no signs of fluid spills, oil slicks, or wet patches. Both visible aisles show shelves stocked with boxes that are neatly aligned, flat against the shelving units, and appear stable with no overhangs or shifting loads.
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 11.3s | TTFT: 3.6s | 104 tokens @ 13.3 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and all visible shelving units

The concrete floor appears dry, clean, and free of any liquid spills, oil slicks, or wet patches. All shelves on both sides contain neatly stacked, uniformly aligned boxes with no signs of tilting, overhanging, crushing, shifting, missing shrink wrap, or jumbled arrangement. Forklift traffic paths are clear. No safety hazards are detectable from this image.
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 35.8s | TTFT: 3.0s | 429 tokens @ 13.3 tok/s

**Model response:**
```
category: proper_stacking  
severity: none  
confidence: 95%  
location: N/A — all shelves are properly stocked with aligned, stable boxes; floor is dry and free of spills. No hazards detected.

---

Wait — let me re-examine the image carefully.

Looking again:

- **Floor**: The concrete appears dusty or stained, but no visible liquid puddles, oil slicks, or chemical leaks. Stains may be old grime — not active spills. → Clean enough for “safe” category.
  
- **Shelves (left side)**: Boxes are uniformly stacked, aligned on pallets, shrink-wrapped where applicable. No tilting, overhanging, crushed boxes, or misalignment. One shelf has empty space — that’s normal inventory management, not a hazard.

- **Shelves (right side)**: Same as left — neat stacking, no instability. Some boxes are wrapped in clear plastic — good practice.

- **Center aisle**: Clear path. No debris, no tripping hazards. Lighting is adequate.

→ All safety criteria met.

But wait — one detail: On the bottom-right shelf (second from right), there’s a small black object near the floor — possibly a wheelbarrow or cart? It doesn’t obstruct traffic significantly. Not a hazard under current conditions.

Also, note: The very top shelves have boxes placed directly on beam edges without full pallet support — this can be considered improper stacking if it violates company policy or OSHA standards regarding load security. However, since these are likely fixed storage units and not mobile racking, and assuming they’re designed for such use, we don’t flag unless clearly unstable.

In reality, OSHA 1910.176(b) requires materials to be stored to prevent shifting/falling — and here, loads appear secure.

Final call: This scene is compliant.

✅ Final Answer:

category: safe  
severity: none  
confidence: 95%  
location: entire warehouse — floor clean, shelves stable, no hazards detected
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 23.4s | TTFT: 3.36s | 263 tokens @ 13.3 tok/s

**Model response:**
```
Looking at the image:

- Floor: The concrete floor appears dry, with no visible liquid puddles, oil slicks, chemical leaks, or wet reflective patches. There are some scattered debris (wooden planks) in the foreground aisle, but these are not fluids or spills — they’re solid objects that may require cleanup but don’t constitute a “spill” under OSHA fluid definition.
  
- Shelves: All shelves on both sides show boxes neatly stacked, flat, aligned, and stable. No overhangs, tilts, crushing, missing wrap, or uneven loads visible. Even distant shelves appear orderly.

→ Therefore, no spill, no improper stacking, no obvious hazards — only minor debris on floor which might be considered low-risk clutter but not a “spill” as defined.

Per instructions: we classify based on *spills* (fluids) and *stacking problems*. Since neither is present, this falls under “safe”. Debris like wood planks isn’t a fluid spill — so it doesn’t trigger “spill” category. If forced to choose from given options, “safe” is correct for the defined criteria.

category: safe
severity: none
confidence: 95%
location: entire warehouse floor and all shelving units
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 14.5s | TTFT: 3.3s | 147 tokens @ 13.4 tok/s

**Model response:**
```
Looking at the image:

- The FLOOR has a large, reflective puddle in the center aisle — clearly liquid on concrete. This qualifies as a spill hazard (risk of slipping).
- Every SHELF shown appears neatly organized: boxes are flat, aligned, not overhanging or crushed. No visible instability or poor stacking.
- Background worker is wearing high-vis gear and appears to be working normally; no immediate action needed from them visible in frame.

Classification:

category: spill  
severity: medium (a large puddle in an active aisle presents slip risk but not imminent danger unless unaddressed)  
confidence: 95%  
location: central concrete floor of main aisle between shelving units
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 5.5s | TTFT: 3.05s | 33 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: large brown fluid puddle on concrete floor in central aisle between shelving units
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 6.9s | TTFT: 3.0s | 52 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle, concrete floor — large wet puddle with reflective sheen indicates liquid spill or leak requiring immediate cleanup to prevent slip/fall hazard and potential chemical exposure risk.
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 6.4s | TTFT: 3.0s | 45 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor — visible reflective puddle with standing water or wet concrete, likely a leak from above or spilled fluid posing slip hazard.
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 6.9s | TTFT: 3.0s | 51 tokens @ 13.4 tok/s

**Model response:**
```
category: proper_stacking
severity: none
confidence: 95%
location: The floor is clean and dry, all items on shelves are neatly organized and stable. There are no spills or improperly stacked boxes visible in the image provided.
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 5.5s | TTFT: 3.0s | 33 tokens @ 13.6 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: liquid puddle on the concrete floor, approximately center-left in the foreground aisle
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 14.1s | TTFT: 3.04s | 145 tokens @ 13.3 tok/s

**Model response:**
```
category: improper_stacking  
severity: high  
confidence: 95%  
location: shelves on both left and right sides — multiple boxes are tilted, jumbled, unevenly aligned, or lacking proper shrink wrap; some loads appear shifted or unstable  

---

**Rationale:**  
While the floor appears clean (no visible spills), numerous shelving units show clear violations of stacking safety standards: boxes are not flat, some hang over edges, others are crushed or misaligned — all potential fall hazards. These issues meet OSHA criteria for improper material handling and storage. The severity is “high” because falling objects could cause serious injury. Confidence is high due to visible evidence across multiple racks in the image.
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 9.1s | TTFT: 3.06s | 80 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire warehouse floor and all shelving units inspected — no spills detected; all boxes on shelves are neatly stacked, aligned, stable, with no overhangs, tilting, crushing, or missing shrink wrap. Concrete floor appears dry and clean under ambient lighting. No visible hazards in this drone-captured view.
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 6.6s | TTFT: 3.0s | 48 tokens @ 13.5 tok/s

**Model response:**
```
safe  
none  
98%  
Entire warehouse floor is dry and clear; all shelves have neatly stacked, stable boxes with no overhangs, tilts, or misalignments visible across both aisles and upper racks.
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 12.0s | TTFT: 3.0s | 119 tokens @ 13.3 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 95%  
location: center aisle, large reflective liquid puddle extending from near foreground toward worker in background — visible on concrete floor, indicating active spill hazard.  

Note: While shelves appear generally organized (no obvious tilted/crushed boxes), the presence of a significant fluid spill on the walking surface constitutes an immediate slip/trip/fall and potential chemical exposure risk, making this a critical safety violation requiring immediate remediation per OSHA standards (e.g., 29 CFR 1910.175).
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 7.7s | TTFT: 3.38s | 58 tokens @ 13.5 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: leftmost rack (second from front), top shelf box tilted and rotated off alignment; also visible on adjacent shelves in same aisle — boxes not flush with edge, some leaning inward/outward.
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 11.5s | TTFT: 3.03s | 112 tokens @ 13.4 tok/s

**Model response:**
```
Looking at the floor, I see no liquid puddles, wet spots, oil slicks, or chemical leaks — the concrete appears dry and clean. Examining all visible shelves on both sides, the boxes are neatly aligned, stable, not tilted, overhanging, crushed, or jumbled; shrink wrap is intact where present; loads are balanced across shelf depths. No items appear unstable or likely to fall.

category: safe
severity: none
confidence: 100%
location: entire warehouse floor and all shelving units shown
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 11.2s | TTFT: 3.07s | 107 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking  
severity: medium  
confidence: 95%  
location: multiple shelving units in foreground and midground (especially right-center shelves) show boxes tilted, rotated, or jumbled — some hanging over edges; left foreground shelf has unaligned, leaning boxes; background racks appear less affected but still show minor misalignment.

Note: No visible spills on floor — concrete is dry and clean. However, stacking issues pose falling hazard risk to personnel walking nearby, warranting medium severity correction priority.
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 92%
- **Time:** 8.1s | TTFT: 3.0s | 68 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking  
severity: medium  
confidence: 92%  
location: multiple shelves on both sides show boxes tilted, leaning, or unevenly stacked—especially noticeable on the left-hand racks where shrink wrap is torn and items are jumbled; right-side shelves also exhibit misaligned loads and instability near top tiers.
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 6.5s | TTFT: 3.0s | 47 tokens @ 13.5 tok/s

**Model response:**
```
Based on the visual analysis of the provided image, here is the OSHA warehouse safety inspection report:

category: safe
severity: none
confidence: 98%
location: Floor area and all visible shelving units
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 6.9s | TTFT: 3.0s | 52 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 100%
location: entire warehouse floor and all shelving units show no spills, wet areas, or improper stacking; boxes are neatly aligned, stable, and properly stored on both sides.
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 7.7s | TTFT: 3.06s | 62 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: The load on the right-hand shelf is heavily tilted, overhanging the edge, with torn shrink wrap and unevenly stacked boxes — indicating instability and risk of falling. Similar minor issues appear on adjacent shelves.
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 19.2s | TTFT: 3.0s | 214 tokens @ 13.4 tok/s

**Model response:**
```
category: spill  
severity: high  
confidence: 95%  
location: center aisle floor, near a dropped box and dark wet patch indicating fluid leak or spill  

---

**Rationale:**  
The image clearly shows a liquid spill (dark, reflective puddle) on the concrete floor in the central aisle — a slip hazard per OSHA standards. Additionally, one cardboard box is knocked over onto the floor next to the spill, increasing risk of entanglement or further contamination. While shelf stacking appears generally neat with no obvious tilting, crushing, or overhangs visible from this angle, the presence of the spill alone qualifies as a critical safety violation under OSHA 1910.22 (walkways) and 1910.147 (hazard communication). The spill’s location in an active walkway makes it high-severity until remediated.

*Note: Shelves may have minor imperfections not fully resolvable at this resolution, but no clear improper_stacking is evident from current view.*
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** None%
- **Time:** 3.9s | TTFT: 3.32s | 9 tokens @ 14.9 tok/s

**Model response:**
```
safe: none
none: N/A
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 10.7s | TTFT: 3.07s | 101 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all visible shelving units  

Rationale: The concrete floor shows no signs of spills, wet patches, or fluid accumulation. All boxes on shelves are neatly aligned, stable, and not tilted, overhanging, crushed, or jumbled. Load distribution appears even across each shelf unit. No torn wrap or shifted loads observed. Lighting is adequate for drone inspection; image clarity supports confident assessment.
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 14.9s | TTFT: 3.24s | 154 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse scene  

Justification: The concrete floor is clean and dry with no visible spills, wet patches, oil slicks, or fluid hazards. All shelves on both sides are properly stocked — boxes are neatly aligned, flat, stable, without overhangs, tilting, crushing, torn wrap, or shifting loads. A single cardboard box lying on the floor in the central aisle appears displaced (possibly from dropped stock), but it does not constitute a spill or stacking hazard — it’s an isolated object, not part of shelving content. No employee is actively loading/unloading at this moment to assess dynamic risk. Overall, conditions meet safety standards for storage and walkways.
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** medium
- **Confidence:** 95%
- **Time:** 7.4s | TTFT: 3.0s | 58 tokens @ 13.5 tok/s

**Model response:**
```
category: improper_stacking
severity: medium
confidence: 95%
location: multiple shelving units on both sides show boxes stacked unevenly, tilted, jumbled, or overhanging — especially noticeable in foreground and mid-right aisles where loads are shifted and unstable
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 90%
- **Time:** 14.8s | TTFT: 3.0s | 156 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual inspection of the image provided:

**Analysis:**
*   **Floor Condition:** The concrete floor appears clean and dry. There are no visible liquid puddles, oil slicks, or fluid leaks in the foreground or background.
*   **Shelf Condition:** On both sides of the aisle, the racking is densely packed with cardboard boxes. Every box appears to be stacked flat, aligned, and supported properly by the shelf below. There are no overhanging boxes, tilted crates, or visible damage that suggests instability.
*   **Conclusion:** The environment shows standard warehouse organization with no immediate safety hazards detected.

category: safe
severity: none
confidence: 90%
location: Entire warehouse aisle and shelving units
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 11.3s | TTFT: 3.34s | 105 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual analysis of the drone camera image:

category: safe
severity: none
confidence: 95%
location: entire warehouse floor and all visible shelving units

**Reasoning:** The concrete floor is clean, dry, and free of any liquid spills or oil slicks. All cardboard boxes on the shelves are stacked neatly in aligned rows, sitting flat without tilting, overhanging, or appearing crushed. There are no signs of instability or messiness that would constitute a hazard.
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 8.0s | TTFT: 3.29s | 62 tokens @ 13.5 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: multiple shelves on both sides — boxes are tilted, unevenly stacked, some overhanging edges, torn/missing shrink wrap, and loads shifted or jumbled (especially visible in the right-hand rack system)
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 6.4s | TTFT: 2.99s | 45 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor — large wet puddle visible with a damaged or displaced box lying near its edge, indicating possible leakage or spillage.
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 85%
- **Time:** 5.2s | TTFT: 3.02s | 30 tokens @ 13.7 tok/s

**Model response:**
```
safe; none; 85%; entire warehouse floor and shelves are clean, dry, and neatly organized with no visible spills or stacking issues.
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 21.8s | TTFT: 3.04s | 248 tokens @ 13.4 tok/s

**Model response:**
```
Looking at the drone camera image of the warehouse:

**Floor Analysis:**
- There are visible wet patches and reflective areas on the concrete floor, particularly in the central aisle and near the forklift.
- These appear to be liquid spills — possibly water or cleaning fluid — not oil slicks (no rainbow sheen) but still hazardous as slip hazards.
- The spill is not contained; it’s spread across the walking/driving path.

**Shelves Analysis:**
- Boxes on both left and right shelves are uniformly stacked, aligned, flat, and stable.
- No tilting, overhanging, crushed boxes, torn wrap, or shifting loads observed.
- Shrink wrapping appears intact where present.
- Overall stacking is orderly and compliant with typical warehouse safety standards.

**Conclusion:**
The primary issue is the **spilled liquid on the floor**, which presents a slip/trip/fall hazard — critical under OSHA guidelines even if small in volume when located in high-traffic areas.

---

category: spill  
severity: critical  
confidence: 95%  
location: central aisle floor, extending from foreground toward middle ground near forklift and warehouse bulkhead
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 10.1s | TTFT: 3.09s | 94 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: no spills or stacking issues detected on floor or shelves; all boxes are neatly aligned, stable, and properly contained; floor is dry and clear of fluid hazards.  

(Note: The lone worker in the center aisle appears to be operating safely within marked walkways — no PPE violation visible from this view. Yellow aisle markings are intact. No debris, damage, or instability observed.)
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 7.8s | TTFT: 2.99s | 64 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor is dry and clean; all shelves on both sides show neatly stacked, stable, and properly aligned boxes with no overhangs, tilts, or crushed items — consistent with OSHA safety standards for storage areas.
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 7.0s | TTFT: 3.0s | 53 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking  
severity: high  
confidence: 95%  
location: right-hand shelving unit — boxes are tilted, stacked unevenly, and appear unstable; one box is leaning significantly outward, risking collapse onto workers or equipment below.
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 100%
- **Time:** 4.4s | TTFT: 3.04s | 19 tokens @ 14.0 tok/s

**Model response:**
```
safe
none
100%
entire visible warehouse (floor and shelves)
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 5.0s | TTFT: 3.08s | 27 tokens @ 13.7 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, between rows of shelving
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 7.9s | TTFT: 3.52s | 59 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse floor and shelves — no visible spills, no improperly stacked boxes, no unstable loads; all items appear neatly organized and secure on racks. Minor dust or wear on concrete is normal and not hazardous.
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 98%
- **Time:** 6.2s | TTFT: 3.01s | 43 tokens @ 13.5 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: floor is clean and dry; all boxes on shelves are neatly stacked, aligned, and stable with no visible spills or stacking hazards
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 15.4s | TTFT: 3.07s | 163 tokens @ 13.4 tok/s

**Model response:**
```
Looking at both the FLOOR and SHELVES in this image:

- The concrete floor appears clean, dry, and free of spills, puddles, oil slicks, or chemical leaks. No reflective wet patches are visible — even under overhead lighting.
- All shelves (both left and right sides) show boxes neatly stacked, aligned, and flat on their surfaces. No boxes are tilted, overhanging, crushed, missing shrink wrap, or jumbled. Loads appear evenly distributed and stable.
- Shelving units themselves look structurally sound — no sagging beams, loose pallets, or damaged racks.

No safety violations or hazards are apparent.

category: safe
severity: none
confidence: 100%
location: entire warehouse floor and all shelving units
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 4.5s | TTFT: 3.3s | 17 tokens @ 14.0 tok/s

**Model response:**
```
safe, none, 95%, no hazards detected in floor or shelf organization
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 12.0s | TTFT: 3.01s | 119 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all shelving units (both aisles)  

Rationale: The concrete floor shows no signs of spills, puddles, or wet patches. All boxes on shelves are neatly aligned, stable, and not overhanging or tilted — no crushed containers, missing wrap, or shifting loads observed. Yellow safety markings are intact and unobstructed. Worker in high-vis vest appears to be operating normally within designated zones. Overall compliance with OSHA stacking and walkway requirements is evident.
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 12.9s | TTFT: 3.35s | 126 tokens @ 13.4 tok/s

**Model response:**
```
Looking at the drone camera image of the warehouse:

- The floor shows a wet, reflective puddle near the center aisle — likely from spilled liquid. This is clearly visible and poses a slip hazard.
- All shelving units on both sides appear to have boxes neatly stacked, aligned horizontally without tilting, overhang, or crushing. No torn shrink wrap or jumbled loads are evident.
- The only issue detected is the fluid on the floor.

Therefore:

category: spill  
severity: high  
confidence: 95%  
location: central aisle floor, approximately midway between rows of shelving
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 100%
- **Time:** 10.8s | TTFT: 3.29s | 98 tokens @ 13.2 tok/s

**Model response:**
```
Based on the visual analysis of the warehouse scene from the drone camera image:

The floor appears clean and dry with no visible liquid puddles, wet patches, or debris. The shelving units are uniformly stocked with cardboard boxes that are aligned in neat rows, flat against the shelf edges, and appear structurally stable with no signs of crushing or falling hazards.

**category:** safe
**severity:** none
**confidence:** 100%
**location:** N/A
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 6.1s | TTFT: 3.39s | 36 tokens @ 13.6 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, large dark liquid puddle with reflective surface indicating wet or hazardous fluid accumulation
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 98%
- **Time:** 10.1s | TTFT: 3.0s | 94 tokens @ 13.3 tok/s

**Model response:**
```
category: spill  
severity: critical  
confidence: 98%  
location: floor, center aisle — visible dark puddle with liquid splatter and reflection indicating fluid on concrete surface

Note: The worker in safety gear appears to be responding to or investigating the spill, which reinforces its hazardous nature under OSHA standards (e.g., 1910.22 requires walkways/floors to be kept free of slips/trips/spills).
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 7.3s | TTFT: 3.0s | 57 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor — dark liquid spill with wet, reflective patch extending across multiple rows; visible drip trail leading toward front-left area. Floor is visibly contaminated and poses slip/trip hazard underfoot.
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 5.5s | TTFT: 2.99s | 34 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: central aisle floor, large reflective liquid puddle extending from near right foreground toward center
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** low
- **Confidence:** 95%
- **Time:** 10.0s | TTFT: 3.02s | 92 tokens @ 13.3 tok/s

**Model response:**
```
Based on the visual analysis of the warehouse scene in the drone image:

category: minor_hazard  
severity: low  
confidence: 95%  
location: A few boxes on the lower shelves (left aisle, mid-height) show slight tilt or misalignment—possibly due to settling—but no immediate fall risk; also, minor scuff marks and faint dust patterns visible on floor but no liquid spills. Overall organization is good but not perfect.
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 85%
- **Time:** 6.2s | TTFT: 3.01s | 42 tokens @ 13.5 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 85%
location: shelves on both sides, especially right foreground and left mid-shelf showing tilted, jumbled, unstable box arrangements
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 23.9s | TTFT: 3.0s | 275 tokens @ 13.3 tok/s

**Model response:**
```
Looking at the image provided — a drone view of a warehouse aisle with tall shelving units on both sides, filled with cardboard boxes.

### Step-by-step analysis:

#### 1. FLOOR CHECK (for spills):
- The concrete floor appears uniformly gray and dry.
- No visible liquid puddles, reflective wet patches, oil slicks, or chemical leaks.
- Some faint yellow safety lines are visible near shelves — typical for designated zones, not fluid.
→ **No spill detected.**

#### 2. SHELF CHECK (for improper stacking):
- Boxes on all visible shelves appear neatly aligned.
- No boxes tilted, rotated, hanging over edges, crushed, or missing shrink wrap.
- Loads are evenly distributed; no shifting to one side.
- Shelves look structurally sound and properly loaded.
→ **No improper stacking observed.**

#### 3. OVERALL SCENE:
- Clean environment.
- No personnel visible (no PPE issues, tripping hazards from movement, etc.).
- Lighting is adequate for visibility in drone imagery.
- Structural elements (walls, ceiling) appear intact.

---

### Classification:

category: safe  
severity: none  
confidence: 95%  
location: entire visible warehouse floor and shelving units across both aisles shown
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 12.1s | TTFT: 3.32s | 116 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire visible warehouse area (floor and all shelving units)

Rationale: The concrete floor is dry with no visible liquid spills, stains, or reflective patches. All boxes on shelves are neatly stacked, aligned in rows, flat, and without overhangs, crushing, or deformation. No torn shrink wrap or uneven loads observed. A worker in high-visibility gear appears to be operating normally without hazardous exposure. The environment meets OSHA warehouse safety standards for floor conditions and material storage.
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 11.3s | TTFT: 3.39s | 104 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual analysis of the drone camera image:

category: safe
severity: none
confidence: 95%
location: N/A (no hazards detected)

**Analysis:**
The floor appears clean, dry, and free of liquid puddles or oil slicks. The shelving units are filled with cardboard boxes stacked neatly in uniform rows; there is no evidence of tilted boxes, overhangs, crushed items, or loose shrink wrap. All visible loads appear stable and properly organized.
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 7.0s | TTFT: 3.29s | 49 tokens @ 13.5 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all visible shelving units show no spills, liquid, or stacking irregularities; boxes are neatly aligned, stable, and uniformly stored.
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 15.1s | TTFT: 3.04s | 159 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and shelves  

**Rationale:**  
- Floor: Clean, dry, no visible spills, oil slicks, or wet patches. Concrete surface appears intact with minor tire marks (normal wear).  
- Shelves: All boxes are neatly stacked in uniform rows, flush against shelf edges, no tilting, overhanging, crushing, or missing wrap. Loads appear balanced and stable across all visible racks on both sides of the aisle.  
- Worker visibility: One worker is centered in the aisle wearing high-vis vest — not a hazard, just standard PPE.  
No unsafe conditions detected under OSHA 1910 standards for storage, housekeeping, or material handling.
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 8.9s | TTFT: 3.47s | 73 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire warehouse floor and all visible shelving units show no spills, wet patches, or stacking irregularities; boxes are uniformly aligned, stable, and properly contained on shelves. No fluid detected on concrete; no tilted, overhanging, crushed, or jumbled loads observed.
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 9.9s | TTFT: 3.02s | 91 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire warehouse floor and all visible shelving units  

Rationale: The concrete floor shows no signs of spills, wet patches, or fluid contamination. All boxes on shelves are neatly stacked, aligned horizontally, without overhangs, tilting, crushing, or missing shrink wrap. Load distribution appears uniform across shelf levels. No evidence of instability or hazard conditions is present.
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 5.8s | TTFT: 3.0s | 37 tokens @ 13.5 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: floor and all shelves are clean, dry, and properly stacked with no visible spills or instability.
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 6.5s | TTFT: 3.0s | 46 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, near right-hand shelving unit — clear liquid puddle with visible wet patch and overturned bucket indicating active or recent fluid release
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 7.1s | TTFT: 3.01s | 54 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual analysis of the warehouse scene, here is the classification:

category: spill
severity: high
confidence: 95%
location: center aisle (a large, dark liquid spill with wet patches visible on the floor in the foreground)
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 7.3s | TTFT: 3.13s | 55 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: right-hand side of aisle — boxes are tilted, leaning, and stacked unevenly on multiple shelves; some appear crushed or shifted laterally, indicating instability risk for falling loads.
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 10.0s | TTFT: 3.04s | 91 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual analysis of the drone camera image:

**category:** spill
**severity:** high
**confidence:** 95%
**location:** The concrete floor in the central aisle shows a large, dark liquid puddle with visible streaks and reflection indicating fluid leakage. Additionally, a person is standing nearby handling what appears to be a chemical drum (labeled 'H'), suggesting active hazardous material handling contributing to the spill risk.
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 8.6s | TTFT: 3.0s | 73 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking  
severity: high  
confidence: 95%  
location: multiple shelving units on both sides of aisles — boxes are tilted, stacked unevenly, overhanging edges, and some appear crushed or shifted (e.g., center-right shelf with top box precariously balanced; left aisle shelves show jumbled alignment)
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 8.3s | TTFT: 3.01s | 70 tokens @ 13.4 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire warehouse scene — floor appears clean and dry with no visible spills; all shelves are uniformly stocked with boxes neatly aligned, stacked flat, without overhangs, crushing, or instability. A worker is visible in the center aisle but not interacting with any hazard.
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 8.2s | TTFT: 3.04s | 68 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all visible shelves — no spills, no unstable stacking, no overhanging boxes, no crushed or tilted loads. All cardboard boxes are neatly aligned with consistent shrink wrap; shelving is organized and within designated yellow-painted zones.
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 7.0s | TTFT: 3.09s | 52 tokens @ 13.5 tok/s

**Model response:**
```
category: improper_stacking
severity: high
confidence: 95%
location: right side, top shelf (box wrapped in plastic is tilted and overhanging), left side middle shelves (boxes are unevenly stacked and some are rotated)
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 11.5s | TTFT: 3.12s | 110 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and all visible shelving units  

Rationale: The concrete floor is dry with no visible spills, puddles, or wet patches. All boxes on shelves are neatly stacked, aligned, and stable — no tilting, overhanging, crushing, or shifting observed. Yellow aisle markings are intact and unobstructed. A worker in high-visibility vest is present but not interacting hazardously. Overall scene meets OSHA warehouse safety standards.
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 4.9s | TTFT: 3.02s | 26 tokens @ 13.7 tok/s

**Model response:**
```
category: safe
severity: none
confidence: 95%
location: entire visible floor and all shelving units
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 7.7s | TTFT: 3.11s | 61 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: floor is dry with no visible spills; all boxes on shelves are neatly stacked, aligned, and stable with no overhangs, tilts, or crushing — conditions appear fully compliant with OSHA warehouse safety standards.
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 95%
- **Time:** 5.3s | TTFT: 2.99s | 31 tokens @ 13.7 tok/s

**Model response:**
```
safe, none, 95%, clean floor with no fluid; all shelved items are neatly aligned and stable without visible tilt or overhang.
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 7.0s | TTFT: 3.1s | 51 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor, near foreground — dark reddish-brown liquid puddle with visible sheen, likely oil or chemical, surrounded by wet patches extending slightly outward.
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 11.7s | TTFT: 3.04s | 114 tokens @ 13.3 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, multiple wet/reflective puddles visible between shelving units

The glossy, reflective patches and irregular shapes indicate liquid pooling on the concrete floor — consistent with spills. This creates slip hazard risk for warehouse workers. No clear evidence of improper stacking (boxes are neatly aligned), but the floor condition alone warrants classification as “spill” with high severity due to potential fall-injury consequences under OSHA standards (29 CFR 1910.22).
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** none
- **Confidence:** 95%
- **Time:** 7.5s | TTFT: 3.0s | 60 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 95%  
location: entire warehouse floor and shelving units — no visible spills or improper stacking. All boxes are neatly arranged, upright, and stable on pallets/shelves; concrete floor is dry and clean with no fluid traces.
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 98%
- **Time:** 8.3s | TTFT: 3.0s | 70 tokens @ 13.4 tok/s

**Model response:**
```
Based on the visual analysis of the provided drone camera image:

category: improper_stacking
severity: critical
confidence: 98%
location: The shelves on both sides of the warehouse aisle, specifically the pallets and boxes stacked at high levels that appear disorganized, tilted, jumbled, and lack proper alignment or securing.
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 8.5s | TTFT: 3.02s | 73 tokens @ 13.4 tok/s

**Model response:**
```
category: improper_stacking  
severity: high  
confidence: 95%  
location: right foreground aisle — boxes are tilted, leaning, jumbled, and unstable on shelving units; multiple shelves show misaligned or overloaded loads posing imminent fall hazard. Left side also shows some minor uneven stacking but less critical. Floor is dry with no visible spills.
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** critical
- **Confidence:** 95%
- **Time:** 6.8s | TTFT: 3.01s | 51 tokens @ 13.4 tok/s

**Model response:**
```
category: spill
severity: critical
confidence: 95%
location: center aisle floor, directly in front of the worker — a large, reflective liquid puddle occupies the walkway between shelving units, creating an immediate slip hazard.
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 5.8s | TTFT: 3.0s | 37 tokens @ 13.5 tok/s

**Model response:**
```
category: spill
severity: high
confidence: 95%
location: center aisle floor, near bottom right of image — large brown liquid puddle clearly visible on concrete
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 13.1s | TTFT: 3.0s | 128 tokens @ 13.3 tok/s

**Model response:**
```
Based on the visual analysis of the warehouse scene provided in the drone camera image:

category: spill
severity: high
confidence: 95%
location: Central aisle floor, specifically a large brownish liquid puddle located approximately halfway down the walkway between the shelves.

**Reasoning:**
The image clearly shows a significant amount of wet liquid on the concrete floor in the center of the aisle. This creates a slip hazard for the worker visible further back. While the boxes appear neatly stacked and stable, the presence of the spill violates standard OSHA requirements for keeping walkways clear of tripping and slipping hazards.
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** high
- **Confidence:** 95%
- **Time:** 18.0s | TTFT: 3.31s | 193 tokens @ 13.4 tok/s

**Model response:**
```
Looking at the image, we analyze:

- **FLOOR**: The concrete floor appears dry and clean — no visible spills, puddles, oil slicks, or wet patches. Some light reflections are present but appear to be from overhead lighting, not fluid.

- **SHELVES (both sides)**: Boxes on shelves are generally neatly stacked with consistent alignment. However, one box in the center-right foreground is visibly tilted and hanging off the shelf edge — it’s rotated at an angle and not flat against the shelf surface. This poses a falling hazard. Additionally, other boxes near that area show slight unevenness or shifting, suggesting potential instability.

This qualifies as **improper_stacking** due to the clearly unstable box overhanging the shelf.

---

category: improper_stacking  
severity: high  
confidence: 95%  
location: center-right foreground shelf — one large cardboard box tilted and hanging off the edge
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** none
- **Confidence:** 98%
- **Time:** 8.6s | TTFT: 3.42s | 70 tokens @ 13.4 tok/s

**Model response:**
```
category: safe  
severity: none  
confidence: 98%  
location: entire warehouse floor and all visible shelving units on both sides — no spills, no unstable stacking, no tilted/overhanging boxes, no crushed or misaligned loads; worker in high-visibility vest appears to be working normally with no immediate hazards observed.
```

---

