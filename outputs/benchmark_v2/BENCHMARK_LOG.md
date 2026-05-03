# HAWKEYE VLM Benchmark v2 — Master Log

**Date:** 2026-03-28 23:55
**Test set:** 0 images (0/category)
**Models:** 1 | **Prompts:** 1 | **Total runs:** 1
**Think:** OFF | **Temperature:** 0.0

---

## hawkeye — `direct`

**Accuracy:** 0/0 = **0.0%**
**Avg time:** 0.0s/image | **Total:** 0.0 min

<details><summary>Prompt used</summary>

```
Classify this warehouse image into exactly one category:
- spill: liquid/fluid on the warehouse floor
- forklift_violation: unsafe forklift operation (raised forks, no seatbelt, pedestrian too close, improper load)
- improper_stacking: boxes on shelves are tilted, overhanging, crushed, torn wrap, unstable
- obstacle: objects blocking the aisle floor (fallen boxes, debris, abandoned equipment, broken pallets)
- safe: clean warehouse, no hazards present

Respond in this format:
category: <one of the above>
description: <one sentence describing what you see>
```
</details>

| Category | P | R | F1 | TP | FP | FN |
|----------|---|---|----|----|----|----|--|
| spill | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |
| forklift_violation | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |
| improper_stacking | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |
| obstacle | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |
| safe | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

<details><summary>Per-image results (click to expand)</summary>

| # | Image | GT | Predicted | OK? | Time | Model Response (truncated) |
|---|-------|----|-----------|-----|------|----------------------------|

</details>

---

# Final Leaderboard

| # | Model | Prompt | Accuracy | Avg Time |
|---|-------|--------|----------|----------|
| 1 | hawkeye | direct | 0.0% | 0.0s |
