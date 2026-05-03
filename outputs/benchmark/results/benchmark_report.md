# VLM Warehouse Hazard Detection Benchmark

Date: 2026-03-13 03:49
Test set: 99 images (33 spill, 33 improper_stacking, 33 safe)

## Summary

| Model | Prompt | NoThink | Overall Acc | Spill F1 | Stacking F1 | Safe F1 | Avg Time |
|-------|--------|---------|-------------|----------|-------------|---------|----------|
| qwen3.5:9b | simple | Yes | 79.8% | 0.952 | 0.667 | 0.762 | 9.9s |
| qwen3.5:9b | simple | No | 85.6% | 0.984 | 0.750 | 0.812 | 115.9s |
| qwen3.5:9b | cot | Yes | 80.6% | 0.954 | 0.653 | 0.780 | 13.5s |
| qwen3.5:9b | cot | No | 81.8% | 0.985 | 0.653 | 0.786 | 55.9s |

## qwen3.5:9b — simple (nothink)

Overall accuracy: **79.8%** (99/99 valid)
Average inference time: **9.9s** per image

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 90.9% | 1.000 | 0.909 | 0.952 | 30 | 0 | 3 |
| improper_stacking | 51.5% | 0.944 | 0.515 | 0.667 | 17 | 1 | 16 |
| safe | 97.0% | 0.627 | 0.970 | 0.762 | 32 | 19 | 1 |

Confusion matrix (rows=ground truth, cols=predicted):

| | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 30 | 0 | 3 | 0 |
| improper_stacking | 0 | 17 | 16 | 0 |
| safe | 0 | 1 | 32 | 0 |

## qwen3.5:9b — simple

Overall accuracy: **85.6%** (90/99 valid)
Average inference time: **115.9s** per image

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 96.9% | 1.000 | 0.969 | 0.984 | 31 | 0 | 1 |
| improper_stacking | 60.0% | 1.000 | 0.600 | 0.750 | 18 | 0 | 12 |
| safe | 100.0% | 0.683 | 1.000 | 0.812 | 28 | 13 | 0 |

Confusion matrix (rows=ground truth, cols=predicted):

| | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 31 | 0 | 1 | 0 |
| improper_stacking | 0 | 18 | 12 | 0 |
| safe | 0 | 0 | 28 | 0 |

## qwen3.5:9b — cot (nothink)

Overall accuracy: **80.6%** (98/99 valid)
Average inference time: **13.5s** per image

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 93.9% | 0.969 | 0.939 | 0.954 | 31 | 1 | 2 |
| improper_stacking | 50.0% | 0.941 | 0.500 | 0.653 | 16 | 1 | 16 |
| safe | 97.0% | 0.653 | 0.970 | 0.780 | 32 | 17 | 1 |

Confusion matrix (rows=ground truth, cols=predicted):

| | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 31 | 0 | 2 | 0 |
| improper_stacking | 1 | 16 | 15 | 0 |
| safe | 0 | 1 | 32 | 0 |

## qwen3.5:9b — cot

Overall accuracy: **81.8%** (99/99 valid)
Average inference time: **55.9s** per image

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 97.0% | 1.000 | 0.970 | 0.985 | 32 | 0 | 1 |
| improper_stacking | 48.5% | 1.000 | 0.485 | 0.653 | 16 | 0 | 17 |
| safe | 100.0% | 0.647 | 1.000 | 0.786 | 33 | 18 | 0 |

Confusion matrix (rows=ground truth, cols=predicted):

| | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 32 | 0 | 1 | 0 |
| improper_stacking | 0 | 16 | 17 | 0 |
| safe | 0 | 0 | 33 | 0 |