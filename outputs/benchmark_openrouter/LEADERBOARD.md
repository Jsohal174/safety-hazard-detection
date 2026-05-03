# OpenRouter benchmark — pedestrian-proximity hard set

_Generated 2026-05-02 19:40_

Test set: 12 forklift_violation images (subtype = pedestrian_proximity), all where the trained 2B got it right.

Prompt: same `direct` prompt as benchmark_v2.py.

| # | Model | Acc (all) | Correct | Refused | Errored | Avg time | Avg tokens (in/out) |
|---|-------|-----------|---------|---------|---------|----------|---------------------|
| 1 | `openai/gpt-5.5` | 100.0% | 12/12 | 0 | 0 | 5.5s | 817/97 |
| 2 | `openai/gpt-5.4` | 100.0% | 12/12 | 0 | 0 | 1.7s | 817/33 |
| 3 | `openai/gpt-5.4-mini` | 100.0% | 12/12 | 0 | 0 | 1.2s | 817/33 |
| 4 | `google/gemini-3.1-pro-preview` | 100.0% | 12/12 | 0 | 0 | 8.1s | 1181/595 |
| 5 | `qwen/qwen3.6-plus` | 100.0% | 12/12 | 0 | 0 | 15.8s | 718/1523 |
| 6 | `x-ai/grok-4.20` | 91.7% | 11/12 | 0 | 0 | 1.3s | 819/32 |
| 7 | `google/gemini-3.1-flash-lite-preview` | 83.3% | 10/12 | 0 | 0 | 1.7s | 1181/28 |
| 8 | `qwen/qwen3.5-397b-a17b` | 75.0% | 9/12 | 0 | 0 | 16.9s | 718/773 |
| 9 | `qwen/qwen3-vl-235b-a22b-thinking` | 66.7% | 8/12 | 0 | 0 | 7.5s | 714/336 |
| 10 | `mistralai/pixtral-large-2411` | 66.7% | 8/12 | 0 | 0 | 1.6s | 2490/31 |
| 11 | `z-ai/glm-4.6v` | 50.0% | 6/12 | 0 | 0 | 6.7s | 873/210 |
| 12 | `meta-llama/llama-4-maverick` | 33.3% | 4/12 | 0 | 0 | 1.4s | 1484/28 |
| 13 | `moonshotai/kimi-k2.6` | 33.3% | 4/12 | 0 | 0 | 14.7s | 912/638 |
| 14 | `meta-llama/llama-4-scout` | 16.7% | 2/12 | 0 | 0 | 1.1s | 1464/34 |
| 15 | `qwen/qwen3.5-9b` | 0.0% | 0/12 | 0 | 0 | 9.1s | 718/600 |

**For comparison:** Trained 2B got 12/12 = 100% on this set.
