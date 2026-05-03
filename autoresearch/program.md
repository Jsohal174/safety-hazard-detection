# HAWKEYE Autoresearch — VLM LoRA Hyperparameter Search

You are an autonomous research agent running on a Mac Mini M2 16GB. Your goal is to find the optimal LoRA hyperparameters for fine-tuning Qwen3.5 VLM on warehouse safety hazard detection.

## In-Scope Files

- `README.md` — this project overview (read-only)
- `prepare.py` — data loading, evaluation function (DO NOT MODIFY)
- `train.py` — LoRA training script with hyperparameters at the top (YOU MODIFY THIS)
- `results.tsv` — experiment log (append-only)
- `program.md` — these instructions (read-only)

## Setup (one-time)

1. Activate MLX venv: `source ~/mlx-env/bin/activate`
2. Create a git repo if not already: `git init && git add -A && git commit -m "initial"`
3. Create experiment branch: `git checkout -b autoresearch/$(date +%b%d)`
4. Read `README.md`, `prepare.py`, and `train.py` to understand the setup
5. Run `python3 prepare.py --check` to verify data and model are ready
6. Initialize `results.tsv`:
   ```
   echo "commit\taccuracy\tval_loss\tpeak_mem_gb\ttime_min\tstatus\tdescription" > results.tsv
   ```

## The Experiment Loop

NEVER STOP. Once the loop begins, do NOT pause to ask the human. Run autonomously.

```
while true:
    1. Read current train.py, results.tsv, and decide what to change
    2. Make ONE change to a hyperparameter in train.py
    3. git add train.py && git commit -m "description of change"
    4. Run: python3 train.py 2>&1 | tee run.log
    5. Extract metrics: grep "^METRIC:" run.log
    6. Append to results.tsv
    7. If accuracy improved → keep (branch advances)
       If accuracy same/worse → git reset --hard HEAD~1
    8. Go to step 1
```

## What to Modify in train.py

Only change the HYPERPARAMETER BLOCK at the top of train.py. The variables you can change:

```python
# === HYPERPARAMETERS (MODIFY THESE) ===
MODEL_NAME = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"  # or 7B
LORA_RANK = 16          # try: 4, 8, 16, 32, 64
LORA_ALPHA = 32         # try: rank*1, rank*2, rank*4
LEARNING_RATE = 1e-4    # try: 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4
LORA_LAYERS = 16        # number of layers to apply LoRA to
BATCH_SIZE = 1          # try: 1, 2, 4
GRAD_ACCUM = 4          # try: 2, 4, 8
NUM_EPOCHS = 3          # try: 2, 3, 5
LORA_DROPOUT = 0.05     # try: 0.0, 0.05, 0.1
TRAIN_STYLE = "all"     # try: "all", "direct", "multi_turn", "reasoning"
WARMUP_STEPS = 10       # try: 0, 5, 10, 20
# === END HYPERPARAMETERS ===
```

## Search Strategy

1. **First: Find the best learning rate** (most impactful parameter)
   - Try: 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4
   - Keep the best one

2. **Second: Optimize rank and alpha**
   - Try rank: 4, 8, 16, 32, 64
   - Try alpha/rank ratio: 1, 2, 4

3. **Third: Training data style**
   - Try: "all", "direct", "multi_turn", "reasoning"

4. **Fourth: Number of layers**
   - Try: 4, 8, 16, 24

5. **Fifth: Fine-tune remaining parameters**
   - Batch size, gradient accumulation, epochs, dropout, warmup

## Rules

- Change ONE parameter per experiment
- If experiment crashes (OOM, error): record status="crash", revert, try smaller config
- If accuracy doesn't change (±0.5%): record status="neutral", revert
- If accuracy improves: record status="keep", advance branch
- Log everything clearly. Future decisions depend on the results history.
- Simpler is better. If two configs give ~same accuracy, prefer fewer parameters.
- NEVER modify prepare.py
- NEVER install new packages
- Target: beat 82.7% accuracy (the zero-shot baseline for Qwen3.5 4b direct nothink)

## Metric

The primary metric is **accuracy** on 50 validation images across 5 categories:
- spill, forklift_violation, improper_stacking, obstacle, safe

Accuracy = correct classifications / total images. Higher is better.

## Time Budget

Each experiment should take approximately 10-15 minutes. The training script has a built-in time limit. If training exceeds 20 minutes, something is wrong — kill it and try a smaller config.
