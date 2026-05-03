"""
HAWKEYE Autoresearch — VLM LoRA Training Script
The autoresearch agent modifies the HYPERPARAMETERS block below.
Uses mlx_vlm.lora for vision-language model fine-tuning WITH images.
"""

import json
import time
import os
import sys
import subprocess

# === HYPERPARAMETERS (MODIFY THESE) ===
MODEL_NAME = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
NUM_EPOCHS = 3
LORA_DROPOUT = 0.0
TRAIN_STYLE = "direct"     # "all", "direct", "multi_turn", "reasoning"
TRAIN_VISION = False        # unfreeze vision layers too
GRAD_ACCUM = 4
GRAD_CHECKPOINT = True      # save memory by recomputing activations
MAX_SEQ_LENGTH = 512
ADAPTER_PATH = "adapters"
# === END HYPERPARAMETERS ===

# === CONSTANTS (DO NOT MODIFY) ===
MAX_TRAINING_TIME = 900   # 15 minutes hard cap
# === END CONSTANTS ===


def run_training():
    """Run VLM LoRA fine-tuning using mlx_vlm.lora CLI."""
    start_time = time.time()

    # Step 1: Build HF dataset
    print("Building HuggingFace dataset...")
    result = subprocess.run(
        ["python3", "prepare.py", "--build-dataset", "--style", TRAIN_STYLE],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] Dataset build failed: {result.stderr}")
        print(f"METRIC:accuracy:0.0000")
        print(f"METRIC:status:crash")
        return

    dataset_path = f"data/hf_dataset/{TRAIN_STYLE}"

    # Calculate iterations
    # Count training examples
    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        num_examples = len(ds)
    except:
        num_examples = 200  # fallback

    steps_per_epoch = max(num_examples // (BATCH_SIZE * GRAD_ACCUM), 1)
    total_iters = steps_per_epoch * NUM_EPOCHS

    print(f"\n{'='*60}")
    print(f"HAWKEYE VLM LoRA Training")
    print(f"{'='*60}")
    print(f"Model:          {MODEL_NAME}")
    print(f"LoRA rank:      {LORA_RANK}")
    print(f"LoRA alpha:     {LORA_ALPHA}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Grad accum:     {GRAD_ACCUM}")
    print(f"Epochs:         {NUM_EPOCHS}")
    print(f"Total iters:    {total_iters}")
    print(f"Train style:    {TRAIN_STYLE}")
    print(f"Train vision:   {TRAIN_VISION}")
    print(f"Max seq len:    {MAX_SEQ_LENGTH}")
    print(f"Examples:       {num_examples}")
    print(f"{'='*60}\n")

    # Build mlx_vlm.lora command
    cmd = [
        "python3", "-m", "mlx_vlm.lora",
        "--model-path", MODEL_NAME,
        "--dataset", dataset_path,
        "--lora-rank", str(LORA_RANK),
        "--lora-alpha", str(LORA_ALPHA),
        "--lora-dropout", str(LORA_DROPOUT),
        "--learning-rate", str(LEARNING_RATE),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(total_iters),
        "--epochs", str(NUM_EPOCHS),
        "--gradient-accumulation-steps", str(GRAD_ACCUM),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--output-path", ADAPTER_PATH,
        "--steps-per-report", "10",
        "--steps-per-eval", str(max(steps_per_epoch, 10)),
        "--train-on-completions",
    ]

    if TRAIN_VISION:
        cmd.append("--train-vision")

    if GRAD_CHECKPOINT:
        cmd.append("--grad-checkpoint")

    print(f"Running: {' '.join(cmd)}\n")

    # Run training with time limit
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                output_lines.append(line.rstrip())

            elapsed = time.time() - start_time
            if elapsed > MAX_TRAINING_TIME:
                print(f"\n[TIMEOUT] Exceeded {MAX_TRAINING_TIME}s. Killing.")
                process.kill()
                break

        process.wait()
        training_time = time.time() - start_time

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print(f"METRIC:accuracy:0.0000")
        print(f"METRIC:training_time_min:{(time.time()-start_time)/60:.2f}")
        print(f"METRIC:status:crash")
        return

    print(f"\nTraining complete in {training_time/60:.1f} minutes")

    # Step 2: Evaluate with images
    print("\n" + "="*60)
    print("Evaluating on validation set WITH images...")
    print("="*60)

    from prepare import evaluate_model
    accuracy, per_cat = evaluate_model(MODEL_NAME, ADAPTER_PATH)

    print(f"\nMETRIC:training_time_min:{training_time/60:.2f}")
    print(f"METRIC:status:ok")


if __name__ == "__main__":
    run_training()
