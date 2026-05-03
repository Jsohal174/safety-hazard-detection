"""
Fine-tune Qwen3-VL on warehouse safety hazard detection.
Usage: python train.py --model 2B --num_images 100 --output_dir /path/to/save
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="2B", choices=["2B", "4B"])
parser.add_argument("--num_images", type=int, default=727)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/hawkeye_data"))
parser.add_argument("--grad_accum", type=int, default=4)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

MODEL_MAP = {
    "2B": "Qwen/Qwen3-VL-2B-Instruct",
    "4B": "Qwen/Qwen3-VL-4B-Instruct",
}
model_name = MODEL_MAP[args.model]

print("=" * 80)
print(f"Training {args.model} ({model_name})")
print(f"Images: {args.num_images}, Epochs: {args.epochs}, LR: {args.lr}")
print(f"LoRA rank: {args.rank}, alpha: {args.alpha}, seed: {args.seed}")
print(f"Output: {args.output_dir}")
print("=" * 80)

# === Load data ===
data_dir = Path(args.data_dir)
images_dir = data_dir / "images"

with open(data_dir / "training_conversations.jsonl") as f:
    all_convos = [json.loads(line) for line in f]

with open(data_dir / "train.jsonl") as f:
    train_meta = [json.loads(line) for line in f]
train_image_names = set(e["image"] for e in train_meta)

train_convos = [c for c in all_convos if c["image"] in train_image_names]
print(f"Total training conversations: {len(train_convos)}")

# Sample by image count for fair comparison
unique_train_images = sorted(set(c["image"] for c in train_convos))
random.shuffle(unique_train_images)
selected_images = set(unique_train_images[: args.num_images])
selected_convos = [c for c in train_convos if c["image"] in selected_images]
print(f"Selected {len(selected_images)} images, {len(selected_convos)} conversations")

# === Load model ===
print(f"\nLoading model {model_name}...")
processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=256 * 256,
    max_pixels=512 * 512,
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
)

# === LoRA ===
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Training ===
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = (len(selected_convos) // args.grad_accum) * args.epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.05),
    num_training_steps=total_steps,
)

model.train()
step = 0
for epoch in range(args.epochs):
    print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
    random.shuffle(selected_convos)

    for i, conv in enumerate(selected_convos):
        img_name = conv["image"].replace(".png", ".jpg")
        img_path = images_dir / img_name
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"  WARNING: missing image {img_path}, skipping")
            continue

        messages = conv.get("messages") or conv.get("turns", [])
        formatted = []
        first_user = True
        for msg in messages:
            if msg["role"] == "user" and first_user:
                formatted.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": msg["content"]},
                    ],
                })
                first_user = False
            else:
                formatted.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

        text = processor.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / args.grad_accum
        loss.backward()

        if (i + 1) % args.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  Step {step}: loss={loss.item() * args.grad_accum:.4f}")

# === Save ===
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
processor.save_pretrained(args.output_dir)
print(f"\nSaved to {args.output_dir}")
