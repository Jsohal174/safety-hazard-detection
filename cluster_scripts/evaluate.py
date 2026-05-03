"""
Evaluate a fine-tuned model on the 186 test images.
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/hawkeye_data"))
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

data_dir = Path(args.data_dir)
images_dir = data_dir / "images"

with open(data_dir / "test.jsonl") as f:
    test_meta = [json.loads(line) for line in f]
full_meta = {}
with open(data_dir / "metadata.jsonl") as f:
    for line in f:
        e = json.loads(line)
        full_meta[e["image"]] = e

print(f"Loaded {len(test_meta)} test images")

print(f"Loading base: {args.base_model}")
processor = AutoProcessor.from_pretrained(
    args.base_model, min_pixels=256 * 256, max_pixels=512 * 512
)
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.base_model, dtype=torch.bfloat16, device_map="cuda"
)

print(f"Loading LoRA: {args.model_dir}")
model = PeftModel.from_pretrained(base_model, args.model_dir)
model.eval()

PROMPT = (
    "Classify this warehouse image into exactly one category: spill, "
    "forklift_violation, improper_stacking, obstacle, or safe. "
    "Respond with the category and a one-sentence description."
)

CATEGORIES = ["forklift_violation", "improper_stacking", "obstacle", "spill", "safe"]


def parse_category(text):
    text_lower = text.lower()
    for line in text_lower.split("\n"):
        match = re.match(r"category:\s*(\w+)", line.strip())
        if match:
            return match.group(1)
    for cat in CATEGORIES:
        if cat in text_lower:
            return cat
    return None


results = []
correct = 0
for i, entry in enumerate(test_meta):
    img_name = entry["image"].replace(".png", ".jpg")
    img_path = images_dir / img_name
    image = Image.open(img_path).convert("RGB")
    full_entry = full_meta.get(entry["image"], entry)
    gt = full_entry["category"]
    accept_also = full_entry.get("accept_also", [])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    pred = parse_category(response)
    is_correct = pred == gt or (pred and pred in accept_also)
    if is_correct:
        correct += 1

    results.append(
        {
            "image": entry["image"],
            "gt": gt,
            "accept_also": accept_also,
            "pred": pred,
            "correct": is_correct,
            "response": response,
        }
    )
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(test_meta)}: {correct}/{i + 1} = {correct / (i + 1) * 100:.1f}%")

acc = correct / len(test_meta)
summary = {
    "total": len(test_meta),
    "correct": correct,
    "accuracy": acc,
    "results": results,
}
with open(args.output, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nFinal: {correct}/{len(test_meta)} = {acc * 100:.1f}%")
print(f"Saved to {args.output}")
