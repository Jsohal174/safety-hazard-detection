"""
HAWKEYE Autoresearch — Data Preparation & Evaluation
DO NOT MODIFY THIS FILE. The agent only modifies train.py.
"""

import json
import os
import sys
import random
from pathlib import Path
from PIL import Image

# === CONSTANTS ===
DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"
TRAIN_CONVOS = DATA_DIR / "train_conversations.jsonl"  # our format
VALID_META = DATA_DIR / "valid_meta.jsonl"               # image + category
HF_DATASET_DIR = DATA_DIR / "hf_dataset"                 # HuggingFace format
CATEGORIES = ["spill", "forklift_violation", "improper_stacking", "obstacle", "safe"]


def build_hf_dataset(style="all"):
    """
    Convert our conversation format to HuggingFace dataset format.
    Creates a dataset with 'images' and 'messages' columns.
    """
    from datasets import Dataset, Features, Value, Sequence, Image as HFImage

    convos_data = [json.loads(l) for l in open(TRAIN_CONVOS)]

    records = []
    for item in convos_data:
        img_name = item["image"].replace(".png", ".jpg")
        img_path = str(IMAGES_DIR / img_name)
        if not os.path.exists(img_path):
            # Try images_all symlink
            img_path = str(DATA_DIR / "images_all" / img_name)
            if not os.path.exists(img_path):
                continue

        for convo in item.get("conversations", []):
            s = convo.get("style", "")
            if style != "all" and s != style:
                continue

            msgs = convo.get("messages") or convo.get("turns", [])
            if not msgs:
                continue

            # Format messages for VLM training
            # First user message should reference the image
            formatted_msgs = []
            first_user = True
            for msg in msgs:
                if msg["role"] == "user" and first_user:
                    # Add image reference to first user message
                    formatted_msgs.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": msg["content"]}
                        ]
                    })
                    first_user = False
                elif msg["role"] == "user":
                    formatted_msgs.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg["content"]}
                        ]
                    })
                else:
                    formatted_msgs.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })

            records.append({
                "image": img_path,
                "messages": formatted_msgs,
            })

    # Create HF dataset
    dataset = Dataset.from_list(records)

    # Cast image column to PIL
    dataset = dataset.cast_column("image", HFImage())

    # Save
    os.makedirs(HF_DATASET_DIR, exist_ok=True)
    dataset.save_to_disk(str(HF_DATASET_DIR / style))

    print(f"Built HF dataset: {len(records)} examples (style={style})")
    print(f"Saved to: {HF_DATASET_DIR / style}")
    return str(HF_DATASET_DIR / style)


def evaluate_model(model_path, adapter_path=None):
    """
    Evaluate a model on the validation set using mlx-vlm inference.
    Passes actual images to the model.
    """
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image

    # Load model
    print(f"Loading model: {model_path}")
    kwargs = {}
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    model, processor = load(model_path, **kwargs)

    # Load validation data
    valid_data = [json.loads(l) for l in open(VALID_META)]

    prompt = (
        "Classify this warehouse image into one category: "
        "spill, forklift_violation, improper_stacking, obstacle, or safe. "
        "Respond with ONLY the category name, nothing else."
    )

    correct = 0
    total = 0
    per_category = {c: {"correct": 0, "total": 0} for c in CATEGORIES}

    for item in valid_data:
        gt = item["category"]
        accept = [gt] + item.get("accept_also", [])
        per_category[gt]["total"] += 1
        total += 1

        # Find image
        img_name = item["image"].replace(".png", ".jpg")
        img_path = str(IMAGES_DIR / img_name)
        if not os.path.exists(img_path):
            img_path = str(DATA_DIR / "images_all" / img_name)

        if not os.path.exists(img_path):
            continue

        try:
            # Run VLM inference with image
            output = generate(
                model, processor,
                prompt=prompt,
                image=img_path,
                max_tokens=30,
                verbose=False,
            )

            response = output.strip().lower()

            # Extract category
            predicted = "unknown"
            for cat in CATEGORIES:
                if cat in response:
                    predicted = cat
                    break

            if predicted in accept:
                correct += 1
                per_category[gt]["correct"] += 1

        except Exception as e:
            print(f"  Error on {item['image']}: {e}")

    accuracy = correct / max(total, 1)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"METRIC:accuracy:{accuracy:.4f}")
    print(f"METRIC:correct:{correct}")
    print(f"METRIC:total:{total}")
    for cat in CATEGORIES:
        ct = per_category[cat]["total"]
        cc = per_category[cat]["correct"]
        cat_acc = cc / max(ct, 1)
        print(f"METRIC:{cat}_acc:{cat_acc:.4f} ({cc}/{ct})")
    print(f"{'='*50}")

    return accuracy, per_category


def check_setup():
    """Verify all data and dependencies are ready."""
    print("=== HAWKEYE Autoresearch Setup Check ===")

    # Check MLX
    try:
        import mlx.core as mx
        print(f"[OK] MLX: {mx.__version__ if hasattr(mx, '__version__') else 'yes'}")
    except ImportError:
        print("[FAIL] MLX not installed")
        return False

    # Check mlx-vlm
    try:
        import mlx_vlm
        print(f"[OK] mlx-vlm available")
    except ImportError:
        print("[FAIL] mlx-vlm not installed. Run: pip install mlx-vlm")
        return False

    # Check datasets
    try:
        import datasets
        print(f"[OK] datasets library available")
    except ImportError:
        print("[FAIL] datasets not installed. Run: pip install datasets")
        return False

    # Check PIL
    try:
        from PIL import Image
        print(f"[OK] PIL available")
    except ImportError:
        print("[FAIL] Pillow not installed. Run: pip install Pillow")
        return False

    # Check data
    if not TRAIN_CONVOS.exists():
        print(f"[FAIL] Training data not found: {TRAIN_CONVOS}")
        return False
    train_count = sum(1 for _ in open(TRAIN_CONVOS))
    print(f"[OK] Training conversations: {train_count}")

    if not VALID_META.exists():
        print(f"[FAIL] Validation data not found: {VALID_META}")
        return False
    valid_count = sum(1 for _ in open(VALID_META))
    print(f"[OK] Validation images: {valid_count}")

    # Check images
    img_count = len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0
    print(f"[OK] Validation images in dir: {img_count}")

    print("\n=== Setup OK ===")
    return True


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_setup()
    elif "--build-dataset" in sys.argv:
        style = "all"
        if "--style" in sys.argv:
            style = sys.argv[sys.argv.index("--style") + 1]
        build_hf_dataset(style)
    elif "--eval" in sys.argv:
        model = sys.argv[sys.argv.index("--eval") + 1]
        adapter = None
        if "--adapter" in sys.argv:
            adapter = sys.argv[sys.argv.index("--adapter") + 1]
        evaluate_model(model, adapter)
    else:
        print("Usage:")
        print("  python3 prepare.py --check")
        print("  python3 prepare.py --build-dataset [--style direct|multi_turn|reasoning|all]")
        print("  python3 prepare.py --eval MODEL [--adapter PATH]")
