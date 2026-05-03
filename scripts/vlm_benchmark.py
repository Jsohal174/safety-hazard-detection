#!/usr/bin/env python3
"""
VLM Benchmark: Test vision models on warehouse hazard detection.
Builds a shuffled 100-image test set, runs each model, saves results + report.

Usage:
    python scripts/vlm_benchmark.py                    # run full benchmark
    python scripts/vlm_benchmark.py --build-only       # just build test set
    python scripts/vlm_benchmark.py --model gemma3:4b   # test single model
    python scripts/vlm_benchmark.py --nothink          # disable thinking for qwen3.5
"""

import base64, json, os, random, shutil, sys, time, urllib.request, urllib.error
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

SERVER = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PROJECT = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection"

SPILL_DIR = f"{PROJECT}/outputs/datasets/images/spill"
STACKING_DIR = f"{PROJECT}/outputs/datasets/images/improper_stacking"
SAFE_DIR = f"{PROJECT}/realistic"

TEST_SET_DIR = f"{PROJECT}/outputs/benchmark/test_images"
RESULTS_DIR = f"{PROJECT}/outputs/benchmark/results"

# Models to benchmark (vision-capable on Mac Mini)
MODELS = [
    "qwen3.5:9b",
    "qwen3-vl:8b",
    "gemma3:4b",
    "qwen3.5:4b",
    "qwen3-vl:2b",
    "granite3.2-vision:2b",
    "llava-phi3:latest",
    "moondream:latest",
]

# Prompts
# Best prompt from A/B testing (p1_simple — 86.7% on quick test)
SIMPLE_PROMPT = """You are an OSHA warehouse safety inspector analyzing a drone camera image. Classify the safety condition of this warehouse scene.

Look at the FLOOR for spills: liquid puddles, wet reflective patches, oil slicks, chemical leaks, any fluid on the concrete.

Look at EVERY SHELF on both sides for stacking problems: boxes tilted or rotated at an angle instead of flat, boxes hanging over the shelf edge, crushed or deformed boxes supporting weight above, torn or missing shrink wrap, boxes stacked unevenly or jumbled instead of neatly aligned in rows, loads shifted to one side of the shelf, anything that looks unstable or could fall.

Classify the scene into ONE of these categories:
- spill: there is liquid/fluid on the warehouse floor
- improper_stacking: boxes or items on shelves are poorly arranged, unstable, tilted, overhanging, crushed, or messy
- minor_hazard: something looks slightly off but not a clear violation
- safe: floor is clean and dry, all items on shelves are neatly organized and stable
- unable_to_determine: image is unclear

Respond in this format:
category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: critical / high / medium / low / none
confidence: 0-100%
location: brief description of where the issue is"""

# Second best — binary questions then classify (p3 — 80.0% on quick test)
COT_PROMPT = """Look at this warehouse image from a drone camera.

Answer these TWO questions:

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES or NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES or NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%"""


def api(endpoint, payload=None, stream=False, timeout=300):
    url = f"{SERVER}{endpoint}"
    if payload:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp if stream else json.loads(resp.read())


def load_model(model):
    print(f"  Loading {model}...")
    api("/api/chat", {"model": model, "messages": [], "keep_alive": -1})
    print(f"  {model} loaded.")


def stop_all():
    data = api("/api/ps")
    for m in data.get("models", []):
        name = m.get("name", "")
        api("/api/chat", {"model": name, "messages": [], "keep_alive": 0})
        print(f"  Unloaded {name}")


def run_inference(model, image_path, prompt, nothink=False):
    """Run model on image, return (response_text, stats_dict)."""
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": True,
        "keep_alive": -1,
    }
    if nothink:
        payload["think"] = False

    start = time.time()
    first_token_time = None
    tokens = []
    resp = api("/api/chat", payload, stream=True)
    final_stats = {}
    for line in resp:
        if not line.strip():
            continue
        chunk = json.loads(line)
        t = chunk.get("message", {}).get("content", "")
        if t:
            if first_token_time is None:
                first_token_time = time.time()
            tokens.append(t)
        if chunk.get("done"):
            final_stats = chunk
            break
    resp.close()

    elapsed = time.time() - start
    text = "".join(tokens)
    eval_count = final_stats.get("eval_count", 0)
    eval_dur = final_stats.get("eval_duration", 0)
    prompt_count = final_stats.get("prompt_eval_count", 0)
    gen_tps = (eval_count / (eval_dur / 1e9)) if eval_dur > 0 else 0
    ttft = (first_token_time - start) if first_token_time else 0

    stats = {
        "total_time": round(elapsed, 1),
        "ttft": round(ttft, 2),
        "tokens_generated": eval_count,
        "gen_tps": round(gen_tps, 1),
        "prompt_tokens": prompt_count,
    }
    return text, stats


def _classify_text(text):
    """Determine category from a chunk of text."""
    t = text.lower().strip().strip("*").strip()
    if "spill" in t:
        return "spill"
    if "improper_stacking" in t or "improper stacking" in t:
        return "improper_stacking"
    # "proper stacking"/"proper_stacking" means safe, not a hazard
    if "proper" in t and "stacking" in t and "improper" not in t:
        return "safe"
    # bare "stacking" = improper stacking
    if "stacking" in t:
        return "improper_stacking"
    if "minor_hazard" in t or "minor hazard" in t:
        return "improper_stacking"
    # "unsafe" should NOT match "safe" — it indicates a hazard
    if "unsafe" in t or "not safe" in t:
        return None  # ambiguous hazard, let later strategies handle it
    if "safe" in t:
        return "safe"
    if "unable" in t:
        return None
    return None


def _extract_confidence(text):
    """Extract a confidence percentage from text."""
    import re
    # Match patterns like "95%", "confidence: 85%", "85"
    m = re.search(r'(\d{1,3})\s*%', text)
    if m:
        return int(m.group(1))
    return None


def _extract_severity(text):
    """Extract severity from text."""
    t = text.lower()
    for s in ["critical", "high", "medium", "low", "none"]:
        if s in t:
            return s
    return None


def parse_response(text):
    """Robust parser — handles many output formats from different models."""
    text_lower = text.lower()
    detected_type = None
    confidence = None
    severity = None

    # === Strategy 1: Look for labeled lines (category:/type:/hazard:) ===
    for line in text_lower.split("\n"):
        stripped = line.strip().lstrip("*").strip()
        # Handle "**category:** safe" markdown format
        stripped = stripped.replace("**", "").strip()

        if stripped.startswith("category:") or stripped.startswith("type:"):
            val = stripped.split(":", 1)[1].strip()
            detected_type = _classify_text(val)
            if detected_type is not None:
                break

        if stripped.startswith("hazard:"):
            val = stripped.split(":", 1)[1].strip()
            if "no" in val:
                detected_type = "safe"
                break

    # === Strategy 1b: If category was "unsafe"/"not safe", scan full text for specific hazard ===
    if detected_type is None:
        for line in text_lower.split("\n"):
            stripped = line.strip().replace("**", "").strip()
            if stripped.startswith("category:") or stripped.startswith("type:"):
                val = stripped.split(":", 1)[1].strip()
                if "unsafe" in val or "not safe" in val:
                    # Category says "unsafe" — check stacking first (more specific), then spill
                    if "stacking" in text_lower or "tilted" in text_lower or "leaning" in text_lower or "overhanging" in text_lower or "jumbled" in text_lower:
                        detected_type = "improper_stacking"
                    elif "spill" in text_lower or "puddle" in text_lower or "liquid" in text_lower:
                        detected_type = "spill"
                    break

    # === Strategy 2: First line/word is the category (no label) ===
    # e.g. "safe\nnone\n85%\n..." or "safe · 100% · ..."
    if detected_type is None:
        first_line = text_lower.strip().split("\n")[0].strip()
        # Try splitting by pipes, dots, commas
        for sep in ["|", "·", ","]:
            if sep in first_line:
                parts = [p.strip() for p in first_line.split(sep)]
                detected_type = _classify_text(parts[0])
                if detected_type is not None:
                    break

        # Try first word/line as-is
        if detected_type is None:
            first_word = first_line.split()[0] if first_line.split() else ""
            detected_type = _classify_text(first_word)

        # Try first line as-is
        if detected_type is None:
            detected_type = _classify_text(first_line)

    # === Strategy 3: First labeled "safe"/"spill"/"stacking" anywhere ===
    if detected_type is None:
        # Check if response contains clear category mentions
        if "spill" in text_lower[:200]:
            if "no spill" not in text_lower[:200] and "no visible spill" not in text_lower[:200]:
                detected_type = "spill"
        if detected_type is None and ("improper_stacking" in text_lower[:300] or "improper stacking" in text_lower[:300]):
            detected_type = "improper_stacking"
        if detected_type is None and "stacking" in text_lower[:200]:
            if "no stacking" not in text_lower[:200]:
                detected_type = "improper_stacking"

    # === Strategy 4: Scan for "safe" as final verdict ===
    if detected_type is None:
        # Look for "classification: safe" or "verdict: safe" patterns
        import re
        m = re.search(r'(?:classification|verdict|result|assessment|conclusion)[:\s]+(\w+)', text_lower)
        if m:
            detected_type = _classify_text(m.group(1))

    # === Extract confidence ===
    for line in text_lower.split("\n"):
        stripped = line.strip().replace("**", "").strip()
        if stripped.startswith("confidence:"):
            confidence = _extract_confidence(stripped)
            break
    if confidence is None:
        confidence = _extract_confidence(text)

    # === Extract severity ===
    for line in text_lower.split("\n"):
        stripped = line.strip().replace("**", "").strip()
        if stripped.startswith("severity:"):
            severity = _extract_severity(stripped)
            break
    if severity is None:
        severity = _extract_severity(text)

    return {
        "type": detected_type,
        "severity": severity,
        "confidence": confidence,
    }


def resize_image(src, dst, max_size=512):
    """Resize image to max_size px on longest side, save as JPG."""
    img = Image.open(src)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = img.convert("RGB")
    img.save(dst, "JPEG", quality=85)


def build_test_set(n_per_class=33):
    """Build a balanced, shuffled test set of resized images."""
    os.makedirs(TEST_SET_DIR, exist_ok=True)

    manifest = []

    # Sample spill images
    spill_files = sorted([f for f in os.listdir(SPILL_DIR) if f.endswith(".png")])
    spill_sample = random.sample(spill_files, min(n_per_class, len(spill_files)))
    for f in spill_sample:
        src = os.path.join(SPILL_DIR, f)
        dst_name = f"spill_{f.replace('.png', '.jpg')}"
        dst = os.path.join(TEST_SET_DIR, dst_name)
        resize_image(src, dst)
        manifest.append({"file": dst_name, "ground_truth": "spill", "source": f"spill/{f}"})

    # Sample stacking images
    stack_files = sorted([f for f in os.listdir(STACKING_DIR) if f.endswith(".png")])
    stack_sample = random.sample(stack_files, min(n_per_class, len(stack_files)))
    for f in stack_sample:
        src = os.path.join(STACKING_DIR, f)
        dst_name = f"stacking_{f.replace('.png', '.jpg')}"
        dst = os.path.join(TEST_SET_DIR, dst_name)
        resize_image(src, dst)
        manifest.append({"file": dst_name, "ground_truth": "improper_stacking", "source": f"improper_stacking/{f}"})

    # Sample safe images
    safe_files = sorted([f for f in os.listdir(SAFE_DIR) if f.endswith(".png")])
    safe_sample = random.sample(safe_files, min(n_per_class, len(safe_files)))
    for f in safe_sample:
        src = os.path.join(SAFE_DIR, f)
        dst_name = f"safe_{f.replace('.png', '.jpg')}"
        dst = os.path.join(TEST_SET_DIR, dst_name)
        resize_image(src, dst)
        manifest.append({"file": dst_name, "ground_truth": "safe", "source": f"realistic/{f}"})

    # Shuffle
    random.shuffle(manifest)

    # Save manifest
    manifest_path = os.path.join(TEST_SET_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Built test set: {len(manifest)} images ({n_per_class} per class)")
    print(f"  Saved to: {TEST_SET_DIR}")
    return manifest


def run_benchmark(model, manifest, prompt_name, prompt_text, nothink=False):
    """Run a model on the entire test set, return results."""
    results = []
    total = len(manifest)

    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_SET_DIR, entry["file"])
        gt = entry["ground_truth"]

        try:
            text, stats = run_inference(model, img_path, prompt_text, nothink=nothink)
            parsed = parse_response(text)

            correct = parsed["type"] == gt
            result = {
                "image": entry["file"],
                "ground_truth": gt,
                "predicted": parsed["type"],
                "severity": parsed["severity"],
                "confidence": parsed["confidence"],
                "correct": correct,
                "raw_response": text,  # full response
                **stats,
            }
            results.append(result)

            status = "OK" if correct else "MISS"
            print(f"  [{i+1}/{total}] {entry['file'][:30]:30s} gt={gt:20s} pred={str(parsed['type']):20s} {status}")

        except Exception as e:
            print(f"  [{i+1}/{total}] {entry['file'][:30]:30s} ERROR: {e}")
            results.append({
                "image": entry["file"],
                "ground_truth": gt,
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

    return results


def compute_metrics(results):
    """Compute per-class and overall metrics."""
    classes = ["spill", "improper_stacking", "safe"]
    metrics = {}

    valid = [r for r in results if r.get("predicted") is not None]

    # Overall accuracy
    correct = sum(1 for r in valid if r["correct"])
    metrics["overall_accuracy"] = round(correct / len(valid) * 100, 1) if valid else 0
    metrics["total_images"] = len(results)
    metrics["valid_responses"] = len(valid)
    metrics["errors"] = len(results) - len(valid)

    # Per-class metrics
    for cls in classes:
        tp = sum(1 for r in valid if r["ground_truth"] == cls and r["predicted"] == cls)
        fp = sum(1 for r in valid if r["ground_truth"] != cls and r["predicted"] == cls)
        fn = sum(1 for r in valid if r["ground_truth"] == cls and r["predicted"] != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        n_class = sum(1 for r in valid if r["ground_truth"] == cls)
        class_correct = sum(1 for r in valid if r["ground_truth"] == cls and r["correct"])
        class_acc = class_correct / n_class * 100 if n_class > 0 else 0

        metrics[cls] = {
            "accuracy": round(class_acc, 1),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "total": n_class,
        }

    # Confusion matrix
    confusion = {}
    for gt_cls in classes:
        confusion[gt_cls] = {}
        for pred_cls in classes + [None]:
            confusion[gt_cls][str(pred_cls)] = sum(
                1 for r in valid if r["ground_truth"] == gt_cls and r["predicted"] == pred_cls
            )
    metrics["confusion_matrix"] = confusion

    # Avg inference time
    times = [r.get("total_time", 0) for r in valid if r.get("total_time")]
    metrics["avg_time_per_image"] = round(sum(times) / len(times), 1) if times else 0

    return metrics


def generate_report(all_results):
    """Generate a markdown report from all benchmark results."""
    lines = []
    lines.append("# VLM Warehouse Hazard Detection Benchmark")
    lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Test set: 99 images (33 spill, 33 improper_stacking, 33 safe)")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Prompt | NoThink | Overall Acc | Spill F1 | Stacking F1 | Safe F1 | Avg Time |")
    lines.append("|-------|--------|---------|-------------|----------|-------------|---------|----------|")

    for run in all_results:
        m = run["metrics"]
        nothink_str = "Yes" if run.get("nothink") else "No"
        lines.append(
            f"| {run['model']} | {run['prompt_name']} | {nothink_str} | "
            f"{m['overall_accuracy']}% | "
            f"{m['spill']['f1']:.3f} | "
            f"{m['improper_stacking']['f1']:.3f} | "
            f"{m['safe']['f1']:.3f} | "
            f"{m['avg_time_per_image']}s |"
        )

    # Detailed per-model sections
    for run in all_results:
        m = run["metrics"]
        lines.append(f"\n## {run['model']} — {run['prompt_name']}" + (" (nothink)" if run.get("nothink") else ""))
        lines.append("")
        lines.append(f"Overall accuracy: **{m['overall_accuracy']}%** ({m['valid_responses']}/{m['total_images']} valid)")
        lines.append(f"Average inference time: **{m['avg_time_per_image']}s** per image")
        lines.append("")

        # Per-class table
        lines.append("| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|----------|-----------|--------|----|----|----|----|")
        for cls in ["spill", "improper_stacking", "safe"]:
            c = m[cls]
            lines.append(f"| {cls} | {c['accuracy']}% | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['tp']} | {c['fp']} | {c['fn']} |")

        # Confusion matrix
        lines.append("")
        lines.append("Confusion matrix (rows=ground truth, cols=predicted):")
        lines.append("")
        lines.append("| | spill | improper_stacking | safe | None |")
        lines.append("|---|---|---|---|---|")
        cm = m["confusion_matrix"]
        for gt in ["spill", "improper_stacking", "safe"]:
            row = cm[gt]
            lines.append(f"| {gt} | {row.get('spill',0)} | {row.get('improper_stacking',0)} | {row.get('safe',0)} | {row.get('None',0)} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true", help="Only build test set")
    parser.add_argument("--model", type=str, help="Test single model")
    parser.add_argument("--nothink", action="store_true", help="Disable thinking (qwen3.5)")
    parser.add_argument("--prompt", choices=["simple", "cot", "both"], default="cot", help="Prompt strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test set")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build or load test set
    manifest_path = os.path.join(TEST_SET_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("\n=== Building test set ===")
        manifest = build_test_set(n_per_class=33)
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Loaded existing test set: {len(manifest)} images")

    if args.build_only:
        return

    # Determine which models to run
    models = [args.model] if args.model else MODELS

    prompts = {}
    if args.prompt in ("simple", "both"):
        prompts["simple"] = SIMPLE_PROMPT
    if args.prompt in ("cot", "both"):
        prompts["cot"] = COT_PROMPT

    all_results = []

    for model in models:
        for prompt_name, prompt_text in prompts.items():
            # Run with thinking enabled first
            nothink_modes = [False]
            if args.nothink or "qwen3.5" in model:
                nothink_modes = [True, False] if not args.nothink else [True]

            for nothink in nothink_modes:
                nothink_label = "_nothink" if nothink else ""
                run_id = f"{model.replace(':', '_').replace('.', '_')}_{prompt_name}{nothink_label}"

                result_file = os.path.join(RESULTS_DIR, f"{run_id}.json")
                if os.path.exists(result_file):
                    print(f"\n=== Skipping {model} ({prompt_name}{nothink_label}) — already done ===")
                    with open(result_file) as f:
                        run_data = json.load(f)
                    all_results.append(run_data)
                    continue

                print(f"\n{'='*60}")
                print(f"  MODEL: {model}")
                print(f"  PROMPT: {prompt_name} | NOTHINK: {nothink}")
                print(f"{'='*60}")

                # Load model (Ollama auto-swaps, no need to unload)
                print(f"  Loading {model}...")
                try:
                    api("/api/chat", {"model": model, "messages": [], "keep_alive": -1}, timeout=300)
                    print(f"  {model} ready")
                except Exception as e:
                    print(f"  ERROR loading {model}: {e}")
                    continue

                # Run benchmark
                results = run_benchmark(model, manifest, prompt_name, prompt_text, nothink=nothink)
                metrics = compute_metrics(results)

                run_data = {
                    "model": model,
                    "prompt_name": prompt_name,
                    "nothink": nothink,
                    "metrics": metrics,
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(run_data)

                # Save individual results (JSON)
                with open(result_file, "w") as f:
                    json.dump(run_data, f, indent=2)

                # Save detailed log file (human-readable, full responses)
                log_file = os.path.join(RESULTS_DIR, f"{run_id}_log.md")
                with open(log_file, "w") as f:
                    f.write(f"# {model} — {prompt_name}" + (" (nothink)" if nothink else "") + "\n\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(f"Test set: {len(manifest)} images\n\n")

                    # Summary metrics
                    f.write("## Results Summary\n\n")
                    f.write(f"**Overall accuracy: {metrics['overall_accuracy']}%** ({metrics['valid_responses']}/{metrics['total_images']} valid)\n")
                    f.write(f"**Average time: {metrics['avg_time_per_image']}s per image**\n\n")
                    f.write("| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |\n")
                    f.write("|-------|----------|-----------|--------|----|----|----|----|" + "\n")
                    for cls in ["spill", "improper_stacking", "safe"]:
                        c = metrics[cls]
                        f.write(f"| {cls} | {c['accuracy']}% | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['tp']} | {c['fp']} | {c['fn']} |\n")

                    # Confusion matrix
                    f.write("\n## Confusion Matrix\n\n")
                    f.write("| Ground Truth \\ Predicted | spill | improper_stacking | safe | None |\n")
                    f.write("|---|---|---|---|---|\n")
                    cm = metrics["confusion_matrix"]
                    for gt in ["spill", "improper_stacking", "safe"]:
                        row = cm[gt]
                        f.write(f"| {gt} | {row.get('spill',0)} | {row.get('improper_stacking',0)} | {row.get('safe',0)} | {row.get('None',0)} |\n")

                    # Full per-image log with complete model responses
                    f.write("\n---\n\n## Detailed Per-Image Results\n\n")
                    for r in results:
                        status = "CORRECT" if r.get("correct") else "WRONG"
                        f.write(f"### {r['image']}\n\n")
                        f.write(f"- **Ground truth:** {r['ground_truth']}\n")
                        f.write(f"- **Predicted:** {r.get('predicted', 'N/A')}\n")
                        f.write(f"- **Result:** {status}\n")
                        f.write(f"- **Severity:** {r.get('severity', 'N/A')}\n")
                        f.write(f"- **Confidence:** {r.get('confidence', 'N/A')}%\n")
                        f.write(f"- **Time:** {r.get('total_time', 'N/A')}s | TTFT: {r.get('ttft', 'N/A')}s | {r.get('tokens_generated', 'N/A')} tokens @ {r.get('gen_tps', 'N/A')} tok/s\n\n")
                        if r.get("raw_response"):
                            f.write(f"**Model response:**\n```\n{r['raw_response']}\n```\n\n")
                        elif r.get("error"):
                            f.write(f"**Error:** {r['error']}\n\n")
                        f.write("---\n\n")

                print(f"\n  Saved: {result_file}")
                print(f"  Log:   {log_file}")

                # Print quick summary
                print(f"\n  --- {model} ({prompt_name}{nothink_label}) ---")
                print(f"  Overall: {metrics['overall_accuracy']}%")
                for cls in ["spill", "improper_stacking", "safe"]:
                    c = metrics[cls]
                    print(f"  {cls:20s}: acc={c['accuracy']}% P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")

    # Generate report
    if all_results:
        report = generate_report(all_results)
        report_path = os.path.join(RESULTS_DIR, "benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n{'='*60}")
        print(f"  REPORT: {report_path}")
        print(f"{'='*60}")
        print(report)


if __name__ == "__main__":
    main()
