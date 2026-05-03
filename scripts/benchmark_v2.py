#!/usr/bin/env python3
"""
HAWKEYE VLM Benchmark v2 — 5 categories, 5 prompts, 10 models.

Logging:
  outputs/benchmark_v2/
  ├── BENCHMARK_LOG.md              # ONE master log: every run, every image, readable
  ├── {model}/{prompt}/
  │   ├── results.json              # metrics
  │   └── per_image_log.jsonl       # machine-readable per-image
  ├── leaderboard.json
  └── test_manifest.json

Variables tested:
  - Model (10 models)
  - Prompt strategy (5 prompts)
  - Think mode (on/off, qwen3.5 only)
  - Temperature (default 0.0)
"""

import argparse
import base64
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT = Path("/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection")
sys.path.insert(0, str(PROJECT))
from vision_api import api

DATASET_DIR = PROJECT / "dataset"
IMAGES_DIR = DATASET_DIR / "images_small"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"
RESULTS_BASE = PROJECT / "outputs" / "benchmark_v2"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
MASTER_LOG = RESULTS_BASE / "BENCHMARK_LOG.md"

CATEGORIES = ["spill", "forklift_violation", "improper_stacking", "obstacle", "safe"]

MODELS = [
    "qwen3.5:9b",
    "qwen3-vl:8b",
    "qwen3.5:4b",
    "gemma3:4b",
    "qwen3.5:2b",
    "qwen3-vl:2b",
    "granite3.2-vision:2b",
    "llava-phi3:latest",
    "qwen3.5:0.8b",
    "moondream:latest",
]

PROMPTS = {
    "direct": """Classify this warehouse image into exactly one category:
- spill: liquid/fluid on the warehouse floor
- forklift_violation: unsafe forklift operation (raised forks, no seatbelt, pedestrian too close, improper load)
- improper_stacking: boxes on shelves are tilted, overhanging, crushed, torn wrap, unstable
- obstacle: objects blocking the aisle floor (fallen boxes, debris, abandoned equipment, broken pallets)
- safe: clean warehouse, no hazards present

Respond in this format:
category: <one of the above>
description: <one sentence describing what you see>""",

    "osha_inspector": """You are an OSHA warehouse safety inspector analyzing a drone camera image.

Examine carefully:
- FLOOR: liquid puddles, wet patches, oil slicks, chemical spills
- FORKLIFT: raised forks while traveling, no seatbelt, pedestrians too close, overloaded/unstable loads
- SHELVES: boxes tilted, overhanging edges, crushed, torn shrink wrap, mixed sizes creating instability
- AISLES: fallen boxes, broken pallets, abandoned equipment (pallet jacks, hand trucks), debris on floor
- If none of the above: scene is safe

Respond in this format:
category: spill / forklift_violation / improper_stacking / obstacle / safe
severity: critical / high / medium / low / none
confidence: 0-100%
description: <one sentence>""",

    "chain_of_thought": """Look at this warehouse image from a drone camera. Answer each question:

Q1: Is there any liquid, wet patch, or puddle on the floor? YES/NO
Q2: Is a forklift doing something unsafe? YES/NO
Q3: Are boxes on shelves tilted, overhanging, crushed, or unstable? YES/NO
Q4: Are there objects on the aisle FLOOR blocking the path? YES/NO
Q5: Is everything clean, organized, and safe? YES/NO

Based on your answers:
category: spill / forklift_violation / improper_stacking / obstacle / safe
description: <one sentence>""",

    "describe_then_classify": """Describe what you see in this warehouse drone image in 2-3 sentences. Focus on floor condition, forklift activity, shelf organization, and aisle clearance.

Then classify:
category: spill / forklift_violation / improper_stacking / obstacle / safe""",

    "json_structured": """Analyze this warehouse safety image. Respond ONLY with valid JSON:
{"category": "spill|forklift_violation|improper_stacking|obstacle|safe", "confidence": 0-100, "description": "one sentence"}""",
}

# ============================================================
# PARSER
# ============================================================

CATEGORY_KEYWORDS = {
    "spill": ["spill", "puddle", "liquid", "wet", "oil", "fluid", "leak"],
    "forklift_violation": ["forklift", "fork lift", "forklift_violation", "forks raised",
                           "no seatbelt", "pedestrian", "no_seatbelt"],
    "improper_stacking": ["stacking", "improper_stacking", "tilted", "overhanging",
                          "leaning", "unstable", "crushed box", "torn wrap"],
    "obstacle": ["obstacle", "debris", "fallen box", "broken pallet", "aisle block",
                 "obstruction", "pallet jack", "scattered"],
    "safe": ["safe", "clean", "no hazard", "no violation", "all clear", "normal"],
}

def parse_response(text):
    text_lower = text.lower().strip()
    for line in text_lower.split("\n"):
        line = line.strip()
        if line.startswith("category:") or line.startswith('"category"'):
            value = line.split(":", 1)[1].strip().strip('"').strip("'").strip(",").strip()
            for cat in CATEGORIES:
                if cat in value: return cat
            if "forklift" in value: return "forklift_violation"
            if "stacking" in value or "improper" in value: return "improper_stacking"
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            data = json.loads(text[s:e])
            cat = data.get("category", "").lower()
            for c in CATEGORIES:
                if c in cat: return c
            if "forklift" in cat: return "forklift_violation"
            if "stacking" in cat: return "improper_stacking"
    except: pass
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower: scores[cat] += 1
    if "unsafe" in text_lower: scores["safe"] = max(0, scores["safe"] - 2)
    mx = max(scores.values())
    if mx > 0:
        for cat in ["spill", "forklift_violation", "improper_stacking", "obstacle", "safe"]:
            if scores[cat] == mx: return cat
    return "unable_to_determine"

# ============================================================
# OLLAMA
# ============================================================

def load_model(model):
    print(f"  Loading {model}...")
    start = time.time()
    api("/api/chat", {"model": model, "messages": [], "keep_alive": -1})
    print(f"  {model} loaded in {time.time()-start:.1f}s")

def stop_all():
    data = api("/api/ps")
    for m in data.get("models", []):
        name = m.get("name", "")
        api("/api/chat", {"model": name, "messages": [], "keep_alive": 0})
        print(f"  Unloaded {name}")

def get_loaded_model():
    data = api("/api/ps")
    models = data.get("models", [])
    return models[0].get("name", "") if models else None

def run_inference(model, image_path, prompt, temperature=0.0, nothink=True):
    MAX_TIME = 120 if nothink else 300  # 2 min nothink, 5 min think
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": True, "keep_alive": -1,
        "options": {"temperature": temperature, "num_predict": 16384 if not nothink else 2048},
        "think": False if nothink else True,
    }
    start = time.time()
    try:
        resp = api("/api/chat", payload, stream=True, timeout=MAX_TIME)
        thinking_tokens = []
        content_tokens = []
        final = {}
        for line in resp:
            if time.time() - start > MAX_TIME:
                resp.close()
                return "TIMEOUT: model stuck", time.time() - start, {"thinking": "".join(thinking_tokens) if thinking_tokens else None}
            if not line.strip(): continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            thinking = msg.get("thinking", "")
            content = msg.get("content", "")
            if thinking:
                thinking_tokens.append(thinking)
            if content:
                content_tokens.append(content)
            if chunk.get("done"):
                final = chunk
                break
        resp.close()

        text = "".join(content_tokens)
        think_text = "".join(thinking_tokens)
        elapsed = time.time() - start

        return text, elapsed, {
            "eval_count": final.get("eval_count", 0),
            "prompt_eval_count": final.get("prompt_eval_count", 0),
            "thinking": think_text if think_text else None,
        }
    except Exception as e:
        return f"ERROR: {e}", time.time() - start, {}

# ============================================================
# TEST SET
# ============================================================

def build_test_set(n_per_category=30, seed=42):
    random.seed(seed)
    metadata = [json.loads(l) for l in open(METADATA_FILE)]
    by_cat = {}
    for m in metadata: by_cat.setdefault(m["category"], []).append(m)
    test_set = []
    for cat in CATEGORIES:
        entries = by_cat.get(cat, [])
        chosen = random.sample(entries, min(n_per_category, len(entries)))
        for c in chosen:
            jpg_name = Path(c["image"]).with_suffix(".jpg").name
            test_set.append({
                "image": c["image"], "image_path": str(IMAGES_DIR / jpg_name),
                "category": c["category"], "accept_also": c.get("accept_also", []),
                "description": c.get("description", ""),
                "subtype": c.get("subtype", ""), "severity": c.get("severity", ""),
            })
    random.shuffle(test_set)
    return test_set

# ============================================================
# MASTER LOG WRITER
# ============================================================

def write_master_header(test_set, models, prompts, nothink, temperature):
    with open(MASTER_LOG, "w") as f:
        f.write("# HAWKEYE VLM Benchmark v2 — Master Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Test set:** {len(test_set)} images ({len(test_set)//5}/category)\n")
        f.write(f"**Models:** {len(models)} | **Prompts:** {len(prompts)} | **Total runs:** {len(models)*len(prompts)}\n")
        f.write(f"**Think:** {'OFF' if nothink else 'ON'} | **Temperature:** {temperature}\n\n")
        f.write("---\n\n")

def append_run_to_master(model, prompt_name, prompt_text, results_data, entries):
    with open(MASTER_LOG, "a") as f:
        acc = results_data["accuracy"] * 100
        f.write(f"## {model} — `{prompt_name}`\n\n")
        f.write(f"**Accuracy:** {results_data['correct']}/{results_data['total']} = **{acc:.1f}%**\n")
        f.write(f"**Avg time:** {results_data['avg_time_s']:.1f}s/image | **Total:** {results_data['total_time_min']:.1f} min\n\n")

        # Prompt used
        f.write("<details><summary>Prompt used</summary>\n\n```\n")
        f.write(prompt_text)
        f.write("\n```\n</details>\n\n")

        # Per-category metrics
        f.write(f"| Category | P | R | F1 | TP | FP | FN |\n")
        f.write(f"|----------|---|---|----|----|----|----|--|\n")
        for cat, m in results_data["category_metrics"].items():
            f.write(f"| {cat} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |\n")
        f.write("\n")

        # Per-image log (collapsible)
        f.write("<details><summary>Per-image results (click to expand)</summary>\n\n")
        f.write("| # | Image | GT | Predicted | OK? | Time | Model Response (truncated) |\n")
        f.write("|---|-------|----|-----------|-----|------|----------------------------|\n")
        for i, e in enumerate(entries, 1):
            ok = "Y" if e.get("correct") else "N"
            resp_short = e.get("model_response", "")[:100].replace("\n", " ").replace("|", "/")
            f.write(f"| {i} | {e['image']} | {e['ground_truth']} | {e['predicted']} | {ok} | {e.get('elapsed_s',0):.1f}s | {resp_short} |\n")
        f.write("\n</details>\n\n")

        # Wrong predictions highlighted
        wrong = [e for e in entries if not e.get("correct", True)]
        if wrong:
            f.write(f"**Wrong predictions ({len(wrong)}):**\n\n")
            for e in wrong[:10]:
                resp_short = e.get("model_response", "")[:200].replace("\n", " ")
                f.write(f"- `{e['image']}`: expected **{e['ground_truth']}**, got **{e['predicted']}**\n")
                f.write(f"  > {resp_short}\n\n")
            if len(wrong) > 10:
                f.write(f"  _(and {len(wrong)-10} more)_\n\n")

        f.write("---\n\n")

# ============================================================
# RUN ONE MODEL x ONE PROMPT
# ============================================================

def run_benchmark(model, prompt_name, test_set, temperature, nothink, resume):
    safe_model = model.replace(":", "_").replace(".", "_")
    mode_suffix = "nothink" if nothink else "think"
    prompt_dir = RESULTS_BASE / safe_model / f"{prompt_name}__{mode_suffix}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    results_file = prompt_dir / "results.json"
    log_file = prompt_dir / "per_image_log.jsonl"

    if resume and results_file.exists():
        print(f"  SKIP (done): {safe_model}/{prompt_name}")
        with open(results_file) as f:
            return json.load(f), []

    completed = {}
    if resume and log_file.exists():
        with open(log_file) as f:
            for line in f:
                e = json.loads(line)
                completed[e["image"]] = e

    print(f"\n  {model} / {prompt_name} | {len(test_set)} imgs ({len(completed)} cached)")

    prompt_text = PROMPTS[prompt_name]
    correct, total = 0, 0
    by_cat = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in CATEGORIES}
    all_entries = []
    log_f = open(log_file, "a" if completed else "w")

    for i, item in enumerate(test_set):
        if item["image"] in completed:
            e = completed[item["image"]]
            all_entries.append(e)
            gt, pred = e["ground_truth"], e["predicted"]
            accept_also = item.get("accept_also", [])
            if pred == gt or pred in accept_also: correct += 1; by_cat[gt]["tp"] += 1
            else:
                if pred in by_cat: by_cat[pred]["fp"] += 1
                by_cat[gt]["fn"] += 1
            total += 1
            continue

        response, elapsed, stats = run_inference(model, item["image_path"], prompt_text, temperature, nothink)
        predicted = parse_response(response)
        gt = item["category"]
        accept_also = item.get("accept_also", [])
        is_correct = predicted == gt or predicted in accept_also
        if is_correct: correct += 1; by_cat[gt]["tp"] += 1
        else:
            if predicted in by_cat: by_cat[predicted]["fp"] += 1
            by_cat[gt]["fn"] += 1
        total += 1

        entry = {
            "image": item["image"],
            "ground_truth": gt, "ground_truth_description": item["description"],
            "ground_truth_subtype": item["subtype"], "ground_truth_severity": item["severity"],
            "predicted": predicted, "correct": is_correct,
            "model_response": response,
            "model_thinking": stats.get("thinking"),
            "elapsed_s": round(elapsed, 2),
            "eval_tokens": stats.get("eval_count", 0),
            "prompt_tokens": stats.get("prompt_eval_count", 0),
        }
        all_entries.append(entry)
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

        acc = correct / total * 100
        tag = "OK" if is_correct else "XX"
        print(f"  [{i+1}/{len(test_set)}] {gt:>22} -> {predicted:<22} [{tag}] {elapsed:.1f}s  acc:{acc:.1f}%")

    log_f.close()

    accuracy = correct / total if total > 0 else 0
    cat_metrics = {}
    for cat in CATEGORIES:
        tp, fp, fn = by_cat[cat]["tp"], by_cat[cat]["fp"], by_cat[cat]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        cat_metrics[cat] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}

    times = [e["elapsed_s"] for e in all_entries if "elapsed_s" in e]
    avg_time = sum(times) / len(times) if times else 0

    results = {
        "model": model, "prompt": prompt_name,
        "mode": "nothink" if nothink else "think", "temperature": temperature,
        "n_images": total, "accuracy": round(accuracy, 4),
        "correct": correct, "total": total,
        "avg_time_s": round(avg_time, 2), "total_time_min": round(sum(times) / 60, 1),
        "category_metrics": cat_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  RESULT: {correct}/{total} = {accuracy*100:.1f}% | {avg_time:.1f}s/img")

    # Append to master log
    append_run_to_master(model, prompt_name, PROMPTS[prompt_name], results, all_entries)

    return results, all_entries

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--n-per-cat", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_set = build_test_set(n_per_category=args.n_per_cat, seed=args.seed)
    models = [args.model] if args.model else MODELS
    prompts = [args.prompt] if args.prompt else list(PROMPTS.keys())
    nothink = not args.think
    total_runs = len(models) * len(prompts)

    print(f"Test set: {len(test_set)} images ({args.n_per_cat}/category)")
    print(f"Plan: {len(models)} models x {len(prompts)} prompts = {total_runs} runs")
    print(f"Think: {'OFF' if nothink else 'ON'} | Temp: {args.temperature}")

    with open(RESULTS_BASE / "test_manifest.json", "w") as f:
        json.dump(test_set, f, indent=2)

    write_master_header(test_set, models, prompts, nothink, args.temperature)

    all_results = []
    run_count = 0
    current_model = get_loaded_model()
    print(f"Currently loaded: {current_model or 'none'}\n")

    for model in models:
        # Check if ALL prompts for this model are already done (resume)
        if args.resume:
            safe_model = model.replace(":", "_").replace(".", "_")
            mode_suffix = "nothink" if nothink else "think"
            all_done = all(
                (RESULTS_BASE / safe_model / f"{p}__{mode_suffix}" / "results.json").exists()
                for p in prompts
            )
            if all_done:
                print(f"\n  SKIP MODEL: {model} — all {len(prompts)} prompts already done ({mode_suffix})")
                # Load cached results for leaderboard
                for p in prompts:
                    rf = RESULTS_BASE / safe_model / f"{p}__{mode_suffix}" / "results.json"
                    with open(rf) as f:
                        all_results.append(json.load(f))
                    run_count += 1
                continue

        print(f"\n{'#'*60}")
        print(f"# {model} — {len(prompts)} prompts x {len(test_set)} images")
        print(f"{'#'*60}")

        if current_model != model:
            stop_all(); time.sleep(1)
            load_model(model)
            current_model = model
        else:
            print(f"  {model} already loaded")

        model_results = []
        for prompt_name in prompts:
            run_count += 1
            print(f"\n>>> Run {run_count}/{total_runs}")
            result, _ = run_benchmark(model, prompt_name, test_set, args.temperature, nothink, args.resume)
            if result:
                all_results.append(result)
                model_results.append(result)

        if model_results:
            safe_model = model.replace(":", "_").replace(".", "_")
            best = max(model_results, key=lambda x: x["accuracy"])
            with open(RESULTS_BASE / safe_model / "model_summary.json", "w") as f:
                json.dump({"model": model, "best_prompt": best["prompt"], "best_accuracy": best["accuracy"],
                           "all_prompts": {r["prompt"]: r["accuracy"] for r in model_results}}, f, indent=2)

    # Leaderboard
    if all_results:
        sorted_r = sorted(all_results, key=lambda x: -x["accuracy"])
        print(f"\n{'='*70}")
        print("LEADERBOARD")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Model':<25} {'Prompt':<22} {'Acc':<8} {'Time':<8}")
        print("-" * 70)
        for i, r in enumerate(sorted_r, 1):
            print(f"{i:<4} {r['model']:<25} {r['prompt']:<22} {r['accuracy']*100:.1f}%   {r['avg_time_s']:.1f}s")

        with open(RESULTS_BASE / "leaderboard.json", "w") as f:
            json.dump(sorted_r, f, indent=2)

        # Append leaderboard to master log
        with open(MASTER_LOG, "a") as f:
            f.write("# Final Leaderboard\n\n")
            f.write("| # | Model | Prompt | Accuracy | Avg Time |\n")
            f.write("|---|-------|--------|----------|----------|\n")
            for i, r in enumerate(sorted_r, 1):
                f.write(f"| {i} | {r['model']} | {r['prompt']} | {r['accuracy']*100:.1f}% | {r['avg_time_s']:.1f}s |\n")

    print("\nDone. (model left loaded for next run)")

if __name__ == "__main__":
    main()
