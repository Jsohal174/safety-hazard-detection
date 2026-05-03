#!/usr/bin/env python3
"""
OpenRouter VLM benchmark on the hard-set pedestrian-proximity images.

Test set (12 images):
  forklift_violation cases with subtype=pedestrian_proximity, where the
  trained 2B classified correctly. These are the "judgement, not perception"
  cases — a worker physically close to a moving/active forklift.

Prompt: same `direct` prompt used in benchmark_v2.py.
Parser: same parse_response from benchmark_v2.py.

Outputs:
  outputs/benchmark_openrouter/
    LEADERBOARD.md
    {model_safe}/per_image_log.jsonl
    {model_safe}/results.json

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python3 scripts/benchmark_openrouter.py            # default model list
  python3 scripts/benchmark_openrouter.py --model openai/gpt-4o
  python3 scripts/benchmark_openrouter.py --resume   # skip already-run models
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

PROJECT = Path("/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection")
sys.path.insert(0, str(PROJECT / "scripts"))
from benchmark_v2 import PROMPTS, parse_response, CATEGORIES  # noqa

DATASET_DIR = PROJECT / "dataset"
IMAGES_DIR = DATASET_DIR / "images_small"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"
RESULTS_BASE = PROJECT / "outputs" / "benchmark_openrouter"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 12 pedestrian-proximity images where the trained 2B got it right.
# Cross-referenced from outputs/hawkeye_test_results_fixed.json + dataset/metadata.jsonl
HARD_SET = [
    "forklift_0007.png",
    "forklift_0226.png",
    "forklift_0216.png",
    "forklift_0120.png",
    "forklift_0109.png",
    "forklift_0207.png",
    "forklift_0135.png",
    "forklift_0077.png",
    "forklift_0004.png",
    "forklift_0172.png",
    "forklift_0051.png",
    "forklift_0106.png",
]

# Vision-capable OpenRouter models — curated May 2026. Edit freely.
# Claude excluded: tested separately via Claude Code on the user's subscription.
DEFAULT_MODELS = [
    # Frontier proprietary
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "x-ai/grok-4.20",
    # Big open-source (current 2026)
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.6-plus",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3.5-9b",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-4.6v",
    "mistralai/pixtral-large-2411",
]


def load_image_b64(image_name: str) -> str:
    jpg = IMAGES_DIR / Path(image_name).with_suffix(".jpg").name
    return base64.b64encode(jpg.read_bytes()).decode("utf-8")


def build_test_set():
    md = {m["image"]: m for m in (json.loads(l) for l in open(METADATA_FILE))}
    out = []
    for img in HARD_SET:
        m = md[img]
        out.append({
            "image": img,
            "category": m["category"],
            "subtype": m.get("subtype", ""),
            "severity": m.get("severity", ""),
            "description": m.get("description", ""),
            "accept_also": m.get("accept_also", []),
        })
    return out


def call_openrouter(model_id: str, prompt: str, image_b64: str, api_key: str,
                    temperature: float = 0.0, timeout: int = 120):
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": 600,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Jsohal174/safety-hazard-detection",
        "X-Title": "HAWKEYE warehouse safety benchmark",
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        choice = body.get("choices", [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        finish = choice.get("finish_reason", "")
        usage = body.get("usage", {}) or {}
        return {
            "ok": True,
            "text": text,
            "elapsed_s": round(elapsed, 2),
            "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "raw_id": body.get("id", ""),
        }
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "text": "",
            "elapsed_s": round(time.time() - start, 2),
            "error_kind": "http",
            "error_status": e.code,
            "error_message": e.read().decode("utf-8", errors="replace")[:400],
        }
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "text": "",
            "elapsed_s": round(time.time() - start, 2),
            "error_kind": "url",
            "error_message": str(e)[:400],
        }
    except Exception as e:
        return {
            "ok": False,
            "text": "",
            "elapsed_s": round(time.time() - start, 2),
            "error_kind": "exception",
            "error_message": f"{type(e).__name__}: {e}"[:400],
        }


REFUSAL_HINTS = (
    "i can't", "i cannot", "i'm unable", "i am unable",
    "i won't", "i will not", "as an ai",
    "i don't have the ability", "i'm not able",
)


def detect_refusal(text: str) -> bool:
    if not text:
        return False
    head = text.lower()[:300]
    return any(h in head for h in REFUSAL_HINTS) and "category" not in head


def safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_").replace(".", "_")


def run_model(model_id: str, test_set, api_key: str, resume: bool = False):
    out_dir = RESULTS_BASE / safe_model_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "per_image_log.jsonl"
    results_file = out_dir / "results.json"

    if resume and results_file.exists():
        print(f"  SKIP (cached): {model_id}")
        return json.loads(results_file.read_text())

    completed = {}
    if resume and log_file.exists():
        with open(log_file) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    completed[e["image"]] = e
                except Exception:
                    pass

    print(f"\n>>> {model_id} | {len(test_set)} images ({len(completed)} cached)")
    log_f = open(log_file, "a" if completed else "w")

    correct = 0
    total = 0
    refused = 0
    errored = 0
    entries = []
    prompt_text = PROMPTS["direct"]

    for i, item in enumerate(test_set, 1):
        if item["image"] in completed:
            e = completed[item["image"]]
            entries.append(e)
            if e.get("correct"): correct += 1
            if e.get("status") == "refused": refused += 1
            if e.get("status") == "error": errored += 1
            total += 1
            continue

        try:
            img_b64 = load_image_b64(item["image"])
        except Exception as e:
            print(f"  [{i}/{len(test_set)}] {item['image']}: image read failed: {e}")
            continue

        result = call_openrouter(model_id, prompt_text, img_b64, api_key)
        gt = item["category"]
        accept_also = item.get("accept_also", [])

        if not result["ok"]:
            entry = {
                "image": item["image"],
                "ground_truth": gt,
                "predicted": "error",
                "correct": False,
                "status": "error",
                "model_response": "",
                "error_kind": result.get("error_kind"),
                "error_status": result.get("error_status"),
                "error_message": result.get("error_message", ""),
                "elapsed_s": result.get("elapsed_s", 0),
            }
            errored += 1
            tag = f"ERR ({result.get('error_status') or result.get('error_kind')})"
        else:
            text = result["text"]
            if detect_refusal(text):
                pred = "refused"
                status = "refused"
                refused += 1
            else:
                pred = parse_response(text)
                status = "ok"
            ok = pred == gt or pred in accept_also
            if ok:
                correct += 1
            entry = {
                "image": item["image"],
                "ground_truth": gt,
                "ground_truth_subtype": item.get("subtype", ""),
                "predicted": pred,
                "correct": ok,
                "status": status,
                "model_response": text,
                "finish_reason": result.get("finish_reason", ""),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "elapsed_s": result["elapsed_s"],
            }
            tag = "OK" if ok else ("REFUSE" if status == "refused" else "WRONG")

        total += 1
        entries.append(entry)
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

        acc = correct / total * 100 if total else 0
        print(f"  [{i:>2}/{len(test_set)}] {item['image']:<22} gt={gt:<20} "
              f"pred={entry['predicted']:<20} [{tag}] {entry['elapsed_s']:.1f}s "
              f"acc={acc:.1f}%")

    log_f.close()

    times = [e.get("elapsed_s", 0) for e in entries]
    pt = [e.get("prompt_tokens", 0) for e in entries if e.get("status") == "ok"]
    ct = [e.get("completion_tokens", 0) for e in entries if e.get("status") == "ok"]

    summary = {
        "model": model_id,
        "n_images": total,
        "correct": correct,
        "refused": refused,
        "errored": errored,
        "accuracy_all": round(correct / total, 4) if total else 0.0,
        "accuracy_completed": round(correct / max(1, total - errored - refused), 4),
        "avg_time_s": round(sum(times) / len(times), 2) if times else 0,
        "avg_prompt_tokens": round(sum(pt) / len(pt)) if pt else 0,
        "avg_completion_tokens": round(sum(ct) / len(ct)) if ct else 0,
        "timestamp": datetime.now().isoformat(),
    }
    results_file.write_text(json.dumps(summary, indent=2))
    print(f"  RESULT: {correct}/{total} = {summary['accuracy_all']*100:.1f}%   "
          f"(refused={refused}, errored={errored})")
    return summary


def write_leaderboard(all_summaries):
    sorted_s = sorted(all_summaries, key=lambda s: -s["accuracy_all"])
    md = ["# OpenRouter benchmark — pedestrian-proximity hard set",
          "",
          f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
          "",
          f"Test set: 12 forklift_violation images (subtype = pedestrian_proximity), "
          f"all where the trained 2B got it right.",
          "",
          "Prompt: same `direct` prompt as benchmark_v2.py.",
          "",
          "| # | Model | Acc (all) | Correct | Refused | Errored | Avg time | Avg tokens (in/out) |",
          "|---|-------|-----------|---------|---------|---------|----------|---------------------|"]
    for i, s in enumerate(sorted_s, 1):
        md.append(
            f"| {i} | `{s['model']}` | "
            f"{s['accuracy_all']*100:.1f}% | "
            f"{s['correct']}/{s['n_images']} | {s['refused']} | {s['errored']} | "
            f"{s['avg_time_s']:.1f}s | {s['avg_prompt_tokens']}/{s['avg_completion_tokens']} |"
        )
    md.append("")
    md.append("**For comparison:** Trained 2B got 12/12 = 100% on this set.")
    md.append("")
    (RESULTS_BASE / "LEADERBOARD.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS_BASE / 'LEADERBOARD.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append",
                    help="OpenRouter model id (repeatable). Default: built-in list.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip models with existing results.json; resume mid-run from per_image_log.jsonl")
    ap.add_argument("--list-models", action="store_true",
                    help="Print default model list and exit")
    ap.add_argument("--list-images", action="store_true",
                    help="Print the 12 hard-set images and exit")
    args = ap.parse_args()

    if args.list_models:
        for m in DEFAULT_MODELS: print(m)
        return
    if args.list_images:
        for img in HARD_SET: print(img)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/settings/keys")
        sys.exit(1)

    models = args.model if args.model else DEFAULT_MODELS
    test_set = build_test_set()

    print(f"Test set: {len(test_set)} images (pedestrian_proximity hard set)")
    print(f"Models:   {len(models)}")
    print(f"Total calls: {len(models) * len(test_set)}")
    print()

    summaries = []
    for m in models:
        try:
            s = run_model(m, test_set, api_key, resume=args.resume)
            summaries.append(s)
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run with --resume to continue.")
            break
        except Exception as e:
            print(f"  FATAL on {m}: {e}")

    if summaries:
        write_leaderboard(summaries)
        sorted_s = sorted(summaries, key=lambda s: -s["accuracy_all"])
        print(f"\n{'='*70}")
        print("LEADERBOARD")
        print(f"{'='*70}")
        print(f"{'#':<3} {'Model':<50} {'Acc':<8} {'Refused':<8} {'Err':<5}")
        for i, s in enumerate(sorted_s, 1):
            print(f"{i:<3} {s['model']:<50} {s['accuracy_all']*100:>5.1f}%   "
                  f"{s['refused']:<8} {s['errored']:<5}")


if __name__ == "__main__":
    main()
