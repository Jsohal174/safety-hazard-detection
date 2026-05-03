#!/usr/bin/env python3
"""
Quick prompt A/B testing — runs on 15-image subset, returns results in ~2 min.
Usage: python3 scripts/quick_prompt_test.py --prompt-id p1
"""
import base64, json, os, sys, time, urllib.request, argparse
from pathlib import Path

SERVER = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PROJECT = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection"
TEST_DIR = f"{PROJECT}/outputs/benchmark/test_images"
QUICK_MANIFEST = f"{TEST_DIR}/quick_manifest.json"

sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from vlm_benchmark import parse_response

# ============================================================
# PROMPT VARIANTS — add new ones here
# ============================================================
PROMPTS = {

"p1_simple": """You are an OSHA warehouse safety inspector analyzing a drone camera image. Classify the safety condition of this warehouse scene.

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
location: brief description of where the issue is""",

"p2_cot": """You are an OSHA warehouse safety inspector conducting a detailed visual inspection from a drone camera.

FLOOR INSPECTION: Carefully examine the entire warehouse floor. Look for any liquid — water puddles, oil slicks, chemical spills, wet or reflective patches on the concrete. Even small puddles count.

SHELF INSPECTION: Examine every shelf level on BOTH sides of the aisle. For each section of shelving, check:
- Are boxes sitting flat and upright, or are any tilted, rotated, or at an angle?
- Are any boxes or items hanging past the edge of the shelf into the aisle?
- Are boxes neatly arranged in rows, or jumbled and disorganized?
- Are any boxes visibly crushed, dented, or deformed from bearing too much weight?
- Is shrink wrap torn, loose, or missing on palletized loads?
- Are loads shifted noticeably to one side?
- Could anything realistically fall if bumped?

Describe in 2-3 sentences what you observe on the floor and shelves. Then classify:

category: spill / improper_stacking / minor_hazard / safe / unable_to_determine
severity: critical / high / medium / low / none
confidence: 0-100%
location: where the issue is""",

"p3_binary_then_classify": """Look at this warehouse image from a drone camera.

Answer these TWO questions:

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES or NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES or NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%""",

"p4_compare": """You are inspecting this warehouse image from a drone camera. Compare the LEFT shelves vs RIGHT shelves carefully.

A well-organized warehouse has:
- All boxes sitting flat and square on shelves
- Uniform rows with consistent spacing
- Intact packaging, no torn cardboard or wrap
- Nothing hanging over shelf edges
- Clean, dry floor

Look for ANY deviation from this standard on EITHER side:
- Boxes at angles, rotated, or not sitting flat
- Uneven or jumbled arrangement
- Boxes extending past shelf edges
- Crushed, torn, or damaged packaging
- Wet spots or puddles on floor

Even small deviations count. What do you find?

category: spill / improper_stacking / safe
severity: critical / high / medium / low / none
confidence: 0-100%
details: what specifically is wrong and where""",

"p5_strict_inspector": """WAREHOUSE SAFETY AUDIT — ZERO TOLERANCE INSPECTION

You are conducting a strict safety audit of this warehouse from drone footage. Your job is to find ANY safety violation, no matter how small. Management expects you to flag issues, not pass everything.

INSPECT:
1. FLOOR — Any liquid, moisture, wet spots, puddles, stains that could be slippery
2. LEFT SHELVES — Every level. Are boxes perfectly flat, aligned, and within shelf boundaries?
3. RIGHT SHELVES — Every level. Same checks.
4. PALLETS — Are loads centered, wrapped properly, no torn wrap or shifted boxes?

If ANYTHING looks even slightly off — a single tilted box, one overhanging item, a suspicious wet spot — flag it.

category: spill / improper_stacking / safe
severity: critical / high / medium / low / none
confidence: 0-100%
location: specific location of the issue""",

"p6_scoring": """Rate this warehouse image on a safety scale. You are viewing from a drone camera.

Score each area 1-10 (10 = perfect):

FLOOR CONDITION: ___/10
(10 = completely dry and clean, 1 = large visible spill or puddle)

SHELF ORGANIZATION: ___/10
(10 = all boxes perfectly flat, aligned, within shelf edges. 1 = boxes falling, tilted, hanging off shelves, crushed)

OVERALL SAFETY: ___/10

Based on the LOWEST score, classify:
- If FLOOR < 5: category = spill
- If SHELF < 7: category = improper_stacking
- If both >= 7: category = safe

category: spill / improper_stacking / safe
floor_score: X/10
shelf_score: X/10
confidence: 0-100%""",

"p7_negative_framing": """This warehouse image was flagged by an automated system as potentially unsafe. Your job is to confirm or dismiss the alert.

The system detected a possible:
- Floor spill (liquid hazard)
- Improper stacking on shelves (falling object hazard)

Examine the image carefully. Is the system's alert justified?

Check the floor: Is there ANY liquid, wet patch, or reflective puddle?
Check the shelves: Are ALL boxes perfectly flat, aligned, and secure? Or are any tilted, hanging off edges, crushed, jumbled, or messy?

If you find a real hazard, classify it. If the scene is actually safe, say so.

category: spill / improper_stacking / safe
confidence: 0-100%
reasoning: one sentence explaining your decision""",


# === ROUND 2: Hybrid prompts combining best of round 1 ===

"p8_hybrid_v1": """You are a warehouse safety inspector analyzing a drone camera image.

STEP 1 — FLOOR: Look at the concrete floor. Is there any liquid, puddle, wet patch, or reflective spot? If yes → spill.

STEP 2 — SHELVES: Compare the left side shelves to the right side shelves. For each side:
- Are boxes sitting flat and square, or are any tilted/rotated/at angles?
- Are all boxes fully within the shelf boundaries, or do any hang over the edge?
- Are boxes in neat uniform rows, or are they jumbled/messy/disorganized?
- Are any boxes visibly crushed, dented, or deformed?

IMPORTANT: Most warehouses have SOME imperfection. Only flag improper_stacking if you see CLEAR issues like boxes at obvious angles, items clearly overhanging edges, visibly crushed packaging, or genuinely messy disorganized shelves. Minor cosmetic variations in box alignment are normal.

If the floor is dry AND shelves look reasonably organized → safe.

category: spill / improper_stacking / safe
confidence: 0-100%
location: brief description""",

"p9_hybrid_v2": """Inspect this warehouse drone image for safety hazards.

CHECK 1 — FLOOR HAZARD: Any liquid/puddle/wet spot on the concrete? (YES/NO)
CHECK 2 — STACKING HAZARD: Any boxes on shelves that are clearly tilted at angles, hanging off shelf edges, visibly crushed under weight, or arranged in a noticeably messy/chaotic way? (YES/NO)

Rules:
- If CHECK 1 = YES → category: spill
- If CHECK 2 = YES → category: improper_stacking
- If both NO → category: safe
- Normal warehouse wear and minor box alignment differences do NOT count as improper stacking

category: spill / improper_stacking / safe
confidence: 0-100%""",

"p10_calibrated": """You are a warehouse safety inspector. This is a drone camera view of a warehouse aisle.

Rate each aspect:

FLOOR: Is there visible liquid, puddles, or wet/reflective patches on the concrete?
→ If YES: this is a SPILL hazard.

SHELVING LEFT SIDE: Are boxes neat and stable, or messy and unstable?
SHELVING RIGHT SIDE: Are boxes neat and stable, or messy and unstable?

For shelving, "messy and unstable" means: boxes visibly tilted 10+ degrees, boxes sticking out past shelf edges, crushed/collapsing boxes bearing load, shrink wrap hanging off in pieces, boxes piled haphazardly rather than in rows.

Normal warehouse conditions (slight spacing variations, minor box wear) = SAFE.
Only genuine stacking problems = IMPROPER_STACKING.

category: spill / improper_stacking / safe
confidence: 0-100%
location: where the issue is""",

"p11_examples": """Classify this warehouse drone image.

SPILL = liquid on the floor (puddle, wet patch, oil slick, reflective pool on concrete)
IMPROPER_STACKING = boxes on shelves are clearly unsafe (examples: box tilted 15+ degrees, box hanging half off a shelf, stack of boxes leaning and about to fall, crushed box collapsing under weight, boxes thrown on shelf randomly instead of stacked in rows)
SAFE = dry floor + boxes on shelves look organized and stable (minor imperfections are normal and still count as safe)

category: spill / improper_stacking / safe
confidence: 0-100%""",

"p12_structured_checklist": """Warehouse Safety Inspection — Drone View

Complete this checklist:

FLOOR:
- [ ] Liquid/puddle visible? (yes/no)
- [ ] Wet or reflective patch? (yes/no)

LEFT SHELVES:
- [ ] Any box tilted more than ~10 degrees? (yes/no)
- [ ] Any box overhanging shelf edge? (yes/no)
- [ ] Any crushed or deformed box? (yes/no)
- [ ] Boxes arranged neatly in rows? (yes/no)

RIGHT SHELVES:
- [ ] Any box tilted more than ~10 degrees? (yes/no)
- [ ] Any box overhanging shelf edge? (yes/no)
- [ ] Any crushed or deformed box? (yes/no)
- [ ] Boxes arranged neatly in rows? (yes/no)

CLASSIFICATION:
- If ANY floor item = yes → category: spill
- If ANY shelf item shows problem → category: improper_stacking
- If all clear → category: safe

category: spill / improper_stacking / safe
confidence: 0-100%""",

}


def api(endpoint, payload=None, stream=False, timeout=300):
    url = f"{SERVER}{endpoint}"
    if payload:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp if stream else json.loads(resp.read())


def run_inference(model, image_path, prompt, nothink=False):
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
    return text, {"total_time": round(elapsed, 1)}


def quick_test(model, prompt_id, nothink=False):
    manifest = json.load(open(QUICK_MANIFEST))
    prompt_text = PROMPTS[prompt_id]

    results = {"correct": 0, "total": 0, "by_class": {}}
    for cls in ["spill", "improper_stacking", "safe"]:
        results["by_class"][cls] = {"tp": 0, "fn": 0, "fp": 0, "total": 0}

    print(f"\n{'='*60}")
    print(f"  QUICK TEST: {model} | {prompt_id} | nothink={nothink}")
    print(f"{'='*60}\n")

    total_time = 0
    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_DIR, entry["file"])
        gt = entry["ground_truth"]

        text, stats = run_inference(model, img_path, prompt_text, nothink=nothink)
        parsed = parse_response(text)
        pred = parsed["type"]
        correct = pred == gt
        total_time += stats["total_time"]

        results["total"] += 1
        if correct:
            results["correct"] += 1

        results["by_class"][gt]["total"] += 1
        if correct:
            results["by_class"][gt]["tp"] += 1
        else:
            results["by_class"][gt]["fn"] += 1
            if pred and pred in results["by_class"]:
                results["by_class"][pred]["fp"] += 1

        status = "OK" if correct else "MISS"
        print(f"  [{i+1}/15] {entry['file'][:35]:35s} gt={gt:20s} pred={str(pred):20s} {status} ({stats['total_time']}s)")

    acc = results["correct"] / results["total"] * 100
    avg_time = total_time / results["total"]

    print(f"\n  --- RESULTS: {prompt_id} ---")
    print(f"  Overall: {acc:.1f}% ({results['correct']}/{results['total']})")
    print(f"  Avg time: {avg_time:.1f}s per image")
    for cls in ["spill", "improper_stacking", "safe"]:
        c = results["by_class"][cls]
        cls_acc = c["tp"] / c["total"] * 100 if c["total"] > 0 else 0
        print(f"  {cls:20s}: {c['tp']}/{c['total']} correct ({cls_acc:.0f}%)")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--prompt-id", default="p1_simple")
    parser.add_argument("--nothink", action="store_true")
    parser.add_argument("--all", action="store_true", help="Test all prompts")
    args = parser.parse_args()

    if args.all:
        summary = []
        for pid in sorted(PROMPTS.keys()):
            r = quick_test(args.model, pid, args.nothink)
            acc = r["correct"] / r["total"] * 100
            stack_acc = r["by_class"]["improper_stacking"]["tp"] / r["by_class"]["improper_stacking"]["total"] * 100
            summary.append((pid, acc, stack_acc))

        print(f"\n{'='*60}")
        print(f"  PROMPT COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Prompt':<25s} {'Overall':>8s} {'Stacking':>10s}")
        print(f"  {'-'*25} {'-'*8} {'-'*10}")
        for pid, acc, stack in sorted(summary, key=lambda x: -x[1]):
            print(f"  {pid:<25s} {acc:>7.1f}% {stack:>9.0f}%")
    else:
        quick_test(args.model, args.prompt_id, args.nothink)
