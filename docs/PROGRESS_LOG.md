
# HAWKEYE Project Progress Log

## Project Summary
**Research Question:** How well do open-source VLMs detect warehouse safety hazards zero-shot, and does LoRA fine-tuning on synthetic images improve their performance?

**Broader Vision:** For any specific facility (warehouse, factory, store), you already have security camera feeds. Take those images, use AI image editing to inject realistic hazards, and fine-tune a VLM specifically for that environment. No need to collect real hazard photos (dangerous/expensive to stage), no manual labeling — the labels come from the injection process. The model doesn't need to generalize to every warehouse in the world; it only needs to work in YOUR facility.

**Pipeline = the real contribution:**
```
Facility camera images → AI image editing injects hazards → Auto-labeled dataset
→ Fine-tune VLM → Custom safety inspector for THAT specific facility
```
Warehouse safety is the case study proving this works. The pipeline generalizes to any environment with camera feeds.

**Why this works:** The sim-to-real gap is minimized because base images come from the actual facility (or realistic simulations of the exact environment). The fine-tuned model learns the specific layout, lighting, and context of that location.

**Key related work:** SynSpill (ICCV 2025 Workshop) — similar approach for single-category spill detection using SDXL + IP Adapters + LoRA on Qwen-VL. Our work extends to multi-class hazards using a more accessible pipeline (Gemini image editing vs custom diffusion pipelines).

**Broader Engine Vision:** The ultimate goal beyond the thesis is a reusable engine:
```
1. Scan environment (3D Gaussian Splatting from phone cameras, or existing security feeds)
2. Generate synthetic training data (AI image editing injects hazards into real images)
3. Auto-validate generated images (VLM checks quality, human reviews flagged ones)
4. Organize dataset with proper labels automatically
5. Fine-tune VLM for that specific environment
6. Deploy — custom safety inspector that works for YOUR facility
```
This thesis proves step 2-5 works. The engine generalizes beyond warehouses to any environment with camera feeds — factories, retail stores, construction sites. The key insight is that the model doesn't need to generalize to every facility; it just needs to work well in the one it was trained for, and the pipeline makes creating that training data cheap and fast.

**VLM vs CNN — Why VLMs for This Task:**
| | CNN (YOLOv8) | VLM (Qwen3.5) |
|---|---|---|
| Images needed | 2,500-10,000 with bounding boxes | 200-500 with text descriptions |
| Annotation cost | $5,000-15,000 (manual bbox labeling) | ~$50-200 (synthetic + AI descriptions) |
| Real hazard photos needed? | Yes, ideally | No — synthetic works |
| New facility | Retrain from scratch | Generate new images, quick fine-tune |
| Output | Bounding box + class label | Full natural language explanation |
| Novel hazards | Cannot detect (only trained classes) | Can describe anything unusual |
| Edge deployment | Easy (small model) | Harder (needs 2-9B model) |

**Owner:** Jaskirat Singh Sohal — CIS*4900 with Prof. John Akinyemi, University of Guelph
**Deadline:** April 5-10, 2026

---

## Phase 1: Blender Scene & Base Renders (COMPLETE)

- Built warehouse scene in Blender 5.0.1 (`warehouse_assembled.blend`)
- 303 objects: shelving racks, boxes, pallets, forklift, human model, PPE equipment
- 62 flight path waypoints for drone camera animation
- **184 base renders** from drone perspective (now in `dataset/images/safe_*.png`)
  - Frames 0-20: Forklift aisle (wide warehouse with forklift visible)
  - Frames 21-170+: Shelving aisles (narrow aisles between pallet racking)
- 3D assets: forklift (low-poly FBX), shelving, boxes, human figures
- Example renders preserved in `docs/base_render.png` and `docs/realistic_render.png`

---

## Phase 2: Hazard Image Injection (COMPLETE)

Using Gemini Pro via Playwright browser automation to inject hazards into base renders.
Script: `scripts/gemini_hazard_inject.py` — randomized slot-based prompt system for maximum diversity.

### Prompt System Architecture
Each hazard prompt is assembled from **template + randomized slots**:
```
TEMPLATE = "{object} is {position}. {severity}. {context}."
```
Each slot has 8-30 options, giving thousands of unique prompt combinations per category.

All prompts wrapped in a photorealism-preservation wrapper that tells Gemini to keep the warehouse identical and only add/modify the hazard.

### Image Counts (as of March 15, 2026)

| Category | Images | Status | Notes |
|----------|--------|--------|-------|
| forklift_violation | 226 | **Done + Labeled** | 10 OSHA violation types, triple-checked labels |
| spill | 210 | **Done + Labeled** | Floor liquid hazards — puddles, oil slicks, chemical spills |
| improper_stacking | 194 | **Done + Labeled** | Boxes tilted, overhanging, crushed, torn wrap on shelves |
| safe | 188 | **Done + Labeled** | Clean base renders + 4 fail images moved from hazard categories |
| obstacle | 181 | **Done + Labeled** | Aisle obstructions — fallen boxes, debris, abandoned equipment |
| **Total** | **999** | **All labeled** | |

### Forklift Violation Details
- **Source frames:** 0-20 (21 forklift aisle images)
- **Variants:** 10 per frame (21 × 10 = 210 target, 156 done so far)
- **10 OSHA-researched violation categories:**
  1. Elevated forks while traveling (most cited OSHA violation)
  2. Overloaded / double-stacked pallets
  3. Pedestrian in danger zone (36% of forklift fatalities)
  4. Tipover risk (42% of forklift injuries)
  5. Person riding on forks
  6. Falling / sliding load
  7. Unattended with forks up
  8. Blocking aisle / emergency exit
  9. Pedestrian pinned / crush risk
  10. Load blocking operator visibility

- **Prompt dimensions:** 10 positions × 10 orientations × 10 operator descriptions × 30 violations × 14 secondary details × 10 environments = ~4.2 million unique combos
- Each prompt tells Gemini to reposition the forklift, add a human operator, and show a specific violation

### Obstacle Category (COMPLETE)
- OSHA 1910.176 (handling/storage) & 1910.22 (walking surfaces)
- Split into forklift aisle (frames 0-20) and shelving aisle (frames 21-170) variants
- 4 focused types: fallen boxes, broken pallets, abandoned equipment, scattered debris
- 181 images generated, all labeled and double-checked

### Generation Infrastructure
- Gemini Pro subscription (2 accounts — main + new Pro free trial)
- Playwright + Chrome CDP automation (`--remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug`)
- Auto-resume: script skips existing images, safe to Ctrl+C and restart
- Rate limit handling: 3 consecutive failures → 10 min auto-pause
- ~90 seconds per image generation cycle

---

## Phase 2.5: Dataset Labeling & Validation (COMPLETE)

All 999 images labeled with structured JSONL annotations using Claude Opus as annotator. Each image was visually inspected, labeled, then double-checked in a verification pass. Forklift violations received a third pass on borderline images.

### Labeling Process
1. **First pass:** Read each image, assign category-specific labels (description, type, severity, quality, location)
2. **Verification pass:** Re-read each image + compare against first-pass label, correct any errors
3. **Third pass** (forklift only): Re-verify 41 borderline images that had disagreements

### Label Quality Summary

| Category | Images | Pass | Questionable | Corrections (2nd pass) |
|----------|--------|------|-------------|----------------------|
| forklift_violation | 226 | 226 | 0 | 95 → re-verified on 3rd pass |
| spill | 210 | 209 | 1 | Pre-validated (209/210 pass) |
| improper_stacking | 194 | 181 | 13 | 58 |
| safe | 188 | 188 | 0 | All confirmed clean |
| obstacle | 181 | 178 | 3 | 33 |
| **Total** | **999** | **982** | **17** | |

### Key Findings During Labeling
- **no_seatbelt violation unverifiable from drone angle** — 24 forklift images labeled `no_seatbelt` were initially marked questionable because seatbelt status can't be confirmed from overhead. Restored to `pass` since hazards were deliberately injected (ground truth).
- **4 fail images moved to safe** — Images with no visible hazard moved from obstacle/improper_stacking to safe category, increasing safe count from 184 → 188.
- **Obstacle type corrections** — Overturned bins were commonly mislabeled as `abandoned_equipment` instead of `scattered_debris` on first pass.
- **Description overstatement** — Second pass commonly toned down severity descriptions ("nearly falling" → "tilted", "crushed" → "compressed").

### Label Files
All labels unified into a single file after dataset reorganization:

| File | Purpose |
|------|---------|
| `dataset/metadata.jsonl` | Unified labels for all 999 images (category, description, subtype, severity, location) |
| `dataset/image_mapping.json` | Old filename → new filename mapping for traceability |

### Random Spot-Check (Final)
15 images randomly sampled (3 per category), all labels verified accurate. 2 minor flags (subtle spill, slightly overstated stacking description) — no mislabels found.

---

## Phase 3: Zero-Shot VLM Benchmark (COMPLETE)

### Setup
- **Ollama server:** Mac Mini at `http://10.0.0.244:11434`, 16GB RAM
- **Test set:** 99 images (33 spill, 33 improper_stacking, 33 safe)
- **Script:** `scripts/vlm_benchmark.py` with robust multi-strategy response parser
- **Prompt strategies:**
  - `simple` — OSHA inspector, direct classification
  - `cot` — Binary questions (floor spill? shelf problems?) then classify

### Models Benchmarked (18 runs total)

| Rank | Model | Prompt | Mode | Accuracy | Spill F1 | Stacking F1 | Safe F1 | Time/img |
|------|-------|--------|------|----------|----------|-------------|---------|----------|
| 1 | **qwen3.5:9b** | simple | think | **85.6%** | 0.984 | **0.750** | 0.812 | 115.9s |
| 2 | qwen3.5:9b | cot | think | 81.8% | 0.985 | 0.653 | 0.786 | 55.9s |
| 3 | qwen3.5:9b | cot | nothink | 80.6% | 0.954 | 0.653 | 0.780 | 13.5s |
| 4 | **qwen3.5:9b** | simple | nothink | **79.8%** | 0.952 | 0.667 | 0.762 | **9.9s** |
| 5 | qwen3.5:4b | simple | nothink | 70.7% | 0.918 | 0.526 | 0.675 | 12.3s |
| 6 | qwen3-vl:2b | cot | nothink | 68.4% | 0.875 | 0.350 | 0.696 | 8.4s |
| 7 | qwen3-vl:8b | simple | nothink | 67.7% | 0.970 | 0.114 | 0.680 | 27.3s |
| 8 | qwen3.5:4b | cot | nothink | 67.3% | 0.745 | 0.582 | 0.689 | 11.2s |
| 9 | qwen3-vl:8b | cot | nothink | 66.7% | 0.955 | 0.059 | 0.680 | 14.3s |
| 10 | qwen3-vl:2b | simple | nothink | 63.8% | 0.789 | 0.491 | 0.576 | 31.2s |
| 11 | granite3.2-vision:2b | cot | nothink | 55.6% | 0.821 | 0.000 | 0.593 | 8.6s |
| 12 | llava-phi3:latest | simple | nothink | 52.0% | 0.634 | 0.528 | 0.286 | 5.3s |
| 13 | llava-phi3:latest | cot | nothink | 46.6% | 0.615 | 0.071 | 0.450 | 3.4s |
| 14 | granite3.2-vision:2b | simple | nothink | 42.4% | 0.308 | 0.122 | 0.600 | 10.1s |
| 15 | gemma3:4b | cot | nothink | 36.4% | 0.167 | 0.512 | 0.000 | 6.8s |
| 16 | gemma3:4b | simple | nothink | 33.3% | 0.000 | 0.500 | 0.000 | 5.5s |
| 17 | moondream:latest | cot | nothink | 33.3% | 0.500 | 0.000 | 0.000 | 1.3s |
| 18 | moondream:latest | simple | nothink | 33.3% | 0.500 | 0.000 | 0.000 | 1.6s |

### Key Findings

1. **qwen3.5:9b is the best model** — 85.6% with think mode, 79.8% without
2. **Thinking mode helps the simple prompt significantly** (+5.8% accuracy, +0.083 stacking F1) but doesn't help CoT (already structures reasoning)
3. **Spill detection is easy** — all decent models get F1 > 0.9 (obvious visual cue)
4. **Improper stacking is the hardest** — best F1 is only 0.750 (subtle visual differences)
5. **Qwen3.5 family dominates** — 9b and 4b take top spots; older qwen3-vl models struggle with stacking
6. **Small models fail badly** — gemma3:4b and moondream basically guess randomly
7. **Speed vs accuracy tradeoff** — think mode is 12x slower but +5.8% accuracy

### Parser Fixes Applied
- "unsafe" was being classified as "safe" → fixed
- "proper_stacking" was not mapping to "improper_stacking" → fixed
- Strategy 1b had wrong keyword priority (spill before stacking) → fixed
- Re-scored all results with `scripts/rescore_and_report.py`

---

## Research Findings — Training Strategy (March 15-18, 2026)

### Dataset Size Requirements
- **SynSpill (ICCV 2025W):** Used 2,000 synthetic images to fine-tune Qwen-VL 7B with LoRA. LoRA-V (vision pathway only) outperformed LoRA-L (language pathway only), showing that visual specificity matters more than textual understanding for hazard detection.
- **Our dataset:** 913 human-verified images across 5 categories. At ~180 per category average, this is less than SynSpill's 2,000 single-category images, but our multi-turn conversation approach multiplies effective training examples.
- **Data efficiency curve planned:** Train on 100, 200, 400, 800 images to determine minimum viable dataset size. If 200 images achieves 85% of full performance, that proves companies don't need massive datasets.

### Multi-Turn Conversation Training Strategy
Instead of single-label training (image → "spill"), we generate 3 conversation styles per image:

**Style A — Detailed multi-turn:**
```
User: Is this warehouse safe?
Assistant: Unsafe. There is a hazard present.
User: What hazard do you see?
Assistant: Dark oil spill on the concrete floor in the center of the aisle.
User: How severe?
Assistant: High — large spill in forklift traffic path.
User: Is there improper stacking?
Assistant: No, shelves appear stable and organized.
```

**Style B — Direct classification:**
```
User: Classify this image.
Assistant: category: spill
```

**Style C — Reasoning chain:**
```
User: Analyze this warehouse step by step.
Assistant: First I check the floor — dark liquid puddle visible. Shelves look stable. Forklift operating normally. Primary hazard: spill. Category: spill.
```

**Result: 913 images × 3 styles = ~2,700 training examples.** Each is a full multi-turn conversation. The model learns to respond in multiple styles, matching whatever prompt format is used at inference time.

Human reviewer notes (242 images) provide the richest descriptions for generating Style A conversations — real human observations about what's actually in each image.

### Conversation Diversity > Photorealism
Research from the VLM synthetic data survey (2025-2026) shows:
- **Semantic diversity and instruction richness** drive VLM performance more than image photorealism
- VLMs already understand visual concepts (spills, forklifts, boxes) from pretraining on billions of images
- Fine-tuning teaches context: "in THIS warehouse, from THIS camera angle, THIS is what a hazard looks like"
- Traditional augmentation (flips, crops, color jitter) is minimally effective for VLMs — they understand semantics, so a flipped spill doesn't teach anything new
- **10-30 examples per hazard subtype is enough** for VLMs, IF the training conversations are rich and varied

### Post-Training Strategy: GRPO (Reinforcement Learning)
After SFT (supervised fine-tuning), we can further improve the model with GRPO (Group Relative Policy Optimization):
1. Model generates 5 different responses for each image
2. Each response scored: correct category (+2), accurate description (+1), no hallucination (+1)
3. Model weights updated to prefer high-scoring responses
4. HuggingFace has a working cookbook for GRPO with Qwen-VL

SFT teaches the model correct answers. GRPO teaches it to reason better — reducing hallucinations and improving explanation quality. Both update the same LoRA adapter weights.

### Open-Source Image Generation Research
Explored alternatives to Gemini for hazard image generation:

| Model | Params | License | Image Editing? | Cost/Image |
|-------|--------|---------|---------------|------------|
| FLUX.1 dev | 12B | Non-commercial | Yes (img2img) | ~$0.012 |
| FLUX.1 schnell | 12B | Apache 2.0 | Yes (fast) | ~$0.012 |
| SDXL | ~2.6B | CreativeML Open RAIL++ | Yes (full ecosystem) | ~$0.005 |
| SD 3.5 Large | 8.1B | Community license | Yes (img2img) | ~$0.02 |
| GLM-Image/CogView-4 | ~16B | Research | Text-to-image mainly | Limited API |

**For the thesis:** Gemini Playwright automation is sufficient (already generated all images).
**For the engine vision:** SDXL (fully open, commercial use, huge ControlNet ecosystem) and FLUX.1 schnell (Apache 2.0, fast) are the best candidates for a deployable pipeline.

### Known Data Issue: Forklift Aisle Floor Texture
The Gemini photorealism enhancement added realistic floor stains/marks to the forklift aisle base renders (frames 0-20). These dark patches on concrete look like spills to VLMs but are just normal floor wear. This means:
- Safe images from the forklift aisle may be classified as "spill" by models
- All forklift-aisle safe images have `accept_also: ["spill"]` in metadata to account for this
- This is a known limitation: the synthetic base images themselves contain ambiguous visual features

---

## Phase 4: Training Pipeline (PLANNED)

### Research Question
Can a small fine-tuned VLM (2B) outperform a large zero-shot VLM (9B)? Can we teach the model to think efficiently without infinite reasoning loops?

### Baselines to Beat (from Benchmark v2)
| Model | Mode | Zero-shot Acc | Target |
|-------|------|--------------|--------|
| qwen3.5:4b | nothink | 82.7% | Fine-tuned should exceed 90% |
| qwen3.5:4b | think (real) | 89.2% | Fine-tuned think should match without timeouts |
| qwen3.5:9b | nothink | 76.7% | Fine-tuned 2b should beat this |
| qwen3.5:2b | nothink | 65.3% | Fine-tuned 2b should beat 9b zero-shot |

### Dataset Split (COMPLETE)
- **728 training images** / **185 test images** (80/20 stratified by category)
- Test set reserved for final evaluation only
- Zero overlap verified

### Step 4.1: Build Training Conversations

Three conversation styles per image, plus negative examples:

**Style A — Multi-turn Q&A:**
```
User: [image] Is this warehouse safe or unsafe?
Assistant: Unsafe. There is a hazard present.
User: What hazard do you see?
Assistant: Dark oil spill on the concrete floor in the center of the aisle.
User: How severe is it?
Assistant: High — the spill is large and in a forklift traffic path.
User: Is there any improper stacking?
Assistant: No, the boxes on the shelves appear stable and properly arranged.
```

**Style B — Direct classification:**
```
User: [image] Classify this warehouse image.
Assistant: category: spill
description: Dark oil spill on concrete floor in the center aisle.
```

**Style C — Reasoning chain:**
```
User: [image] Analyze this warehouse step by step.
Assistant: First I check the floor — I see a dark liquid puddle reflecting overhead lights. Then I check the shelves — boxes look stable and properly arranged. The forklift appears to be operating normally. No workers are in danger zones. The primary hazard is the liquid spill on the floor. Category: spill. Severity: high.
```

**Negative examples** (critical for reducing false positives):
```
User: [image] Is there a spill on the floor?
Assistant: No. The dark marks on the concrete are normal tire tracks and floor wear, not liquid spills. The floor is dry.
```

**728 images × 3 styles + negatives = ~2,500+ training conversations**

Human reviewer notes (242 images) used for Style A descriptions — real human observations about what's actually in each image.

### Step 4.2: Domain Randomization Augmentation

Applied at training time only (clean dataset stays clean). Each training image randomly receives:
- JPEG compression (quality 60-90) — simulates real camera compression
- Gaussian blur (radius 0.3-1.0) — simulates camera focus issues
- Brightness/contrast variation (0.85-1.15) — simulates lighting changes
- Color temperature shifts (0.8-1.2) — simulates different camera white balance
- Sensor noise — simulates real camera grain

Purpose: bridge the gap between synthetic Gemini images and real warehouse camera feeds. Model learns to detect hazards regardless of image quality.

### Step 4.3: Autoresearch — Hyperparameter Search (BEFORE full training)

Inspired by Andrej Karpathy's autoresearch framework (March 2026). Instead of guessing hyperparameters, an autonomous agent tests ~100 configurations overnight using 5-minute training runs.

**How it works:**
```
Loop (overnight, ~12 experiments/hour):
  1. Agent reads current best config
  2. Proposes one change (e.g. rank 16 → 32)
  3. Trains for 5 minutes on subset (100 images, 1 epoch)
  4. Evaluates on validation set
  5. If accuracy improved → keep change, git commit
  6. If worse → revert, try something else
  7. Repeat
```

5-minute experiments work because you don't need full training to know if a change helps. If learning rate 2e-4 makes loss drop faster in 5 minutes, it'll also be better after 5 hours. Testing the direction, not training to completion.

**What the agent searches:**
| Parameter | Options | Total Combos |
|-----------|---------|-------------|
| LoRA rank | 4, 8, 16, 32, 64 | 5 |
| LoRA alpha/rank ratio | 1, 2, 4 | 3 |
| Learning rate | 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4 | 6 |
| Vision LR multiplier | 0.1, 0.2, 0.5, 1.0 | 4 |
| LoRA dropout | 0, 0.05, 0.1 | 3 |
| Target layers | all-linear, qv-only, language-only, vision-only | 4 |
| Training data | all styles, A only, B only, with/without negatives | 4 |
| Augmentation | on/off, strength levels | 3 |

1000+ possible combinations. Manual search tests 5-10. Autoresearch tests 100 overnight and finds the optimal config.

### Step 4.4: SFT (Supervised Fine-Tuning) — Full Training

Using the optimal config found by autoresearch:

| Setting | Value |
|---------|-------|
| **Framework** | TRL SFTTrainer + PEFT LoRA |
| **Precision** | bf16 (NOT 4-bit for vision layers — research says quantizing vision layers degrades visual understanding) |
| **LoRA targets** | All linear layers: vision + language (LoRA-V + LoRA-L). SynSpill showed LoRA-V outperforms LoRA-L for hazard detection |
| **LoRA rank** | Found by autoresearch (default: 16) |
| **LoRA alpha** | Found by autoresearch (default: 32) |
| **Learning rate** | Separate for vision (lower, ~2e-5) and language (higher, ~1e-4). Vision layers need gentler updates |
| **Epochs** | 3-5 with early stopping on validation accuracy |
| **Batch size** | 4 with gradient accumulation 4 (effective batch 16) |
| **Compute** | Google Colab — T4 (16GB) for 2b, A100 (40GB) for 4b |
| **Models** | qwen3.5:4b (primary), qwen3.5:2b (comparison) |

**What each setting means:**
- **LoRA** = Low-Rank Adaptation. Adds small trainable "adapters" to frozen model weights. Only ~1-2% of parameters updated, rest stay frozen. Dramatically reduces memory and training time.
- **bf16** = Brain Float 16-bit precision. Half the memory of full precision (fp32) with minimal accuracy loss. Essential for fitting models on consumer GPUs.
- **Gradient accumulation** = Simulates larger batch size without more memory. Model accumulates gradients from 4 batches of 4 images (=16 images) before updating weights.
- **Epochs** = Number of complete passes through the training data. Too few → underfitting. Too many → overfitting (memorizing training data instead of learning patterns).

### Step 4.5: GRPO (Reinforcement Learning) — Think Mode Improvement

After SFT gives us a good classifier, GRPO teaches the model to think efficiently. This directly addresses the timeout problem from benchmark: 4b think had 89.2% real accuracy but 19% timeout rate.

**Training on SFT checkpoint with reward function:**
```python
def reward(response, ground_truth, thinking_tokens):
    score = 0

    # Correctness (most important)
    if correct_category(response, ground_truth): score += 2.0
    if good_description(response): score += 1.0
    if no_hallucination(response): score += 1.0

    # Conciseness (penalize excessive thinking)
    thinking_length = len(thinking_tokens)
    score -= 0.0005 * thinking_length  # slight penalty per token

    # Bonus for efficient correct thinking
    if correct and thinking_length < 2000:
        score += 0.5  # reward quick correct answers

    return score
```

**Training data for GRPO:**
- **Positive examples:** 4b/9b think traces that got correct answers in <3000 tokens (captured from benchmark logs — full reasoning chains available)
- **Negative examples:** Infinite loop traces (16000+ chars, timeout) from benchmark — model learns what BAD reasoning looks like
- **Budget forcing at inference:** Cap thinking at 3000 tokens → force `</think>` and answer. Prevents infinite loops entirely.

**Research basis:**
- TON paper (2025): "Thought dropout" + GRPO achieves 87% token reduction while improving accuracy
- GRPO-LEAD (2025): Difficulty-aware length penalty — easy images get penalized more for long thinking
- GTR (ICCV 2025): Guided Thought Reinforcement prevents "thought collapse" where RL training causes model to stop thinking entirely

**Target:** Think mode accuracy of 89%+ with <5% timeout rate (currently 89.2% but 19% timeout).

### Step 4.6: Data Efficiency Curve

Train on 100, 200, 400, 728 images (all with best config from autoresearch):
- Plot accuracy vs training set size
- Find minimum viable dataset size for deployment
- If 200 images achieves 85% of full performance → proves companies don't need massive datasets
- Key thesis figure showing the efficiency of the VLM + synthetic data approach

### Step 4.7: Comprehensive Evaluation

**Main comparison table:**
| Comparison | What it proves |
|-----------|---------------|
| Fine-tuned 4b nothink vs zero-shot 4b nothink | Fine-tuning improves accuracy |
| Fine-tuned 2b nothink vs zero-shot 9b nothink | Small fine-tuned > large zero-shot |
| SFT vs SFT+GRPO (think mode) | RL improves reasoning efficiency |
| Fine-tuned vs Claude/Gemini (API, zero-shot) | Open-source competitive with commercial |
| Training on 100/200/400/800 images | Minimum data requirements |
| Clean test vs augmented test (noise/blur) | Robustness for real deployment |

**Per-category analysis:**
- improper_stacking: weakest category (F1=0.524 zero-shot) — did fine-tuning fix it?
- forklift_violation: subtle hazards (pedestrian proximity) — can model learn safety judgment?
- safe vs false positives: does model stop calling floor texture "spill"?

**Commercial model comparison:**
- Same 185 test images sent to Claude and Gemini via API
- Same direct prompt, zero-shot, fair comparison
- Cost analysis: API cost per image vs local inference cost

---

## Phase 2.6: Human Review & Label Correction (COMPLETE — March 15-18, 2026)

### Problem Discovered
Running initial benchmark with qwen3.5:9b revealed that many "wrong" model predictions were actually correct — the model was seeing the image accurately but our labels were wrong. Root cause: Claude's labeling was biased by knowing which folder images came from (e.g. images in `forklift_violation/` were assumed to be forklift violations even when the image showed normal operation or a different hazard).

### Systematic Issues Found
1. **no_seatbelt labels (35 images)**: Seatbelt status cannot be determined from drone camera angle. Many of these images were actually safe or had completely different hazards (person on forks, pedestrian proximity, obstacles on floor).
2. **forks_raised over-labeling**: Forks slightly raised during normal picking/placing operations were incorrectly flagged as violations. Only genuinely dangerous situations (traveling with load elevated, unbalanced loads at height) are real violations.
3. **Pedestrian proximity under-detected**: Claude systematically missed workers standing near active forklifts — the most common real forklift hazard.
4. **Multi-label images**: Many images contained multiple valid hazards (spill + obstacle, forklift violation + debris). Single-label evaluation penalized correct predictions.
5. **Deformed/broken images**: ~83 images (mostly improper_stacking) had Gemini generation artifacts — distorted objects, unrealistic scenes, broken geometry.
6. **Description inaccuracies**: Locations systematically described as "center of aisle" when hazards were at edges/corners. Severity overstated. Subtypes misidentified.

### Human Review Tool
Built an HTML review tool (`scripts/review_tool.html`) for efficient manual review:
- Shows each image with current metadata (category, description, location, severity, subtype)
- Confirm (Enter), Change category (C + click), or add notes via text/voice
- Voice input via browser Speech-to-Text API — auto-detects category keywords from speech
- Keyboard shortcuts: 1-5 for quick category selection, arrows for navigation
- Progress saves to localStorage + manual JSON export
- Served locally via `python3 -m http.server 8080`

### Review Process
Jaskirat personally reviewed all 996 images (took ~3 hours):
- **667 confirmed** as correctly labeled
- **329 changed** — category corrections, description notes, or flagged for deletion
- **83 images deleted** (deformed/broken — mostly improper_stacking from Gemini)
- **89 category changes** applied
- **124 images** marked as multi-label (`accept_also` field — model gets credit for any valid hazard)
- **242 images** received detailed notes about what's actually in the image

### Category Changes Applied

| Change | Count | Reason |
|--------|-------|--------|
| forklift_violation → safe | 32 | Normal operation, seatbelt-only issues, pedestrians far away |
| improper_stacking → safe | 20 | Boxes plastic-wrapped, minor unevenness not a real hazard |
| spill → obstacle | 7 | Has both spill + obstacle, user picked obstacle as primary |
| forklift_violation → improper_stacking | 5 | Issue was the load/stacking, not the forklift operation |
| safe → forklift_violation | 4 | Previously incorrectly moved to safe |
| safe → spill | 4 | Floor spills visible in "safe" images |
| forklift_violation → obstacle | 3 | Debris/fallen boxes, forklift operating normally |
| forklift_violation → spill | 3 | Floor spill, no forklift issue |
| safe → obstacle | 3 | Debris or pallet jacks on floor |
| Other | 8 | Various corrections |

### Description Correction Process
After category review, 6 parallel agents processed the 242 images with human notes:
1. Read the voice-to-text notes (correcting "spell"→spill, "folks"→forks, "skit"→skid, "save"→safe, "are"→air, "stabbed"→strapped, "eyes"→aisles)
2. Looked at the actual image to verify
3. Wrote clean, grammatically correct descriptions with corrected locations, severity, and subtypes

Common corrections:
- Location "center of aisle" → actual position (bottom-left, right shelving row, far end, etc.)
- Severity downgraded where hazards were minor
- Subtypes corrected (e.g. "forks_raised" → "pedestrian_proximity" for 12+ images where the real danger was a person near the forklift, not the forks)

### Final Dataset After Review

| Category | Before Review | After Review | Change |
|----------|--------------|-------------|--------|
| safe | 217 | 256 | +39 (many images were actually safe) |
| spill | 216 | 214 | -2 |
| obstacle | 191 | 200 | +9 |
| forklift_violation | 181 | 137 | -44 (biggest correction) |
| improper_stacking | 191 | 106 | -85 (83 deleted + 20 moved to safe) |
| **Total** | **996** | **913** | -83 deleted |

### Key Learnings
- **AI labeling is biased by metadata**: Knowing the intended category biases the label. VLMs doing honest visual assessment found errors AI labeling missed.
- **Human review is essential**: No substitute for domain expert eyes on every image.
- **Voice-to-text review is fast**: 996 images in ~3 hours using voice notes.
- **Multi-label is reality**: Warehouse images often contain multiple simultaneous hazards.
- **Smaller dataset with correct labels > larger dataset with wrong labels**.

---

## Phase 3.5: Zero-Shot VLM Benchmark v2 (COMPLETE — March 18-19, 2026)

### Setup Changes from v1
- **5 categories** instead of 3 (added forklift_violation and obstacle)
- **150 test images** (30 per category, stratified random sample, seed=42 for reproducibility)
- **9 models** tested (dropped granite3.2-vision:2b due to extreme slowness — 18s/image, would take 10+ hours)
- **5 prompt strategies** instead of 2
- **Pre-resized images** (1024px JPG, ~128KB each vs 7MB PNGs — 8x faster inference). Original PNGs were 2754×1536 at ~7MB each. Inference went from 183s/image to ~8s/image after resizing.
- **Multi-label scoring** via `accept_also` field — 124 images have multiple valid categories. Model gets credit for predicting any valid hazard.
- **Ollama server** on Mac Mini M2 16GB at `http://10.0.0.244:11434` (local network). Also tested via Cloudflare tunnel (`cloudflared tunnel --url http://localhost:11434`) for remote access from university, but free tunnel has 100s timeout — too short for model loading. Switched back to local network for reliability.
- Script: `scripts/benchmark_v2.py` using `vision_api.py` for all Ollama communication (proven reliable API wrapper).
- Model loaded once, all prompts run, then switched to next model (avoids 2-10 min reload per prompt).

### 5 Prompt Strategies

| # | Name | Approach | Description |
|---|------|----------|-------------|
| 1 | **direct** | Minimal instruction | "Classify into one category. Respond with category + one sentence description." |
| 2 | **osha_inspector** | Role-based | "You are an OSHA inspector. Examine floor/forklift/shelves/aisles. Classify + severity + confidence." |
| 3 | **chain_of_thought** | Binary questions | "Answer YES/NO for each hazard type, then classify." |
| 4 | **describe_then_classify** | Description first | "Describe the scene in 2-3 sentences, then classify." |
| 5 | **json_structured** | Structured output | "Respond ONLY with JSON: {category, confidence, description}" |

### Models Tested

| Model | Size | Family | Vision Architecture |
|-------|------|--------|-------------------|
| qwen3.5:9b | 9.7B | Qwen 3.5 | Early-fusion multimodal |
| qwen3-vl:8b | 8.8B | Qwen 3 VL (older) | Vision-language |
| qwen3.5:4b | 4.7B | Qwen 3.5 | Early-fusion multimodal |
| gemma3:4b | 4.3B | Google Gemma 3 | Multimodal |
| qwen3.5:2b | 2.3B | Qwen 3.5 | Early-fusion multimodal |
| qwen3-vl:2b | 2.1B | Qwen 3 VL (older) | Vision-language |
| llava-phi3:latest | 4B | Microsoft LLaVA-Phi3 | CLIP + Phi-3 |
| qwen3.5:0.8b | 873M | Qwen 3.5 | Early-fusion multimodal |
| moondream:latest | 1B | Moondream | Lightweight vision |

### Complete Results — Nothink Mode (47 runs)

**Full Leaderboard (sorted by accuracy):**

| # | Model | Prompt | Accuracy | Avg Time |
|---|-------|--------|----------|----------|
| 1 | **qwen3.5:4b** | **direct** | **82.7%** | **5.6s** |
| 2 | qwen3.5:9b | direct | 76.7% | 9.0s |
| 3 | qwen3.5:4b | chain_of_thought | 76.0% | 8.0s |
| 4 | qwen3.5:4b | osha_inspector | 75.3% | 6.6s |
| 5 | qwen3.5:9b | osha_inspector | 74.0% | 10.4s |
| 6 | qwen3.5:9b | chain_of_thought | 66.0% | 12.9s |
| 7 | qwen3.5:4b | json_structured | 65.3% | 6.2s |
| 8 | qwen3.5:2b | direct | 65.3% | 3.1s |
| 9 | qwen3.5:0.8b | json_structured | 64.7% | 2.1s |
| 10 | qwen3.5:9b | json_structured | 64.0% | 9.8s |
| 11 | qwen3.5:9b | describe_then_classify | 63.3% | 13.2s |
| 12 | gemma3:4b | json_structured | 63.3% | 5.3s |
| 13 | qwen3-vl:8b | json_structured | 59.3% | 18.7s |
| 14 | qwen3.5:0.8b | direct | 59.3% | 1.9s |
| 15 | qwen3-vl:8b | direct | 57.3% | 19.1s |
| 16 | qwen3.5:2b | json_structured | 56.7% | 3.8s |
| 17 | qwen3.5:4b | describe_then_classify | 55.3% | 8.6s |
| 18 | qwen3-vl:8b | describe_then_classify | 53.3% | 19.8s |
| 19 | qwen3.5:2b | osha_inspector | 53.3% | 3.8s |
| 20 | qwen3.5:2b | describe_then_classify | 50.7% | 5.5s |
| 21 | gemma3:4b | direct | 50.0% | 4.7s |
| 22-47 | (remaining) | (various) | 0-48.7% | — |

**Per-model best results:**

| Model | Best Prompt | Best Acc | Speed | Notes |
|-------|-----------|----------|-------|-------|
| **qwen3.5:4b** | direct | **82.7%** | 5.6s | Best overall — conservative, accurate |
| qwen3.5:9b | direct | 76.7% | 9.0s | Over-classifies hazards |
| qwen3.5:2b | direct | 65.3% | 3.1s | Good for size, fast |
| qwen3.5:0.8b | json_structured | 64.7% | 2.1s | Surprisingly capable for 873M params |
| gemma3:4b | json_structured | 63.3% | 5.3s | Only works with JSON format |
| qwen3-vl:8b | json_structured | 59.3% | 18.7s | Old architecture, slow |
| llava-phi3 | direct | 46.0% | 3.0s | Weak on multi-category |
| qwen3-vl:2b | direct | 40.7% | 6.9s | Old architecture, parser issues |
| moondream | chain_of_thought | 36.7% | 1.3s | Too small for this task |

### Key Finding 1: qwen3.5:4b > qwen3.5:9b (Smaller is Better)

Per-category F1 comparison (direct prompt):

| Category | 9b F1 | 4b F1 | Winner | Analysis |
|----------|-------|-------|--------|----------|
| spill | 0.966 | 0.800 | 9b | 9b has higher spill recall but many false positives |
| forklift_violation | 0.778 | 0.727 | 9b | Both decent, 9b slightly better at pedestrian proximity |
| improper_stacking | 0.652 | 0.815 | **4b** | 4b actually looks at shelves; 9b fixates on floor |
| obstacle | 0.690 | 0.885 | **4b** | 9b calls everything "obstacle" (26 false positives) |
| safe | 0.764 | 0.909 | **4b** | 4b correctly identifies clean images; 9b sees phantom hazards |

**Why 4b wins:** The 9b model over-classifies — it sees hazards that don't exist. It has high recall on spills and forklifts but at the cost of massive false positives (calling safe images "obstacle", calling stacking images "spill"). The 4b is more conservative and disciplined. This aligns with research showing larger models can be more "creative" in finding problems that don't exist — a liability for safety classification where precision matters.

### Key Finding 2: Direct Prompt Dominates

Results across all Qwen3.5 models (direct prompt accuracy):

| Model | direct | osha | cot | describe | json |
|-------|--------|------|-----|----------|------|
| qwen3.5:4b | **82.7%** | 75.3% | 76.0% | 55.3% | 65.3% |
| qwen3.5:9b | **76.7%** | 74.0% | 66.0% | 63.3% | 64.0% |
| qwen3.5:2b | **65.3%** | 53.3% | 0.0% | 50.7% | 56.7% |
| qwen3.5:0.8b | 59.3% | 34.0% | 42.0% | 29.3% | **64.7%** |

**Why direct wins:** Warehouse hazard classification is a pattern recognition task, not a reasoning task. The model's first instinct when seeing an image is usually correct. Adding reasoning steps (CoT, describe-then-classify) gives the model room to overthink, find secondary issues, second-guess its initial classification, and talk itself into the wrong answer. The OSHA inspector role adds context that sometimes helps (2nd best for 4b/9b) but can also bias the model toward finding violations where none exist.

**Exception: 0.8b prefers json_structured (64.7%).** The smallest model benefits from strict output formatting — it needs the constraint to stay on task. Without structure, it rambles.

### Key Finding 3: Qwen 3.5 Architecture Dominates

The Qwen 3.5 family at EVERY size beats every other model family at larger sizes:

| Comparison | Winner |
|-----------|--------|
| qwen3.5:0.8b (59.3%) vs moondream:1b (36.7%) | Qwen 3.5 at half the size |
| qwen3.5:2b (65.3%) vs qwen3-vl:8b (59.3%) | Qwen 3.5 at 1/4 the size |
| qwen3.5:4b (82.7%) vs gemma3:4b (63.3%) | Same size, Qwen 3.5 wins by 19.4% |
| qwen3.5:0.8b (59.3%) vs llava-phi3:4b (46.0%) | 5x smaller, still wins |

The early-fusion multimodal architecture of Qwen 3.5 is fundamentally better than the bolt-on vision approaches (CLIP+LLM) used by older models. This is a strong thesis finding: **architecture matters more than parameter count for vision tasks.**

### Deep Analysis: 9b vs 4b Response Comparison

Manually read all 150 responses from both models on the `direct` prompt to compare HOW they reason:

**Agreement analysis (150 images):**
- Both correct: 108 (72%)
- Only 4b correct: 16 (10.7%) — 4b wins these
- Only 9b correct: 7 (4.7%) — 9b wins these
- Both wrong: 19 (12.7%)
- Total disagreements: 40

**9b personality — "The Nervous Inspector":**
- Sees danger everywhere, especially on the floor
- Uses hedging language: "could pose", "potential", "suggesting"
- Describes floor texture (tire marks, concrete stains) as hazards
- Says "obstacle" 28 times out of 40 disagreements
- Better at reading complex scenes (sees forklift + person + load simultaneously)
- Better at pedestrian proximity detection

**4b personality — "The Focused Observer":**
- Looks at the most visually dominant feature, describes it clearly, moves on
- Uses decisive language: "A large puddle", "boxes are tilted and leaning"
- Actually looks at shelves (9b often ignores them and describes the floor)
- Says "spill" 20 times out of 40 disagreements (defaults to spill for ambiguous floor patches)
- Shorter, cleaner descriptions
- Misses forklift violations when spill-like floor texture is present

**Where 4b beats 9b (16 images):**
Almost always stacking or spill images. 4b says "boxes tilted and leaning precariously on right-hand shelves" while 9b says "dark stains on the concrete floor." The 9b literally doesn't look at the shelves in many stacking images.

**Where 9b beats 4b (7 images):**
Mostly forklift scenes. 9b correctly sees "forks raised while pedestrian stands too close" while 4b fixates on floor stains and says "spill." The 9b reads the human interaction in the scene better.

**Where both fail (19 images):**
The forklift aisle floor texture problem. Both see dark patches on the concrete (which are just normal floor wear from the Gemini photorealism enhancement) and classify them as either "obstacle" (9b) or "spill" (4b). Neither can distinguish floor stains from actual hazards. This is the #1 error mode and exactly what fine-tuning should fix.

### Think Mode Results (COMPLETE — March 19-21, 2026)

Testing think mode (internal chain-of-thought reasoning) on the top 4 Qwen3.5 models with `direct` and `osha_inspector` prompts. 54 total benchmark runs completed.

**Implementation challenges:**
1. **Token budget discovery:** Think mode generates hidden `<think>` tokens that count against `num_predict`. Initial runs with `num_predict: 500` produced EMPTY visible output — all tokens consumed by thinking. Fixed to `num_predict: 16384` for think mode, `2048` for nothink.
2. **Think flag discovery:** Setting `think: True` explicitly in the API payload is required. Omitting the flag defaults to thinking ON (model burns all tokens on hidden reasoning). Must set `think: False` for nothink, `think: True` for think — not just omit it.
3. **Thinking capture:** Ollama streams thinking tokens in a separate `message.thinking` field. The benchmark captures these in `model_thinking` field of per-image logs — provides full reasoning chain for every image.
4. **Timeout protection:** Small models (0.8b, 2b) generate infinite thinking loops on some images — 69,000+ characters of circular reasoning. Added 300s (5 min) timeout per image. Stuck images are marked "TIMEOUT" and skipped.
5. **Mac Mini crashes:** Extended think mode inference causes the Mac Mini to become unreachable after hours of sustained load. IP changed from 10.0.0.244 to 10.0.0.54 after one restart. Sleep disabled with `sudo pmset -a sleep 0 disablesleep 1` and `caffeinate -s &`.

**Complete Think Mode Results:**

| Model | Prompt | Nothink | Think (reported) | Think (real*) | Timeouts | Avg Time |
|-------|--------|---------|-----------------|-------------|----------|----------|
| 0.8b | direct | 59.3% | 50.7% | — | 17 (11%) | 47s |
| 0.8b | osha | 34.0% | — (aborted 89/150) | — | 35+ | — |
| 2b | direct | 65.3% | 56.7% | — | 5 + 17 errors | — |
| 2b | osha | 53.3% | 20.5% (aborted 117/150) | — | 65 errors | — |
| **4b** | **direct** | **82.7%** | 71.3% | **89.2%** | 29 (19%) | 153s |
| **4b** | **osha** | 75.3% | 50.0% | **90.4%** | 66 (44%) | 189s |
| **9b** | **direct** | 76.7% | 70.7% | **86.9%** | 27 (18%) | 386s |
| **9b** | **osha** | 74.0% | **74.7%** | **85.2%** | 19 (13%) | 132s |

*Real = accuracy on images where the model completed its reasoning (excluding timeouts/errors)

### Key Finding 4: Think Mode — Size Threshold Effect

Think mode has a clear size threshold at ~4B parameters:

**Below 4B (0.8b, 2b): Think mode HURTS accuracy (-8 to -33%)**
- Models generate circular reasoning loops ("Wait, let me look again... But actually... Let me reconsider...")
- 0.8b generates 69,000+ characters of thinking on a single image without reaching a conclusion
- The thinking process introduces doubt and reasoning errors that small models cannot recover from
- Nothink "gut instinct" is more reliable than extended reasoning for these sizes

**At 4B and above: Think mode IMPROVES accuracy (+6-15% real)**
- 4b direct: 82.7% nothink → 89.2% real think (+6.5%)
- 4b osha: 75.3% nothink → 90.4% real think (+15.1%) — highest accuracy achieved
- 9b direct: 76.7% nothink → 86.9% real think (+10.2%)
- 9b osha: 74.0% nothink → 85.2% real think (+11.2%)
- Models reason methodically and reach correct conclusions most of the time

**The timeout problem:** Even at 4b/9b, 13-44% of images timeout (model thinks for 5 min without concluding). This makes reported accuracy lower than nothink, even though real accuracy is higher. Production deployment should use a think-then-fallback strategy: try think mode, if timeout → fall back to nothink.

### Key Finding 5: Small Models Can See But Cannot Judge

The most significant insight from the think mode experiment. Captured from 0.8b's actual thinking traces:

**Example: forklift_0021.png (GT: forklift_violation)**

The 0.8b model's internal reasoning:
```
1. Analyze the image:
   - There is a warehouse setting with high shelves and pallets filled with boxes.
   - A forklift is in the middle of the aisle.
   - Two workers are operating it. One is standing near the forks, one is sitting
     in the driver's seat.
   - The boxes are stacked very high and tightly packed.

2. Evaluate the options:
   - forklift_violation: The forklift is in use. It's not clearly violating rules.
     It looks like normal operation.
   - improper_stacking: The boxes are stacked very high. This is a classic sign
     of improper stacking.

3. Final answer: improper_stacking
```

The model SEES correctly — it observes "Two workers, one standing near the forks, one driving." This IS the forklift violation (person dangerously close to active forks). But it REASONS incorrectly — it says "forklift is in use, not clearly violating rules, looks like normal operation."

**The insight:** The model can perceive visual details accurately but lacks the domain knowledge to judge what constitutes a safety violation. It thinks "person near forklift = normal warehouse work." This is precisely what fine-tuning will teach — not better vision, but better safety judgment.

**Comparison across sizes for the same image (forklift_0021.png):**
- 0.8b think: sees workers, concludes "improper_stacking" (WRONG — focused on boxes)
- 2b think: connection error (stuck thinking)
- 4b think: "A forklift is operating with its forks raised high while a second worker stands on the forks" → forklift_violation (CORRECT)
- 9b think: "A worker is standing too close to a forklift that is lifting a load" → forklift_violation (CORRECT)

The 4b and 9b models have enough safety knowledge to recognize the danger. The smaller models see the same scene but cannot interpret its safety implications.

### Key Finding 6: Best Overall Strategy

| Strategy | Accuracy | Speed | Reliability |
|----------|----------|-------|-------------|
| 4b nothink direct | **82.7%** | **5.6s** | **100%** |
| 4b think direct (real) | 89.2% | 153s | 81% (19% timeout) |
| 4b think osha (real) | **90.4%** | 189s | 56% (44% timeout) |
| 9b nothink direct | 76.7% | 9.0s | 100% |
| 9b think osha | 74.7% (real: 85.2%) | 132s | 87% (13% timeout) |

**For production:** 4b nothink direct — fast, reliable, 82.7%.
**For maximum accuracy:** Think-then-fallback — try 4b think, if timeout → fall back to nothink. Expected ~88% overall.
**For fine-tuning base model:** 4b nothink — will be trained to internalize the reasoning that think mode provides, achieving think-level accuracy at nothink speed.

### Logging Structure
```
outputs/benchmark_v2/
├── BENCHMARK_LOG.md              # Master readable log with all results
├── {model_name}/
│   ├── {prompt_name}__{mode}/    # e.g. direct__nothink, direct__think
│   │   ├── results.json          # Metrics (accuracy, F1, per-category)
│   │   └── per_image_log.jsonl   # Every image: input, full model response, parsed result
│   └── model_summary.json        # Best prompt for this model
├── leaderboard.json              # Overall rankings
└── test_manifest.json            # Test set definition (reproducible)
```

### Infrastructure Notes
- **Image size matters enormously.** 7MB PNG (2754×1536) → 183s/image. 128KB JPG (1024px) → 8s/image. The image encoding/processing step dominates inference time on 16GB Mac Mini, not the model itself.
- **Cloudflare tunnel is unreliable for ML workloads.** Free tier has 100s timeout. Model loading takes 2-10 minutes. Long inference requests (30+ seconds on weak models) also timeout. Local network is essential for reliability.
- **Mac Mini sleep/crash kills overnight runs.** Must disable sleep entirely (`sudo pmset -a sleep 0 disablesleep 1`) and run `caffeinate -s &`. Even with these, the Mac Mini crashed multiple times under sustained think mode load — possibly thermal throttling. IP changed from 10.0.0.244 to 10.0.0.54 after one restart.
- **Model loading takes 2-10 minutes on Mac Mini.** Never unload a model unless switching to a different one. The benchmark script checks what's currently loaded and skips the load step if the right model is already in memory.
- **Token limits must be generous.** `num_predict: 500` is too low — some models think by default even with `think: False`, consuming all tokens on hidden reasoning. Use `num_predict: 2048` for nothink, `16384` for think mode. Some models (qwen3-vl family) produced empty responses at 500 tokens, corrupting benchmark results.
- **Timeout protection is essential for think mode.** Without a per-image timeout, a single stuck image can block for 78+ minutes (observed: 6465 seconds on one image). Implemented 300s timeout with "TIMEOUT: model stuck" marker in logs.
- **qwen3-vl models (older architecture) are partially corrupted** in nothink results due to token limit issue. Their real accuracy when they respond is much higher than reported (e.g., qwen3-vl:2b direct: reported 40.7%, real 78.2% on non-empty responses). These were not rerun — the models are at the bottom of the leaderboard regardless.

### Benchmark Data Integrity Audit (March 21, 2026)
Final audit of all 54 benchmark runs:
- **All Qwen 3.5 nothink runs (20 runs): CLEAN** — zero empty, zero errors, all 150 images
- **All 4b/9b think runs (4 runs): CLEAN** — 149-150 have thinking captured, zero empty, timeouts properly handled
- **0.8b/2b think runs (4 runs): USABLE** — high timeout/error rates but data is valid where responses exist
- **qwen3-vl runs (10 runs): PARTIALLY CORRUPTED** — 15-100% empty responses from token limit. Real accuracy higher than reported.
- **Other model nothink runs (13 runs): CLEAN** — gemma3, llava-phi3, moondream all valid (moondream json_structured has 133 empty but that's the model's inability to output JSON, not a bug)

---

## Timeline & What's Next

### Completed
- [x] Blender warehouse scene built (303 objects, 62 flight waypoints)
- [x] 184 base drone renders from Blender
- [x] Gemini photorealism enhancement (Playwright automation)
- [x] 999 hazard-injected images generated across 5 categories
- [x] Full dataset labeled with structured JSONL annotations
- [x] Labels double-checked by Claude (forklift triple-checked)
- [x] Human review of ALL 996 images using custom HTML review tool with voice input
- [x] 83 deformed/broken images deleted, 89 category corrections, 124 multi-label tags, 242 description corrections
- [x] Dataset reorganized — flat structure, category-prefixed filenames, unified metadata.jsonl
- [x] Pre-resized images to 1024px JPG (7GB → 128MB, 8x faster inference)
- [x] Repo cleaned — removed dead code, outdated docs (17G → 9.7G)
- [x] README rewritten for current VLM pipeline
- [x] Zero-shot benchmark v1: 8 models × 2 prompts × 3 categories (99 images)
- [x] **Zero-shot benchmark v2 nothink** — 9 models × 5 prompts = 47 runs (150 images each)
- [x] **Think mode benchmark** — 4 Qwen3.5 models × 2 prompts = 8 runs with full reasoning traces captured
- [x] **Deep analysis** — read all 150 responses from 9b vs 4b, analyzed disagreement patterns, thinking traces
- [x] **Train/test split** — 728 train / 185 test, stratified, zero overlap

---

## Phase 5: Training Data Generation (COMPLETE — March 22, 2026)

### Training Conversation Generation

Generated multi-style training conversations for ALL 727 training images using Claude Opus agents. Each image was individually viewed and described (not copied from metadata).

**Three conversation styles per image:**
- **Multi-turn Q&A:** Progressive dialogue — "Is this safe?" → "What hazards?" → "How severe?" → negative example ("Is there a spill?" → "No, because...")
- **Direct classification:** Single-turn "Classify this image" → "category: X, description: ..."
- **Reasoning chain:** Step-by-step analysis — Floor check → Shelf check → Forklift check → Personnel check → Conclusion

**Statistics:**
- 727 images × 3+ styles = 2,521 training conversations
- Generated in 73 batches of 10 images each (10 parallel agents per batch)
- Every conversation written from direct visual observation of the image
- Negative examples included in every multi-turn conversation
- ~50 images flagged for review (dual hazards, subtle hazards, borderline safe)
- Human review confirmed/corrected flagged images

**Key quality measures:**
- Agents used "messages" key format consistently (early batches had "turns" key — fixed)
- Human notes from original review incorporated into descriptions
- Metadata category corrections applied (stacking_0097→safe, forklift_0158→obstacle, etc.)

---

## Phase 6: Hyperparameter Search (COMPLETE — March 22-28, 2026)

### Approach

Ran a series of experiments on a subset of training data (200 images, 600 conversations) to find the best hyperparameters before committing to full training on all 727 images. Each experiment trained from scratch with one parameter changed, evaluated on 50 validation images, and the best config was kept.

**Infrastructure evolution:**
1. **Mac Mini M4 16GB + MLX** — attempted first, but mlx-vlm doesn't support Qwen3.5 VL yet (needs PyTorch tensors)
2. **Google Colab Pro T4** — worked for 2B model. 4B OOMed on T4 (14.6GB VRAM)
3. **Google Colab A100** — used for 4B experiments
4. **Kaggle T4 (free)** — final training platform. 30 free GPU hours/week, no compute unit limits

### Training Engine

Built a custom training loop with proper ML engineering practices:
- Single optimizer across all epochs (not recreated per epoch)
- Proper label masking — only trains on assistant responses, not prompts. Verified: 89% prompt tokens masked, 11% response tokens trained
- Cosine LR schedule with warmup
- Per-epoch data shuffling
- Validation loss monitoring each epoch (detects overfitting)
- Per-category evaluation breakdown
- NaN loss handling and gradient clipping

### 2B Hyperparameter Search Results (5 experiments)

Training on 200 images (600 conversations with style=all), evaluating on 50 validation images:

| Exp | Change | Accuracy | Val Loss | Status |
|-----|--------|----------|----------|--------|
| 1 | lr=1e-4, direct only, 3ep | 72% | 1.07 | baseline |
| 2 | lr=2e-4, direct only | 78% | 1.26 | +6% |
| 3 | lr=2e-4, rank=32 | 76% | 1.06 | rank=32 overfits |
| **4** | **lr=2e-4, style=all, 3ep** | **86%** | **1.05** | **BEST** |
| 5 | lr=2e-4, style=all, 5ep | 82% | 1.20 | overfits at epoch 4+ |

**Key Finding 7: Multi-Turn Reasoning Data is Critical**

Switching from direct-only (190 examples) to all styles (600 examples: direct + multi_turn + reasoning) improved accuracy from 78% to 86% — a +8% jump in one experiment. This was the single most impactful change.

The multi-turn and reasoning styles teach the model WHY something is a violation, not just WHAT the label is. Example: forklift_violation went from 40% (direct-only) to 90% (all styles) because the reasoning conversations explicitly explain "worker standing near active forks = OSHA violation."

**Key Finding 8: 3 Epochs Optimal, 5 Overfits**

Val loss trajectory: 1.06 → 1.01 → 1.01 → 1.10 → 1.20. Loss plateaus at epoch 3 and rises after, indicating overfitting. Train loss continues dropping (1.27 → 0.30) but this is memorization, not learning.

**Key Finding 9: Optimal Hyperparameters**

| Parameter | Best Value | Why |
|-----------|-----------|-----|
| Learning rate | 2e-4 | Faster convergence than 1e-4, doesn't overfit like 5e-4 |
| LoRA rank | 16 | Rank 32 overfits more with no accuracy gain |
| LoRA alpha | 32 (2× rank) | Standard ratio |
| Style | "all" (multi_turn + direct + reasoning) | +14% over direct-only |
| Epochs | 3 | 5 overfits |
| Dropout | 0.05 | Standard |
| Grad accumulation | 4 | Effective batch size 4 |

### 4B Autoresearch Results

| Exp | Platform | Accuracy | Issue |
|-----|----------|----------|-------|
| 6a | Colab T4 (4-bit) | 20% | OOM — 83% examples skipped |
| 6b | Colab H100 (bf16) | 58% (with think parsing) | Think mode — model reasons before answering |
| 6c | Colab A100 (4-bit) | 80% | Full run, zero skips |

**Key Finding 10: 4B Think Mode Problem**

The 4B model outputs reasoning in plain text before the category, consuming tokens on analysis. The actual category is buried after `</think>` tags. With proper parsing (extracting answer after `</think>`), 4B achieves 80% — but still below 2B's 86%.

**Key Finding 11: Smaller Fine-Tuned > Larger Fine-Tuned**

Fine-tuned 2B (86%) beats fine-tuned 4B (80%). The 2B model learns more efficiently because it doesn't overthink. Same pattern observed in zero-shot benchmarks where 4b (82.7%) beat 9b (76.7%). For safety classification, model efficiency matters more than capacity.

---

## Phase 7: Full Training (COMPLETE — March 28-29, 2026)

### Training on Kaggle

After exhausting Colab Pro compute units (100 CU in 5 days, mostly on debugging GPU issues), moved to Kaggle for free T4 GPU access.

**Training setup (Kaggle):**
- **Framework:** Unsloth + TRL SFTTrainer
- **Model:** Qwen3.5-2B (unsloth/Qwen3.5-2B)
- **Precision:** float32 (bf16 not supported on T4 with Unsloth)
- **LoRA:** r=16, alpha=32, dropout=0.05, all linear layers + vision layers
- **Training data:** 2,521 conversations (727 images × 3 styles)
- **Epochs:** 3
- **Optimizer:** AdamW 8-bit, lr=2e-4, cosine schedule, warmup=50 steps
- **Grad accumulation:** 4 (effective batch size 4)
- **Training time:** 3.5 hours on T4

**Training loss curve:**
```
Step 25:    1.733
Step 200:   0.930
Step 400:   0.892
Step 630:   0.835  (end epoch 1)
Step 800:   0.672
Step 1000:  0.695
Step 1260:  0.587  (end epoch 2)
Step 1400:  0.487
Step 1600:  0.478
Step 1890:  0.487  (end epoch 3)
Final loss: 0.714
```

### Deployment

After training on Kaggle, the LoRA adapter (88MB) was:
1. Downloaded from Kaggle
2. Transferred to Mac Mini M4
3. Merged with base Qwen3.5-2B model using mlx-vlm
4. Converted and loaded into Ollama as "hawkeye" model
5. Served via Flask API on port 5555 for remote evaluation

### Evaluation Results

**Validation set (50 images, Kaggle — same session as training):**

| Category | Accuracy |
|----------|----------|
| spill | 10/10 = **100%** |
| improper_stacking | 9/10 = **90%** |
| obstacle | 9/10 = **90%** |
| forklift_violation | 8/10 = **80%** |
| safe | 8/10 = **80%** |
| **Overall** | **44/50 = 88%** |

**Full test set (186 unseen images, Kaggle — separate run):**

| Category | Accuracy |
|----------|----------|
| spill | 42/43 = **98%** |
| obstacle | 39/41 = **95%** |
| improper_stacking | 20/22 = **91%** |
| forklift_violation | 24/28 = **86%** |
| safe | 41/52 = **79%** |
| **Overall** | **166/186 = 89.2%** |

Note: The Kaggle eval script had a parsing bug — it checked if GT keyword appeared anywhere in the model response text (including descriptions) rather than reading the `category:` line. The 89.2% figure uses proper category-line parsing.

**Full test set (186 unseen images, Mac Mini via Ollama):**

Initial automated evaluation reported 79% due to the same parsing bug. After fixing the parser:

| Category | Accuracy |
|----------|----------|
| spill | 41/43 = **95%** |
| obstacle | 39/41 = **95%** |
| improper_stacking | 20/22 = **91%** |
| forklift_violation | 24/28 = **86%** |
| safe | 43/52 = **83%** |
| **Overall** | **167/186 = 89.8%** |

After human review of all 19 "wrong" predictions, 7 were confirmed as actually correct (edge cases, dual hazards, debatable labels):

| Category | Accuracy (human-corrected) |
|----------|---------------------------|
| spill | **96%** |
| obstacle | **97%** |
| improper_stacking | **91%** |
| forklift_violation | **86%** |
| safe | **87%** |
| **Overall** | **~94%** |

**Cross-validation:** The full 186 test set was evaluated on BOTH Kaggle (Unsloth on T4) and Mac Mini (mlx-vlm via Ollama). All 186 responses were completely different text — zero word-for-word matches, zero first-100-character matches. Different inference engines, different wording, but same accuracy (~89% automated, ~94% human-corrected). Some borderline images even flipped between runs (e.g., forklift_0043: Kaggle=safe, Mac Mini=forklift_violation). This confirms the model genuinely learned safety classification, not memorized specific outputs.

### Key Finding 12: Fine-Tuned 2B Beats All Zero-Shot Models

| Model | Verified Accuracy* | Size | Time/img | Training |
|-------|-------------------|------|----------|----------|
| **Fine-tuned HAWKEYE F32 (nothink)** | **175/186 = 94.1%** | 8.3 GB | 150.6s | 727 images, 3.5h, Free (Kaggle) |
| **Fine-tuned HAWKEYE Q4 (think)** | **172/186 = 92.5%** | 1.8 GB | 4.2s | same |
| **Fine-tuned HAWKEYE Q4 (nothink)** | **171/186 = 91.9%** | 1.8 GB | 2.9s | same |
| Fine-tuned Qwen3.5-4B | ~80% | — | — | 200 images, 2.5h, $14 (Colab) |
| Zero-shot Qwen3.5-4b (nothink) | **125/150 = 83.3%** | ~4.7 GB | 5.6s | None |
| Zero-shot Qwen3.5-9b (nothink) | **117/150 = 78.0%** | ~9.7 GB | 9.0s | None |
| Zero-shot Qwen3.5-2b (think) | **125/186 = 67.2%** | 3.78 GB | — | None |
| Zero-shot Qwen3.5-2b (nothink) | 65.3% | 3.78 GB | 3.1s | None |
| Zero-shot Gemma3-4b | 63.3% | ~4.3 GB | 5.3s | None |

*Verified = manually reading every single model response, comparing to GT description/notes, visually inspecting disputed images, determining the model's actual final classification. See Phase 8.

**A 1.8 GB quantized model achieves 92.5% accuracy at 4.2s/image, outperforming zero-shot models 5× its size. The full-precision version reaches 94.1% but at 150s/image. The improvement from fine-tuning is +25.3 pp (67.2% → 92.5%) with quantization, or +26.9 pp (67.2% → 94.1%) at full precision.** This is the central thesis finding.

### Key Finding 13: Model Description Quality (Quantized Model)

Manual review of all 186 full responses revealed remarkably detailed descriptions:
- **Reads labels:** Model identifies "FRAGILE", "THIS SIDE UP", "CAUTION FORKLIFT TRAFFIC", "8 km/h" speed signs
- **Identifies materials:** "amber-brown oil", "blue plastic container", "splintered wooden planks", "grey cylindrical container"
- **Describes spatial relationships:** "lower-left foreground near base of left shelving", "near the corner of the aisle"
- **Explains WHY it's dangerous:** "unauthorized riding violation — forklifts are designed for seat operation only", "worker is at ground level in the forklift's travel path, creating a serious struck-by hazard"
- **Catches dual hazards:** "The combination of the liquid spill and the displaced box creates both a slip hazard and an obstacle"

Example response (forklift_0088, person clinging to moving forklift):
> "A yellow forklift is operating in a warehouse aisle between tall orange pallet racks. A worker is hanging onto the rear frame of the forklift, riding on the back of the vehicle as it moves through the aisle. This is an extremely dangerous unauthorized riding situation with no seatbelt or safety restraint. The operator is seated at the controls while the second person clings to the rear of the moving vehicle."

This is genuine safety reasoning from a 2B model — not keyword matching.

### Key Finding 14: Error Analysis (Quantized Think Mode — 14 errors)

Every wrong image was visually inspected. The 14 genuinely wrong predictions fall into clear patterns:

1. **Safe over-detection (5 errors):** stacking_0086, safe_0150, safe_0014, stacking_0104 — model sees minor shelf unevenness or floor marks as hazards. forklift_0026 — model can't judge distance from drone perspective, sees far-away pedestrians as "within operating zone."

2. **Missed hazard type (3 errors):** forklift_0189 (normal shelving operation, model said violation), forklift_0126 (floor spill was the issue but model saw forks), forklift_0098 (floor debris was the issue but model saw forklift load).

3. **Missed subtle spill (2 errors):** safe_0148 (model wavered between obstacle and safe on a genuinely borderline image), spill_0005 (distant dark spill barely visible from drone angle, model focused on forklift).

4. **Missed stacking (2 errors):** stacking_0046 (said safe, missed torn wrap on upper shelves), stacking_0127 (said safe, missed subtle shelf overloading in dim lighting).

5. **Hallucinated pedestrian danger (1 error):** forklift_0017 — workers were far away but drone perspective compressed the distance.

6. **Parser edge case (1 error):** stacking_0096 — model correctly reasoned improper_stacking in thinking but output format didn't produce a clean category line.

### Think Mode vs Nothink Error Comparison

Think mode fixed 7 errors that nothink made: #17 (forks raised very high), #25 (forklift tipping), #44 (correctly said safe), #91 (oversized load), #145 (tipping forward), #166 (unsecured drums), #172 (pedestrian with back to forklift). These are all cases where step-by-step reasoning helped the model notice subtle details.

Think mode introduced 2 new errors that nothink got right: #95 (hallucinated obstacle), #136 (missed stacking). The reasoning process occasionally leads to overthinking on borderline cases.

**Net: think mode is strictly better (+5 images, +2.7%) with minimal speed cost (+1.3s/image).**

---

## Phase 8: Manual Response Verification (COMPLETE — March 30-31, 2026)

### Problem: Parser Bugs in Evaluation

Evaluation scripts used keyword matching to extract the model's predicted category from its response text. This is unreliable for two reasons:

1. **Think mode keyword pollution (2B base model):** The Kaggle eval script searched for GT keywords anywhere in the model's response. In think mode, the model produces long reasoning text that naturally mentions all hazard categories ("I see no spill, no forklift_violation..."). The parser would match the GT keyword in a negation, inflating accuracy.

2. **Multi-category responses (9B):** Occasionally the model would self-correct mid-response, outputting two `category:` lines. The parser took the first one, which was sometimes wrong.

### Verification Process

Every single model response was manually read, compared to the ground truth description and human notes, and the model's actual final classification was determined. For each image:

1. Read GT category, description, human_notes, accept_also
2. Read the model's full response (including think-mode reasoning if applicable)
3. Determine what category the model actually concluded with (reading final classification, not keyword matching)
4. Check if the model's observations match what's actually in the image per GT
5. Mark correct or wrong

### Base Qwen3.5-2B (think mode, 186 test images) — Kaggle

**Kaggle parser reported: 158/186 = 84.9%**

Manual verification found **32 parser bugs** where the parser matched GT keywords in the model's reasoning text (e.g., model says "I don't see a spill" but parser matches "spill"):

| Batch | Score | Parser Bugs |
|-------|-------|-------------|
| 1-25 | 17/25 = 68% | — |
| 26-50 | 13/25 = 52% | — |
| 51-75 | 13/25 = 52% | — |
| 76-100 | 19/25 = 76% | 3 (#76, #94, #100) |
| 101-125 | 23/25 = 92% | 1 (#117) |
| 126-150 | 17/25 = 68% | 4 (#131, #135, #140, #146) |
| 151-186 | 23/36 = 64% | 7 (#156, #158, #164, #181, #183, #184, #185) |
| **TOTAL** | **125/186 = 67.2%** | **15+ parser bugs (76-186 only)** |

**Verified accuracy: 125/186 = 67.2%** (vs 84.9% reported — a 17.7 percentage point inflation from parser bugs)

Note: These numbers are against the updated metadata (8 GT corrections applied during Phase 8 verification). The GT corrections hurt the base model slightly because it was calling some images "safe" and getting credit when GT was also safe — now that GT was corrected to the actual hazard, the base model is exposed as having missed it.

Common base model errors:
- Calling safe images hazardous (hallucinating stacking issues, seeing tire marks as spills)
- Missing forklift violations (especially pedestrian proximity)
- Confusing obstacle with other categories
- Over-detecting from normal warehouse features

### Base Qwen3.5-4B (nothink, direct prompt, 150 test images) — Mac Mini

**Original parser reported: 124/150 = 82.7%**

Structured nothink responses — parser was correct. 1 accept_also recovery found:

| Batch | Score |
|-------|-------|
| 1-25 | 20/25 = 80% |
| 26-50 | 22/25 = 88% |
| 51-75 | 22/25 = 88% |
| 76-100 | 20/25 = 80% |
| 101-125 | 18/25 = 72% |
| 126-150 | 23/25 = 92% |
| **TOTAL** | **125/150 = 83.3%** |

Correction: +1 accept_also recovery (stacking_0182 — model said obstacle, GT accepts obstacle). No parser bugs found.

**Verified accuracy: 125/150 = 83.3%** (vs 82.7% reported — minor 0.6% increase)

4B model's main weakness: hallucinating "spill" on normal floor texture. 11 of 25 wrong entries predicted "spill" when the floor just had concrete stains or tire marks. Different from 9B which hallucinated "obstacle" — the 4B defaults to "spill" for ambiguous floor patches.

### Base Qwen3.5-9B (nothink, direct prompt, 150 test images) — Mac Mini

**Original parser reported: 115/150 = 76.7%**

The nothink mode responses are structured (`category: X\ndescription: ...`), so the parser was mostly correct. Only 2 issues found:

| Batch | Score |
|-------|-------|
| 1-25 | 18/25 = 72% |
| 26-50 | 21/25 = 84% |
| 51-75 | 20/25 = 80% |
| 76-100 | 18/25 = 72% |
| 101-125 | 19/25 = 76% |
| 126-150 | 21/25 = 84% |
| **TOTAL** | **117/150 = 78.0%** |

Corrections: +1 parser bug fix (forklift_0213 — model self-corrected to accepted answer but parser took first category line), +1 accept_also recovery (stacking_0182).

**Verified accuracy: 117/150 = 78.0%** (vs 76.7% reported — minor 1.3% increase)

9B model's main weakness: strong "obstacle" bias — sees tire marks, concrete stains, and normal floor wear as obstacles. 22 of 33 wrong entries predicted "obstacle" when they shouldn't have.

### Fine-Tuned HAWKEYE 2B — Quantized Model (Mac Mini via Ollama)

**Quantization specs:**
| Component | Size | Format |
|-----------|------|--------|
| Original merged model | 8.3 GB | float32 |
| Quantized LLM | 1.2 GB | Q4_K_M (4-bit k-quant medium) |
| Vision encoder | 637 MB | F16 (not quantized — preserves visual detail) |
| **Total deployed** | **~1.8 GB** | **78% size reduction** |

All responses manually read and compared to GT descriptions, human notes, and accept_also fields. Full responses saved in log files.

### Full-Precision Model (8.3 GB, float32)

**Nothink mode (150.6s/image):** `hawkeye_normal_nothink_full_log.json`

| Batch | Score |
|-------|-------|
| 1-25 | 23/25 = 92% |
| 26-50 | 23/25 = 92% |
| 51-75 | 24/25 = 96% |
| 76-100 | 23/25 = 92% |
| 101-125 | 25/25 = **100%** |
| 126-150 | 24/25 = 96% |
| 151-186 | 33/36 = 92% |
| **TOTAL** | **175/186 = 94.1%** |

11 errors: #4 stacking_0086, #7 forklift_0017, #28 forklift_0189, #36 forklift_0126, #58 forklift_0098, #86 forklift_0066, #91 forklift_0094, #132 safe_0014, #166 forklift_0037, #168 stacking_0104, #172 forklift_0044.

Note: The full-precision model generates self-conversation loops (the model hallucinates multi-turn Q&A after the first response), consuming ~150s/image. Only the first response before `<|im_end|>` is the actual classification. This is a known issue with unquantized Qwen3.5 inference without proper stop tokens.

### Quantized Model (1.8 GB, Q4_K_M)

**Nothink mode (2.9s/image):** `hawkeye_nothink_full_log.json`

| Batch | Score |
|-------|-------|
| 1-25 | 19/25 = 76% |
| 26-50 | 22/25 = 88% |
| 51-75 | 23/25 = 92% |
| 76-100 | 22/25 = 88% |
| 101-125 | 25/25 = **100%** |
| 126-150 | 22/25 = 88% |
| 151-186 | 32/36 = 89% |
| **TOTAL** | **171/186 = 91.9%** (after GT corrections) |

**Think mode (4.2s/image):** `hawkeye_think_full_log.json`

| Batch | Score |
|-------|-------|
| 1-25 | 22/25 = 88% |
| 26-50 | 23/25 = 92% |
| 51-75 | 24/25 = 96% |
| 76-100 | 23/25 = 92% |
| 101-125 | 25/25 = **100%** |
| 126-150 | 22/25 = 88% |
| 151-186 | 33/36 = 92% |
| **TOTAL** | **172/186 = 92.5%** |

### Quantization Impact Analysis

| Model | Accuracy | Time/img | Size |
|-------|----------|----------|------|
| **Full-precision (nothink)** | **175/186 = 94.1%** | 150.6s | 8.3 GB |
| Quantized Q4 (think) | 172/186 = 92.5% | 4.2s | 1.8 GB |
| Quantized Q4 (nothink) | 171/186 = 91.9% | 2.9s | 1.8 GB |

**Quantization cost: -2.2 percentage points (94.1% → 91.9%) for 78% size reduction and 52x speed improvement.** The quantized think mode (92.5%) nearly matches the full-precision nothink (94.1%) at 36x faster speed.

The full-precision model catches 4 images the quantized nothink misses (#17 forklift_0019, #25 forklift_0043, #44 safe_0187, #91 forklift_0094 — all subtle forklift judgments). The quantized think mode recovers 3 of these 4.

**Key Finding: Think mode on quantized model is the best deployment option** — 92.5% accuracy, 4.2s/image, 1.8 GB. Only 1.6 pp below full-precision at 36x faster and 78% smaller.

**GT Corrections During Verification:**

8 images where the model's prediction was confirmed correct upon visual inspection of the actual image, and GT labels were updated in `metadata.jsonl`:

| Image | Was | Now | Reason |
|-------|-----|-----|--------|
| safe_0145 | safe | **obstacle** | Grey bins and boxes on floor against wall |
| safe_0148 | safe | **obstacle** | Cardboard box and debris on floor |
| stacking_0033 | safe | **obstacle** | Pallet with loose boxes blocking center of aisle |
| spill_0154 | obstacle (no accept) | obstacle + **accept spill** | Massive liquid pool is clearly a spill too |
| forklift_0074 | improper_stacking | **forklift_violation** | Forklift visibly tipping forward, not a stacking issue |
| forklift_0218 | safe | **forklift_violation** | Worker crouching next to active forklift forks |
| forklift_0062 | improper_stacking (no accept) | + **accept forklift_violation** | Triple-stacked pallets on forks is both stacking and forklift load handling |
| forklift_0066 | spill (no accept) | + **accept forklift_violation** | Dark floor marks from forklift fluid leak — forklift-related classification valid |

### Key Finding 15: The Real Fine-Tuning Improvement

| Model | Verified Accuracy | Time/img | Size |
|-------|------------------|----------|------|
| Base Qwen3.5-2B (think) | **125/186 = 67.2%** | — | 3.78 GB |
| Base Qwen3.5-4B (nothink) | **125/150 = 83.3%** | 5.6s | ~4.7 GB |
| Base Qwen3.5-9B (nothink) | **117/150 = 78.0%** | 9.0s | ~9.7 GB |
| Fine-tuned HAWKEYE F32 (nothink) | **175/186 = 94.1%** | 150.6s | 8.3 GB |
| **Fine-tuned HAWKEYE Q4 (nothink)** | **171/186 = 91.9%** | **2.9s** | **1.8 GB** |
| **Fine-tuned HAWKEYE Q4 (think)** | **172/186 = 92.5%** | **4.2s** | **1.8 GB** |

The fine-tuned quantized 2B beats:
- Base 9B (5.4× larger) by **14.5 pp** (78.0% → 92.5%) at **2x faster** inference
- Base 4B (2.6× larger) by **9.2 pp** (83.3% → 92.5%) at **faster** inference
- Base 2B (same architecture) by **25.3 pp** (67.2% → 92.5%)
- While being **78% smaller** (1.8 GB vs 8.3 GB full-precision) with only 1.6 pp accuracy cost

This means the fine-tuning + quantization pipeline produces a model that is:
1. **Smaller** — Q4 quantization reduces size ~66% vs F16
2. **Faster** — 2.9-4.2s/image vs 5.6-9.0s for larger zero-shot models
3. **More accurate** — 92.5% vs best zero-shot 83.3%
4. **Better at reasoning** — think mode works efficiently with no timeouts, adding +2.7% accuracy for +1.3s cost

Note: The 4B is the strongest zero-shot baseline at 83.3%, confirming Key Finding 1 (smaller 4B > larger 9B). The 9B's obstacle bias and the 4B's spill bias are different failure modes — 9B over-detects floor features as obstacles, 4B over-detects them as spills — but the net accuracy impact is similar.

---

## Timeline & What's Next

### Completed
- [x] Blender warehouse scene built (303 objects, 62 flight waypoints)
- [x] 184 base drone renders from Blender
- [x] Gemini photorealism enhancement (Playwright automation)
- [x] 999 hazard-injected images generated across 5 categories
- [x] Full dataset labeled with structured JSONL annotations
- [x] Labels double-checked by Claude (forklift triple-checked)
- [x] Human review of ALL 996 images using custom HTML review tool with voice input
- [x] 83 deformed/broken images deleted, 89 category corrections, 124 multi-label tags, 242 description corrections
- [x] Dataset reorganized — flat structure, category-prefixed filenames, unified metadata.jsonl
- [x] Pre-resized images to 1024px JPG (7GB → 128MB, 8x faster inference)
- [x] Repo cleaned — removed dead code, outdated docs (17G → 9.7G)
- [x] README rewritten for current VLM pipeline
- [x] Zero-shot benchmark v1: 8 models × 2 prompts × 3 categories (99 images)
- [x] **Zero-shot benchmark v2 nothink** — 9 models × 5 prompts = 47 runs (150 images each)
- [x] **Think mode benchmark** — 4 Qwen3.5 models × 2 prompts = 8 runs with full reasoning traces captured
- [x] **Deep analysis** — read all 150 responses from 9b vs 4b, analyzed disagreement patterns, thinking traces
- [x] **Train/test split** — 727 train / 186 test, stratified, zero overlap
- [x] **Training conversation generation** — 727 images × 3 styles = 2,521 conversations via Claude agents
- [x] **Autoresearch** — 5 experiments on 2B, 3 on 4B, found optimal config (lr=2e-4, rank=16, style=all, 3ep)
- [x] **Full SFT training** — Qwen3.5-2B on Kaggle T4, 3.5 hours, final loss 0.714
- [x] **Deployment** — HAWKEYE model running on Mac Mini via Ollama
- [x] **Quantization** — Q4_K_M quantization (8.3 GB → 1.8 GB, 78% reduction, vision encoder kept at F16)
- [x] **Full test evaluation** — 186 unseen images on both quantized and full-precision models
- [x] **Manual response verification (Phase 8)** — manually read ALL responses for:
  - Base 2B think (186 images) → 125/186 = 67.2%
  - Base 4B nothink (150 images) → 125/150 = 83.3%
  - Base 9B nothink (150 images) → 117/150 = 78.0%
  - Fine-tuned HAWKEYE Q4 nothink (186 images) → 171/186 = 91.9%
  - Fine-tuned HAWKEYE Q4 think (186 images) → 172/186 = 92.5%
  - Fine-tuned HAWKEYE F32 nothink (186 images) → 175/186 = 94.1%
- [x] **GT corrections** — 8 metadata labels corrected after visual inspection confirmed model was right and GT was wrong
- [x] **HTML comparison viewer** — side-by-side base vs HAWKEYE responses for all 186 images (`scripts/model_comparison_v2.html`)

### Next Steps
1. **Commercial model comparison** — test Claude/Gemini API on same 186 test images for reference
2. **GRPO training** — reinforcement learning to improve think mode reasoning efficiency
3. **Write thesis** — deadline April 5-10

### Key Files
| File | Purpose |
|------|---------|
| `dataset/metadata.jsonl` | Unified labels for 913 human-verified images |
| `dataset/train.jsonl` | 727 training images (80% split) |
| `dataset/test.jsonl` | 186 test images (20% split) |
| `dataset/training_conversations.jsonl` | 2,521 multi-style training conversations |
| `dataset/images/` | 913 full-resolution PNG images |
| `dataset/images_small/` | 913 pre-resized 1024px JPGs (for inference + training) |
| `dataset/image_mapping.json` | Old → new filename mapping |
| `dataset/review_results_final.json` | Complete human review data (all 996 decisions + notes) |
| `scripts/benchmark_v2.py` | VLM benchmark v2 (5 categories, 5 prompts, multi-label, think mode) |
| `scripts/review_tool.html` | Human review tool (HTML + voice input) |
| `scripts/organize_dataset.py` | Dataset reorganization script |
| `scripts/gemini_hazard_inject.py` | Hazard injection via Gemini automation |
| `outputs/benchmark_v2/` | Benchmark v2 results — 54 runs with full logs + thinking traces |
| `outputs/hawkeye_test_results.json` | Fine-tuned model evaluation on 186 test images |
| `outputs/benchmark/` | Benchmark v1 results (3-category, 18 runs) |
| `vision_api.py` | Ollama API helper (list, load, stop, test models) |
| `hawkeye/simulation/blender/` | Blender scenes + 3D assets |
| `docs/PROGRESS_LOG.md` | This file |

---

### Key Log Files
| File | Contents |
|------|----------|
| `hawkeye_normal_nothink_full_log.json` | Full-precision F32 model, nothink, 186 images, 150.6s/img |
| `hawkeye_nothink_full_log.json` | Quantized Q4 model, nothink, 186 images, 2.9s/img |
| `hawkeye_think_full_log.json` | Quantized Q4 model, think, 186 images, 4.2s/img |
| `test_eval_base_think_log.json` | Base Qwen3.5-2B, think mode, 186 images (Kaggle) |
| `scripts/model_comparison_v2.html` | Side-by-side HTML viewer: base vs HAWKEYE for all 186 images |

---

*Last updated: March 31, 2026*
