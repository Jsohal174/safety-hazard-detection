# HAWKEYE: Benchmarking VLMs for Warehouse Safety Hazard Detection

## Project Overview

Evaluating open-source Vision Language Models (VLMs) for warehouse safety hazard detection using synthetic images. We generate warehouse images with injected hazards, benchmark multiple VLMs zero-shot with different prompt strategies, then LoRA fine-tune the best performer and measure improvement.

**Research Question:** "How well do open-source VLMs detect warehouse safety hazards zero-shot, and does LoRA fine-tuning on synthetic images improve their performance?"

## Owner

Jaskirat Singh Sohal (Jas)

- CIS\*4900 research project with Prof. John Akinyemi, University of Guelph
- Deadline: April 5-10, 2026

## Pipeline

```
1. BASE RENDERS (Blender)
   Warehouse scene → Drone-perspective renders (60-80 clean images)

2. HAZARD INJECTION (Gemini API / Stable Diffusion img2img)
   Clean renders → Add hazards → 300 images (60 per category)

3. ZERO-SHOT VLM BENCHMARK
   4 VLMs × 3 prompt strategies × 300 images = 3,600 evaluations
   → Precision, Recall, F1, BERTScore per model/strategy

4. LORA FINE-TUNING
   Best VLM + LoRA on 240 train images → Evaluate on 60 test
   → Compare fine-tuned vs zero-shot

5. THESIS
   Results, analysis, figures → Research paper
```

## Hazard Categories

| ID  | Category             | Description                                           |
| --- | -------------------- | ----------------------------------------------------- |
| 0   | `spill`              | Liquid puddle on warehouse floor                      |
| 1   | `missing_ppe`        | Worker without hard hat                               |
| 2   | `forklift_violation` | Unsafe forklift operation (no seatbelt, forks raised) |
| 3   | `improper_stacking`  | Unstable/overloaded boxes or pallets                  |
| 4   | `safe`               | Clean warehouse, no hazards                           |

**Target:** 300 images total, 60 per category.

## VLMs to Benchmark

| Model            | Size | Source          |
| ---------------- | ---- | --------------- |
| Qwen 2.5 VL      | 7B   | Alibaba/Qwen    |
| LLaMA 3.2 Vision | 11B  | Meta            |
| InternVL 2.5     | ~8B  | Shanghai AI Lab |
| Gemma 3          | 12B  | Google          |

## Prompt Strategies

1. **Direct** — Classify this image into one category
2. **Descriptive** — Describe hazards, then classify
3. **Chain-of-thought** — Step-by-step safety analysis, then classify

## Tech Stack

| Category         | Tool                                                    |
| ---------------- | ------------------------------------------------------- |
| 3D Rendering     | Blender 5.0.1                                           |
| Image Generation | Gemini API, Stable Diffusion (diffusers)                |
| VLM Inference    | HuggingFace Transformers                                |
| Fine-tuning      | PEFT (LoRA)                                             |
| Quantization     | bitsandbytes (4-bit)                                    |
| Evaluation       | bert-score, scikit-learn                                |
| Config           | Hydra                                                   |
| Tracking         | Weights & Biases                                        |
| Compute          | MacBook Pro M2 16GB (dev) + Google Colab Pro (training) |

## Project Structure

```
hawkeye/
├── simulation/blender/          # Blender assets, scenes, scripts
│   ├── assets/                  # 3D models (shelves, boxes, forklift, etc.)
│   ├── scenes/                  # .blend files
│   └── scripts/                 # Blender Python scripts
├── data/                        # Dataset management
│   ├── hazard_injector.py       # Gemini/SD img2img hazard injection
│   ├── dataset.py               # Dataset class + splits
│   └── prompts.py               # Hazard injection prompts
├── evaluation/                  # VLM benchmarking
│   ├── vlm_runner.py            # Load & run VLMs
│   ├── prompt_strategies.py     # 3 prompt strategies
│   ├── response_parser.py       # Parse VLM output → category
│   ├── metrics.py               # BERTScore, P/R/F1
│   └── results_analyzer.py      # Tables, plots, confusion matrices
├── finetuning/                  # LoRA fine-tuning
│   ├── prepare_lora_data.py     # Format dataset for training
│   └── lora_trainer.py          # LoRA training script
configs/
├── config.yaml                  # Main config
├── vlm/                         # Per-model configs
├── prompts/                     # Prompt strategy configs
├── lora/                        # LoRA training config
scripts/
├── 01_render_base_images.py     # Generate clean warehouse renders
├── 02_inject_hazards.py         # Add hazards via AI editing
├── 03_benchmark_vlms.py         # Run VLM evaluation
├── 04_finetune_lora.py          # LoRA fine-tuning
├── 05_evaluate_finetuned.py     # Evaluate fine-tuned model
├── 06_generate_report.py        # Results figures/tables
notebooks/
├── vlm_benchmark.ipynb          # Colab: heavy inference
├── lora_finetuning.ipynb        # Colab: training
outputs/
├── renders/base/                # Clean Blender renders
├── datasets/images/             # Hazard-injected images (by category)
├── results/                     # Benchmark results + analysis
├── checkpoints/                 # LoRA adapter weights
```

## Commands

```bash
# Step 1: Render base warehouse images (requires Blender)
python scripts/01_render_base_images.py

# Step 2: Inject hazards into base renders
python scripts/02_inject_hazards.py

# Step 3: Run VLM benchmark
python scripts/03_benchmark_vlms.py

# Step 4: LoRA fine-tune best model
python scripts/04_finetune_lora.py

# Step 5: Evaluate fine-tuned model
python scripts/05_evaluate_finetuned.py

# Step 6: Generate report figures
python scripts/06_generate_report.py
```

## Timeline (8 weeks)

| Week | Dates        | Task                                   |
| ---- | ------------ | -------------------------------------- |
| 1    | Feb 18-24    | Restructure repo + base renders        |
| 2    | Feb 25-Mar 3 | Hazard injection pipeline → 300 images |
| 3    | Mar 4-10     | Build VLM evaluation framework         |
| 4    | Mar 11-17    | Run full benchmark on Colab            |
| 5    | Mar 18-24    | Analysis + LoRA data prep              |
| 6    | Mar 25-31    | LoRA fine-tuning + evaluation          |
| 7    | Apr 1-7      | Write thesis                           |
| 8    | Apr 7-10     | Buffer + submission                    |

## Current Progress

- [x] Blender warehouse scene built (warehouse_assembled.blend)
- [x] 3D assets collected (boxes, shelves, forklift, human, PPE)
- [x] Project restructured for VLM benchmarking
- [ ] Base warehouse renders from drone perspective
- [ ] Hazard injection pipeline
- [ ] VLM evaluation framework
- [ ] Zero-shot benchmark
- [ ] LoRA fine-tuning
- [ ] Thesis
