# Synthetic Specialization for Warehouse Safety Hazard Detection

A 1.8 GB vision-language model, trained on AI-generated synthetic warehouse data, that runs offline on a Mac Mini and matches GPT-5.5 and Gemini 3.1 Pro on the hardest forklift-violation cases.

Research project at the University of Guelph supervised by Dr. John Akinyemi. The contribution is the pipeline (Blender, photorealistic enhancement, hazard injection, LoRA, quantization), not the warehouse model itself. The same approach generalizes to any domain where real data is scarce but the deployment context is known.

## Headline Numbers

Held-out test set, n=186 images across five hazard categories. Inference on a Mac Mini M4 via Ollama.

| Model                                       | Accuracy | Size    | Latency  |
| ------------------------------------------- | -------- | ------- | -------- |
| Trained 2B, Q4 no-think (deployed)          | 91.9%    | 1.8 GB  | ~3 s     |
| Trained 2B, Q4 think                        | 92.5%    | 1.8 GB  | ~4.2 s   |
| Trained 2B, F32 no-think (full precision)   | 94.1%    | 8.3 GB  | n/a      |
| Qwen3.5-VL-2B base (zero-shot, same arch)   | 65.3%    | 3.78 GB | ~3.1 s   |

Training delta on the deployed model: **+26.6 percentage points**, same base, same prompt, same hardware.

## Frontier Comparison on the Hard Set

12 forklift-violation images where workers stand inside the operating zone of an active forklift. Same direct-classification prompt sent through OpenRouter, May 2026.

| Model                              | Accuracy | Notes                          |
| ---------------------------------- | -------- | ------------------------------ |
| **Trained 2B (this repo)**         | **12/12** | 1.8 GB, runs offline           |
| GPT-5.5                            | 12/12    | Frontier, hosted               |
| GPT-5.4 / GPT-5.4-mini             | 12/12    | Frontier, hosted               |
| Gemini 3.1 Pro                     | 12/12    | Frontier, hosted               |
| Qwen3.6-plus                       | 12/12    | Frontier-tier, hosted          |
| Grok-4.20                          | 11/12    | Frontier-tier, hosted          |
| Qwen3.5-397B-a17b                  | 9/12     | Open-source flagship           |
| Qwen3-VL-235B (thinking)           | 8/12     | Open-source reasoning, 235B    |
| Pixtral-Large                      | 8/12     | Open-source                    |
| GLM-4.6v                           | 6/12     | Open-source                    |
| Llama-4-Maverick                   | 4/12     | Open-source                    |
| Kimi-k2.6                          | 4/12     | Open-source                    |
| Llama-4-Scout                      | 2/12     | Open-source                    |

Full leaderboard with per-image responses: [`outputs/benchmark_openrouter/LEADERBOARD.md`](outputs/benchmark_openrouter/LEADERBOARD.md).

## Pipeline

1. **3D environment.** A warehouse scene in Blender 5.0.1 with ~3,300 objects across roughly 100 by 60 metres, including pallet racking, shelving, a forklift, worker models, boxes, crates, and bins.
2. **Drone-perspective rendering.** A Python script flies a virtual camera through nine aisles at ~3.5 m, pitched 55 degrees down, capturing frames every 2 m in a serpentine path. 184 base renders.
3. **Photorealism (Nano Banana Pro / Gemini).** Each raw render is rewritten into a photorealistic image while preserving camera angle, layout, and object positions.
4. **Hazard injection.** Prompted image editing injects realistic hazards (spills, forklift violations, improper stacking, aisle obstructions) into clean photoreal frames. 999 generated, 913 retained after manual review.
5. **LoRA fine-tuning.** Qwen3.5-VL-2B base, trained on 2,521 multi-style conversations from 727 images. 3.5 hours on a free Kaggle T4 GPU.
6. **Quantization.** Merge LoRA into base, then quantize language weights to 4-bit (Q4_K_M) via llama.cpp. Vision encoder kept at F16. Result: 1.8 GB GGUF, runs in Ollama.

Total project cost: ~$25.

## Quick Start

Run the trained model in Ollama:

```bash
# Install Ollama: https://ollama.com
# Pull the GGUF straight from HuggingFace
ollama pull hf.co/Jsohal174/hawkeye-warehouse-safety-gguf

# Classify an image
ollama run hf.co/Jsohal174/hawkeye-warehouse-safety-gguf \
  "Classify this warehouse image into exactly one of: spill, forklift_violation, improper_stacking, obstacle, safe. Respond with the category and a one-line description." \
  /path/to/image.jpg
```

For the ~3 s no-think configuration, prefix the prompt with `/no_think`.

## Dataset

913 labeled images across five hazard categories, post-review.

| Category            | Images |
| ------------------- | ------ |
| safe                | 256    |
| spill               | 214    |
| obstacle            | 200    |
| forklift_violation  | 137    |
| improper_stacking   | 106    |

Each image carries structured metadata (category, subtype, severity, location, free-text description, multi-label tags). Forklift violations include 10 OSHA-derived subtypes such as `pedestrian_proximity`, `unauthorized_rider`, `forks_raised_traveling`. Every image was reviewed by hand using a custom voice-to-text labeling tool.

Dataset metadata and the splits (`train.jsonl`, `test.jsonl`) live in [`dataset/`](dataset/). The full image files are hosted on HuggingFace.

## Released Artifacts

- **Model (Q4, Ollama-ready, 1.8 GB):** https://huggingface.co/Jsohal174/hawkeye-warehouse-safety-gguf
- **Dataset (913 labeled images):** https://huggingface.co/datasets/Jsohal174/warehouse-safety-hazard-dataset
- **GitHub (this repo):** https://github.com/Jsohal174/safety-hazard-detection

## Reproducing the Benchmarks

### Trained model evaluation

```bash
# Inference on the held-out test set, scored against ground truth
python scripts/benchmark_v2.py --model trained-2b-q4 --prompt direct
```

### Frontier benchmark via OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python scripts/benchmark_openrouter.py
# Resume mid-run from per-image logs:
python scripts/benchmark_openrouter.py --resume
```

The 12 hard-set images are listed in [`scripts/benchmark_openrouter.py`](scripts/benchmark_openrouter.py). Per-image logs and per-model results land in `outputs/benchmark_openrouter/<model>/`.

### Training

LoRA training scripts and the prepared conversation dataset are under [`autoresearch/`](autoresearch/). The training config is in [`configs/lora/`](configs/lora/). The notebook target is a free Kaggle T4 GPU.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── scripts/                  # Benchmarking, dataset prep, generation tools
│   ├── benchmark_v2.py             # Trained model + zero-shot Ollama benchmark
│   ├── benchmark_openrouter.py     # 13-model OpenRouter frontier benchmark
│   ├── organize_dataset.py
│   ├── gemini_automate.py          # Photorealistic enhancement automation
│   ├── gemini_hazard_inject.py     # Prompted hazard injection
│   ├── review_tool.html            # Voice-to-text labeling tool
│   └── ...
├── autoresearch/             # LoRA training (Kaggle T4)
│   ├── prepare.py
│   ├── train.py
│   └── program.md
├── cluster_scripts/          # Cluster job helpers
├── configs/
│   └── lora/                       # LoRA hyperparameters
├── dataset/
│   ├── metadata.jsonl              # Per-image labels and metadata
│   ├── train.jsonl / test.jsonl    # Splits
│   ├── training_conversations.jsonl  # 2,521 conversations from 727 images
│   ├── image_mapping.json
│   └── review_results_final.json
├── outputs/
│   ├── benchmark_openrouter/       # Frontier benchmark results
│   ├── benchmark_v2/               # Trained-model benchmark results
│   └── hawkeye_test_results_fixed.json
├── hawkeye/
│   └── simulation/blender/scripts/ # Render pipeline
└── docs/
    ├── paper.tex                   # Research paper source
    └── PROGRESS_LOG.md             # Build log
```

## Citation

```bibtex
@misc{sohal2026synthetic,
  author = {Sohal, Jaskirat Singh},
  title  = {Synthetic Specialization: Fine-Tuning and Compressing Foundation Models for Domain-Specific Tasks. A Case Study in Warehouse Safety Hazard Detection},
  year   = {2026},
  howpublished = {University of Guelph},
  note   = {Supervisor: Dr. John Akinyemi},
  url    = {https://github.com/Jsohal174/safety-hazard-detection}
}
```

## Acknowledgments

Supervised by Dr. John Akinyemi, University of Guelph.

Inference on consumer hardware is enabled by [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), and [Unsloth](https://github.com/unslothai/unsloth). The frontier benchmark uses [OpenRouter](https://openrouter.ai).
