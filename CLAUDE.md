# Warehouse Safety Hazard Detection

## Project Overview
End-to-end ML system for detecting safety hazards in warehouse environments. Using synthetic data generation (3D rendering + Stable Diffusion/ControlNet) to train object detection models.

## Owner
Jaskirat Singh Sohal (Jas)
- Research project with Prof. John Akinyemi, University of Guelph
- Deadline: April 5-6, 2026

## What We're Building
A computer vision pipeline that:
1. Generates synthetic warehouse images with hazards
2. Trains object detection models (YOLOv8 + RT-DETR)
3. Deploys as production-ready API

## Hazard Classes to Detect
- Spills (liquid on floor)
- Obstacles in aisles (boxes, debris)
- Missing PPE (no hard hat, no vest)
- Forklift violations
- Blocked exits
- Damaged racking

## Tech Stack
| Category | Tool |
|----------|------|
| 3D Modeling | Blender |
| Image Generation | Stable Diffusion, ControlNet |
| Labeling | CVAT |
| Data Versioning | DVC |
| Augmentation | Albumentations |
| Training | PyTorch, Ultralytics (YOLOv8), RT-DETR |
| Experiment Tracking | Weights & Biases |
| Distributed Training | PyTorch DDP |
| Cloud GPU | RunPod or Lambda Labs |
| Optimization | ONNX, TensorRT |
| Serving | FastAPI, Uvicorn |
| Containers | Docker, nvidia-docker |
| Cloud Deployment | AWS SageMaker or GCP Vertex |
| MLOps | MLflow, GitHub Actions |

## Timeline (11 weeks)

### Month 1: Data
- **Week 1 (Jan 19-25):** Build 3D warehouse in Blender, render 500+ base images
- **Week 2 (Jan 26-Feb 1):** Stable Diffusion + ControlNet pipeline, generate 2000+ images
- **Week 3 (Feb 2-8):** Label with CVAT, define hazard classes, train/val/test split
- **Week 4 (Feb 9-15):** DVC setup, PyTorch DataLoader, Albumentations augmentation

### Month 2: Training & Evaluation
- **Week 5 (Feb 16-22):** W&B setup, train baseline YOLOv8
- **Week 6 (Feb 23-Mar 1):** Train RT-DETR, compare architectures, hyperparameter tuning
- **Week 7 (Mar 2-8):** Mixed precision (FP16), distributed training on 2 GPUs
- **Week 8 (Mar 9-15):** Evaluation - mAP, error analysis, edge case testing

### Month 3: Optimization, Deployment & MLOps
- **Week 9 (Mar 16-22):** ONNX export, INT8 quantization, benchmark speed vs accuracy
- **Week 10 (Mar 23-29):** FastAPI + Docker + nvidia-docker, batched inference
- **Week 11 (Mar 30-Apr 5):** Deploy to SageMaker/Vertex, GitHub Actions CI/CD, MLflow registry

## Project Structure (Target)
```
warehouse-safety-detection/
├── CLAUDE.md                 # This file
├── README.md                 # Project documentation
├── data/
│   ├── raw/                  # 3D rendered base images
│   ├── synthetic/            # Stable Diffusion outputs
│   ├── labeled/              # CVAT annotations
│   └── processed/            # Final train/val/test splits
├── notebooks/                # Exploration, experiments
├── src/
│   ├── data/
│   │   ├── generate.py       # Stable Diffusion + ControlNet pipeline
│   │   ├── augment.py        # Albumentations
│   │   └── dataloader.py     # PyTorch DataLoader
│   ├── models/
│   │   ├── yolov8.py         # YOLOv8 training
│   │   └── rtdetr.py         # RT-DETR training
│   ├── evaluation/
│   │   └── metrics.py        # mAP, precision, recall
│   ├── optimization/
│   │   ├── quantize.py       # ONNX, TensorRT
│   │   └── benchmark.py      # Speed vs accuracy
│   └── serving/
│       └── api.py            # FastAPI endpoint
├── configs/                  # Training configs
├── docker/
│   └── Dockerfile            # GPU inference container
├── .github/
│   └── workflows/
│       └── train.yml         # CI/CD for retraining
├── dvc.yaml                  # Data versioning
├── requirements.txt
└── pyproject.toml
```

## Current Week
Week 1 (Jan 19-25): Building 3D warehouse model in Blender

## Commands Reference
```bash
# Data versioning
dvc init
dvc add data/
dvc push

# Training
python src/models/yolov8.py --config configs/yolov8.yaml
python src/models/rtdetr.py --config configs/rtdetr.yaml

# Evaluation
python src/evaluation/metrics.py --model runs/best.pt --data data/test

# Export & Quantize
python src/optimization/quantize.py --model runs/best.pt --format onnx
python src/optimization/benchmark.py --models runs/best.pt runs/best.onnx runs/best_int8.onnx

# Serve
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Docker
docker build -t warehouse-detector -f docker/Dockerfile .
docker run --gpus all -p 8000:8000 warehouse-detector
```

## Goals
1. Learn full ML engineer stack through one project
2. Compare CNN (YOLOv8) vs Transformer (RT-DETR) architectures
3. Production-ready deployment with MLOps
4. Research paper/documentation for Prof. Akinyemi

## Success Metrics
- mAP@50 > 0.80
- Inference < 50ms per image
- Deployed and accessible via API
- CI/CD pipeline working
- All experiments tracked in W&B