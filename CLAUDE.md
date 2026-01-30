# HAWKEYE: Drone-Based Warehouse Safety Hazard Detection

## Project Overview
Autonomous drone-based warehouse hazard detection system using RGB-D fusion trained on synthetic data. The system detects hazards from a drone's aerial perspective, tracks them across frames, and sends alerts to a dashboard.

## Owner
Jaskirat Singh Sohal (Jas)
- Research project with Prof. John Akinyemi, University of Guelph
- Deadline: April 5-6, 2026

## What We're Building
A computer vision pipeline that:
1. Generates synthetic warehouse images from drone perspective (Blender + SD/ControlNet)
2. Trains RGB-D fusion object detection models (4-channel YOLOv8)
3. Tracks hazards across frames in simulated drone flights
4. Deploys with real-time alert dashboard

## Key Differentiators from Standard Approach
- **Drone perspective** (aerial view at ~2.5m height, 45-60° pitch)
- **RGB-D fusion** (depth camera adds distance/3D information)
- **Multi-object tracking** (IoU-based tracker for consistency across frames)
- **Real-time dashboard** (React frontend with WebSocket updates)

## Hazard Classes to Detect
| ID | Class | Description |
|----|-------|-------------|
| 0 | `spill` | Liquid on floor |
| 1 | `obstacle` | Boxes/debris in aisle |
| 2 | `missing_ppe` | Worker without hard hat/vest |
| 3 | `forklift_violation` | Unsafe forklift operation |
| 4 | `blocked_exit` | Exit door obstructed |
| 5 | `damaged_rack` | Bent/broken racking |

## Tech Stack
| Category | Tool |
|----------|------|
| 3D Modeling | Blender 4.0+ |
| Image Enhancement | Stable Diffusion, ControlNet |
| Depth Rendering | OpenEXR |
| Config Management | Hydra |
| Augmentation | Albumentations |
| Training | PyTorch, PyTorch Lightning, Ultralytics (YOLOv8) |
| Experiment Tracking | Weights & Biases |
| Cloud GPU | RunPod or Lambda Labs |
| Optimization | ONNX, TorchScript |
| Backend | FastAPI, WebSockets |
| Frontend | React, Tailwind CSS |
| Evaluation | pycocotools |

## Timeline (8 weeks - Already in Week 1-2)

### Weeks 1-2: Simulation Foundation
- Build warehouse scene with drone camera setup
- RGB-D rendering pipeline
- Hazard spawning system

### Week 3: Data Pipeline
- Domain randomization
- Flight path generation
- SD + ControlNet enhancement
- Generate dataset (15000+ frames)

### Weeks 4-5: Perception Model
- RGB-D dataset and dataloader
- Train RGB-only baseline
- Train fusion model (4-channel YOLOv8)
- Ablation studies

### Week 6: Drone Integration
- Inference pipeline optimization
- Multi-object tracker
- Simulation integration

### Week 7: Alert System
- FastAPI backend
- React dashboard
- WebSocket real-time updates

### Week 8: Polish & Evaluation
- Full evaluation suite
- Demo materials
- Documentation

## Project Structure
```
hawkeye/
├── simulation/
│   ├── blender/
│   │   ├── assets/           # Warehouse, hazards, props
│   │   ├── scripts/          # Generation scripts
│   │   └── scenes/           # .blend files
│   └── generation/           # Flight paths, annotations
├── perception/
│   ├── models/               # Fusion YOLOv8, dual encoder
│   ├── datasets/             # RGB-D dataset class
│   ├── training/             # Lightning module
│   └── evaluation/           # Metrics
├── drone/
│   ├── planning/             # Path planning
│   ├── inference/            # Real-time detection
│   └── control/              # Tracker
├── alert_system/
│   ├── backend/              # FastAPI
│   └── dashboard/            # React
├── configs/                  # Hydra configs
├── scripts/                  # Entry points
├── outputs/
│   ├── datasets/             # Generated data
│   ├── checkpoints/          # Model weights
│   └── results/              # Evaluation outputs
└── docs/
```

## Commands Reference
```bash
# Generate dataset
python scripts/generate_dataset.py

# Train model
python scripts/train_model.py model=fusion_yolo training.epochs=100

# Evaluate
python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt

# Run demo
python scripts/run_demo.py

# Start alert backend
uvicorn hawkeye.alert_system.backend.main:app --reload

# Start dashboard (dev)
cd hawkeye/alert_system/dashboard && npm run dev
```

## Goals
1. Learn full ML engineer stack with drone/robotics angle
2. Compare RGB-only vs RGB-D fusion performance
3. Build end-to-end system from synthetic data to live dashboard
4. Research paper/documentation for Prof. Akinyemi

## Success Metrics
- mAP@50 > 0.80 (target 0.85)
- RGB-D fusion improves over RGB-only by >5%
- Real-time inference > 15 FPS
- Working alert dashboard with live visualization
- All experiments tracked in W&B
