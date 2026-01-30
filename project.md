# HAWKEYE: Drone-Based Warehouse Hazard Detection

## Protocol for AI Assistant

**CRITICAL:** When completing any step, follow this strict protocol:

1. **The Briefing:**
   - **Goal:** What specific part are we building?
   - **Tech Stack:** Which libraries/tools (e.g., "Using PyTorch Lightning for training")
   - **Context:** How does this connect to previous steps?

2. **The Implementation:**
   - Provide **complete** file content. No placeholders like `# ... rest of code`
   - Include file paths (e.g., `hawkeye/perception/models/fusion.py`)

3. **The Debrief:**
   - Explain the code under the hood
   - Highlight specific logic decisions (why this approach vs alternatives)

4. **Verification:**
   - Provide a specific command or action to prove it works
   - **Git Checkpoint:** After each phase, commit and push

---

## Project Overview

Autonomous drone-based warehouse hazard detection system using RGB-D fusion trained on synthetic data. The system detects hazards from a drone's perspective, tracks them across frames, and sends alerts to a dashboard.

**Owner:** Jaskirat Singh Sohal (Jas)
- Research project with Prof. John Akinyemi, University of Guelph
- Deadline: April 5-6, 2026

---

## Progress Tracking

```
Phase 1: Project Setup ✅ (Mostly Complete)
├── [x] 1.1 - Python Environment Setup
│   ├── [x] 1.1.1 - Create virtual environment
│   ├── [x] 1.1.2 - Install core dependencies (requirements.txt)
│   └── [x] 1.1.3 - Verify PyTorch MPS backend works on M2
├── [x] 1.2 - Blender Setup
│   ├── [x] 1.2.1 - Install Blender 4.x (have 5.0.1)
│   ├── [x] 1.2.2 - Install BlenderKit addon (free assets)
│   └── [x] 1.2.3 - Learn basic navigation (see blender/BLENDER_BASICS.md)
├── [x] 1.3 - Asset Collection
│   ├── [x] 1.3.1 - Download warehouse shell/building
│   ├── [x] 1.3.2 - Download industrial shelving/racking
│   ├── [x] 1.3.3 - Download forklift model
│   ├── [x] 1.3.4 - Download pallets and boxes
│   ├── [x] 1.3.5 - Download debris pack (for obstacles)
│   └── [x] 1.3.6 - Download human models + PPE (hard hat, vest)
├── [x] 1.4 - HAWKEYE Repository Structure
│   ├── [x] 1.4.1 - Create hawkeye/ directory structure
│   ├── [x] 1.4.2 - Update requirements.txt with new dependencies
│   └── [x] 1.4.3 - Configure Hydra for config management
├── [ ] 1.5 - Git checkpoint: "Phase 1 complete - HAWKEYE setup"

Phase 2: Simulation Environment (Blender)
├── [ ] 2.1 - Scene Assembly (using collected assets)
│   ├── [ ] 2.1.1 - Create base warehouse layout
│   ├── [ ] 2.1.2 - Set up industrial lighting
│   ├── [ ] 2.1.3 - Configure drone camera angles (aerial perspective)
│   └── [ ] 2.1.4 - Test render from drone viewpoint
├── [ ] 2.2 - Hazard Insertion
│   ├── [ ] 2.2.1 - Spill materials and placement
│   ├── [ ] 2.2.2 - Obstacle placement (boxes in aisles)
│   ├── [ ] 2.2.3 - Missing PPE scenarios
│   ├── [ ] 2.2.4 - Forklift violation setups
│   ├── [ ] 2.2.5 - Blocked exit scenarios
│   └── [ ] 2.2.6 - Damaged racking textures/geometry
├── [ ] 2.3 - Drone Camera System
│   ├── [ ] 2.3.1 - RGB camera setup (1920x1080)
│   ├── [ ] 2.3.2 - Depth camera setup (EXR output)
│   ├── [ ] 2.3.3 - Depth noise model implementation
│   └── [ ] 2.3.4 - Camera pose randomization
├── [ ] 2.4 - Domain Randomization
│   ├── [ ] 2.4.1 - Lighting randomization
│   ├── [ ] 2.4.2 - Texture randomization
│   ├── [ ] 2.4.3 - Camera jitter
│   └── [ ] 2.4.4 - Post-render artifacts (noise, blur, compression)
├── [ ] 2.5 - Flight Path Generation
│   ├── [ ] 2.5.1 - Full warehouse patrol (serpentine)
│   ├── [ ] 2.5.2 - Single aisle inspection
│   └── [ ] 2.5.3 - Targeted hazard inspection
├── [ ] 2.6 - Batch Rendering Pipeline
│   ├── [ ] 2.6.1 - Python script for batch rendering
│   ├── [ ] 2.6.2 - Randomize drone positions along flight paths
│   ├── [ ] 2.6.3 - Randomize object placements
│   ├── [ ] 2.6.4 - Output segmentation masks (for auto-labeling)
│   └── [ ] 2.6.5 - Render 500+ base RGB-D images
├── [ ] 2.7 - Stable Diffusion Enhancement
│   ├── [ ] 2.7.1 - Set up diffusers locally (M2 MPS) for testing
│   ├── [ ] 2.7.2 - ControlNet depth conditioning
│   ├── [ ] 2.7.3 - img2img for realism enhancement
│   ├── [ ] 2.7.4 - Set up cloud GPU (RunPod) for batch generation
│   └── [ ] 2.7.5 - Generate 2000+ enhanced images (cloud)
├── [ ] 2.8 - Git checkpoint: "Phase 2 complete - Simulation environment"

Phase 3: Perception Model
├── [ ] 3.1 - Dataset & Dataloader
│   ├── [ ] 3.1.1 - PyTorch Dataset class for RGB-D
│   ├── [ ] 3.1.2 - Synchronized augmentations (Albumentations)
│   └── [ ] 3.1.3 - DataLoader configuration
├── [ ] 3.2 - Model Architecture
│   ├── [ ] 3.2.1 - Early Fusion YOLOv8 (4-channel input)
│   └── [ ] 3.2.2 - (Optional) Late Fusion dual encoder
├── [ ] 3.3 - Training Pipeline
│   ├── [ ] 3.3.1 - PyTorch Lightning module
│   ├── [ ] 3.3.2 - Loss function (CIoU + Focal)
│   ├── [ ] 3.3.3 - W&B logging integration
│   └── [ ] 3.3.4 - Checkpointing strategy
├── [ ] 3.4 - Training Stages
│   ├── [ ] 3.4.1 - RGB-only baseline
│   ├── [ ] 3.4.2 - Depth-only baseline
│   ├── [ ] 3.4.3 - Early fusion training
│   └── [ ] 3.4.4 - (Optional) Late fusion training
├── [ ] 3.5 - Evaluation
│   ├── [ ] 3.5.1 - mAP metrics (0.5, 0.75, 0.5:0.95)
│   ├── [ ] 3.5.2 - Per-class analysis
│   ├── [ ] 3.5.3 - Ablation studies
│   └── [ ] 3.5.4 - Failure case analysis
├── [ ] 3.6 - Git checkpoint: "Phase 3 complete - Perception model"

Phase 4: Drone Integration
├── [ ] 4.1 - Path Planning Module
│   ├── [ ] 4.1.1 - Warehouse map representation
│   ├── [ ] 4.1.2 - Coverage path planning (Boustrophedon)
│   └── [ ] 4.1.3 - Reactive inspection paths
├── [ ] 4.2 - Real-Time Inference
│   ├── [ ] 4.2.1 - Model export (TorchScript/ONNX)
│   ├── [ ] 4.2.2 - Inference pipeline optimization
│   └── [ ] 4.2.3 - Multi-object tracker (IoU-based)
├── [ ] 4.3 - Simulation Integration
│   ├── [ ] 4.3.1 - Blender simulation runner
│   └── [ ] 4.3.2 - Detection overlay visualization
├── [ ] 4.4 - Git checkpoint: "Phase 4 complete - Drone integration"

Phase 5: Alert System
├── [ ] 5.1 - Backend API (FastAPI)
│   ├── [ ] 5.1.1 - Alert CRUD endpoints
│   ├── [ ] 5.1.2 - WebSocket for real-time updates
│   ├── [ ] 5.1.3 - Drone status endpoint
│   └── [ ] 5.1.4 - Statistics endpoint
├── [ ] 5.2 - Dashboard Frontend (React)
│   ├── [ ] 5.2.1 - Video feed component with overlays
│   ├── [ ] 5.2.2 - Warehouse map component
│   ├── [ ] 5.2.3 - Alert list with filtering
│   └── [ ] 5.2.4 - Statistics panel
├── [ ] 5.3 - Git checkpoint: "Phase 5 complete - Alert system"

Phase 6: Evaluation & Documentation
├── [ ] 6.1 - Comprehensive Evaluation
│   ├── [ ] 6.1.1 - Full benchmark script
│   ├── [ ] 6.1.2 - Speed vs accuracy tradeoffs
│   └── [ ] 6.1.3 - Visualization report
├── [ ] 6.2 - Demo Materials
│   ├── [ ] 6.2.1 - Demo video (2-3 min)
│   └── [ ] 6.2.2 - Sample detection images
├── [ ] 6.3 - Documentation
│   ├── [ ] 6.3.1 - README and installation guide
│   ├── [ ] 6.3.2 - Technical documentation
│   └── [ ] 6.3.3 - Research paper draft
├── [ ] 6.4 - Git checkpoint: "v1.0 Release"
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION PIPELINE (Blender + SD)                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  Warehouse   │─────►│    Hazard    │─────►│   Domain     │
  │  + Drone Cam │      │  Spawner     │      │  Randomizer  │
  └──────────────┘      └──────────────┘      └──────────────┘
         │                                           │
         ▼                                           ▼
  ┌─────────────────────────────────┐    ┌─────────────────────┐
  │     Drone Flight Simulator      │    │   Batch Renderer    │
  │  ┌─────────┐    ┌───────────┐  │    │  RGB + Depth + Mask │
  │  │  RGB    │    │   Depth   │  │    └─────────────────────┘
  │  │ Camera  │    │  Camera   │  │              │
  │  └─────────┘    └───────────┘  │              ▼
  └─────────────────────────────────┘    ┌─────────────────────┐
                                         │  Stable Diffusion   │
                                         │  + ControlNet       │
                                         │  (Realism Enhance)  │
                                         └─────────────────────┘
                                                   │
                                                   ▼
                               ┌─────────────────────────────────┐
                               │   RGB-D Frames + Annotations    │
                               │        (COCO Format)            │
                               └─────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   RGB-D      │─────►│   Fusion     │─────►│  Detection   │
  │  DataLoader  │      │   Model      │      │    Head      │
  └──────────────┘      └──────────────┘      └──────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   Synchronized          4-Channel            Boxes + Classes
   Augmentations         YOLOv8 or            + Confidence
                         Dual Encoder


┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │    Drone     │─────►│   Real-Time  │─────►│   Tracker    │
  │  Simulation  │      │  Inference   │      │  (IoU-based) │
  └──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      ALERT SYSTEM                             │
  │  ┌─────────────┐         ┌──────────────────────────────┐   │
  │  │  FastAPI    │◄───────►│     React Dashboard          │   │
  │  │  Backend    │   WS    │  ┌────────┐ ┌────────────┐  │   │
  │  └─────────────┘         │  │ Video  │ │ Map View   │  │   │
  │                          │  │ Feed   │ │ + Alerts   │  │   │
  │                          │  └────────┘ └────────────┘  │   │
  │                          └──────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────┘
```

---

## Hazard Classes

| Class ID | Hazard Type | Description | Detection Difficulty |
|----------|-------------|-------------|---------------------|
| 0 | `spill` | Liquid on floor | Medium - requires texture recognition |
| 1 | `obstacle` | Boxes/debris in aisle | Easy - clear object boundaries |
| 2 | `missing_ppe` | Worker without hard hat/vest | Hard - requires person detection + PPE check |
| 3 | `forklift_violation` | Unsafe forklift operation | Medium - context-dependent |
| 4 | `blocked_exit` | Exit door obstructed | Medium - requires scene understanding |
| 5 | `damaged_rack` | Bent/broken racking | Hard - subtle visual defects |

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Simulation** | Blender 4.0+ (Python API) | 3D scene generation, RGB-D rendering |
| **Simulation** | OpenEXR | Depth map storage |
| **Image Gen** | Stable Diffusion + ControlNet | Realism enhancement |
| **Image Gen** | diffusers (HuggingFace) | SD pipeline |
| **Perception** | PyTorch 2.0+ | Deep learning framework |
| **Perception** | Ultralytics (YOLOv8) | Object detection backbone |
| **Perception** | PyTorch Lightning | Training framework |
| **Perception** | Albumentations | Augmentation |
| **Perception** | timm | Backbone models (for late fusion) |
| **Tracking** | Custom IoU tracker | Multi-object tracking |
| **Config** | Hydra | Configuration management |
| **Logging** | Weights & Biases | Experiment tracking |
| **Backend** | FastAPI + Uvicorn | Alert API |
| **Backend** | WebSockets | Real-time updates |
| **Frontend** | React + Tailwind CSS | Dashboard |
| **Evaluation** | pycocotools | mAP metrics |

---

## Directory Structure

```
hawkeye/
├── simulation/
│   ├── blender/
│   │   ├── assets/
│   │   │   ├── warehouse/       # Floor, walls, racks
│   │   │   ├── hazards/         # Hazard object prefabs
│   │   │   └── props/           # Boxes, pallets, equipment
│   │   ├── scripts/             # Blender Python scripts
│   │   │   ├── warehouse_generator.py
│   │   │   ├── hazard_spawner.py
│   │   │   ├── domain_randomizer.py
│   │   │   └── render_pipeline.py
│   │   └── scenes/              # Saved .blend files
│   └── generation/
│       ├── flight_paths.py      # Path planning
│       └── annotation_exporter.py
├── perception/
│   ├── models/
│   │   ├── fusion_yolo.py       # 4-channel YOLOv8
│   │   └── dual_encoder.py      # Late fusion (optional)
│   ├── datasets/
│   │   └── rgbd_dataset.py
│   ├── training/
│   │   ├── train.py
│   │   └── lightning_module.py
│   └── evaluation/
│       └── metrics.py
├── drone/
│   ├── planning/
│   │   └── path_planner.py
│   ├── inference/
│   │   └── realtime_detector.py
│   └── control/
│       └── tracker.py           # IoU-based MOT
├── alert_system/
│   ├── backend/
│   │   ├── main.py              # FastAPI app
│   │   ├── models.py            # Pydantic models
│   │   └── database.py          # Alert storage
│   └── dashboard/
│       ├── src/
│       │   ├── components/
│       │   └── App.jsx
│       └── package.json
├── configs/
│   ├── config.yaml              # Main Hydra config
│   ├── simulation/
│   ├── model/
│   └── training/
├── scripts/
│   ├── generate_dataset.py
│   ├── train_model.py
│   └── run_demo.py
├── tests/
├── outputs/
│   ├── datasets/
│   │   └── hawkeye_v1/
│   │       ├── train/
│   │       │   ├── rgb/
│   │       │   ├── depth/
│   │       │   └── annotations.json
│   │       ├── val/
│   │       └── test/
│   ├── checkpoints/
│   └── results/
└── docs/
```

---

## Configuration (Hydra)

### Main Config Structure

```yaml
# configs/config.yaml
defaults:
  - simulation: warehouse
  - model: fusion_yolo
  - training: default

simulation:
  warehouse:
    dimensions: [50, 30]  # meters
    rack_height_levels: 4
    aisle_width: 3.5

  hazards:
    types: [aisle_obstruction, fallen_items, unstable_stacking, person_restricted_zone]
    per_scene_range: [0, 5]

  camera:
    rgb_resolution: [1920, 1080]
    depth_resolution: [1280, 720]
    fov: 84
    height: 2.5
    pitch_range: [45, 60]

  domain_randomization:
    lighting_intensity_range: [500, 2000]
    color_temp_range: [3500, 6500]
    texture_variations: 6

dataset:
  train_scenes: 1500
  val_scenes: 200
  test_scenes: 300
  output_resolution: [640, 640]

model:
  architecture: fusion_yolo  # or dual_encoder
  backbone: yolov8m
  input_channels: 4  # RGB + Depth
  num_classes: 6     # spill, obstacle, missing_ppe, forklift_violation, blocked_exit, damaged_rack

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  optimizer: adamw
  scheduler: cosine
  early_stopping_patience: 20
  mixed_precision: true

evaluation:
  iou_thresholds: [0.5, 0.75]
  confidence_threshold: 0.5
```

---

## Success Criteria

### Minimum Viable (Must Have)
- [ ] Working data generation pipeline producing RGB-D + COCO annotations
- [ ] Trained fusion model achieving **>80% mAP@50** on synthetic test set
- [ ] Simulated drone demo showing detection working
- [ ] Basic alert logging to console/file

### Target (Should Have)
- [ ] **>85% mAP@50** on synthetic data
- [ ] Ablation study showing fusion improvement **>5%** over RGB-only
- [ ] Real-time inference **>15 FPS** on laptop GPU
- [ ] Full React dashboard with live visualization
- [ ] Comprehensive evaluation report

### Stretch (Nice to Have)
- [ ] **>90% mAP@50** on synthetic data
- [ ] Testing on real RGB-D data
- [ ] Late fusion architecture comparison
- [ ] Published paper or technical report

---

## Timeline (8 Weeks)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Simulation Foundation | Blender warehouse generator, RGB-D rendering, basic hazard spawning |
| 3 | Data Pipeline | Domain randomization, flight paths, annotations, initial dataset (5000 frames) |
| 4-5 | Perception Model | Dataset/dataloader, RGB baseline, fusion model trained |
| 6 | Integration | Inference pipeline, tracking, simulation integration |
| 7 | Alert System | FastAPI backend, React dashboard |
| 8 | Polish & Evaluation | Full evaluation, demo materials, documentation |

---

## Compute Strategy

| Task | Where | Why |
|------|-------|-----|
| Blender scene building | **Local (M2)** | GUI-based, Metal acceleration |
| Blender batch rendering | **Local (M2)** | No upload needed, GPU works well |
| Code development | **Local (M2)** | Fast iteration |
| Model training (test) | **Local (M2)** | Quick validation on small subset |
| Model training (full) | **Cloud GPU** | CUDA required for speed |
| Dashboard development | **Local (M2)** | Fast iteration |

---

## Current Status

**Current Phase:** Phase 1 - Project Setup (Week 1-2)
**Completed:** Python environment, Blender setup, asset collection
**Current Task:** Restructure repository for HAWKEYE drone-based approach

**Key Change from Original:**
- Same hazard classes (spill, obstacle, missing_ppe, forklift_violation, blocked_exit, damaged_rack)
- **Drone perspective** instead of fixed security cameras
- **RGB-D fusion** (depth camera added)
- **Real-time tracking** across frames
- **Alert dashboard** for monitoring

**Next Steps:**
1. Create hawkeye/ directory structure
2. Set up Hydra config
3. Assemble warehouse scene with drone camera viewpoint
4. Begin hazard insertion scripting

---

## Resources

### Blender Assets
- [BlenderKit Free Assets](https://www.blenderkit.com/)
- [Sketchfab Warehouse Models](https://sketchfab.com/tags/warehouse)

### Documentation
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Docs](https://hydra.cc/docs/intro/)
- [Albumentations Docs](https://albumentations.ai/docs/)

### Papers
- [Domain Randomization for Sim-to-Real Transfer](https://arxiv.org/abs/1703.06907)
- [RGB-D Fusion for Object Detection](https://arxiv.org/abs/2012.12089)
