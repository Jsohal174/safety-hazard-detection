# Warehouse Safety Hazard Detection - Project Roadmap

## Protocol for AI Assistant

**CRITICAL:** When completing any step, follow this strict protocol:

1. **The Briefing:**
   - **Goal:** What specific part are we building?
   - **Tech Stack:** Which libraries/tools (e.g., "Using Albumentations for augmentation")
   - **Context:** How does this connect to previous steps?

2. **The Implementation:**
   - Provide **complete** file content. No placeholders like `# ... rest of code`
   - Include file paths (e.g., `src/data/generate.py`)

3. **The Debrief:**
   - Explain the code under the hood
   - Highlight specific logic decisions (why this approach vs alternatives)

4. **Verification:**
   - Provide a specific command or action to prove it works
   - **Git Checkpoint:** After each phase, commit and push

---

## Progress Tracking

Use this checklist to track completion. Mark with `[x]` when done.

```
Phase 1: Environment & 3D Assets (Week 1)
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
├── [ ] 1.4 - Scene Assembly
│   ├── [ ] 1.4.1 - Create base warehouse layout
│   ├── [ ] 1.4.2 - Set up lighting (industrial overhead)
│   ├── [ ] 1.4.3 - Configure multiple camera angles
│   └── [ ] 1.4.4 - Test render a few images
├── [ ] 1.5 - Git checkpoint: "Phase 1 complete - Blender setup"

Phase 2: Synthetic Data Generation (Week 2)
├── [ ] 2.1 - Blender Rendering Pipeline
│   ├── [ ] 2.1.1 - Python script for batch rendering
│   ├── [ ] 2.1.2 - Randomize camera positions
│   ├── [ ] 2.1.3 - Randomize object placements
│   ├── [ ] 2.1.4 - Output segmentation masks (for auto-labeling)
│   └── [ ] 2.1.5 - Render 500+ base images
├── [ ] 2.2 - Hazard Insertion
│   ├── [ ] 2.2.1 - Spill materials and placement
│   ├── [ ] 2.2.2 - Obstacle placement (boxes in aisles)
│   ├── [ ] 2.2.3 - Missing PPE scenarios
│   ├── [ ] 2.2.4 - Forklift violation setups
│   ├── [ ] 2.2.5 - Blocked exit scenarios
│   └── [ ] 2.2.6 - Damaged racking textures/geometry
├── [ ] 2.3 - Stable Diffusion Enhancement
│   ├── [ ] 2.3.1 - Set up diffusers locally (M2 MPS) for testing
│   ├── [ ] 2.3.2 - ControlNet depth conditioning (test locally)
│   ├── [ ] 2.3.3 - img2img for realism enhancement (test locally)
│   ├── [ ] 2.3.4 - Set up cloud GPU (RunPod) for batch generation
│   ├── [ ] 2.3.5 - Batch processing pipeline (cloud)
│   └── [ ] 2.3.6 - Generate 2000+ enhanced images (cloud)
├── [ ] 2.4 - Git checkpoint: "Phase 2 complete - Data generation"

Phase 3: Labeling & Data Pipeline (Week 3-4)
├── [ ] 3.1 - CVAT Setup
│   ├── [ ] 3.1.1 - Install CVAT (Docker or cloud)
│   ├── [ ] 3.1.2 - Create project with hazard classes
│   ├── [ ] 3.1.3 - Import images
│   └── [ ] 3.1.4 - Configure annotation format (YOLO/COCO)
├── [ ] 3.2 - Auto-Labeling from Blender
│   ├── [ ] 3.2.1 - Convert segmentation masks to bounding boxes
│   ├── [ ] 3.2.2 - Map object IDs to hazard classes
│   └── [ ] 3.2.3 - Import pre-annotations to CVAT
├── [ ] 3.3 - Manual Annotation
│   ├── [ ] 3.3.1 - Review and correct auto-labels
│   ├── [ ] 3.3.2 - Label SD-generated images
│   └── [ ] 3.3.3 - Quality check (random sample review)
├── [ ] 3.4 - Dataset Split
│   ├── [ ] 3.4.1 - Export annotations (YOLO format)
│   ├── [ ] 3.4.2 - Create train/val/test split (70/20/10)
│   └── [ ] 3.4.3 - Verify class distribution across splits
├── [ ] 3.5 - DVC Setup
│   ├── [ ] 3.5.1 - Initialize DVC
│   ├── [ ] 3.5.2 - Add data to DVC tracking
│   ├── [ ] 3.5.3 - Configure remote storage (GDrive/S3)
│   └── [ ] 3.5.4 - First dvc push
├── [ ] 3.6 - PyTorch DataLoader
│   ├── [ ] 3.6.1 - Create custom Dataset class
│   ├── [ ] 3.6.2 - Implement Albumentations augmentation
│   └── [ ] 3.6.3 - Test DataLoader iteration
├── [ ] 3.7 - Git checkpoint: "Phase 3 complete - Data pipeline"

Phase 4: Model Training (Week 5-7)
├── [ ] 4.1 - W&B Setup
│   ├── [ ] 4.1.1 - Create W&B account/project
│   ├── [ ] 4.1.2 - Configure API key
│   └── [ ] 4.1.3 - Test logging
├── [ ] 4.2 - YOLOv8 Baseline
│   ├── [ ] 4.2.1 - Create training config (configs/yolov8.yaml)
│   ├── [ ] 4.2.2 - Implement src/models/yolov8.py
│   ├── [ ] 4.2.3 - Train on local M2 (small subset test)
│   ├── [ ] 4.2.4 - Set up cloud GPU (RunPod/Lambda)
│   ├── [ ] 4.2.5 - Full training run
│   └── [ ] 4.2.6 - Log metrics to W&B
├── [ ] 4.3 - RT-DETR Training
│   ├── [ ] 4.3.1 - Create training config (configs/rtdetr.yaml)
│   ├── [ ] 4.3.2 - Implement src/models/rtdetr.py
│   ├── [ ] 4.3.3 - Train on cloud GPU
│   └── [ ] 4.3.4 - Compare with YOLOv8 in W&B
├── [ ] 4.4 - Hyperparameter Tuning
│   ├── [ ] 4.4.1 - Learning rate sweep
│   ├── [ ] 4.4.2 - Augmentation strength tuning
│   ├── [ ] 4.4.3 - Image size experiments
│   └── [ ] 4.4.4 - Select best model
├── [ ] 4.5 - Advanced Training
│   ├── [ ] 4.5.1 - Mixed precision (FP16) training
│   ├── [ ] 4.5.2 - Multi-GPU training (PyTorch DDP)
│   └── [ ] 4.5.3 - Final training run with best config
├── [ ] 4.6 - Git checkpoint: "Phase 4 complete - Training"

Phase 5: Evaluation (Week 8)
├── [ ] 5.1 - Metrics Implementation
│   ├── [ ] 5.1.1 - Implement mAP calculation
│   ├── [ ] 5.1.2 - Per-class precision/recall
│   ├── [ ] 5.1.3 - Confusion matrix
│   └── [ ] 5.1.4 - Inference speed benchmarking
├── [ ] 5.2 - Error Analysis
│   ├── [ ] 5.2.1 - Identify failure cases
│   ├── [ ] 5.2.2 - Per-hazard-class analysis
│   ├── [ ] 5.2.3 - False positive analysis
│   └── [ ] 5.2.4 - Edge case identification
├── [ ] 5.3 - Model Comparison
│   ├── [ ] 5.3.1 - YOLOv8 vs RT-DETR analysis
│   ├── [ ] 5.3.2 - Speed vs accuracy tradeoff
│   └── [ ] 5.3.3 - Select production model
├── [ ] 5.4 - Git checkpoint: "Phase 5 complete - Evaluation"

Phase 6: Optimization (Week 9)
├── [ ] 6.1 - ONNX Export
│   ├── [ ] 6.1.1 - Export best model to ONNX
│   ├── [ ] 6.1.2 - Verify ONNX inference
│   └── [ ] 6.1.3 - Benchmark ONNX vs PyTorch
├── [ ] 6.2 - Quantization
│   ├── [ ] 6.2.1 - INT8 quantization
│   ├── [ ] 6.2.2 - Accuracy vs speed comparison
│   └── [ ] 6.2.3 - Select deployment format
├── [ ] 6.3 - TensorRT (Optional - for NVIDIA deployment)
│   ├── [ ] 6.3.1 - TensorRT conversion
│   └── [ ] 6.3.2 - Benchmark TensorRT
├── [ ] 6.4 - Git checkpoint: "Phase 6 complete - Optimization"

Phase 7: API & Deployment (Week 10)
├── [ ] 7.1 - FastAPI Development
│   ├── [ ] 7.1.1 - Create inference endpoint
│   ├── [ ] 7.1.2 - Add image upload handling
│   ├── [ ] 7.1.3 - Return detection results (JSON)
│   ├── [ ] 7.1.4 - Add batch inference endpoint
│   └── [ ] 7.1.5 - Add health check endpoint
├── [ ] 7.2 - Docker Container
│   ├── [ ] 7.2.1 - Create Dockerfile (GPU-enabled)
│   ├── [ ] 7.2.2 - Test container locally
│   └── [ ] 7.2.3 - Optimize container size
├── [ ] 7.3 - Local Testing
│   ├── [ ] 7.3.1 - Test with sample images
│   ├── [ ] 7.3.2 - Load testing
│   └── [ ] 7.3.3 - Error handling verification
├── [ ] 7.4 - Git checkpoint: "Phase 7 complete - API ready"

Phase 8: Cloud & MLOps (Week 11)
├── [ ] 8.1 - Cloud Deployment
│   ├── [ ] 8.1.1 - Choose platform (SageMaker vs Vertex)
│   ├── [ ] 8.1.2 - Deploy model endpoint
│   ├── [ ] 8.1.3 - Configure auto-scaling
│   └── [ ] 8.1.4 - Set up monitoring
├── [ ] 8.2 - MLflow Registry
│   ├── [ ] 8.2.1 - Set up MLflow server
│   ├── [ ] 8.2.2 - Register model versions
│   └── [ ] 8.2.3 - Model staging workflow
├── [ ] 8.3 - CI/CD Pipeline
│   ├── [ ] 8.3.1 - GitHub Actions for testing
│   ├── [ ] 8.3.2 - Automated retraining trigger
│   └── [ ] 8.3.3 - Deployment automation
├── [ ] 8.4 - Documentation
│   ├── [ ] 8.4.1 - API documentation
│   ├── [ ] 8.4.2 - Model card
│   └── [ ] 8.4.3 - Research paper draft
├── [ ] 8.5 - Git checkpoint: "v1.0 Release"
```

---

## 1. Architectural Overview

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA GENERATION PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   Blender    │─────►│    Stable    │─────►│    CVAT      │
  │  3D Renders  │      │  Diffusion   │      │  Labeling    │
  └──────────────┘      └──────────────┘      └──────────────┘
        │                      │                     │
        ▼                      ▼                     ▼
   Base Images +         Enhanced          Bounding Box
   Segmentation          Realistic         Annotations
   Masks                 Images            (YOLO format)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │     DVC      │─────►│   PyTorch    │─────►│    W&B       │
  │  Versioning  │      │  DataLoader  │      │  Tracking    │
  └──────────────┘      └──────────────┘      └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    │  ┌───────┬───────┐  │
                    │  │YOLOv8 │RT-DETR│  │
                    │  └───────┴───────┘  │
                    └─────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │    ONNX      │─────►│   FastAPI    │─────►│   Docker     │
  │  Optimized   │      │   Serving    │      │  Container   │
  └──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Cloud Deploy    │
                                          │ (SageMaker/GCP)  │
                                          └──────────────────┘
```

### Hazard Classes

| Class ID | Hazard Type | Description | Detection Difficulty |
|----------|-------------|-------------|---------------------|
| 0 | `spill` | Liquid on floor | Medium - requires texture recognition |
| 1 | `obstacle` | Boxes/debris in aisle | Easy - clear object boundaries |
| 2 | `missing_ppe` | Worker without hard hat/vest | Hard - requires person detection + PPE check |
| 3 | `forklift_violation` | Unsafe forklift operation | Medium - context-dependent |
| 4 | `blocked_exit` | Exit door obstructed | Medium - requires scene understanding |
| 5 | `damaged_rack` | Bent/broken racking | Hard - subtle visual defects |

### Tech Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| 3D Rendering | Blender 4.x | Base image generation |
| Image Enhancement | Stable Diffusion + ControlNet | Photorealism |
| Labeling | CVAT | Bounding box annotation |
| Data Versioning | DVC | Dataset tracking |
| Augmentation | Albumentations | Training augmentation |
| Training | Ultralytics (YOLOv8), RT-DETR | Object detection |
| Experiment Tracking | Weights & Biases | Metrics logging |
| Optimization | ONNX, TensorRT | Inference speed |
| Serving | FastAPI + Uvicorn | REST API |
| Containerization | Docker + nvidia-docker | Deployment |
| Cloud | AWS SageMaker / GCP Vertex | Production hosting |
| CI/CD | GitHub Actions | Automation |
| Model Registry | MLflow | Version management |

### Compute Strategy (Hybrid: Local + Cloud)

**Why hybrid?** Your M2 Mac handles development and Blender well, but lacks CUDA for heavy ML workloads.

| Task | Where | Why |
|------|-------|-----|
| Blender scene building | **Local (M2)** | GUI-based, M2 GPU works great |
| Blender batch rendering | **Local (M2)** | Metal acceleration, no upload needed |
| Code development | **Local (M2)** | Fast iteration, IDE access |
| SD testing (small batches) | **Local (M2)** | MPS backend works, ~30s/image |
| SD production (2000+ images) | **Cloud GPU** | 10x faster with CUDA |
| Model training (test runs) | **Local (M2)** | Quick validation on small subset |
| Model training (full) | **Cloud GPU** | CUDA required, multi-GPU support |
| CVAT labeling | **Local (Docker)** | Browser-based, runs anywhere |
| API development | **Local (M2)** | Fast iteration |
| Final deployment | **Cloud** | Production hosting |

**Cloud GPU Options:**
- **RunPod:** Pay-per-hour, RTX 4090 (~$0.44/hr), good for training
- **Lambda Labs:** Similar pricing, reliable
- **Google Colab Pro:** $10/mo, good for experiments (not production)

**When to use cloud:** Phase 2 (heavy SD generation) and Phase 4+ (training)

---

## 2. Data Schema

### Directory Structure

```
data/
├── raw/                          # Blender outputs
│   ├── images/                   # Rendered RGB images
│   │   ├── scene_001_cam_01.png
│   │   └── ...
│   ├── masks/                    # Segmentation masks
│   │   ├── scene_001_cam_01_mask.png
│   │   └── ...
│   └── metadata.json             # Scene configurations
│
├── synthetic/                    # Stable Diffusion outputs
│   ├── images/
│   │   ├── sd_001.png
│   │   └── ...
│   └── generation_log.json       # SD parameters used
│
├── labeled/                      # CVAT exports
│   ├── annotations/
│   │   ├── scene_001_cam_01.txt  # YOLO format
│   │   └── ...
│   └── classes.txt               # Class definitions
│
└── processed/                    # Final dataset
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml                 # Dataset config for YOLO
```

### Annotation Format (YOLO)

```
# Each .txt file: one line per object
# Format: class_id center_x center_y width height (normalized 0-1)

# Example: scene_001_cam_01.txt
0 0.453 0.672 0.124 0.089   # spill
1 0.234 0.445 0.056 0.123   # obstacle
2 0.678 0.334 0.089 0.234   # missing_ppe
```

### data.yaml Structure

```yaml
# data/processed/data.yaml
path: /path/to/data/processed
train: train/images
val: val/images
test: test/images

nc: 6  # number of classes
names:
  0: spill
  1: obstacle
  2: missing_ppe
  3: forklift_violation
  4: blocked_exit
  5: damaged_rack
```

---

## 3. Configuration Files

### YOLOv8 Training Config

```yaml
# configs/yolov8.yaml
model: yolov8m.pt          # Medium model (good balance)
data: data/processed/data.yaml
epochs: 100
imgsz: 640
batch: 16
device: 0                   # GPU index
workers: 8
patience: 20                # Early stopping
save_period: 10             # Checkpoint frequency

# Augmentation
augment: true
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10
translate: 0.1
scale: 0.5
flipud: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.1

# W&B
project: warehouse-safety
name: yolov8m-baseline
```

### RT-DETR Training Config

```yaml
# configs/rtdetr.yaml
model: rtdetr-l.pt          # Large RT-DETR
data: data/processed/data.yaml
epochs: 100
imgsz: 640
batch: 8                    # Lower due to memory
device: 0
workers: 8
patience: 20

# W&B
project: warehouse-safety
name: rtdetr-l-baseline
```

---

## 4. API Specification

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/detect` | Single image detection |
| POST | `/detect/batch` | Batch detection |
| GET | `/model/info` | Model metadata |

### Request/Response Examples

**POST /detect**

Request:
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@warehouse_image.jpg" \
  -F "confidence=0.5"
```

Response:
```json
{
  "success": true,
  "inference_time_ms": 45.2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "spill",
      "confidence": 0.92,
      "bbox": {
        "x1": 234,
        "y1": 456,
        "x2": 389,
        "y2": 521
      }
    },
    {
      "class_id": 1,
      "class_name": "obstacle",
      "confidence": 0.87,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 180,
        "y2": 340
      }
    }
  ],
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

---

## 5. Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| mAP@50 | > 0.80 | Critical |
| mAP@50:95 | > 0.60 | High |
| Inference time | < 50ms/image | Critical |
| Per-class AP (all classes) | > 0.70 | High |
| False positive rate | < 10% | Medium |
| API latency (p99) | < 100ms | Medium |

---

## 6. Current Status

**Current Phase:** Phase 1 - Environment & 3D Assets
**Current Week:** Week 1 (Jan 19-25)
**Current Task:** Asset collection and Blender setup

**Blockers:** None

**Next Steps:**
1. Set up Python environment
2. Install Blender
3. Download warehouse 3D assets from Sketchfab/BlenderKit

---

## 7. Resources & Links

### 3D Asset Sources
- [Sketchfab Warehouse Models](https://sketchfab.com/tags/warehouse)
- [BlenderKit Free Assets](https://www.blenderkit.com/)
- [CGTrader Free Models](https://www.cgtrader.com/free-3d-models)
- [TurboSquid Free](https://www.turbosquid.com/Search/3D-Models/free/warehouse)

### Documentation
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Stable Diffusion + ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [CVAT Documentation](https://opencv.github.io/cvat/docs/)
- [DVC Documentation](https://dvc.org/doc)

### Cloud GPU
- [RunPod](https://www.runpod.io/)
- [Lambda Labs](https://lambdalabs.com/)
