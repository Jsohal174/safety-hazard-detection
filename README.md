# Warehouse Safety Hazard Detection

An end-to-end ML system for detecting safety hazards in warehouse environments using synthetic data and modern object detection architectures.

## Overview

This project builds a computer vision pipeline that:
1. Generates synthetic warehouse images using 3D rendering + Stable Diffusion/ControlNet
2. Trains and compares CNN (YOLOv8) vs Transformer (RT-DETR) object detection models
3. Optimizes for production with quantization (ONNX, TensorRT)
4. Deploys as a production-ready API with MLOps pipeline

**Research Project** - University of Guelph, supervised by Prof. John Akinyemi

## Hazard Classes

| Class | Description |
|-------|-------------|
| Spill | Liquid on floor |
| Obstacle | Boxes, debris in aisles |
| No PPE | Missing hard hat or vest |
| Forklift Violation | Unsafe forklift operation |
| Blocked Exit | Exit path obstructed |
| Damaged Racking | Structural damage to shelving |

## Results

| Model | mAP@50 | mAP@50:95 | FPS | Size |
|-------|--------|-----------|-----|------|
| YOLOv8 | - | - | - | - |
| RT-DETR | - | - | - | - |
| YOLOv8 (INT8) | - | - | - | - |

*Results will be updated as training completes*

## Installation

```bash
# Clone repository
git clone https://github.com/Jsohal174/warehouse-safety-detection.git
cd warehouse-safety-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
warehouse-safety-detection/
├── data/
│   ├── raw/                  # 3D rendered base images
│   ├── synthetic/            # Stable Diffusion outputs
│   ├── labeled/              # CVAT annotations
│   └── processed/            # Final train/val/test splits
├── notebooks/                # Exploration, experiments
├── src/
│   ├── data/                 # Data generation & loading
│   ├── models/               # YOLOv8, RT-DETR training
│   ├── evaluation/           # Metrics & analysis
│   ├── optimization/         # Quantization, benchmarking
│   └── serving/              # FastAPI endpoint
├── configs/                  # Training configs
├── docker/                   # Deployment containers
└── .github/workflows/        # CI/CD pipelines
```

## Usage

### Generate Synthetic Data

```bash
# Generate images with Stable Diffusion + ControlNet
python src/data/generate.py --input data/raw --output data/synthetic --num_images 2000
```

### Train Models

```bash
# Train YOLOv8
python src/models/yolov8.py --config configs/yolov8.yaml

# Train RT-DETR
python src/models/rtdetr.py --config configs/rtdetr.yaml
```

### Evaluate

```bash
python src/evaluation/metrics.py --model runs/best.pt --data data/test
```

### Export & Optimize

```bash
# Export to ONNX
python src/optimization/quantize.py --model runs/best.pt --format onnx

# Quantize to INT8
python src/optimization/quantize.py --model runs/best.onnx --format int8

# Benchmark
python src/optimization/benchmark.py --models runs/best.pt runs/best.onnx runs/best_int8.onnx
```

### Run API

```bash
# Local
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Docker
docker build -t warehouse-detector -f docker/Dockerfile .
docker run --gpus all -p 8000:8000 warehouse-detector
```

### API Endpoints

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Health check
curl "http://localhost:8000/health"
```

## Tech Stack

| Category | Tool |
|----------|------|
| 3D Modeling | Blender |
| Image Generation | Stable Diffusion, ControlNet |
| Labeling | CVAT |
| Data Versioning | DVC |
| Augmentation | Albumentations |
| Training | PyTorch, Ultralytics, RT-DETR |
| Experiment Tracking | Weights & Biases |
| Optimization | ONNX, TensorRT |
| Serving | FastAPI |
| Containers | Docker |
| Cloud | AWS SageMaker / GCP Vertex AI |
| MLOps | MLflow, GitHub Actions |

## Experiment Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai/).

View runs: `wandb login` then check your W&B dashboard.

## Timeline

- **Month 1 (Jan-Feb):** Data generation, labeling, pipeline setup
- **Month 2 (Feb-Mar):** Model training, evaluation, comparison
- **Month 3 (Mar-Apr):** Optimization, deployment, MLOps

## License

MIT

## Author

**Jaskirat Singh Sohal**
- GitHub: [@Jsohal174](https://github.com/Jsohal174)
- LinkedIn: [jaskiratsohal](https://linkedin.com/in/jaskiratsohal)
- Email: jsohal03@uoguelph.ca

## Acknowledgments

- Prof. John Akinyemi (University of Guelph) - Research Supervision
- DeepLearning.AI - Deep Learning & PyTorch Certifications