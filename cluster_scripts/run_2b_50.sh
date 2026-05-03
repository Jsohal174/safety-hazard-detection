#!/bin/bash
#SBATCH --account=def-drjohn
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/2b_50_%j.out

module load python/3.10
source ~/hawkeye_env/bin/activate

mkdir -p logs results checkpoints

python ~/hawkeye_data/train.py --model 2B --num_images 50 --output_dir checkpoints/2b_50
python ~/hawkeye_data/evaluate.py --model_dir checkpoints/2b_50 --base_model Qwen/Qwen2.5-VL-3B-Instruct --output results/2b_50.json
