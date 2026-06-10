#!/bin/bash
#SBATCH --job-name=chess-train
#SBATCH --partition=gpu            # GPU partition (confirm with: sinfo -s)
#SBATCH --gres=gpu:1               # one GPU; add a type if required, e.g. gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # enough to load a few float32 chunks into RAM
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs

module load anaconda3/2024.06
module load cuda/12.1              # confirm with: module avail cuda
source activate chess

python3 Training/final_train.py
