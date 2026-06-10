#!/bin/bash
#SBATCH --job-name=chess-parse
#SBATCH --partition=short          # CPU partition (confirm with: sinfo -s)
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/parse_%j.out
#SBATCH --error=logs/parse_%j.err

mkdir -p logs Training

module load anaconda3/2024.06       # confirm name with: module avail anaconda
source activate chess               # the env you create in Stage 2

# Writes dataset_*.npz into Training/. It runs until the PGN is exhausted; you
# can `scancel` this job once a few chunks exist (check: ls -lh Training/).
python3 DataAndParser/lichessparse.py
