#!/bin/bash
##SBATCH --job-name=eval
#SBATCH --output=%x_%j.out    
#SBATCH --error=logs/%x_%j.out 
##SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100



# Load CUDA module (adjust version to match your system)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source ~/envs/thesis/bin/activate  # activate your virtual environment
# Install any needed packages
# pip install --no-cache-dir -r requirements.txt
 
python -m pipeline.eval_test -i /home/xchen6/breaking_updates_rl/data/prompts_no_comments