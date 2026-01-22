#!/bin/bash
##SBATCH --job-name=eval
#SBATCH --output=%x_%j.out    
#SBATCH --error=%x_%j.out 
##SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_h100


ml load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/envs/unsloth/bin/activate

python -m pipeline.distillation.filter_distillation 