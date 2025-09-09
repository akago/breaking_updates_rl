#!/bin/bash
#SBATCH --job-name=snellius-intro-exercise
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681


# Load CUDA module (adjust version to match your system)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0



source ~/envs/thesis/bin/activate  # activate your virtual environment
# Install any needed packages
pip install --no-cache-dir -r requirements.txt
 
python gpu_pytorch_mnist.py