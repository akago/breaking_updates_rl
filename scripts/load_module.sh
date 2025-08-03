#!/bin/bash

ml load Python/3.11.3-GCCcore-12.3.0 
ml load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
ml load Transformers/4.39.3-gfbf-2023a
# export GH_TOKEN=$(grep -v '^#' Breaking-Updates-Repair/.env | xargs)
