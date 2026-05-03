#!/bin/bash
# V2 Script to only run jobs that failed last time due to path issue
#SBATCH --job-name=vulnllm_eval
#SBATCH --partition=gpu          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a30:2         
#SBATCH --time=06:00:00
#SBATCH --output=results/hprc_eval_%j.log

# 1. SETUP ENVIRONMENT (Placeholders for TAMU HPRC)
# 1. SETUP ENVIRONMENT (Placeholders for TAMU HPRC)
module load GCC/12.3.0 CUDA/12.1.1 Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate vulnscan
module load WebProxy

export PYTHONPATH=$PYTHONPATH:$(pwd)

python js-model-training_v2.py