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

# Properly initialize conda for non-interactive bash scripts
eval "$(conda shell.bash hook)"
conda activate vulnscan
module load WebProxy

export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "--- Job 1: Baseline Reproduction ---"
# python -m vulscan.test.test \
#     --output_dir results/v2/baseline_verify \
#     --dataset_path ./datasets/test/function_level/ \
#     --language c python java \
#     --model UCSB-SURFI/VulnLLM-R-7B \
#     --save \
#     --use_cot \
#     --batch_size 4 \
#     --vllm --tp 2 \
#     --max_tokens 8192 \
#     --random_cwe 

echo "--- Job 2: Zero-Shot JavaScript (Original Model) ---"
python -m vulscan.test.test \
  --output_dir results/v2/js_zeroshot \
  --dataset_path ./datasets/test/function-level/ \
  --language javascript \
  --model UCSB-SURFI/VulnLLM-R-7B \
  --requests_per_minute 1000 \
  --save \
  --use_cot \
  --batch_size 4 \
  --tp 2 \
  --vllm \
  --max_tokens 8192 \
  --random_cwe

echo "All evaluations complete. Check results/ directory for JSON and logs."