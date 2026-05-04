#!/bin/bash
#SBATCH --job-name=vulnllm_eval_v3
#SBATCH --partition=gpu          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a30:2         
#SBATCH --time=06:00:00
#SBATCH --output=results/v3/hprc_eval_%j.log

module load GCC/12.3.0 CUDA/12.1.1 Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate vulnscan
module load WebProxy

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Job 1: Zero-Shot JavaScript Baseline (fixed CWE hints) ---"
python -m vulscan.test.test \
  --output_dir results/v3/js_baseline \
  --dataset_path ./datasets/test/function_level/ \
  --language javascript \
  --model UCSB-SURFI/VulnLLM-R-7B \
  --save --use_cot \
  --batch_size 4 --tp 2 --vllm \
  --max_tokens 8192 \
  --random_cwe

echo "--- Job 2: JS SFT V2 (fixed CWE hints) ---"
python -m vulscan.test.test \
  --output_dir results/v3/js_sft_v2 \
  --dataset_path ./datasets/test/function_level/ \
  --language javascript \
  --model models/VulnLLM-R-7B-JS-V2 \
  --save --use_cot \
  --batch_size 4 --tp 2 --vllm \
  --max_tokens 8192 \
  --random_cwe

echo "All evaluations complete."