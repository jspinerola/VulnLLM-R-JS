#!/bin/bash
# HPRC Master Evaluation Script for VulnLLM-R & JS-SFT
# NOTE: Update placeholders (partitions, account info) based on your TAMU HPRC quota.

#SBATCH --job-name=vulnllm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=results/hprc_eval_%j.log

# 1. SETUP ENVIRONMENT (Placeholders for TAMU HPRC)
# module load GCC/12.3.0 CUDA/12.1.1 Anaconda3/2023.09-0
# source activate vulnscan

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. MERGE MODEL (Runs once)
if [ ! -d "models/VulnLLM-R-7B-JS" ]; then
    echo "Merging JS-SFT Adapter..."
    python merge_js_model.py
fi

# 3. RUN EVALUATIONS
echo "--- Job 1: Baseline Reproduction (C/Python/Java) ---"
python -m vulscan.test.test \
    --output_dir results/baseline_verify \
    --dataset_path ./datasets/test/function_level/ \
    --language c python java \
    --model UCSB-SURFI/VulnLLM-R-7B \
    --save --use_cot --use_policy --vllm --tp 2 --batch_size 16

echo "--- Job 2: Zero-Shot JavaScript (Original Model) ---"
python -m vulscan.test.test \
    --output_dir results/js_zeroshot \
    --dataset_path ./datasets/test/function_level/javascript/ \
    --language javascript \
    --model UCSB-SURFI/VulnLLM-R-7B \
    --save --use_cot --use_policy --vllm --tp 2 --batch_size 16

echo "--- Job 3: JavaScript SFT Evaluation (Tuned Model) ---"
python -m vulscan.test.test \
    --output_dir results/js_sft_eval \
    --dataset_path ./datasets/test/function_level/javascript/ \
    --language javascript \
    --model models/VulnLLM-R-7B-JS \
    --save --use_cot --use_policy --vllm --tp 2 --batch_size 16

echo "--- Job 4: Regression Testing (Tuned Model on C/Py/Java) ---"
python -m vulscan.test.test \
    --output_dir results/regression_check \
    --dataset_path ./datasets/test/function_level/ \
    --language c python java \
    --model models/VulnLLM-R-7B-JS \
    --save --use_cot --use_policy --vllm --tp 2 --batch_size 16

echo "All evaluations complete. Check results/ directory for JSON and logs."
