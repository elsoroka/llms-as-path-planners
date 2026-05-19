#!/bin/sh  
#SBATCH --job-name=qwen-coder-30b-vllm
#SBATCH -p batch  
#SBATCH --nodes=1
#SBATCH -A marlowe-m000186-pm05
#SBATCH -G 4
#SBATCH -c 2
#SBATCH --mem-per-gpu=64G
#SBATCH --time=4:00:00
#SBATCH --output=vllm_qwen-coder-30b-instruct_%A.out
#SBATCH --error=vllm_qwen-coder-30b-instruct_error_%A.out
#SBATCH --export=ALL

module load conda
CONDA_BASE="/marlowe/apps/Mambaforge/24.3.0-0"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate /scratch/m000186/esoroka/

vllm serve "Qwen/Qwen3-Coder-30B-A3B-Instruct" --port 8000 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 32768 &

echo "Waiting for vLLM to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
PROVIDER_ARGS="--provider vllm --model $MODEL --base-url http://localhost:8000/v1"
TESTSETS="test-sets/ICL_test_set_updated.json test-sets/ICL_test_set_5x5_updated.json test-sets/ICL_test_set_7x7_updated.json test-sets/ICL_test_set_moreobsts_updated.json"

mkdir -p outputs

for TESTSET in $TESTSETS; do
    STEM=$(basename $TESTSET .json)

    echo "--- Baseline: $STEM ---"
    python run_baseline.py $TESTSET outputs/baseline_${STEM}_qwen-30b.jsonl \
        --max-samples 100 $PROVIDER_ARGS

    echo "--- Code-form: $STEM ---"
    python code_form.py $TESTSET outputs/code_form_${STEM}_qwen-30b.jsonl \
        --max-samples 100 $PROVIDER_ARGS
done

echo "All jobs done."
