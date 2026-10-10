#!/bin/bash
#SBATCH --job-name=llama_test_alpha_0.5
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=23:59:00
#SBATCH --chdir=/home/903614/thesis_type_correction
#SBATCH --output=./logs/llama_test_alpha_0.5_%j.out
#SBATCH --error=./logs/llama_test_alpha_0.5_%j.err

echo "=== CodeLlama 7B Testing ==="
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load environment
source /home/903614/thesis_type_correction/codellama_env/bin/activate

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run testing
echo "=== Starting Python Test Script ==="
srun python -u /home/903614/thesis_type_correction/scripts/testscript_codellama.py

echo "=== Testing Complete ==="
echo "Job finished at: $(date)"
