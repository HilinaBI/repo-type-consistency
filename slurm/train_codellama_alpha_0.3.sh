#!/bin/bash
#SBATCH --job-name=llama_alpha_0.5_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=23:59:00
#SBATCH --chdir=/home/903614/thesis_type_correction
#SBATCH --output=./logs/llama_alpha_0.3_%j.out
#SBATCH --error=./logs/llama_alpha_0.3_%j.err

echo "=== CodeLlama 7B Training ==="
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load environment
source /home/903614/thesis_type_correction/codellama_env/bin/activate

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
echo "=== Starting Python Script ==="
srun python -u /home/903614/thesis_type_correction/scripts/codellama_alpha_0.3_train.py

echo "=== Job Complete ==="
echo "Job finished at: $(date)"
