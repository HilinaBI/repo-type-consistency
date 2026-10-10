#!/bin/bash
#SBATCH --job-name=t5_New_normalized_α1.0
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --chdir=/home/903614/thesis_type_correction
#SBATCH --output=./logs/t5_New_Normalized_clean_repo_with_variants_α0.1_%j.out
#SBATCH --error=./logs/t5_New_Normalized_clean_repo_with_variants_α0.1_%j.err

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "Job started at: $(date)"

source /home/903614/thesis_type_correction/venv/bin/activate

srun /home/903614/thesis_type_correction/venv/bin/python /home/903614/thesis_type_correction/scripts/t5_train_alpha_0.0_MULTI.py

echo "Job finished at: $(date)"
