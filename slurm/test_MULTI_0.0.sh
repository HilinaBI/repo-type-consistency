#!/bin/bash
#SBATCH --job-name=test_new_normalized_alpha_1.0
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/test_new_clean_repo_normalized_alpha_1.0_with_variant_inference_%j.out
#SBATCH --error=./logs/test_new_clean_repo_normalized_alpha_1.0_with_variant_inference_%j.err
#SBATCH --partition=gpus
#SBATCH --mem=8GB

cd /home/903614/thesis_type_correction
source venv/bin/activate

python scripts/test_MULTI_0.0.py
