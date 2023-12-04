#!/bin/bash
#
# Copyright (c) 2023. Imen Ben Amor
#

#SBATCH --job-name=exp
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=output_shap.log
#SBATCH --error=error_shap.log
#SBATCH --mem=64G


source /etc/profile.d/conda.sh
conda activate ba_lr

python3 Step3/attribute_explainer_shap.py

conda deactivate