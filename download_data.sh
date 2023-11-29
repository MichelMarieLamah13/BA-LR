#!/bin/bash
#
# Copyright (c) 2023. Imen Ben Amor
#

#SBATCH --job-name=exp
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log
# # #SBATCH --mem=32G


source /etc/profile.d/conda.sh
conda activate ba_lr

cp -r /local_disk/clytie/ibenamor/phd_experiments/data/Explainability/BA/* ./data/

conda deactivate