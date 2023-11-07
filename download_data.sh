#!/bin/bash
#
# Copyright (c) 2023. Imen Ben Amor
#

#SBATCH --job-name=ba_lr
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --output=ba_lr_output.log
#SBATCH --error=ba_lr_error.log
#SBATCH --mem=32G


source /etc/profile.d/conda.sh
conda activate ba_lr

wget "https://filesender.renater.fr/download.php?token=66c1be29-2f57-476a-be3a-ce470c437a54&files_ids=30866254"

conda deactivate