#!/bin/bash
#
# Copyright (c) 2023. Imen Ben Amor
#

#SBATCH --job-name=exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --mem=32G


source /etc/profile.d/conda.sh
conda activate ba_lr


# cp -r /local_disk/clytie/ibenamor/phd_experiments/data/Explainability/BA/* ./data/BA/
# python3 Step3/attribute_explainer.py

python3 Step3/attribute_explainer_lime.py
# python3 -m pdb Step3/attribute_explainer_lime.py
# python3 -m pdb Step3/attribute_explainer.py
# python3 fix_problem.py
# python3 correct_files.py

conda deactivate