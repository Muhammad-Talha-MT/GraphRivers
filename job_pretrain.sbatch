#!/bin/bash
#SBATCH -J fb_pretrain
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:v100-32:3
#SBATCH --mem=68G
#SBATCH -t 24:00:00
#SBATCH -A ees250003p
#SBATCH -o logs/%x_%j.out

set -e
cd $PROJECT/GraphRivers
source $PROJECT/miniforge3/bin/activate $PROJECT/envs/graphrivers

nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python trainwavenetwu_hourly.py --config config_frenchbroad.yaml
