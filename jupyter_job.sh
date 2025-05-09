#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --output=jupyter_output_%j.log

module load gnu9/9.4.0
module load cuda/10.2

source ~/miniconda3/etc/profile.d/conda.sh

conda activate ImgProc_GPU_env

# Run Jupyter Notebook
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0


# watch -n 1 'nvidia-smi; echo ""; top -b -n 1 | head -20'