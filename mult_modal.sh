#!/bin/bash
#
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
#SBATCH -p gpu_high
#SBATCH -o log/%j.out
#SBATCH -e log/%j.err

source "/home/xiaoyam/anaconda3/bin/activate"
#module load python3
python src/grid_search.py configs/mult_modal_conf0.json
