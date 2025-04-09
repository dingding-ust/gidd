#!/bin/bash
#SBATCH -p gpu3090
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -J ar-minimal
#SBATCH -o /scratch/PI/makchen/ddingab/gidd/logs/ar-minimal_%j.out
#SBATCH -e /scratch/PI/makchen/ddingab/gidd/logs/ar-minimal_%j.err

# 加载环境
source ~/.bashrc
module load cuda/11.2
conda activate gidd_env

# 禁用SDPA
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_ATTN_IMPLEMENTATION="eager"
export PYTHONPATH=$PYTHONPATH:/home/ddingab/gidd

# 项目根目录
cd /home/ddingab/gidd

# 最简化命令 - 只添加关键参数
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name ar \
  logging.wandb_project=GIDD-Experiments \
  logging.run_name=small-ar-owt-minimal \
  ++training.compile_model=False \
  ++model.attn_implementation=eager