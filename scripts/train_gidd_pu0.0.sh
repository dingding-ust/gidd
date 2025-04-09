#!/bin/bash
#SBATCH -p math
#SBATCH -w hhnode-ib-236
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -J gidd-train-pu0.0
#SBATCH -o /scratch/PI/makchen/ddingab/gidd/logs/gidd-train-pu0.0_%j.out
#SBATCH -e /scratch/PI/makchen/ddingab/gidd/logs/gidd-train-pu0.0_%j.err

# 1. 加载环境
source ~/.bashrc
module load cuda/11.2
conda activate gidd_env

# NCCL配置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 2. 设置数据缓存路径
export HF_DATASETS_CACHE=/scratch/PI/makchen/ddingab/datasets/huggingface/datasets
export TRANSFORMERS_CACHE=/scratch/PI/makchen/ddingab/datasets/huggingface/transformers
export TORCH_HOME=/scratch/PI/makchen/ddingab/datasets/torch
export PYTHONPATH=$PYTHONPATH:/home/ddingab/gidd

# 设置CUDA内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 3. 确保切换到项目根目录
cd /home/ddingab/gidd

# 添加环境变量以获取完整错误信息
export HYDRA_FULL_ERROR=1

# 使用较小的批次和序列长度 - 修复参数格式
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd \
  logging.wandb_project=GIDD-Experiments \
  logging.run_name=small-gidd-owt-pu0.0-fixed \
  model.p_uniform=0.0 \
  training.train_batch_size=8 \
  training.eval_batch_size=8 \
  +training.gradient_accumulation_steps=8 \
  +data.max_seq_length=2048 \
  +data.truncation=true \
  +data.preprocessing.do_truncation=true \
  +data.preprocessing.max_length=2048 \
  training.seed=1 \
  logging.save_freq=5000 \
  training.compile_model=False \
  'hydra.run.dir=/scratch/PI/makchen/ddingab/gidd/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'