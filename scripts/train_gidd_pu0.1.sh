#!/bin/bash
#SBATCH -p math                        # 分区名称
#SBATCH -w hhnode-ib-235               # 指定节点
#SBATCH --gres=gpu:8                   # 申请8块GPU
#SBATCH --ntasks=8                     # 总进程数
#SBATCH --mem=64G                      # 64GB内存
#SBATCH -t 7-00:00:00                  # 最长运行时间7天
#SBATCH -J gidd-train-pu0.1            # 作业名称
#SBATCH -o /scratch/PI/makchen/ddingab/gidd/logs/gidd-train-pu0.1_%j.out  # 标准输出日志
#SBATCH -e /scratch/PI/makchen/ddingab/gidd/logs/gidd-train-pu0.1_%j.err  # 标准错误日志

# 1. 加载环境
source ~/.bashrc
module load cuda/11.2
conda activate gidd_env


# NCCL配置（关键修改）
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

# 使用较小的批次和序列长度
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd \
  logging.run_name="'small-gidd+-owt-pu=0.0-reduced'" \
  training.train_batch_size=8 \
  training.eval_batch_size=8 \
  training.gradient_accumulation_steps=8 \
  +data.max_seq_length=512 \
  training.compile_model=False \
  hydra.run.dir="/scratch/PI/makchen/ddingab/gidd/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"