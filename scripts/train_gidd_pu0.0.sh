#!/bin/bash
#SBATCH -p math                        # 分区名称 (比如gpu)
#SBATCH -w hhnode-ib-236               # 指定节点 (你说只有这台GPU较好)
#SBATCH --gres=gpu:8                   # 申请8块GPU
#SBATCH --ntasks=8                     # 总进程数 (和GPU数量对应)
#SBATCH --mem=64G                      # 64GB内存 (可根据需要调整)
#SBATCH -t 7-00:00:00                  # 最长运行时间7天
#SBATCH -J gidd-train-pu0.0            # 作业名称
#SBATCH -o ../watch_folder/gidd-train-pu0.0_%j.out  # 标准输出日志
#SBATCH -e ../watch_folder/gidd-train-pu0.0_%j.err  # 标准错误日志 (可选)

# 1. 加载环境
source ~/.bashrc
module load cuda/11.2
conda activate gidd_env

# 2. 运行命令: p_u=0.0
cd ..
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd logging.run_name="'small-gidd+-owt-pu=0.0'"
