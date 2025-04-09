   #!/bin/bash
   #SBATCH -p gpu3090
   #SBATCH --gres=gpu:8
   #SBATCH --ntasks=8
   #SBATCH --mem=64G
   #SBATCH -t 7-00:00:00
   #SBATCH -J ar-train-no-sdpa
   #SBATCH -o /scratch/PI/makchen/ddingab/gidd/logs/ar-train-no-sdpa_%j.out
   #SBATCH -e /scratch/PI/makchen/ddingab/gidd/logs/ar-train-no-sdpa_%j.err

   # 加载环境
   source ~/.bashrc
   module load cuda/11.2
   conda activate gidd_env

   # NCCL配置
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1
   export NCCL_P2P_DISABLE=1

   # 数据缓存路径
   export HF_DATASETS_CACHE=/scratch/PI/makchen/ddingab/datasets/huggingface/datasets
   export TRANSFORMERS_CACHE=/scratch/PI/makchen/ddingab/datasets/huggingface/transformers
   export TORCH_HOME=/scratch/PI/makchen/ddingab/datasets/torch
   export PYTHONPATH=$PYTHONPATH:/home/ddingab/gidd

   # CUDA内存分配
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

   # 项目根目录
   cd /home/ddingab/gidd

   # 环境变量
   export HYDRA_FULL_ERROR=1
   # 禁用SDPA
   export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
   export TRANSFORMERS_ATTN_IMPLEMENTATION="eager"

   # AR模型命令
   torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name ar \
     logging.wandb_project=GIDD-Experiments \
     logging.run_name=small-ar-owt \
     +training.train_batch_size=8 \
     +training.eval_batch_size=8 \
     +training.gradient_accumulation_steps=8 \
     +data.max_seq_length=2048 \
     +data.truncation=true \
     +training.seed=1 \
     +model.attn_implementation="eager" \
     +logging.save_freq=5000 \
     +training.compile_model=False \
     'hydra.run.dir=/scratch/PI/makchen/ddingab/gidd/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'