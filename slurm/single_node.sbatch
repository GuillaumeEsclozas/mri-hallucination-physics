#!/bin/bash
set -euo pipefail

#SBATCH --job-name=mri-halluc-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# NOTE: These scripts target generic SLURM clusters.
# UGent HPC uses a Torque/PBS frontend over SLURM. To adapt:
#   Replace #SBATCH with #PBS directives
#   Replace sbatch with qsub
#   Use: #PBS -l nodes=1:ppn=16:gpus=4
#   Use: #PBS -l walltime=04:00:00
# See https://docs.hpc.ugent.be for UGent specifics.

mkdir -p logs

# ---- Environment ----
module purge
module load Python/3.10
module load CUDA/12.1
source $HOME/envs/mri-halluc/bin/activate

# ---- NCCL configuration ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# ---- Reproducibility ----
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- Launch ----
torchrun \
    --standalone \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    scripts/train_ddp.py \
    training=multi_gpu \
    data.data_dir=/scratch/$USER/fastmri/singlecoil_train \
    data.val_dir=/scratch/$USER/fastmri/singlecoil_val \
    checkpoint_dir=/scratch/$USER/checkpoints/run_${SLURM_JOB_ID} \
    seed=42
