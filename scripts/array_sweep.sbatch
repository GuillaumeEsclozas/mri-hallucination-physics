#!/bin/bash
set -euo pipefail

#SBATCH --job-name=mri-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-2

# Sweep across detector configurations using SLURM array jobs.
# Each array index maps to a different detector config.
# IMPORTANT: keep --array range in sync with the DETECTORS array below.

mkdir -p logs

DETECTORS=(
    "unet_baseline"
    "deep_ensemble"
    "mc_dropout"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#DETECTORS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds DETECTORS array (size=${#DETECTORS[@]})"
    exit 1
fi

DETECTOR=${DETECTORS[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID: detector=$DETECTOR"

# ---- Environment ----
module purge
module load Python/3.10
module load CUDA/12.1
source $HOME/envs/mri-halluc/bin/activate

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- Launch ----
# Deep ensemble configs automatically train multiple members
# (train_from_config loops over seeds in the detector config).
torchrun \
    --standalone \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    scripts/train_ddp.py \
    training=multi_gpu \
    detector=$DETECTOR \
    data.data_dir=/scratch/$USER/fastmri/singlecoil_train \
    data.val_dir=/scratch/$USER/fastmri/singlecoil_val \
    checkpoint_dir=/scratch/$USER/checkpoints/${DETECTOR}_${SLURM_ARRAY_JOB_ID} \
    seed=42
