# HPC Infrastructure for MRI Hallucination Detection

Distributed training and scaling infrastructure for the
`mri-hallucination-physics` repository. Benchmarks 22 hallucination
detectors on fastMRI single coil knee data with PyTorch DDP.

## Project Structure
```
configs/             Hydra config files (data, training, detector)
scripts/
  train_ddp.py       Main distributed training entry point
  scaling_benchmark.py   1/2/4 GPU throughput comparison
  aggregate_results.py   Collect results from sweep runs
  prepare_data.py    Validate fastMRI directory structure
  sweep_detectors.sh Sequential detector sweep (local testing)
slurm/
  single_node.sbatch Single node multi GPU job
  array_sweep.sbatch SLURM array job for detector sweep
docker/
  Dockerfile         NGC PyTorch base image
  singularity_build.sh   Apptainer/Singularity for HPC clusters
results/scaling/     Benchmark CSV and plots
```

## Quick Start

Single GPU (Colab):
```bash
python train_ddp.py synthetic=true training.epochs=5
```

Multi GPU (torchrun):
```bash
torchrun --standalone --nproc_per_node=4 train_ddp.py training=multi_gpu
```

Deep ensemble (trains 3 members with different seeds):
```bash
python train_ddp.py detector=deep_ensemble
```

## Key Features

DDP with proper gradient synchronization, SyncBatchNorm, and
distributed evaluation via all_gather. Mixed precision with
GradScaler, gradient clipping, and DDP safe NaN handling (always
calls backward to avoid deadlock, checks gradients after sync).
Linear warmup followed by cosine annealing LR schedule. Checkpoint
resume saves full state (model, optimizer, scheduler, scaler, epoch).
Structured JSON logging (train_log.jsonl). Optional wandb integration
gated behind `use_wandb=true`. Hydra config composition for sweeps.

## HPC Notes

SLURM scripts target generic clusters. UGent HPC (joltik/accelgor)
uses a Torque/PBS frontend over SLURM. Adapt #SBATCH to #PBS
directives and use qsub instead of sbatch. See comments in the
sbatch files.

Apptainer images must be stored on $VSC_SCRATCH on UGent systems.
See docker/singularity_build.sh for build instructions.

## References

Küstner et al. (2024). Predictive uncertainty in deep learning based
MR image reconstruction using deep ensembles: Evaluation on the
fastMRI data set. Magnetic Resonance in Medicine, 92(1), 289-302.

Bhadra et al. (2021). On hallucinations in tomographic image
reconstruction. IEEE TMI, 40(11), 3249-3260.

Zbontar et al. (2018). fastMRI: An open dataset and benchmarks for
accelerated MRI.
