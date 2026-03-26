#!/bin/bash
# Launch detector sweep. Two modes:
#
# SLURM (preferred): submit as array job
#   sbatch slurm/array_sweep.sbatch
#
# Local sequential (for testing without SLURM):
#   bash scripts/sweep_detectors.sh <data_dir> <val_dir> <output_root>
#
# Deep ensemble configs train multiple members automatically
# (train_from_config loops over the seeds list in the config).

set -euo pipefail

DATA_DIR=${1:?Usage: sweep_detectors.sh <data_dir> <val_dir> <output_root>}
VAL_DIR=${2:?}
OUTPUT_ROOT=${3:?}
NGPUS=${4:-1}

DETECTORS=("unet_baseline" "deep_ensemble" "mc_dropout")

for det in "${DETECTORS[@]}"; do
    echo "=== Training detector: $det ==="

    if [ "$NGPUS" -gt 1 ]; then
        torchrun --standalone --nproc_per_node=$NGPUS \
            scripts/train_ddp.py \
            detector=$det \
            data.data_dir=$DATA_DIR \
            data.val_dir=$VAL_DIR \
            checkpoint_dir=$OUTPUT_ROOT/$det \
            training=multi_gpu \
            seed=42
    else
        python scripts/train_ddp.py \
            detector=$det \
            data.data_dir=$DATA_DIR \
            data.val_dir=$VAL_DIR \
            checkpoint_dir=$OUTPUT_ROOT/$det \
            seed=42
    fi

    echo ""
done

echo "=== Aggregating results ==="
python scripts/aggregate_results.py $OUTPUT_ROOT $OUTPUT_ROOT/summary.csv
