#!/bin/bash
# Build an Apptainer (formerly Singularity) container from the Docker image.
#
# UGent HPC uses Apptainer. Key constraints:
#   1. SIF images MUST be stored on $VSC_SCRATCH, not $VSC_HOME or $VSC_DATA.
#      The centrally provided apptainer command refuses to run images from
#      other filesystems.
#   2. Direct docker:// pulls are not supported on UGent infrastructure.
#      Build the SIF locally or on a build node, then copy to $VSC_SCRATCH.
#   3. GPU access requires the --nv flag.
#
# Build workflow:
#   On a machine with root/fakeroot (laptop, CI, or interactive build node):
#     docker build -t mri-halluc:latest -f docker/Dockerfile .
#     docker push ghcr.io/<user>/mri-halluc:latest
#     apptainer build container.sif docker://ghcr.io/<user>/mri-halluc:latest
#
#   On UGent cluster:
#     cp container.sif $VSC_SCRATCH/containers/
#     apptainer exec --nv $VSC_SCRATCH/containers/container.sif python scripts/train_ddp.py
#
# Target clusters at UGent:
#   joltik:  4x NVIDIA V100 (32GB) per node, 32 cores
#   accelgor: 4x NVIDIA A100 (80GB) per node, 48 cores

set -euo pipefail

IMAGE_URI="${1:-docker://ghcr.io/guillaumeesclozas/mri-halluc:latest}"
OUTPUT_SIF="${2:-container.sif}"

echo "Building Apptainer image from: $IMAGE_URI"
echo "Output: $OUTPUT_SIF"

apptainer build "$OUTPUT_SIF" "$IMAGE_URI"

echo ""
echo "Build complete. Test with:"
echo "  apptainer exec --nv $OUTPUT_SIF python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "On UGent, copy to scratch before running:"
echo "  cp $OUTPUT_SIF \$VSC_SCRATCH/containers/"
