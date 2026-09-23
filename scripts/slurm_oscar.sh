#!/bin/bash
# Train SPANet on one GPU of Brown's Oscar cluster.
#
# Submit from the repository root, after `mkdir -p outputs`:
#   sbatch scripts/slurm_oscar.sh [DATA_DIR] [OUTPUT_DIR] [spanet.train options...]
# Defaults: DATA_DIR=data/datasets/mc20260908-v1, OUTPUT_DIR=outputs/<job id>,
# options "-g 1 -b 1024". Resources below can be overridden on the sbatch
# command line, for example --time=08:00:00.
#
# The gpu partition's cards all work with the environment's PyTorch 2.3
# (CUDA 12.1). Blackwell cards (B200, RTX PRO 6000 Blackwell) on gpu-he do not;
# the check below stops such a job before training.
#SBATCH --job-name=spanet-vcb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
data="${1:-data/datasets/mc20260908-v1}"
output="${2:-outputs/${SLURM_JOB_ID:-local}}"
shift $(( $# < 2 ? $# : 2 ))
options=("$@")
[ "${#options[@]}" -gt 0 ] || options=(-g 1 -b 1024)

# Module and conda activation scripts may read unset variables.
set +u
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spanet-gpu
set -u

echo "host $(hostname), job ${SLURM_JOB_ID:-none}, data $data, output $output, options ${options[*]}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Fail early if this PyTorch build cannot run kernels on the allocated GPU.
python -c "import torch; x = torch.ones(8, device='cuda'); \
print(torch.__version__, torch.cuda.get_device_name(), 'check', (x @ x).item())"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SPANET_PYTHON=python scripts/train.sh "$data" "$output" "${options[@]}"
