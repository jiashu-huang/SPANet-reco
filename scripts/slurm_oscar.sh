#!/bin/bash
# Train SPANet on one GPU of Brown's Oscar cluster.
#
# Submit from the repository root, after `mkdir -p outputs`:
#   sbatch scripts/slurm_oscar.sh [DATA_DIR] [OUTPUT_DIR] [options...]
# Defaults: DATA_DIR=data/datasets/mc20260908-v1, OUTPUT_DIR=outputs/<job id>,
# options "-g 1 -b 1024". Options are those of scripts/train.sh: spanet.train
# options plus --alpha (mass chi-square loss) and --seed, for example
#   sbatch scripts/slurm_oscar.sh data/datasets/mc20260908-v1 outputs/a095-s1 -g 1 -b 1024 --alpha 0.95 --seed 1 Resources below follow the group's Oscar template and
# can be overridden on the sbatch command line, for example --time=08:00:00 or
# --mail-user=you@brown.edu for the end-of-job email.
#
# The L40S cards of l40s-gcondo work with the environment's PyTorch 2.3
# (CUDA 12.1). Blackwell cards (B200, RTX PRO 6000 Blackwell) do not; the check
# below stops a job on such a card before training.
#SBATCH --job-name=spanet-vcb
#SBATCH --partition=l40s-gcondo
#SBATCH --time=5:00:00             # run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=32G                  # memory per node
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --error=outputs/spanet-vcb-%A.err   # %A: job ID
#SBATCH --output=outputs/spanet-vcb-%A.out
##SBATCH --mail-type=begin         # email when the job begins
#SBATCH --mail-type=end            # email when the job ends
##SBATCH --mail-user=email@brown.edu   # or pass --mail-user to sbatch
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
