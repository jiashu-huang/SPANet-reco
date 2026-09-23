#!/usr/bin/env bash
# Train SPANet on a dataset written by spanet_reco.build_dataset.
#
# Usage: scripts/train.sh DATA_DIR OUTPUT_DIR [options...]
#   DATA_DIR   directory with train.h5 and validation.h5
#   OUTPUT_DIR run directory; checkpoints go to OUTPUT_DIR/vcb/version_N
# Options are those of spanet.train, which override configs/options-vcb.json, and
# the mass chi-square settings of spanet_reco.train (--alpha, default 1 = plain
# SPANet loss), for example on one GPU:
#   scripts/train.sh data/datasets/mc20260908-v1 outputs/run1 -g 1 -b 1024 --alpha 0.5
# SPANET_PYTHON selects the Python of an environment with SPANet (default: python).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,11p' "$0" >&2
  exit 2
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data="$(cd -- "$1" && pwd)"
output="$2"
shift 2
for split in train validation; do
  [ -f "$data/$split.h5" ] || { echo "missing $data/$split.h5" >&2; exit 1; }
done
mkdir -p "$output"
# Record which dataset and code revision this run used.
cp "$data/summary.json" "$output/dataset-summary.json"
git -C "$repo" describe --always --dirty > "$output/spanet-reco-revision.txt" 2>/dev/null || true

# spanet_reco.train runs from this checkout; the package need not be installed.
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/spanet-matplotlib}" \
  "${SPANET_PYTHON:-python}" -u -m spanet_reco.train \
  -ef "$repo/configs/event-vcb.yaml" \
  -of "$repo/configs/options-vcb.json" \
  -tf "$data/train.h5" \
  -vf "$data/validation.h5" \
  -l "$output" -n vcb "$@" 2>&1 | tee -a "$output/train.log"
