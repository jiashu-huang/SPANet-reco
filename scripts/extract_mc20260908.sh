#!/usr/bin/env bash
# Extract the September 8 Summer24 processed MC with configs/extract-vcb.yaml:
# signal TTtoLNuCB (all 155 files) and background TTtoLNu2Q (first 160 files),
# as seven parallel parts. Run from the SPANet-reco root; it takes about 2 minutes.
# The extractor refuses to overwrite existing parts. Check logs/status.txt for
# one exit=0 line per part.
set -uo pipefail
M=/isilon/export/home/jhuan166/Vcb/MC
X=../nano-spanet-extractor/.venv/bin/nano-spanet-extract
O=data/extracted/mc20260908
mkdir -p "$O/signal" "$O/background" "$O/logs"
SIG=$M/TTtoLNuCB_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNuCB_syst/roots
BKG=$M/TTtoLNu2Q_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNu2Q_syst/roots
run() {  # sample input-dir part pattern
  "$X" --input-dir "$2" --pattern "$4" --config configs/extract-vcb.yaml \
    --output "$O/$1/part$3.h5" > "$O/logs/$1-part$3.log" 2>&1
  echo "$1 part$3 exit=$?" >> "$O/logs/status.txt"
}
run signal "$SIG" 0 'batch_0[0-4][0-9].root' &
run signal "$SIG" 1 'batch_0[5-9][0-9].root' &
run signal "$SIG" 2 'batch_1[0-4][0-9].root' &
run signal "$SIG" 3 'batch_15[0-9].root' &
run background "$BKG" 0 'batch_00[0-4][0-9].root' &
run background "$BKG" 1 'batch_00[5-9][0-9].root' &
run background "$BKG" 2 'batch_01[0-5][0-9].root' &
wait
echo "all done" >> "$O/logs/status.txt"
