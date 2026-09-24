# Running on Oscar from the command line

A step-by-step route from a built dataset on BRUX to trained and compared SPANet
runs on Brown's Oscar cluster, using only a terminal. Commands marked
**BRUX** run there; all others run on Oscar. [Running the pipeline](running.md)
explains each step in more detail.

```text
one time:  log in -> clone -> create environment
per data:  BRUX: choose features, extract, build -> rsync -> verify checksums
per run:   git pull -> smoke test -> sbatch -> monitor -> results -> compare -> copy back
```

Paths below assume the repository at `/oscar/home/jhuan166/Vcb/SPANet-reco`.
Run every Oscar command from that directory unless stated otherwise.

## 1. Log in

```bash
ssh jhuan166@ssh.ccv.brown.edu
```

It asks for the Brown password and Duo. Long steps (environment creation,
`rsync`) survive a dropped connection inside `tmux` (`tmux new -s spanet`;
reattach with `tmux attach -t spanet`).

## 2. One-time setup

```bash
cd /oscar/home/jhuan166/Vcb
git clone https://github.com/jiashu-huang/SPANet-reco.git
cd SPANet-reco
mkdir -p data/datasets outputs

module load anaconda3/2023.09-0-aqbc
conda env create -f environment-spanet-gpu.yml
PY=$(conda run -n spanet-gpu python -c 'import sys; print(sys.executable)')
$PY -c "import torch, spanet; print(torch.__version__, torch.version.cuda)"
```

Creating the environment takes 10-30 minutes on a login node; if it is slow, add
`--solver=libmamba`. The check prints `2.3.x 12.1`. The login node has no GPU;
the job script checks the GPU when a job starts.

Do not skip this step. A job submitted before the environment exists fails
within seconds, and its `.err` file reads
`EnvironmentNameNotFound: Could not find conda environment: spanet-gpu`.
`ls ~/.conda/envs` shows whether `spanet-gpu` exists.

A faster route is `mamba` from the `miniforge3` module, which built the
environment in about 12 minutes. It must be told where to put the environment:
without `-p`, it tries the read-only module tree and stops with
`cannot create directories: Permission denied`.

```bash
module load miniforge3/25.3.0-3-a6hh
mamba env create -y -p "$HOME/.conda/envs/spanet-gpu" -f environment-spanet-gpu.yml
```

Both modules look in `~/.conda/envs` first, so the job's `conda activate
spanet-gpu` (through `anaconda3`) finds an environment created either way.
Afterwards, in a new shell, run the `module load anaconda3` and `PY=...` lines
above to check it.

`PY` is the environment's Python. Sections 8 and 9 use it; in a new shell, set it
again with the `module load` and `PY=...` lines above. The Slurm job activates
the environment itself.

## 3. Copy the dataset

**BRUX:**

```bash
rsync -ah --partial --info=progress2 \
  /isilon/export/home/jhuan166/Vcb/SPANet-reco/data/datasets/mc20260908-v1 \
  jhuan166@ssh.ccv.brown.edu:/oscar/home/jhuan166/Vcb/SPANet-reco/data/datasets/
```

If the transfer stops, run the same command again to resume. Then, on Oscar:

```bash
cd data/datasets/mc20260908-v1 && sha256sum -c SHA256SUMS && cd -
```

Expect four lines ending in `OK`. A new dataset directory is copied the same way;
write its `SHA256SUMS` on BRUX first (see [running.md](running.md#3-copy-to-the-gpu-cluster)).

## 4. Update the code

```bash
git pull
```

A queued job reads `scripts/train.sh` and the Python code when it starts, not
when it is submitted. Pull before submitting, and avoid pulling while jobs that
should use the previous code are still waiting.

## 5. Smoke test

One epoch on 5% of the training data checks the environment, the GPU, and the
data path in a few minutes:

```bash
sbatch --time=00:30:00 scripts/slurm_oscar.sh data/datasets/mc20260908-v1 outputs/smoke -g 1 -b 1024 -e 1 -p 5
```

The job log `outputs/spanet-vcb-<jobid>.out` should show the PyTorch version, the
GPU name with `check 8.0`, `Training on Full Events only.`, and
`` `Trainer.fit` stopped: `max_epochs=1` reached. ``; `outputs/smoke/vcb/version_0/checkpoints/`
should hold a checkpoint. Errors go to `outputs/spanet-vcb-<jobid>.err`.

If the log instead ends with `IndexError: pop from empty list` raised in
`rich_progress.py`, the environment has Rich 14 or newer, whose `clear_live`
Lightning 2.4 cannot use. The environment file pins `rich<14`; fix an existing
environment with

```bash
module load anaconda3/2023.09-0-aqbc
conda install -n spanet-gpu --override-channels -c pytorch -c nvidia -c conda-forge --solver=libmamba "rich<14"
```

## 6. Submit training

[`scripts/slurm_oscar.sh`](../scripts/slurm_oscar.sh) requests one L40S GPU in
`l40s-gcondo`, 4 CPUs, 32 GB, and 5 hours. Its arguments are the dataset, the
output directory, and training options (default `-g 1 -b 1024`, 15 epochs):

```bash
sbatch --mail-user=you@brown.edu scripts/slurm_oscar.sh \
  data/datasets/mc20260908-v1 outputs/a1-s1 -g 1 -b 1024 --alpha 1 --seed 1
```

Plain SPANet against the mass chi-square loss, three seeds each (six jobs):

```bash
for seed in 1 2 3; do
  for alpha in 1 0.95; do
    sbatch -J "a$alpha-s$seed" --mail-user=you@brown.edu scripts/slurm_oscar.sh \
      data/datasets/mc20260908-v1 "outputs/a$alpha-s$seed" -g 1 -b 1024 --alpha "$alpha" --seed "$seed"
  done
done
```

To train on another set of features, add `-ef configs/event-vcb-<name>.yaml` to
the options; it replaces the default event file. [Choosing training features](features.md)
explains how to write that file, and how to extract a branch that is not yet in
the dataset.

Each run writes `outputs/a<alpha>-s<seed>/`: `train.log`, `dataset-summary.json`,
`spanet-reco-revision.txt`, and `vcb/version_0/` with checkpoints, TensorBoard
events, `options.json`, `event.yaml`, and `mass_chi2.json`. Other options:
`-e` epochs, `--time` for the time limit, `--top-mass` and the other mass
settings (see [SPANet configuration](spanet-config.md#mass-chi-square-loss)).

## 7. Monitor

```bash
squeue -u $USER                                   # queued and running jobs
sacct -u $USER -S today --format=JobID,JobName%16,State,Elapsed,MaxRSS,NodeList
tail -c 3000 outputs/spanet-vcb-<jobid>.out | tr '\r' '\n' | tail -5   # latest progress
srun --jobid=<jobid> --overlap nvidia-smi         # GPU use of a running job
scancel <jobid>                                   # stop a job
```

The progress bar rewrites one line, hence the `tr`. Unlike the local copy,
official SPANet draws its progress bar with Rich when Rich is installed, as it is
here (SPANet needs it), and a log file may show little of it. To follow training
regardless, list the checkpoints (one per finished epoch) or print the logged
metrics (section 8). `sacct` also shows finished, failed, and timed-out jobs.

## 8. Read results

Checkpoint names carry the validation accuracy of their epoch; SPANet keeps the
best three and `last.ckpt`:

```bash
ls outputs/a*/vcb/version_*/checkpoints/
```

Per-epoch validation metrics and training losses from the TensorBoard events:

```bash
$PY - outputs/a0.95-s1/vcb/version_0 <<'EOF'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
events = EventAccumulator(glob.glob(sys.argv[1] + "/events.*")[0], size_guidance={"scalars": 0})
events.Reload()
for tag in ("validation_average_jet_accuracy", "validation_accuracy",
            "loss/spanet_loss", "loss/mass_chi2"):
    if tag in events.Tags()["scalars"]:
        print(tag, " ".join(f"{e.value:.3f}" for e in events.Scalars(tag)[-15:]))
EOF
```

`validation_accuracy` is the fraction of validation events with both tops
correct; `validation_average_jet_accuracy` selects the best checkpoint. The
losses are logged every 50 steps; the command shows the last 15 values. It
also works while a job runs.

## 9. Compare runs

[`scripts/compare_checkpoints.py`](../scripts/compare_checkpoints.py) evaluates
each run's best checkpoint on the same 20.6k validation events, the first 10% of `validation.h5`, per sample. It
loads each run's training data, so run it as a CPU job rather than on the login
node:

```bash
sbatch -J compare -t 01:00:00 -c 4 --mem=16G -o outputs/compare-%j.out --wrap \
  "PYTHONPATH=src PYTHONNOUSERSITE=1 $PY scripts/compare_checkpoints.py \
   outputs/a1-s1/vcb/version_0 outputs/a0.95-s1/vcb/version_0 \
   --output outputs/comparison-s1.json"
cat outputs/compare-<jobid>.out
```

Add more `version_0` directories to compare more runs. The run directories
record absolute Oscar paths in `options.json`, so compare runs on Oscar.

## 10. Resume a run that hit the time limit

`sacct` shows `TIMEOUT`. Continue from the last checkpoint into the same output
directory; SPANet writes the continuation to the next `version_N`:

```bash
sbatch --time=08:00:00 scripts/slurm_oscar.sh data/datasets/mc20260908-v1 outputs/a0.95-s1 \
  -g 1 -b 1024 --alpha 0.95 --seed 1 -cf outputs/a0.95-s1/vcb/version_0/checkpoints/last.ckpt
```

Use the same options as the original run. Compare the newest `version_N`.

## 11. Copy results back to BRUX

**BRUX:**

```bash
mkdir -p /isilon/export/home/jhuan166/Vcb/SPANet-reco/outputs/oscar
rsync -ah --partial --info=progress2 \
  jhuan166@ssh.ccv.brown.edu:/oscar/home/jhuan166/Vcb/SPANet-reco/outputs/ \
  /isilon/export/home/jhuan166/Vcb/SPANet-reco/outputs/oscar/
```

`outputs/` is ignored by Git in both places.

## Quick reference

| Task | Command |
| --- | --- |
| Log in | `ssh jhuan166@ssh.ccv.brown.edu` |
| Environment's Python | `module load anaconda3/2023.09-0-aqbc; PY=$(conda run -n spanet-gpu python -c 'import sys; print(sys.executable)')` |
| Update code | `git pull` |
| Submit | `sbatch scripts/slurm_oscar.sh DATA_DIR OUTPUT_DIR [options]` |
| Other features | add `-ef configs/event-vcb-<name>.yaml` to the options |
| Jobs | `squeue -u $USER`, `sacct -u $USER -S today` |
| Progress | `tail -c 3000 outputs/spanet-vcb-<jobid>.out \| tr '\r' '\n' \| tail -5` |
| Cancel | `scancel <jobid>` |
| Checkpoints | `ls outputs/*/vcb/version_*/checkpoints/` |
