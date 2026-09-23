# Running the pipeline

Four steps turn processed MC into a trained model. The first two run on BRUX,
where the MC is stored; training runs on a GPU node. Run commands from the
repository root.

## 1. Extract features (BRUX)

```bash
scripts/extract_mc20260908.sh
```

This runs `nano-spanet-extract` with [`configs/extract-vcb.yaml`](../configs/extract-vcb.yaml)
on the September 8 Summer24 production, in seven parallel parts, and writes
`data/extracted/mc20260908/{signal,background}/part*.h5`. It took about two
minutes on 2026-09-22 with extractor 0.2.0:

| Sample | Files | Input events | Written | Rejected (selected charge -1) | Fully matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTtoLNuCB (signal) | 155 (all) | 5,648,574 | 5,637,440 | 11,134 | 985,035 (17.5%) |
| TTtoLNu2Q (background) | 160 (`batch_0000`-`batch_0159`) | 5,981,834 | 5,966,371 | 15,463 | 1,237,378 (20.7%) |

## 2. Build training, validation, and test files (BRUX)

```bash
micromamba run -n spanet-reco python -m spanet_reco.build_dataset \
  --signal data/extracted/mc20260908/signal/part*.h5 \
  --background data/extracted/mc20260908/background/part*.h5 \
  --output-dir data/datasets/mc20260908-v1
```

- Each source ROOT file goes entirely to one split: 80% of each sample's files
  to training, 10% to validation, and 10% to test, chosen with seed 20260922
  (`--fractions` and `--seed` change them).
- Training and validation keep fully matched events only, with the same number
  from each sample.
- The test split keeps every event of its files, so it also measures the
  fraction of events that are fully matched.
- Every event gets `sin_phi`, `cos_phi`, and `META/sample_id` (0 TTtoLNu2Q,
  1 TTtoLNuCB). Rows are shuffled, so the samples are interleaved.
- `META/source_file` indexes the combined `PROVENANCE/files`, and
  `META/source_entry` gives the ROOT entry, so each event can be traced back.
  `summary.json` and `PROVENANCE/build` record the settings, the files in each
  split, and the counts.

The build took two minutes and 1.1 GB of memory, and wrote 700 MB:

| Split | TTtoLNuCB | TTtoLNu2Q | Source files | Contents |
| --- | ---: | ---: | ---: | --- |
| `train.h5` | 784,060 | 784,060 | 123 + 128 | fully matched, balanced |
| `validation.h5` | 103,324 | 103,324 | 16 + 16 | fully matched, balanced |
| `test.h5` | 558,462 (97,651 matched) | 599,047 (124,367 matched) | 16 + 16 | all events |

`--train-limit N` caps training events per sample, for a quick run.

## 3. Copy to the GPU cluster

Only the built dataset directory is needed; MC and extracted parts stay on BRUX.
Write checksums before copying, so the copy can be verified:

```bash
cd data/datasets/mc20260908-v1
sha256sum summary.json train.h5 validation.h5 test.h5 > SHA256SUMS
```

On the cluster, clone this repository, create the dataset directory, and create
the SPANet environment once. On Brown's Oscar, conda comes from a module and the
environment is built on a login node (it needs internet access):

```bash
cd /oscar/home/jhuan166/Vcb
git clone https://github.com/jiashu-huang/SPANet-reco.git
mkdir -p SPANet-reco/data/datasets SPANet-reco/outputs
module load anaconda3/2023.09-0-aqbc
conda env create -f SPANet-reco/environment-spanet-gpu.yml
```

[`environment-spanet-gpu.yml`](../environment-spanet-gpu.yml) pins the versions
used to develop the configuration (Python 3.11, PyTorch 2.3 with CUDA 12.1,
Lightning 2.4) and SPANet at upstream commit `46c6805`. Training needs only
SPANet, the configs, and the scripts; the `spanet_reco` package is not required
on the cluster.

Then, from BRUX, copy the dataset and verify it on the cluster:

```bash
rsync -ah --partial --info=progress2 \
  /isilon/export/home/jhuan166/Vcb/SPANet-reco/data/datasets/mc20260908-v1 \
  jhuan166@transfer.ccv.brown.edu:/oscar/home/jhuan166/Vcb/SPANet-reco/data/datasets/
```

```bash
cd /oscar/home/jhuan166/Vcb/SPANet-reco/data/datasets/mc20260908-v1 && sha256sum -c SHA256SUMS
```

`--partial` keeps an interrupted file, so rerunning the same command resumes it.

## 4. Train

On Oscar, submit from the repository root:

```bash
cd /oscar/home/jhuan166/Vcb/SPANet-reco
sbatch scripts/slurm_oscar.sh
```

[`scripts/slurm_oscar.sh`](../scripts/slurm_oscar.sh) requests one GPU in the
`gpu` partition, 4 CPUs, 16 GB, and 6 hours; it activates `spanet-gpu`, checks
that PyTorch can run on the allocated GPU, and trains on
`data/datasets/mc20260908-v1` into `outputs/<job id>` with `-g 1 -b 1024`. The
job log is `outputs/slurm-<job id>.out`. Arguments select another dataset,
output directory, or options, and sbatch options override the resources:

```bash
sbatch --time=12:00:00 scripts/slurm_oscar.sh data/datasets/mc20260908-v1 outputs/run2 -g 1 -b 2048 -e 20
```

Every card in the `gpu` partition works with PyTorch 2.3. The Blackwell cards
on `gpu-he` (B200, RTX PRO 6000 Blackwell) need a newer PyTorch; the GPU check
stops such a job before training.

Elsewhere, [`scripts/train.sh`](../scripts/train.sh) runs training directly:

```bash
SPANET_PYTHON=/path/to/spanet-environment/bin/python \
  scripts/train.sh data/datasets/mc20260908-v1 outputs/run1 -g 1 -b 1024
```

`scripts/train.sh` runs `spanet.train` with the event and options files,
`train.h5`, and `validation.h5`. Options after the output directory override
[`configs/options-vcb.json`](../configs/options-vcb.json): `-g 1` uses one GPU,
`-b` sets the batch size, and `-e` the number of epochs (15 by default). The
run directory receives `train.log`, the dataset's `summary.json`, the
repository revision, and SPANet's `vcb/version_N/` with checkpoints,
TensorBoard logs, and copies of the options and event files.

The options keep the pilot's batch size of 128 and learning rate of 0.001. At
batch 128 an epoch is about 12,300 steps; `-b 1024` needs about 1,500. The
learning rate is not scaled with the batch size.

SPANet keeps the checkpoint with the best `validation_average_jet_accuracy`.
On 2026-09-22 a CPU check with an earlier 90/5/5 build of the same extraction
trained on 1% of the training events for one epoch (138 steps), validated on the
whole validation file, and saved a checkpoint at 0.401. That run only confirms
the setup.

## After training

Evaluation on `test.h5` is not yet implemented here. SPANet's own
`python -m spanet.test RUN/vcb/version_0 -tf test.h5` and
`python -m spanet.predict` provide overall metrics and predicted assignments.
