# How the training data were extracted

A short record of how `data/datasets/mc20260908-v1` was made on BRUX on
2026-09-22. The data are not in Git; the full procedure is in
[Running the pipeline](docs/running.md).

## Source MC

Processed NanoAOD from `Calib_ChargeTagger_JH` commit `d212f09` (Summer24,
September 8 production), under `/isilon/export/home/jhuan166/Vcb/MC`:

| Sample | Production | Files used |
| --- | --- | --- |
| TTtoLNuCB (signal, W to cb) | `TTtoLNuCB_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNuCB_syst/roots` | all 155 |
| TTtoLNu2Q (background, W to qq') | `TTtoLNu2Q_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNu2Q_syst/roots` | `batch_0000` to `batch_0159` (160 of 3672) |

## 1. Extraction

`scripts/extract_mc20260908.sh` runs `nano-spanet-extractor` 0.2.0 (commit
`26fae68`) with [`configs/extract-vcb.yaml`](configs/extract-vcb.yaml) in seven
parallel parts (about 2 minutes). The mapping keeps the seven leading jets with
pT > 25 GeV and abs(eta) < 2.4 among the ten saved jets, shows the b-tag and
charge scores of the three highest-b-tag jets (with the `tag_selected`
indicator), rejects events where a selected jet has no charge score, and reads
the trigger lepton, MET, and the `Gen*JetIdx` truth assignments.

| Sample | Input events | Written | Rejected (no charge score) | Fully matched |
| --- | ---: | ---: | ---: | ---: |
| TTtoLNuCB | 5,648,574 | 5,637,440 | 11,134 | 985,035 (17.5%) |
| TTtoLNu2Q | 5,981,834 | 5,966,371 | 15,463 | 1,237,378 (20.7%) |

## 2. Dataset build

```bash
micromamba run -n spanet-reco python -m spanet_reco.build_dataset \
  --signal data/extracted/mc20260908/signal/part*.h5 \
  --background data/extracted/mc20260908/background/part*.h5 \
  --output-dir data/datasets/mc20260908-v1
```

Source files are split 80/10/10 per sample (seed 20260922). Training and
validation keep fully matched events, balanced between samples; the test
split keeps every event. Each event gets `sin_phi`, `cos_phi`, and
`META/sample_id` (0 TTtoLNu2Q, 1 TTtoLNuCB).

| File | TTtoLNuCB | TTtoLNu2Q | Size |
| --- | ---: | ---: | ---: |
| `train.h5` | 784,060 | 784,060 | 387 MB |
| `validation.h5` | 103,324 | 103,324 | 51 MB |
| `test.h5` | 558,462 (97,651 matched) | 599,047 (124,367 matched) | 262 MB |

Each file records its mapping in `PROVENANCE/config` and its build settings,
split files, and counts in `PROVENANCE/build`; `summary.json` holds the same
build record.
