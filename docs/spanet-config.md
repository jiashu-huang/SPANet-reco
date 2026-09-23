# SPANet configuration

SPANet selects the two hadronic-W daughter jets in each event, as an unordered
pair, together with the hadronic and leptonic top b jets. It does not decide
which W jet is the up- or down-type quark; b-tagging of the selected W jets is
applied separately afterwards.

Three files define a training run:

| File | Read by | Defines |
| --- | --- | --- |
| [`configs/extract-vcb.yaml`](../configs/extract-vcb.yaml) | `nano-spanet-extract` | Which ROOT branches become features and targets, and every cut |
| [`configs/event-vcb.yaml`](../configs/event-vcb.yaml) | SPANet (`-ef`) | Which HDF5 features the model reads, their preprocessing, the particles to assign, and their symmetry |
| [`configs/options-vcb.json`](../configs/options-vcb.json) | SPANet (`-of`) | Network size, loss, optimizer, and training settings |

`tests/test_configs.py` checks that the event file and the extraction mapping
agree on feature and target names.

## Preparing the input

[`spanet_reco.build_dataset`](running.md#2-build-training-validation-and-test-files-brux)
adds the derived features while building the training files. For a single
extracted file, the same step is available on its own:

```bash
micromamba run -n spanet-reco python -m spanet_reco.features extracted.h5 model-input.h5
```

It copies the file unchanged and adds `sin_phi` and `cos_phi` to every
`INPUTS` group that has `phi` (Jets, Lepton, Met), with zero on padded jets.
`PROVENANCE/angular_features` records the step. Raw `phi` stays in the file
for traceability; the event file does not read it.

Raw phi jumps from +pi to -pi although both are the same direction, so about a
quarter of evenly spread jet pairs have a misleading raw difference. On the unit
circle, nearby directions always have nearby inputs, and cos(delta phi), which
enters every pair mass, is a product of the inputs:
cos(phi1) cos(phi2) + sin(phi1) sin(phi2).

## Event file

- `INPUTS`: group and feature names are HDF5 paths (`INPUTS/Jets/pt`). Only
  listed features are read, in the listed order; `MASK` is read automatically.
  `SEQUENTIAL` must come before `GLOBAL` for this SPANet version.
- Preprocessing: `log_normalize` for pT and mass (log(x + 1), then training-set
  mean and standard deviation over real objects), `normalize` for eta, sin phi,
  and cos phi, and `none` for the scores, `tag_selected`, and lepton charge,
  which are already in [0, 1] or equal to -1 or +1.
- `EVENT`: `had_top` (b, q1, q2) and `lep_top` (b), all drawn from `Jets`. The
  names are also the target paths `TARGETS/<particle>/<daughter>`.
- `PERMUTATIONS`: `had_top: [q1, q2]` makes the W pair unordered in the output,
  the loss, and decoding. Upstream labels are ordered (`GenHadQ1` down-type,
  `GenHadQ2` up-type), but that order is deliberately not learned.

The b-tag and charge scores remain model inputs. They help identify the top b
jets, but they also influence which jets SPANet places in the W. A b-tag yield
measured on the selected W jets therefore depends on the trained selection, which
must be taken from MC.

## Options file

The values are those of the pilot study
(`spanet-test/options_files/kinematics.json`) with these changes:

| Option | Pilot | Here | Reason |
| --- | --- | --- | --- |
| `partial_events` | true | false | Train on fully matched events only, as in the [target contract](target-contract.md) |
| `detection_loss_scale` | 1.0 | 0.0 | With fully matched events only, the detection target is always 1 |
| `dataset_randomization` | unset | 20260912 | Shuffle rows before any subset (`-p`) or split taken from the training file; built datasets are already shuffled |
| `batch_size`, `epochs`, `num_gpu`, `num_dataloader_workers` | command line | 128, 15, 0, 0 | The pilot's launcher values, recorded in the file |

`spanet.train` builds its options in this order: dataset paths from
`-ef`/`-tf`/`-vf`, then the JSON file, then `-e`, `-b`, `-g`, `-f`, and `-r`.
Dataset paths therefore stay out of the JSON, where they would override the
command line. `-r N` sets `dataset_randomization`, not a PyTorch seed. Each run
saves its `options.json` and `event.yaml` in `version_N/`.

SPANet drops incomplete batches, including in validation. A validation set
smaller than one batch produces no validation metric and no best checkpoint.

## Training command

[`scripts/train.sh`](../scripts/train.sh) runs `python -m spanet_reco.train`
with these files and a built dataset; see [Running the pipeline](running.md).
`spanet_reco.train` accepts every `spanet.train` option. With a separate
validation file, SPANet validates on all of it, and `train_validation_split` is
unused.

## Mass chi-square loss

[`spanet_reco.model`](../src/spanet_reco/model.py) trains SPANet's unchanged
network with the loss

    L = alpha * L_SPANet + (1 - alpha) * <chi2>
    chi2(b, q1, q2) = ((m(b q1 q2) - 172.5) / 20)^2 + ((m(q1 q2) - 80.4) / 15)^2

- `L_SPANet` is SPANet's own loss: the cross-entropy of the true assignment,
  averaged over `had_top` and `lep_top`.
- `chi2` is computed for every jet triplet (b, q1, q2) of the hadronic top, with
  q1 and q2 the W jets. Jet four-vectors come from the model inputs (mass, pt,
  eta, sin phi, cos phi), before SPANet standardizes them.
- `<chi2>` is the average of `chi2` under the network's `had_top` probabilities
  P(b, q1, q2), over the events of a batch. The true assignment does not enter
  it, and the choice of the single most probable triplet has no gradient; the
  probability-weighted average moves probability toward triplets with top- and
  W-like masses.
- `alpha = 1` (the default) reproduces SPANet's loss exactly. The network is
  unchanged, so checkpoints load in `spanet.test` and `spanet.predict`, and the
  best checkpoint is still chosen by `validation_average_jet_accuracy`.

The settings are options of `spanet_reco.train`: `--alpha`, `--top-mass`,
`--top-width`, `--w-mass`, and `--w-width`, plus `--seed`, which seeds Python,
NumPy, and PyTorch (`spanet.train` sets no seed; its `-r` only shuffles the
data). They are saved as `mass_chi2.json` in the run's `version_N/` and cannot
go in `options-vcb.json`, because SPANet rejects unknown option names.
TensorBoard receives `loss/spanet_loss`, `loss/mass_chi2` (with its
`_top` and `_w` parts), and `loss/combined_loss`.

`spanet_reco.train` replaces the model class that `spanet.train` builds and then
runs `spanet.train` unmodified. `MassChi2Model.spanet_loss` copies SPANet's loss
code from the pinned commit `46c6805`, so that the network runs once per batch
for both terms; `tests/spanet_model_checks.py` checks that `alpha = 1` equals
SPANet's own loss. Run those checks with
`SPANET_PYTHON=/path/to/spanet/bin/python pytest`.

### CPU test runs (2026-09-23)

Three short runs on BRUX differ only in `alpha`: 4% of `mc20260908-v1/train.h5`
(59.6k training events, with 3.1k events from the same slice for validation),
batch 256, 3 epochs, `--seed 20260923`, 4 CPU threads. Each run's best
checkpoint (the last epoch in every case) was then evaluated on the first 10% of
`validation.h5` (10,353 TTtoLNu2Q and 10,312 TTtoLNuCB events), which no run
used. "min chi2" chooses the triplet with the smallest chi-square, with no
network. Windows are 40 GeV around 172.5 and 30 GeV around 80.4; widths are the
interquartile range divided by 1.349.

| TTtoLNu2Q / TTtoLNuCB | alpha 1 | alpha 0.95 | alpha 0.5 | min chi2 | true jets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both tops correct (%) | 64.9 / 44.7 | 65.0 / 46.4 | 38.7 / 30.7 | | |
| Hadronic top correct (%) | 66.1 / 45.3 | 66.3 / 47.1 | 39.1 / 31.0 | 34.7 / 33.2 | |
| W pair correct (%) | 74.6 / 49.4 | 75.3 / 51.7 | 48.8 / 38.2 | 43.7 / 41.7 | |
| Leptonic b correct (%) | 83.4 / 77.6 | 82.9 / 77.0 | 79.5 / 72.7 | | |
| m(bjj) in window (%) | 82.4 / 80.1 | 86.0 / 83.6 | 85.8 / 82.6 | 96.8 / 96.7 | 88.8 / 85.6 |
| m(jj) in window (%) | 86.9 / 80.6 | 90.6 / 86.0 | 87.4 / 85.6 | 99.4 / 99.5 | 93.7 / 93.7 |
| m(bjj) width (GeV) | 26.0 / 26.1 | 24.3 / 24.1 | 24.7 / 24.9 | 12.8 / 13.2 | 23.4 / 22.9 |
| m(jj) width (GeV) | 16.0 / 19.5 | 15.1 / 17.3 | 16.8 / 18.0 | 7.5 / 7.5 | 13.7 / 13.6 |

Per-epoch training averages of `<chi2>` were 117 → 21 → 13.7 for `alpha = 1`
(recorded, not in its loss), 60 → 8.0 → 5.5 for 0.95, and 58 → 8.0 → 5.3 for 0.5;
SPANet's loss ended at 1.02, 1.05, and 2.00.

- `alpha = 0.5` loses assignment accuracy: the chi-square term, tens of units
  against about 2 for SPANet's loss, dominates, and its minimum is usually not
  the true triplet (median chi-square 1.3-1.5 for the true triplet, 0.45 for the
  smallest one), so the network moves toward the min-chi2 choice.
- `alpha = 0.95` matches or exceeds plain SPANet in accuracy and moves the chosen
  masses toward the top and W peaks (3-5 more points inside the windows,
  narrower peaks). Its chi-square term dominates only in the first epoch.
- These are single-seed, 3-epoch runs on 4% of the data; accuracies have a
  statistical uncertainty of about 0.5 points. The signal gains of `alpha = 0.95`
  (+1.7 both tops, +2.3 W pair) need full-data runs with several seeds.

To repeat a run and the comparison (SPANet environment, repository root):

```bash
PYTHONPATH=src python -m spanet_reco.train -ef configs/event-vcb.yaml \
  -of configs/options-vcb.json -tf data/datasets/mc20260908-v1/train.h5 \
  -l outputs/test-chi2 -n alpha-0.95 -e 3 -b 256 -p 4 --alpha 0.95 --seed 20260923
PYTHONPATH=src python scripts/compare_checkpoints.py \
  outputs/test-chi2/alpha-1.0/version_0 outputs/test-chi2/alpha-0.95/version_0 \
  outputs/test-chi2/alpha-0.5/version_0
```

## Not yet decided

- The value of `alpha`, and whether to select checkpoints by the pilot's
  per-sample full-assignment accuracy instead of SPANet's jet accuracy.
- Evaluation on the test split, reported separately for each sample.
