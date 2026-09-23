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

The extractor output is followed by one derived-feature step:

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
| `dataset_randomization` | unset | 20260912 | Shuffle before the train/validation split, so validation mixes both samples |
| `batch_size`, `epochs`, `num_gpu`, `num_dataloader_workers` | command line | 128, 15, 0, 0 | The pilot's launcher values, recorded in the file |

`spanet.train` builds its options in this order: dataset paths from
`-ef`/`-tf`/`-vf`, then the JSON file, then `-e`, `-b`, `-g`, `-f`, and `-r`.
Dataset paths therefore stay out of the JSON, where they would override the
command line. `-r N` sets `dataset_randomization`, not a PyTorch seed. Each run
saves its `options.json` and `event.yaml` in `version_N/`.

SPANet drops incomplete batches, including in validation. A validation set
smaller than one batch produces no validation metric and no best checkpoint.

## Training command

With SPANet's environment and a prepared training file:

```bash
PYTHONNOUSERSITE=1 /isilon/export/home/jhuan166/Vcb/external/SPANet/environment/bin/python \
  -m spanet.train -ef configs/event-vcb.yaml -of configs/options-vcb.json \
  -tf model-input.h5 -l outputs -n vcb
```

On 2026-09-22 this ran for two epochs on the development fixture (190 of 200
events fully matched after the jet cuts), confirming that the three files load
together. The fixture is too small for a meaningful result.

## Not yet decided

- Training, validation, and test samples, their sizes, and a per-event sample label.
- Whether to keep SPANet's standard loss and checkpoint selection, or use the
  pilot's per-sample full-assignment selection.
