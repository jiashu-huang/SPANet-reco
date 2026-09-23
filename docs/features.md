# Choosing training features

Two YAML files decide what SPANet trains on. The extraction mapping decides
which ROOT branches reach the dataset; the event file decides which of those
the model reads, and how each is preprocessed.

| File | Decides | After a change |
| --- | --- | --- |
| [`configs/extract-vcb.yaml`](../configs/extract-vcb.yaml) | Branches written to the dataset, jet and tag selections, targets | Extract, build, copy, and train again |
| [`configs/event-vcb.yaml`](../configs/event-vcb.yaml) | Features the model reads, their preprocessing, particles and symmetry | Train again only |
| `spanet_reco.build_dataset` | Derived features: `sin_phi` and `cos_phi` for every input with `phi` | Build, copy, and train again |
| [`configs/options-vcb.json`](../configs/options-vcb.json) | Network size and training settings; input sizes follow the event file | Nothing |

Choose the route by what the feature is:

- **Already in the dataset** (drop a feature, or change its preprocessing):
  route A, an event file only.
- **A branch of the processed MC that is not extracted yet:** route B, a new
  mapping and dataset.
- **Computed from other features:** route C, code in the build step.

Keep the defaults for the reference run and add each variant as a new file,
`configs/event-vcb-<name>.yaml` (and `configs/extract-vcb-<name>.yaml` for
route B), so that runs remain comparable. `tests/test_configs.py` checks every
`configs/event-*.yaml` against `extract-<name>.yaml` when it exists and
`extract-vcb.yaml` otherwise.

## Features in the current dataset

`data/datasets/mc20260908-v1` holds these model features (`MASK` marks real jets):

| Input | Features |
| --- | --- |
| `Jets` (sequential, 7 slots) | `mass`, `pt`, `eta`, `phi`, `sin_phi`, `cos_phi`, `btag_raw`, `charge_raw`, `tag_selected` |
| `Lepton` (global) | `pt`, `eta`, `phi`, `sin_phi`, `cos_phi`, `charge` |
| `Met` (global) | `pt`, `phi`, `sin_phi`, `cos_phi` |

The default event file reads all of them except raw `phi`. To list the features
of any built file:

```bash
micromamba run -n spanet-reco python -c "import h5py; f = h5py.File('data/datasets/mc20260908-v1/train.h5'); [print(g, list(f['INPUTS'][g])) for g in f['INPUTS']]"
```

## Route A: choose among extracted features

1. Copy the default event file and give the variant a name:

   ```bash
   cp configs/event-vcb.yaml configs/event-vcb-nocharge.yaml
   ```

2. Edit its `INPUTS`. Delete a line to drop a feature, or change its
   preprocessing:

   | Value | Effect | Typical use |
   | --- | --- | --- |
   | `none` | Used as stored | Scores in [0, 1], indicators, charges |
   | `normalize` | Standardized with the training set's mean and width | eta, sin phi, cos phi |
   | `log_normalize` | log(x + 1), then standardized | pT, mass, MET: positive, steeply falling |
   | `log` | log(x + 1) only | Rarely needed |

   log(x + 1) is infinite at x = -1, the value NanoAOD stores for an unavailable
   score. Use `none` or `normalize` for any feature that can be -1 or below.

   Keep these rules, which the tests check:
   - Group and feature names match the dataset exactly; their order is free.
   - `SEQUENTIAL` comes before `GLOBAL`.
   - Jets keep `mass`, `pt`, `eta`, `sin_phi`, and `cos_phi`: the training code
     builds jet four-vectors from them for the mass chi-square, whatever `alpha` is.
   - Raw `phi` stays out; `sin_phi` and `cos_phi` are read together or not at all.
   - `EVENT` and `PERMUTATIONS` stay as they are.

   A whole global input can be left out by deleting its group, for example
   `Met:` and its features.

3. Check the file:

   ```bash
   micromamba run -n spanet-reco python -m pytest tests/test_configs.py -q
   ```

4. Commit and push it, so that the checkout on Oscar has it.
5. Train with it by adding `-ef` to the training options; it replaces the
   default event file:

   ```bash
   sbatch scripts/slurm_oscar.sh data/datasets/mc20260908-v1 outputs/nocharge-s1 \
     -g 1 -b 1024 --seed 1 -ef configs/event-vcb-nocharge.yaml
   ```

   The run directory keeps a copy as `vcb/version_0/event.yaml`. The same
   `--seed` as a reference run makes the comparison fair.

## Route B: extract a new branch

1. Find the branch name in the processed MC, on BRUX:

   ```bash
   ../nano-spanet-extractor/.venv/bin/nano-spanet-extract --list-branches --input-dir \
     /isilon/export/home/jhuan166/Vcb/MC/TTtoLNuCB_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNuCB_syst/roots
   ```

2. Copy the mapping to `configs/extract-vcb-<name>.yaml` and add the feature as
   `feature name: branch`. Per-jet branches use the slot template `{i}` under
   `sequential: Jets: features:`; event-level branches go under a group in
   `global:`, which may be a new group:

   ```yaml
   sequential:
     Jets:
       features:
         qvg: ak4JetbtagPNetQvG{i}  # ParticleNet quark-vs-gluon score, slots 0 to 9
   global:
     Lepton:
       mass: TriggerLeptonMass      # an event-level branch
   ```

   Feature names used by `jet_selection` (`pt`, `eta`) and `tag_selection`
   (`btag_raw`, `charge_raw`) must stay consistent with those blocks. Any new cut
   must be declared, with its threshold, in the mapping; the mapping format is
   described in the README of the sibling `nano-spanet-extractor` checkout
   ("Mapping format" and "Cuts and thresholds").

   Only reconstructed quantities may be inputs. Generator-level branches, such as
   `ak4JetHadronFlavour{i}`, `ak4JetPartonFlavour{i}`, `ak4JetGenJetIdx{i}`,
   `ak4JetMatchedGenJetPt{i}`, and `ak4JetFlavSplit{i}`, would leak the truth into
   the model.

3. Check that every file has the branches, for both samples (the background
   pattern covers the files the extraction script reads):

   ```bash
   X=../nano-spanet-extractor/.venv/bin/nano-spanet-extract
   MC=/isilon/export/home/jhuan166/Vcb/MC
   $X --dry-run --config configs/extract-vcb-<name>.yaml \
     --input-dir $MC/TTtoLNuCB_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNuCB_syst/roots
   $X --dry-run --config configs/extract-vcb-<name>.yaml --pattern 'batch_0[01][0-9][0-9].root' \
     --input-dir $MC/TTtoLNu2Q_Summer24MiniAODv6/NanoAOD-processed/prod_20260908_TTtoLNu2Q_syst/roots
   ```

4. Extract into a new directory, then build a new dataset. The default seed and
   fractions give the same file split as `mc20260908-v1`, and, with unchanged
   cuts, the same events:

   ```bash
   CONFIG=configs/extract-vcb-<name>.yaml OUT=data/extracted/mc20260908-<name> \
     scripts/extract_mc20260908.sh
   micromamba run -n spanet-reco python -m spanet_reco.build_dataset \
     --signal data/extracted/mc20260908-<name>/signal/part*.h5 \
     --background data/extracted/mc20260908-<name>/background/part*.h5 \
     --output-dir data/datasets/mc20260908-<name>
   ```

5. Write `configs/event-vcb-<name>.yaml` that reads the new feature (route A,
   steps 1-3), and run the tests.
6. Commit and push the mapping and event file. Write checksums, copy the dataset
   to Oscar, and verify it ([Running on Oscar](oscar.md#3-copy-the-dataset)).
7. Train with the new dataset and event file:

   ```bash
   sbatch scripts/slurm_oscar.sh data/datasets/mc20260908-<name> outputs/<name>-s1 \
     -g 1 -b 1024 --seed 1 -ef configs/event-vcb-<name>.yaml
   ```

8. Record the new dataset in [README-extraction.md](../README-extraction.md).

A reference run for comparison should use the same dataset, or a dataset built
with the same selection, so that only the features differ.

## Route C: derived features

`spanet_reco.build_dataset` adds `sin_phi` and `cos_phi` to every input with
`phi`, using `sin_cos` in [`features.py`](../src/spanet_reco/features.py). Other
derived quantities, such as a jet's distance to the lepton, need similar code in
`build_dataset` with tests, a rebuilt dataset, and an event file that reads them.
Nothing else in the pipeline changes.
