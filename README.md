# SPANet-reco

Event reconstruction for the Vcb analysis with SPANet, in semileptonic ttbar
signal (TTtoLNuCB, W to cb) and background (TTtoLNu2Q) MC. The immediate goal is
to select the two hadronic-W daughter jets in each event, as an unordered pair;
b-tagging of those jets is applied afterwards, outside SPANet.

## Status

2026-09-23

- The September 8 Summer24 MC is extracted and built into training, validation,
  and test files (`data/datasets/mc20260908-v1`, not in Git); see
  [README-extraction.md](README-extraction.md).
- The SPANet event definition, options, and an optional hadronic top and W mass
  chi-square loss are in place and tested on CPU.
- A Slurm job trains on one GPU of Brown's Oscar cluster.
- Next: full-data training on Oscar, comparing plain SPANet with the mass
  chi-square loss over several seeds, and evaluation on the test split.

## Pipeline

| Step | Tool | Where it runs | Guide |
| --- | --- | --- | --- |
| 1. Extract features and truth assignments from processed MC | `nano-spanet-extractor` with [`configs/extract-vcb.yaml`](configs/extract-vcb.yaml), via [`scripts/extract_mc20260908.sh`](scripts/extract_mc20260908.sh) | BRUX | [Running the pipeline](docs/running.md#1-extract-features-brux) |
| 2. Build training, validation, and test files | `python -m spanet_reco.build_dataset` | BRUX | [Running the pipeline](docs/running.md#2-build-training-validation-and-test-files-brux) |
| 3. Train | [`scripts/slurm_oscar.sh`](scripts/slurm_oscar.sh) or [`scripts/train.sh`](scripts/train.sh), which run `python -m spanet_reco.train` | GPU node | [Running the pipeline](docs/running.md#4-train) |
| 4. Compare runs | [`scripts/compare_checkpoints.py`](scripts/compare_checkpoints.py) | anywhere with SPANet | [Running the pipeline](docs/running.md#after-training) |

## Model

| Object | Selection | Features |
| --- | --- | --- |
| Jets | Up to seven jets with corrected pT > 25 GeV and abs(eta) < 2.4 (AN-25-214), ordered by decreasing pT | Mass, transverse momentum, eta, sin(phi), cos(phi) |
| Jets ranked by b-tag score | Up to three retained jets with the highest available PNet B-tag scores | PNet B-tag score and the same jet's ParTPosvsNeg charge score (fillers 0 and 0.5 on other jets), and a 0/1 `tag_selected` indicator |
| Lepton | Trigger lepton | Transverse momentum, eta, sin(phi), cos(phi), charge |
| MET | Event MET | Magnitude, sin(phi), cos(phi) |

SPANet assigns jets to `had_top` (b, q1, q2), with q1 and q2 interchangeable, and
to `lep_top` (b). Training uses fully matched events only. The loss is SPANet's
cross-entropy of the true assignment, optionally combined with a hadronic top and
W mass chi-square: `alpha * L_SPANet + (1 - alpha) * <chi2>`.

## Documentation

| Document | Contents |
| --- | --- |
| [README-extraction.md](README-extraction.md) | How the current dataset was made, with event counts |
| [docs/running.md](docs/running.md) | Commands for every step, including the transfer to Oscar and Slurm |
| [docs/spanet-config.md](docs/spanet-config.md) | Event and options files, the mass chi-square loss, and its test runs |
| [docs/custom-loss.md](docs/custom-loss.md) | How to add a physics term to the training loss |
| [docs/input-contract.md](docs/input-contract.md) | Input branches, jet selection, tag scores, and event rejection |
| [docs/target-contract.md](docs/target-contract.md) | Assignment targets, symmetry, fully matched baseline, and evaluation |
| [docs/development-fixture.md](docs/development-fixture.md) | The 200-event development fixture and early checks |

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/` | Extraction mapping, SPANet event file, and SPANet options |
| `src/spanet_reco/` | Target reading (`root_io`, `targets`), dataset build (`build_dataset`, `features`), and training (`model`, `train`) |
| `scripts/` | Extraction, training, Slurm, comparison, and fixture preparation |
| `tests/` | Tests with synthetic ROOT and HDF5 files; `spanet_model_checks.py` needs SPANet |
| `environment.yml`, `environment-spanet-gpu.yml` | Development environment, and the SPANet training environment for a GPU node |
| `data/`, `outputs/` | Local datasets and training runs, ignored by Git |

## Development setup

Requires `micromamba`. Run all commands from the repository root. Create the
Python 3.11 environment once, then install the package in editable mode with its
development tools:

```bash
micromamba create -f environment.yml
micromamba run -n spanet-reco python -m pip install -e ".[dev]"
```

Rerun the installation after changing dependencies or package metadata. ROOT
reading uses Uproot and HDF5 access uses h5py; the `dev` extra adds pytest,
Ruff, and PyYAML. SPANet and PyTorch run in a separate environment: locally
`external/SPANet/environment`, and on a GPU node the one from
`environment-spanet-gpu.yml`. The training code runs there from this checkout,
without installation.

Check the environment:

```bash
micromamba run -n spanet-reco python -m pip check
micromamba run -n spanet-reco python -c "import spanet_reco; print(spanet_reco.__file__)"
```

### Check code quality

Run these checks before committing Python changes:

```bash
micromamba run -n spanet-reco python -m pytest tests -q
micromamba run -n spanet-reco ruff check src tests scripts
micromamba run -n spanet-reco ruff format --check src tests scripts
```

Tests use small synthetic ROOT and HDF5 files and need no private MC. The model
checks need SPANet and are skipped unless `SPANET_PYTHON` names its Python:

```bash
SPANET_PYTHON=/isilon/export/home/jhuan166/Vcb/external/SPANet/environment/bin/python \
  micromamba run -n spanet-reco python -m pytest tests -q
```

To apply formatting, run `micromamba run -n spanet-reco ruff format src tests scripts`
and review the changes with `git diff`.

## Development fixture

The [fixture guide](docs/development-fixture.md) describes 100 signal and 100
background events recovered from the pilot study in `spanet-test`, stored under
`data/fixtures/pilot-v1/`. Check their upstream assignments with:

```bash
micromamba run -n spanet-reco python -m spanet_reco.root_io data/fixtures/pilot-v1/signal.root
```

Each file should report 100 events, all fully matched before the extractor's
jet cuts. The fixture is for development only.

## Open decisions

- The weight `alpha` of the mass chi-square loss, and whether to select
  checkpoints by per-sample full-assignment accuracy instead of SPANet's jet
  accuracy.
- Evaluation on the test split, reported separately for each sample, including
  the fraction of events that are fully matched.
- Event weights (none are used now).
