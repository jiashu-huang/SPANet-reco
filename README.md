# SPANet-reco

Event reconstruction for the Vcb analysis using SPANet, focusing on
reconstructing top and antitop decays in signal and background MC samples.
The immediate goal is to select the two hadronic-W daughter jets in each
event, as an unordered pair; b-tagging of those jets is applied afterwards.

## Status 

2026-09-22

Package setup, target validation, and ROOT target reading are implemented.
A small development fixture is available. Feature extraction is handled by the
sibling `nano-spanet-extractor` repository. Its cuts (jets with pT > 25 GeV and
abs(eta) < 2.4, following AN-25-214) and tag-score fillers are declared in the
version 2 extraction mapping [`configs/extract-vcb.yaml`](configs/extract-vcb.yaml);
the extractor has no built-in cut values or cut options.
The SPANet event definition, training options, and the step adding sin(phi) and
cos(phi) inputs are in place and load in SPANet (see
[SPANet configuration](docs/spanet-config.md)). Dataset preparation at scale,
training runs, and evaluation are not implemented yet.

## Model inputs

| Object    | Selection | Features  |
| ---       | ---       | ---       |
| Jets                          | Up to seven jets with corrected pT > 25 GeV and abs(eta) < 2.4, ordered by decreasing pT | Mass, transverse momentum, eta, sin(phi), cos(phi) |
| MET                           | Event MET                                     | Magnitude, sin(phi), cos(phi) |
| Lepton                        | Trigger lepton only                           | Transverse momentum, eta, sin(phi), cos(phi), charge |
| Jets ranked by b-tag score    | Up to three retained jets with the highest available PNet B-tag scores | PNet B-tag score and corresponding ParTPosvsNeg jet charge score (fillers 0 and 0.5 on other retained jets), and a 0/1 `tag_selected` indicator |

Each jet charge score must remain associated with the same jet as its 
accompanying b-tag score. The exact branch mapping and preparation rules are 
described in the [input contract](docs/input-contract.md). To change a selection
threshold, edit the extraction mapping; the output records it in `PROVENANCE/config`.
The [target contract](docs/target-contract.md) describes the assignment task, and
[SPANet configuration](docs/spanet-config.md) describes the event and options files.

## Development setup

Requires `micromamba`. Run all commands from the repository root.

To start: create the Python 3.11 environment once.

```bash
micromamba create -f environment.yml
```

Install the package in editable mode with its development tools:

```bash
micromamba run -n spanet-reco python -m pip install -e ".[dev]"
```

Editable installation makes source changes available without reinstalling.
Rerun the installation command after changing dependencies or package
metadata. ROOT reading uses Uproot and HDF5 access uses h5py. The `dev` extra
installs pytest, Ruff, and PyYAML for the configuration consistency tests.
SPANet itself runs in its own environment, `external/SPANet/environment`.

Commands below explicitly select the environment, so shell activation
is unnecessary.

### Verification

```bash
micromamba run -n spanet-reco python --version
micromamba run -n spanet-reco python -c "import spanet_reco; print(spanet_reco.__file__)"
micromamba run -n spanet-reco python -m pip check
```

Expect Python `3.11.x`, an import path pointing to this checkout's
`src/spanet_reco/__init__.py`, and no broken dependency requirements.

### Check code quality

Run these checks before committing Python changes:

```bash
micromamba run -n spanet-reco python -m pytest tests -q
micromamba run -n spanet-reco ruff check src tests scripts
micromamba run -n spanet-reco ruff format --check src tests scripts
```

Tests use small NumPy arrays and temporary synthetic ROOT/HDF5 files.
They require no private MC files.

To apply formatting:

```bash
micromamba run -n spanet-reco ruff format src tests scripts
```

Review the resulting changes with `git diff` before staging them.

## Development fixture

The [fixture guide](docs/development-fixture.md) describes the 100 signal
and 100 background events recovered using training-event identities from
`spanet-test`. It includes preparation commands and source provenance.
Local data under `data/` are ignored by Git.

Check the upstream assignments in a fixture:

```bash
micromamba run -n spanet-reco python -m spanet_reco.root_io data/fixtures/pilot-v1/signal.root
micromamba run -n spanet-reco python -m spanet_reco.root_io data/fixtures/pilot-v1/background.root
```

Each command should report 100 events, 100 fully matched, and zero exclusions.
These counts describe the upstream assignments before the extractor's jet cuts.
The reader loads only `nJets` and the four assignment branches; fully matched
training must check the final extracted target masks, as specified in the
[target contract](docs/target-contract.md). This fixture is for development;
it does not define the eventual training or evaluation datasets.

## Decisions required before training

- Training, validation, and test samples, their sizes, and a per-event sample label.
- Whether to keep SPANet's standard loss and checkpoint selection.
