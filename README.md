# SPANet-reco

Event reconstruction for the Vcb analysis using SPANet, focusing on
reconstructing top and antitop decays in signal and background MC samples.

## Status 

2026-09-21T13:41

Initial repository setup. Data preparation, training, and evaluation
are not implemented yet.

## Proposed model inputs

| Object    | Selection | Features  |
| ---       | ---       | ---       |
| Jets                          | Up to seven jets, ordered by decreasing pT    | Mass, transverse momentum, eta, phi |
| MET                           | Event MET                                     | Magnitude, phi |
| Lepton                        | Trigger lepton only                           | Transverse momentum, eta, phi, charge |
| Jets ranked by b-tag score    | Three jets with the highest PNetb-tag scores  | PNet B-tag score and corresponding ParTPosVsNegjet charge score |

Each jet charge score must remain associated with the same jet as its 
accompanying b-tag score. The exact branch mapping and preparation rules are 
described in the [input contract](docs/input-contract.md). 
The [target contract](docs/target-contract.md) describes the assignment task.

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
metadata. The `dev` extra installs pytest and Ruff.

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
micromamba run -n spanet-reco ruff check src
micromamba run -n spanet-reco ruff format --check src
```

To apply formatting:

```bash
micromamba run -n spanet-reco ruff format src
```

Review the resulting changes with `git diff` before staging them.

## Decisions required before implementation

- Padding and masks for events with fewer than the required objects.
- Custom loss function.
- Training, validation, and test split definitions.
