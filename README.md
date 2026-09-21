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
| Jets      | Up to seven jets, ordered by decreasing transverse momentum | Mass, transverse momentum, eta, phi |
| Missing transverse momentum | Event MET | Magnitude, phi |
| Lepton | Trigger lepton only | Transverse momentum, eta, phi, charge |
| Jets ranked by b-tag score | Three jets with the highest b-tag scores | B-tag score and corresponding jet charge score |

Each jet charge score must remain associated with the same jet as
its accompanying b-tag score.

## Decisions required before implementation

- Exact input branches, units, and definitions of the tag and charge scores.
- Padding and masks for events with fewer than the required objects.
- Reconstruction targets and truth-matching rules.
- Custom loss function.
- Training, validation, and test split definitions.
