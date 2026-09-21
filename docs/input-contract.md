# Input contract

## Source

SPANet-reco consumes MC processed by `Calib_ChargeTagger_JH`, normally stored 
under `Vcb/MC`. The input is the ROOT Events tree.

Object definitions, corrections, and trigger-lepton selection are inherited 
from the upstream production.

Each prepared dataset must record its source production and the upstream code 
revision used to produce it.

Here, i denotes a saved jet slot. The upstream skimmer saves up to
ten jets ordered by decreasing corrected transverse momentum.

| Quantity                  | Input branches |
| ---                       | --- |
| Jet kinematics            | ak4JetMass{i}, ak4JetPt{i}, ak4JetEta{i}, ak4JetPhi{i} |
| MET                       | METPt, METPhi |
| Trigger lepton            | TriggerLeptonPt, TriggerLeptonEta, TriggerLeptonPhi, TriggerLeptonCharge |
| ParticleNet b-tag score   | ak4JetbtagPNetB{i} |
| Jet charge score          | ak4JetParTPosvsNeg{i} |

Momenta and masses are in GeV. Phi is in radians. Eta is dimensionless.

## Jet selection and score association

1. Retain up to seven (7) real jets in the upstream pT ordering.
2. Rank these retained jets by decreasing ParticleNet b-tag score.
3. Select up to three real jets from that ranking.
4. Retrieve each selected jet's charge score from the same saved slot.
5. Preserve the selected slot indices during preparation so that
   kinematics, tag scores, and truth labels remain traceable.

Empty upstream slots contain -99999. They must not participate in
jet ranking or be treated as measured features.

## Available truth assignments

The upstream branches are:

- GenHadBJetIdx
- GenLepBJetIdx
- GenHadQ1JetIdx
- GenHadQ2JetIdx

These refer to the ten saved jet slots. An unmatched parton has index -1.

After retaining seven jets, an upstream index of 7, 8, or 9 refers
to a jet outside the model input. It cannot be used unchanged as
a valid assignment target.

## Decisions still required

- Additional analysis selections, if any.
- Model padding values and masks.
- Handling of invalid tag scores and equal-score ranking ties.
- Feature transformations and normalization.
- Event weights, dataset splits, and the training loss.