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

Every threshold below is declared in the extraction mapping,
[`configs/extract-vcb.yaml`](../configs/extract-vcb.yaml) (mapping `version: 2`).
The extractor has no built-in cut values, no defaults, and no command-line cut
options. A mapping that omits a cut does not apply it, and a version 1 mapping
is rejected. Changing a threshold therefore means editing the mapping, and each
output records the mapping it used in `PROVENANCE/config`.

1. Read all ten saved jet slots (`sequential.Jets.source_objects: 10`).
   Remove jets with abs(eta) >= 2.4, then jets with processed, corrected
   pT <= 25 GeV. This is the AN-25-214 jet definition (Sec. 3.3, Table 9).
   Both cuts are strict, declared as `jet_selection.max_abs_eta: 2.4` and
   `jet_selection.min_pt: 25`.
2. Write the first seven (7) remaining jets, in the upstream pT ordering, to
   slots 0 through 6 (`sequential.Jets.max_objects: 7`). These are the seven
   leading jets passing both cuts.
3. Rank the written jets with available ParticleNet b-tag scores in decreasing
   score order. Equal scores retain their slot order. The ranking's own limit,
   `tag_selection.max_abs_eta: 2.4`, equals the jet cut and so removes nothing more.
4. Select up to three jets from that ranking (`tag_selection.top_n: 3`).
5. Retrieve each selected jet's charge score from the same jet.
6. Record each written slot's saved upstream slot in
   `META/jet_selection/Jets/source_slot` (`-1` for padding), so that
   kinematics, tag scores, and truth labels remain traceable.

Empty upstream slots contain -99999. They must not participate in
jet ranking or be treated as measured features.

Removed jets leave no gaps: a jet failing the eta cut in, for example, saved
slot 2 lets later passing jets move up, including jets from saved slots 7 to 9.
Keep events with fewer than seven passing jets and pad the remaining slots at the
end. Padded slots have all features zeroed. Jets at eta = +2.4 or -2.4 fail. The
selection uses input precision before feature casting. Nonfinite pT or eta on a
real saved jet is an error.

The jet mask means "one of the seven leading jets passing the pT and eta cuts"
and is contiguous. The original jet count and the number of written jets are
recorded separately in metadata. Events with more than seven passing jets, or
with more than ten jets saved upstream, are counted as truncated. The
thresholds and slot policy are saved in `PROVENANCE/jet_selection`, and the
tag-score policy in `PROVENANCE/tag_score_policy`.

## Tag-score availability and event selection

The extractor recognizes `-1` as an unavailable tag-score sentinel. It excludes
jets with this b-tag value from score ranking while retaining their kinematics
and availability for truth assignment. If fewer than three eligible central jets have
available b-tag scores, expose the available ones.

This is an extraction convention: the upstream NanoAOD writer can also encode
an exact-zero discriminator output as `-1`; the stored sentinel does not retain
the reason that the score was unavailable.

When including charge scores (`tag_selection.include_charge: true`), reject an
event if **any of the selected top-three jets** has charge score `-1`. Do not substitute a lower-ranked jet. An unavailable
charge on an unselected jet does not cause rejection. Exact zero is accepted as
a valid score; other nonfinite or out-of-range scores where used are errors.

Retained events keep their pT-ordered written jets. On written jets whose scores
are not shown, the scores are replaced by the declared fillers
(`tag_selection.fill`): b-tag 0, since real scores are never exactly 0 and 0 reads
as "not b-like", and charge 0.5, the neutral point of ParTPosvsNeg. The
`tag_selected` indicator (`tag_selection.indicator`) is a learned 0/1 jet
feature equal to 1 exactly where the scores are shown, for both b-tag and charge.
Padded slots have zero for every feature, including the indicator. The
same selected slots are saved as a diagnostic bitmap in
`META/tag_selection/Jets`. The jet object mask identifies retained and padded
slots; it does not mask individual features.

Apply this event selection in `nano-spanet-extractor`, with the same row selection
for inputs, truth targets, and event metadata. Record input, accepted, and rejected
event counts, including counts per source file. Truth matching and tag-score
availability are separate requirements.

## Available truth assignments

The upstream branches are:

- GenHadBJetIdx
- GenLepBJetIdx
- GenHadQ1JetIdx
- GenHadQ2JetIdx

These refer to the ten saved jet slots. An unmatched parton has index -1.

The extractor rewrites each assignment as the written slot of its jet: after
compaction, saved slot 3 may become slot 1, and saved slot 8 may become slot 6.
An assignment is unavailable after extraction when its jet fails a cut (pT at or
below 25 GeV, or abs(eta) >= 2.4) or is not among the seven leading passing jets.
Represent it as target `-1` and set the corresponding particle target mask false.
Apply the fully matched baseline to these final targets.

## Model features

After extraction, `python -m spanet_reco.features` adds `sin_phi` and
`cos_phi` for jets, the trigger lepton, and MET, zero on padded jets. The model
reads these instead of raw phi, which stays in the file. The model's feature
list and preprocessing are defined in the event file
[`configs/event-vcb.yaml`](../configs/event-vcb.yaml); see
[SPANet configuration](spanet-config.md).

## Decisions still required

- Additional analysis selections, if any. Any new cut must be declared, with
  its threshold, in the extraction mapping.
- Event weights, dataset splits, and the training loss.
