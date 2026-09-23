# Target contract

## Task

The initial task is jet assignment in semileptonic ttbar events.

The assignment branches are named had_top and lep_top according to
their decay modes.

Mapping these branches to top and antitop, and reconstructing their four-momenta, 
will be specified separately.

## Target mapping

| Target        | Upstream branch |
| ---           | --- |
| had_top/b     | GenHadBJetIdx |
| had_top/q1    | GenHadQ1JetIdx |
| had_top/q2    | GenHadQ2JetIdx |
| lep_top/b     | GenLepBJetIdx |

Each target is an integer index into the retained, pT-ordered jet
collection: the seven leading jets passing pT > 25 GeV and abs(eta) < 2.4,
written to slots 0 through 6. The extractor rewrites each upstream index,
which refers to the ten saved slots, as its jet's written slot. All retained
jets are eligible assignment candidates.

The top-three b-tag feature selection does not restrict which jets
may be assigned to these roles: a retained jet whose scores are not shown
remains a candidate.

Generator information is used for targets, training selection, and
diagnostics. It is excluded from model input features.

## Symmetry

The had_top/q1 and had_top/q2 targets are interchangeable. The had_top/b and 
lep_top/b roles remain distinct from each other and from both W-daughter roles. 

## First training baseline

Use the upstream truth assignments without recomputing matching.

An event is fully matched within the retained inputs when:

- All four assigned jets pass the corrected pT > 25 GeV and abs(eta) < 2.4
  jet requirements.
- All four assigned jets are among the seven leading passing jets, so each has
  a written slot from 0 through 6.
- All four indices are distinct.

An upstream index of -1 denotes an unmatched parton. The extractor also sets a
target to -1 if its jet fails the pT or eta requirement, or is not among the
seven leading passing jets. Events containing any of these cases are excluded
from this first training baseline. Validate final extracted target masks: the
ROOT target reader checks upstream assignments before the jet requirements. It
accepts assignments in all ten saved slots (`N_INPUT_JETS = 10`), because
compaction can retain jets from saved slots 7 through 9, so it cannot tell which
assigned jets the extractor will keep.

Duplicate nonnegative assignments, indices outside the upstream
range, and assignments to empty upstream slots are validation errors.
They must be reported rather than silently repaired.

Never clip an invalid target into the valid index range.

## Preparation diagnostics

Record counts separately for each source sample:

- Events considered.
- Events with at least one unmatched parton.
- Events with at least one assigned jet failing the pT or eta requirement.
- Events with at least one assigned jet passing both but not among the seven
  leading passing jets.
- Total excluded events.
- Fully matched events retained.

Exclusion reasons may overlap. Count each excluded event only once
in the total.

The extraction and dataset build currently record events considered and fully
matched per sample (see [README-extraction.md](../README-extraction.md) and each
dataset's `summary.json`). The breakdown by exclusion reason is not yet produced.

## Evaluation

A complete assignment is correct when both top-b assignments are
correct and the unordered pair of predicted W jets matches truth.

Report assignment accuracy on the fully matched evaluation subset,
together with the fraction of events retained by that requirement.
