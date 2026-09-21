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
collection. All real retained jets are eligible assignment candidates.

The top-three b-tag feature selection does not restrict which jets
may be assigned to these roles.

Generator information is used for targets, training selection, and
diagnostics. It is excluded from model input features.

## Symmetry

The had_top/q1 and had_top/q2 targets are interchangeable. The had_top/b and 
lep_top/b roles remain distinct from each other and from both W-daughter roles. 

## First training baseline

Use the upstream truth assignments without recomputing matching.

An event is fully matched within the retained inputs when:

- All four indices refer to real, unpadded jets among slots 0 through 6.
- All four indices are distinct.

An upstream index of -1 denotes an unmatched parton.
Indices 7 through 9 refer to jets outside the retained inputs.
Events containing either case are excluded from this first training
baseline.

Duplicate nonnegative assignments, indices outside the upstream
range, and assignments to empty upstream slots are validation errors.
They must be reported rather than silently repaired.

Never clip an invalid target into the valid index range.

## Preparation diagnostics

Record counts separately for each source sample:

- Events considered.
- Events with at least one unmatched parton.
- Events with at least one match outside the retained seven jets.
- Total excluded events.
- Fully matched events retained.

Exclusion reasons may overlap. Count each excluded event only once
in the total.

## Evaluation

A complete assignment is correct when both top-b assignments are
correct and the unordered pair of predicted W jets matches truth.

Report assignment accuracy on the fully matched evaluation subset,
together with the fraction of events retained by that requirement.