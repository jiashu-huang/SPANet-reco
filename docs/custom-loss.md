# Adding physics to the training loss

This guide explains how to add a physics term to SPANet's training loss in this
repository. The hadronic top and W mass chi-square in
[`spanet_reco/model.py`](../src/spanet_reco/model.py) is the worked example; its
settings and test results are in [SPANet configuration](spanet-config.md#mass-chi-square-loss).

## Where the loss is computed

SPANet computes its loss in `JetReconstructionTraining.training_step`
(`spanet/network/jet_reconstruction/jet_reconstruction_training.py`, lines
203-290 at the pinned commit `46c6805`). For each particle in the event file it
takes -log P(true jet combination), averages over events, and averages the
particles; optional detection, KL, regression, and classification terms are off
in [`configs/options-vcb.json`](../configs/options-vcb.json).

SPANet itself is not modified: the cluster installs the upstream commit. The
change lives in this repository instead:

```text
scripts/train.sh
  -> python -m spanet_reco.train      reads --alpha, masses, widths, --seed;
                                      sets MassChi2Model.settings; replaces the
                                      model class that spanet.train builds
    -> spanet.train (unmodified)      options, datasets, trainer, checkpoints
      -> MassChi2Model.training_step  once per batch: SPANet's loss + new term
```

- `MassChi2Model` ([model.py:101](../src/spanet_reco/model.py)) subclasses SPANet's
  `JetReconstructionModel` and overrides `training_step` (line 150).
- `MassChi2Model.spanet_loss` (line 168) is SPANet's loss code, copied so the
  network runs once per batch for both terms. It must match the pinned SPANet
  version; a test checks that `alpha = 1` gives exactly SPANet's loss.
- [`spanet_reco/train.py`](../src/spanet_reco/train.py) declares the command-line
  settings (`MASS_OPTIONS`, line 17) and swaps the class in (line 70).

## What `training_step` can use

| Object | Content |
| --- | --- |
| `batch.sources[i]` | `(data, mask)` for each `INPUTS` group, in event-file order: Jets `(events, 7, 8)`, Lepton `(events, 1, 5)`, Met `(events, 1, 3)`. Find an input with `self.event_info.input_names.index("Jets")`. |
| Units of `data` | Features marked `log` or `log_normalize` hold log(x + 1); all others are raw. Standardization happens later, inside the network, so `data` is not standardized. `self.event_info.input_features["Jets"]` lists each feature's name and `log_scale`, in column order; look columns up by name. |
| `outputs = self.forward(batch.sources)` | `outputs.assignments[p]` holds log-probabilities over jet combinations for particle `p`, in `EVENT` order: `had_top` `(events, 7, 7, 7)` with axes `(b, q1, q2)`, symmetric in q1 and q2; `lep_top` `(events, 7)`. Combinations with a padded or repeated jet are exactly `-inf`, so P = 0. |
| `batch.assignment_targets[p]` | `indices` `(events, daughters)` with -1 for a missing jet, `mask`, and `weight`. |
| Particle and daughter names | `self.event_particle_names`; daughters are `self.event_info.product_particles[name].names`. SPANet's `self.product_particle_names` holds only the first daughter. |
| Helpers in `model.py` | `jet_four_vectors` (inputs to `(E, px, py, pz)` in GeV), `invariant_mass`, and `hadronic_chi2` (terms and a validity mask for every `(b, q1, q2)`). |

The trigger lepton and MET are also available, but a leptonic top mass needs the
neutrino pz, for example from MET and a W mass constraint, which has zero or two
solutions. Only the hadronic top and W masses follow directly from jets.

## Steps for a new term

1. **Make the term differentiable in the network's output.** Write it as an
   average over the network's probabilities, `<f> = sum over c of P(c) f(c)`,
   where `c` runs over jet combinations. A term evaluated on the true combination
   alone does not depend on the network, and a term on the single most probable
   combination has no gradient; neither can train anything.
2. **Evaluate `f` for every combination** as a tensor with the same shape and
   axes as `outputs.assignments[p]`, by broadcasting over the jet axes (see
   `hadronic_chi2`, line 75). Compute it under `torch.no_grad()`, since the
   inputs need no gradient, and zero it on invalid combinations: 0 times an
   infinite value is NaN.
3. **Combine it with SPANet's loss** in `training_step`: run `self.forward` once,
   take `self.spanet_loss(outputs, batch)`, add the new term with its weight, and
   log each part with `self.log("loss/...")` so that it appears in TensorBoard.
   Choose the weights by looking at both terms' sizes at the start and end of
   training.
4. **Add its settings** as fields of a frozen dataclass like `MassChi2Settings`
   (line 33), validated in `__post_init__`, and as options in
   `spanet_reco/train.py`. They are saved to `mass_chi2.json` in the run
   directory. They cannot go in `options-vcb.json`: SPANet raises `KeyError` on an
   unknown option name when it prints its settings.
5. **Test it** in [`tests/spanet_model_checks.py`](../tests/spanet_model_checks.py):
   compare the physics quantity with an independent NumPy calculation, check
   that the weight which disables the term reproduces SPANet's loss exactly, and
   check that gradients are finite. Run the checks with
   `SPANET_PYTHON=/path/to/spanet/bin/python micromamba run -n spanet-reco python -m pytest tests`.
6. **Measure it.** Train a short run and a baseline with the same `--seed` and
   data slice (`-p`, `-e`), then compare them on the same unseen events with
   [`scripts/compare_checkpoints.py`](../scripts/compare_checkpoints.py). Confirm
   an improvement with full-data runs and several seeds.

In outline, a new term fits `training_step` as follows:

```python
outputs = self.forward(batch.sources)
spanet_loss = self.spanet_loss(outputs, batch)
with torch.no_grad():
    f, valid = ...  # per combination, same shape as outputs.assignments[p]
term = torch.where(valid, outputs.assignments[p].exp() * f, 0).flatten(1).sum(1).mean()
loss = weight_spanet * spanet_loss + weight_term * term
```

## Lessons from the mass chi-square

- **A physics optimum is not the truth.** In most events a wrong triplet fits
  the masses better than the true one: the median chi-square is 1.3-1.5 for the
  true triplet and 0.45 for the smallest one, and the smallest-chi-square triplet
  is the true hadronic top in only 33-35% of events. Minimizing the term alone
  pulls the network toward that choice.
- **Scale decides the outcome.** `<chi2>` starts near 60-120 while SPANet's loss
  is about 2. With `alpha = 0.5` the chi-square dominated and accuracy fell by
  14-26 points; with `alpha = 0.95` it dominated only the first epoch,
  accuracy matched or exceeded plain SPANet, and the chosen masses moved toward
  the peaks.
- **SPANet learns masses on its own.** Without the term, `<chi2>` fell from 117
  to 14 in three epochs. A physics term adds a push in the same direction; it
  should guide the cross-entropy, not replace it.

## Constraints to keep

- **Checkpoints.** Changing only the loss keeps the network, so checkpoints
  still load in `spanet.test`, `spanet.predict`, and `compare_checkpoints.py`,
  which build a plain `JetReconstructionModel`. Changing the network breaks that.
- **Checkpoint selection** still uses `validation_average_jet_accuracy`. The new
  term is not computed during validation; override `validation_step` to track it.
- **Memory.** A table over n jet slots per daughter has 7^n entries per event:
  343 for the hadronic top. Larger tables grow quickly with the batch size.
- **SPANet version.** `spanet_loss` copies code from commit `46c6805`. When the
  pinned SPANet changes, update the copy; the `alpha = 1` check will fail until
  it matches.

## Other places for physics

- **Inputs:** add derived features, such as pair masses, in
  `spanet_reco.build_dataset` and the event file. The network learns from them;
  the loss is unchanged.
- **Scores before the softmax:** a per-combination term added in SPANet's
  `BranchDecoder` (`spanet/network/layers/branch_decoder.py`, line 173, where
  SPANet left a commented-out `combinatorial_scale` hook) acts on predictions
  directly. It changes the network, so it needs its own test and predict code.
- **Regression outputs:** a `REGRESSIONS` block in the event file with
  `regression_loss_scale` trains extra outputs, but needs target values in the
  HDF5 files. SPANet supports event-level regressions fully.
- **After training:** choose among SPANet's most probable combinations with a
  physics criterion, with no retraining.
