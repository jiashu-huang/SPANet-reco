# Development fixture

The first fixture contains 100 signal and 100 background events from MC
processed by `Calib_ChargeTagger_JH`. It exercises ROOT input and target
validation while model feature preparation is developed.

## Why recover events from ROOT?

The fully matched files in `spanet-test` are converted HDF5 datasets. The
pilot stored ten jet slots, encoded ParticleNet scores into bins, applied
extra object selections, and recomputed truth matching from parent NanoAOD.
Its features and labels therefore differ from our input and target contracts.

We use the pilot **training** file to identify original processed ROOT
entries. We then copy their original branches and require four distinct
upstream truth assignments in real saved slots. pilot-v1 required slots 0
through 6; the current recipe accepts all ten saved slots (see
[Recorded provenance and counts](#recorded-provenance-and-counts)). The pilot's
assignment labels and transformed features are not used as model inputs or targets.

The selected events still inherit the pilot's earlier selections through
their membership in that file. They are a deliberately selected development
sample, unsuitable for measuring acceptance or reconstruction performance.
The final dataset construction and split policy remain separate decisions.

## Local files

All binary data and the generated manifest live in the Git-ignored directory
`data/fixtures/pilot-v1/`:

| File | Contents |
| --- | --- |
| `signal.root` | 100 events from `TTtoLNuCB_Summer24MiniAODv6` |
| `background.root` | 100 events from `TTtoLNu2Q_Summer24MiniAODv6` |
| `reference-train.h5` | Unmodified copy of the legacy pilot training file, retained for traceability |
| `manifest.json` | Source paths, checksums, ROOT UUIDs, selected entry numbers, event IDs, and counts |

Each ROOT fixture has an `Events` TTree with 84 branches: event identifiers,
`nJets`, four truth assignments, MET, the trigger lepton, and the kinematics,
ParticleNet score, charge score, and original NanoAOD index for each of the
ten upstream jet slots. All copied values and dtypes are preserved.
Keeping ten slots in this source fixture does not change the seven-jet
model contract. Feature truncation, padding, and score ranking come later.

Weights, systematic variations, and other production trees are not included.
These reduced files are not complete production samples.

## Preparation on BRUX

Run from the repository root after installing `.[dev]`:

```bash
micromamba run -n spanet-reco python scripts/prepare_pilot_fixture.py \
  --reference /isilon/export/home/jhuan166/Vcb/spanet-test/datasets/ttbar_top3_btag_charge_lepton_charge_mc20260908_fullmatch_8k/train.h5 \
  --output-dir data/fixtures/pilot-v1 \
  --events-per-sample 100
```

The recipe requires the original ROOT files at the paths recorded in the
pilot. It rejects an existing output directory; use a new directory for a
fresh reconstruction of the fixture.

For each sample it sorts source paths, chooses the first source with enough
eligible pilot training entries, and takes the first requested entries in
their HDF5 row order. It verifies file UUIDs, run/luminosity-block/event IDs,
and jet counts before copying. It validates the saved upstream assignments
and checks the resulting ROOT files before publishing the output directory.
Original files are opened read-only.

## Recorded provenance and counts

The [versioned manifest](pilot-fixture-manifest.json) records the fixture
prepared on 2026-09-21. The pilot records upstream revision
`d212f09313e4418bd8b14d96bfa61356b3ee9c9d` for its September 8 production.
The manifest preserves that declaration and the pilot selection settings.
Source checksums identify the actual files read.

The candidate counts below refer only to pilot training rows from the one
chosen source file per sample, before taking the first 100 eligible events:

| Sample | Candidates | Unmatched upstream | Match outside seven | Total excluded | Eligible | Copied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Background (`batch_0491.root`) | 2027 | 160 | 31 | 189 | 1838 | 100 |
| Signal (`batch_001.root`) | 1974 | 126 | 35 | 161 | 1813 | 100 |

The two exclusion reasons can overlap; total exclusions count each event
once. These are fixture preparation diagnostics, not production efficiencies.

These counts, and the manifest's selection rule, reflect the recipe on
2026-09-21, when eligible assignments had to lie in saved slots 0 through 6
(`N_INPUT_JETS = 7`). Since the extractor now compacts the leading passing jets
from all ten saved slots, the recipe uses `N_INPUT_JETS = 10`, so "match outside
seven" no longer excludes events. A rebuild with the current recipe can
therefore select different rows than pilot-v1, including events with an
assignment in saved slots 7 to 9. With the same rule, rebuilding selects the same
logical rows from unchanged sources. Fresh ROOT files can have different UUIDs
and byte checksums; the versioned manifest describes this particular prepared copy.

## Inspecting targets

```bash
micromamba run -n spanet-reco python -m spanet_reco.root_io data/fixtures/pilot-v1/signal.root
micromamba run -n spanet-reco python -m spanet_reco.root_io data/fixtures/pilot-v1/background.root
```

Each should report:

```json
{
  "events": 100,
  "unmatched": 0,
  "outside_retained": 0,
  "excluded": 0,
  "fully_matched": 100
}
```

For a smaller range, add `--entry-start 0 --entry-stop 10`; the stop is
exclusive. Invalid ranges, missing branches, and malformed truth labels
raise errors. Only the five branches needed for target validation are read.

Automated tests construct synthetic ROOT and HDF5 files, so they can run
without this local fixture or access to the BRUX MC storage.

## Extracting top-three b-tag and charge inputs

The version 2 extraction mapping [`configs/extract-vcb.yaml`](../configs/extract-vcb.yaml)
declares everything the sibling `nano-spanet-extractor` applies: the seven
leading jets with pT > 25 GeV and abs(eta) < 2.4 (the AN-25-214 jet definition)
among the ten saved slots, compacted into slots 0 through 6; top-three b-tag
selection with fillers 0 (b-tag) and 0.5 (charge) and the `tag_selected`
indicator; and charge-availability rejection. The command takes no cut options.
Events with fewer than seven passing jets are retained with padded slots. From
the SPANet-reco repository root, using the extractor's existing development
environment:

```bash
../nano-spanet-extractor/.venv/bin/nano-spanet-extract \
  --input-dir data/fixtures/pilot-v1 \
  --config configs/extract-vcb.yaml \
  --output data/fixtures/pilot-v1/extract-vcb-v2.h5
```

Choose a fresh output filename to repeat the command, or use `--overwrite` to
deliberately replace a generated HDF5 file. The ROOT fixtures remain the inputs.
Checked on 2026-09-22 with extractor commit `9b17b71` (jet compaction):

| Sample | Input events | Missing selected charge | Written events | Fully matched after jet cuts |
| --- | ---: | ---: | ---: | ---: |
| Signal | 100 | 0 | 100 | 98 |
| Background | 100 | 0 | 100 | 92 |
| Total | 200 | 0 | 200 | 190 |

Each event keeps three to seven jets in contiguous slots and shows exactly three
b-tag and charge scores. Every padded slot has zero-valued features, including
the indicator. Compared with the earlier output below, the eta cut removes an
assigned jet in eight more events, which the fully matched baseline then
excludes. Two signal and three background events have more than seven passing
jets and are reported as truncated. In this fixture, no written jet comes from
saved slots 7 to 9; on the full processed samples, compaction changes the
retained jets of about 1% of events.

The output records total and per-file counts in `PROVENANCE/cutflow`, the jet
cuts in `PROVENANCE/jet_selection`, the score policy, fillers, and indicator in
`PROVENANCE/tag_score_policy`, each written jet's saved slot in
`META/jet_selection/Jets/source_slot`, and original source file/entry
identifiers under `META`. See the [input contract](input-contract.md) for the
complete policy.

### Earlier output (2026-09-21)

`data/fixtures/pilot-v1/top3-charge-pt25-eta24.h5` was prepared with extractor
0.1.0 under the earlier rules. It read a version 1 mapping with built-in cuts and
took the top-three selection from the since-removed options
`--top-n-btag 3 --include-charge`. It applied abs(eta) < 2.4 only to score
ranking, so it keeps 85 jets with abs(eta) >= 2.4 in 73 events as assignment
candidates. It zero-fills unselected scores and has no indicator. Its
`PROVENANCE/config` has no `jet_selection` block, and its
`PROVENANCE/jet_selection` records the threshold as `min_pt_gev`.

| Sample | Input events | Missing selected charge | Written events | Fully matched after pT cut |
| --- | ---: | ---: | ---: | ---: |
| Signal | 100 | 0 | 100 | 100 |
| Background | 100 | 0 | 100 | 98 |
| Total | 200 | 0 | 200 | 198 |

All 200 events retain their original event identities and upstream truth indices
in metadata. In two background events, the pT cut removes a truth-assigned jet;
the affected output target becomes `-1` and its particle mask becomes false.
Those events remain in the extracted file but must be excluded by the fully
matched training baseline, leaving 198 events. All written events have three
selected central jets with available b-tag and charge scores. Jet arrays have
seven slots with four to seven passing jets per event; every masked-out slot has
zero-valued features.

An independent comparison against the ROOT fixture verified all model features,
event and source identities, masks, and truth targets of that earlier output. The
local report is saved as `data/fixtures/pilot-v1/top3-charge-pt25-eta24-audit.json`.
The 31 events reported as truncated refer to input events with more than seven
jets, counted before the charge requirement. This remains a development sample.

### Investigation of unavailable charge scores before the pT cut

Before applying an extractor pT requirement, nine events had a selected jet with
unavailable charge. All nine affected jets fail the current pT > 25 GeV cut and
no longer participate in score ranking. The affected signal jet has processed
pT 23.58 GeV, which passed the earlier 20 GeV requirement. The current fixture
has no missing selected charge scores after selection, but a corrected-pT
threshold alone does not guarantee charge availability in other samples.

These nine jets were matched to their parent NanoAOD records using the production
job input lists, `(run, luminosityBlock, event)`, and `ak4JetNanoIdx`. The b-tag scores
match exactly, and all four parent `Jet_ParT*` charge discriminators are `-1` for
each of these jets. The unavailable values therefore precede the processed skim
and the SPANet extractor.

For all nine jets, raw pT reconstructed as `Jet_pt * (1 - Jet_rawFactor)` is between
12.68 and 14.90 GeV, with `|eta| < 2.5`. For example, signal event 14534513 has a
selected jet in saved slot 5 (NanoAOD jet 9) with raw pT 14.44 GeV, NanoAOD pT
15.85 GeV, processed pT 23.58 GeV, and ParticleNet b-tag score 0.12585.

The local CMSSW code provides a consistent mechanism:

- `RecoBTag/FeatureTools/plugins/ParticleTransformerAK4TagInfoProducer.cc`
  marks jets below 15 GeV or outside `|eta| <= 2.5` as unfilled.
- `RecoBTag/ONNXRuntime/plugins/ParticleTransformerAK4ONNXJetTagsProducer.cc`
  initializes outputs to `-1` and runs inference only for filled tag information.
- `RecoBTag/ONNXRuntime/python/pfParticleNetFromMiniAODAK4_cff.py` sets the PUPPI
  central tagger's minimum jet pT to zero in this checkout.
- `PhysicsTools/NanoAOD/python/jetsAK4_Puppi_cff.py` stores a discriminator only
  when it is strictly positive; otherwise it writes `-1`. This also loses the
  distinction between an unavailable score and an exact zero.

These observations strongly support a mismatch in tagger pT acceptance. The
raw-pT values are reconstructed from stored NanoAOD quantities; the original
per-jet inference inputs have not been replayed. The local diagnostic
`data/fixtures/pilot-v1/charge-score-audit.json` records the nine event identities,
source paths and entries, jet indices, kinematics, and parent scores.
