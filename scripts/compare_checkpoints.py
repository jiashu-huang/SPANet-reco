"""Compare trained runs on the same fully matched events.

For each run directory (SPANet's version_N), the checkpoint with the highest
validation_average_jet_accuracy is evaluated on the first FRACTION of a built
file (default: validation.h5). Results are reported per sample: full-event,
hadronic top, W pair, and leptonic b accuracy, with the W pair unordered, and
the masses of the chosen hadronic top and W jets. The true jets, and the
triplet with the smallest mass chi-square (no network), are shown for reference.

Run from the repository root in the SPANet environment:
    PYTHONPATH=src python scripts/compare_checkpoints.py RUN/version_0 [RUN/version_0 ...]
"""

import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from spanet import JetReconstructionModel, Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from torch.utils.data import DataLoader

from spanet_reco.model import MassChi2Settings, hadronic_chi2, invariant_mass, jet_four_vectors

SAMPLES = {"background": 0, "signal": 1}
TOP_WINDOW, W_WINDOW = 40.0, 30.0  # GeV around the chi-square masses


def load_model(version: Path) -> tuple[JetReconstructionModel, str]:
    options = Options.load(str(version / "options.json"))
    options.num_gpu = 0
    model = JetReconstructionModel(options)
    candidates = [p for p in (version / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt"]
    best = max(candidates, key=lambda p: float(p.stem.rsplit("=", 1)[1]))
    model.load_state_dict(torch.load(best, map_location="cpu")["state_dict"])
    model.eval()
    return model, best.name


def masses(p4: np.ndarray, jets: np.ndarray) -> np.ndarray:
    events = np.arange(len(p4))[:, None]
    return invariant_mass(torch.from_numpy(p4[events, jets].sum(1))).numpy()


def evaluate(data, model=None) -> dict[str, np.ndarray]:
    """Per-event results for a model, or for the true and minimum chi-square triplets."""
    features = {
        f.name: (i, bool(f.log_scale)) for i, f in enumerate(data.event_info.input_features["Jets"])
    }
    names = list(data.event_info.product_particles)
    had, lep = names.index("had_top"), names.index("lep_top")
    rows = {}
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=2048, shuffle=False):
            jets = batch.sources[0]
            p4 = jet_four_vectors(jets.data.float(), features)
            true_had = batch.assignment_targets[had].indices.numpy()
            if model is not None:
                predicted = model.predict(batch.sources).assignments
                chosen, lep_ok = (
                    predicted[had],
                    predicted[lep][:, 0] == (batch.assignment_targets[lep].indices.numpy()[:, 0]),
                )
            else:
                top, w, valid = hadronic_chi2(p4, jets.mask, MassChi2Settings())
                chi2 = torch.where(valid, top + w, torch.inf).flatten(1)
                chosen = np.stack(np.unravel_index(chi2.argmin(1).numpy(), top.shape[1:]), 1)
                lep_ok = np.full(len(chosen), np.nan)
            w_ok = (np.sort(chosen[:, 1:], 1) == np.sort(true_had[:, 1:], 1)).all(1)
            p4 = p4.numpy()
            values = {
                "w_pair": w_ok,
                "had_top": w_ok & (chosen[:, 0] == true_had[:, 0]),
                "lep_b": lep_ok,
                "m_top": masses(p4, chosen),
                "m_w": masses(p4, chosen[:, 1:]),
                "m_top_true": masses(p4, true_had),
                "m_w_true": masses(p4, true_had[:, 1:]),
            }
            for key, value in values.items():
                rows.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in rows.items()}


def summarize(result: dict[str, np.ndarray], select: np.ndarray, prefix: str = "") -> dict:
    out = {"events": int(select.sum())}
    if prefix == "":
        full = result["had_top"] & (result["lep_b"] == 1)
        out |= {"full": full[select].mean()} if not np.isnan(result["lep_b"]).all() else {}
        for key in ("had_top", "w_pair", "lep_b"):
            if not np.isnan(result[key].astype(float)).all():
                out[key] = result[key][select].astype(float).mean()
    for key, center, window in (("m_top", 172.5, TOP_WINDOW), ("m_w", 80.4, W_WINDOW)):
        m = result[key + prefix][select]
        out[f"{key}_window"] = (np.abs(m - center) < window).mean()
        out[f"{key}_median"] = float(np.median(m))
        out[f"{key}_width"] = float((np.percentile(m, 75) - np.percentile(m, 25)) / 1.349)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="SPANet version_N directories")
    parser.add_argument(
        "--data", type=Path, default=Path("data/datasets/mc20260908-v1/validation.h5")
    )
    parser.add_argument("--event-file", type=Path, default=Path("configs/event-vcb.yaml"))
    parser.add_argument("--fraction", type=float, default=0.1, help="leading fraction of --data")
    parser.add_argument("--output", type=Path, help="also write the results as JSON")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    data = JetReconstructionDataset(str(args.data), str(args.event_file), limit_index=args.fraction)
    with h5py.File(args.data, "r") as hdf5_file:
        sample = hdf5_file["META/sample_id"][: len(data)]
    reference = evaluate(data)
    columns = {}
    for run in args.runs:
        model, checkpoint = load_model(run)
        result = evaluate(data, model)
        columns[str(run)] = {"checkpoint": checkpoint} | {
            name: summarize(result, sample == value) for name, value in SAMPLES.items()
        }
    columns["min_chi2"] = {name: summarize(reference, sample == v) for name, v in SAMPLES.items()}
    columns["true_jets"] = {
        name: summarize(reference, sample == v, "_true") for name, v in SAMPLES.items()
    }

    for name in SAMPLES:
        print(f"\n{name} ({int((sample == SAMPLES[name]).sum())} events)")
        keys = [k for k in next(iter(columns.values()))[name] if k != "events"]
        labels = [
            Path(run).parent.name if Path(run).name.startswith("version") else run
            for run in columns
        ]
        print(f"{'':16s}" + "".join(f"{label:>16s}" for label in labels))
        for key in keys:
            cells = [columns[run][name].get(key) for run in columns]
            print(
                f"{key:16s}"
                + "".join(f"{c:16.3f}" if c is not None else f"{'':16s}" for c in cells)
            )
    if args.output:
        args.output.write_text(json.dumps(columns, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
