"""Recover a small ROOT development fixture from legacy pilot training identities.

The HDF5 file supplies event identities only. Features and truth assignments
are copied from the original processed ROOT files without re-encoding or
rematching. This deliberately selected fixture is not an evaluation sample.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import uproot

from spanet_reco.root_io import TargetBatch, read_target_batch
from spanet_reco.targets import N_INPUT_JETS, N_SAVED_JETS, TARGET_BRANCHES

IDENTITY_BRANCHES = ("run", "luminosityBlock", "event")
JET_FIELDS = ("Mass", "Pt", "Eta", "Phi", "btagPNetB", "ParTPosvsNeg", "NanoIdx")
FIXTURE_BRANCHES = (
    *IDENTITY_BRANCHES,
    "nJets",
    *TARGET_BRANCHES,
    "METPt",
    "METPhi",
    "TriggerLeptonPt",
    "TriggerLeptonEta",
    "TriggerLeptonPhi",
    "TriggerLeptonCharge",
    *(f"ak4Jet{field}{slot}" for field in JET_FIELDS for slot in range(N_SAVED_JETS)),
)
SAMPLES = {0: "background", 1: "signal"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_reference(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False) or handle.attrs.get("split") != "train":
            raise ValueError("Reference must be a completed pilot training split")
        if not all(np.all(handle[f"TARGETS/{top}/MASK"][:]) for top in ("ht", "lt")):
            raise ValueError("Reference must contain fully matched pilot events")
        names = (
            "sample_id",
            "source_file_id",
            "source_entry",
            "n_production_jets",
            *IDENTITY_BRANCHES,
        )
        meta = {name: handle[f"META/{name}"][:] for name in names}
        report = json.loads(handle.attrs["report"])
        provenance = {
            "reference_hdf5": str(path.resolve()),
            "reference_sha256": sha256(path),
            "reference_split": "train",
            "legacy_selection": json.loads(handle.attrs["selection"]),
            "legacy_feature_config": json.loads(handle.attrs["feature_config"]),
            "source_files": report["source_files"],
        }

    size = len(meta["sample_id"])
    for name, values in meta.items():
        if values.shape != (size,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Reference META/{name} must be a one-dimensional integer array")
    if set(np.unique(meta["sample_id"])) != set(SAMPLES):
        raise ValueError("Reference must contain background (0) and signal (1)")
    if np.any(meta["source_entry"] < 0):
        raise ValueError("Reference contains negative ROOT entry numbers")
    keys = list(
        zip(*(meta[key].tolist() for key in ("sample_id", *IDENTITY_BRANCHES)), strict=True)
    )
    if len(set(keys)) != len(keys):
        raise ValueError("Reference contains duplicate event identities within a sample")
    return meta, provenance


def select_sample(meta: dict, provenance: dict, sample_id: int, count: int) -> tuple[dict, dict]:
    """Choose the first source by path that supplies enough eligible pilot rows."""
    sample_rows = meta["sample_id"] == sample_id
    sources = provenance["source_files"]
    file_ids = sorted(
        np.unique(meta["source_file_id"][sample_rows]),
        key=lambda file_id: sources[str(int(file_id))],
    )
    for file_id in file_ids:
        source = Path(sources[str(int(file_id))])
        rows = np.flatnonzero(sample_rows & (meta["source_file_id"] == file_id))
        entries = meta["source_entry"][rows]
        batch = read_target_batch(source)
        if np.any(entries >= len(batch.n_jets)):
            raise ValueError(f"Reference entry outside source tree: {source}")

        with uproot.open(source) as root_file:
            source_uuid = str(root_file.file.uuid)
            # This is the file-ID convention used by the existing pilot dataset.
            actual_id = int.from_bytes(
                hashlib.blake2b(source_uuid.encode(), digest_size=8).digest(), "little"
            )
            if actual_id != int(file_id):
                raise ValueError(f"Source ROOT UUID disagrees with reference: {source}")
            tree = root_file["Events"]
            missing = sorted(set(FIXTURE_BRANCHES) - set(tree.keys()))
            if missing:
                raise ValueError(f"{source}: missing fixture branches: {', '.join(missing)}")
            columns = tree.arrays(list(FIXTURE_BRANCHES), library="np")

        for name in IDENTITY_BRANCHES:
            if not np.array_equal(columns[name][entries], meta[name][rows]):
                raise ValueError(f"Reference event identity disagrees with source {name}: {source}")
        if not np.array_equal(batch.n_jets[entries], meta["n_production_jets"][rows]):
            raise ValueError(f"Reference jet counts disagree with source: {source}")

        eligible = batch.fully_matched[entries]
        if np.count_nonzero(eligible) < count:
            continue
        chosen_rows = rows[eligible][:count]
        chosen_entries = meta["source_entry"][chosen_rows]
        selected = {name: values[chosen_entries] for name, values in columns.items()}
        candidate_batch = TargetBatch(
            batch.targets[entries], batch.n_jets[entries], batch.fully_matched[entries]
        )
        details = {
            "sample_id": sample_id,
            "source_root": str(source.resolve()),
            "source_root_uuid": source_uuid,
            "source_root_sha256": sha256(source),
            "source_reference_candidates": candidate_batch.counts(),
            "events_copied": count,
            "reference_rows": chosen_rows.tolist(),
            "source_entries": chosen_entries.tolist(),
            "event_identity": {name: selected[name].tolist() for name in IDENTITY_BRANCHES},
        }
        return selected, details
    raise ValueError(f"No single source file supplies {count} eligible {SAMPLES[sample_id]} events")


def prepare(reference: Path, output: Path, count: int) -> dict:
    if count < 1:
        raise ValueError("events-per-sample must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    meta, provenance = read_reference(reference)
    selected = {
        name: select_sample(meta, provenance, sample_id, count)
        for sample_id, name in SAMPLES.items()
    }
    provenance.pop("source_files")
    manifest = {
        "schema_version": 1,
        "purpose": "development_fixture_only",
        "selection_rule": (
            "For each sample, use the first ROOT source in path order with enough eligible "
            "pilot training rows; take the first requested rows in HDF5 order after requiring "
            "four distinct upstream Gen*JetIdx assignments in real slots 0 through "
            f"{N_INPUT_JETS - 1}."
        ),
        "truth_source": "upstream Gen*JetIdx branches; pilot targets are not copied",
        "copied_branches": list(FIXTURE_BRANCHES),
        **provenance,
        "samples": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-fixture-", dir=output.parent) as temporary:
        staging = Path(temporary)
        reference_copy = staging / "reference-train.h5"
        shutil.copyfile(reference, reference_copy)
        if sha256(reference_copy) != provenance["reference_sha256"]:
            raise ValueError("Reference changed while preparing the fixture")
        for name, (columns, details) in selected.items():
            path = staging / f"{name}.root"
            with uproot.create(path) as root_file:
                tree = root_file.mktree(
                    "Events", {branch: values.dtype for branch, values in columns.items()}
                )
                tree.extend(columns)
            counts = read_target_batch(path).counts()
            if counts["fully_matched"] != count or counts["excluded"] != 0:
                raise ValueError(f"Written {name} fixture did not pass target validation")
            manifest["samples"][name] = {
                **details,
                "file": path.name,
                "file_sha256": sha256(path),
                "validation": counts,
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events-per-sample", type=int, default=100)
    args = parser.parse_args()
    manifest = prepare(args.reference, args.output_dir, args.events_per_sample)
    print(
        json.dumps(
            {name: item["validation"] for name, item in manifest["samples"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
