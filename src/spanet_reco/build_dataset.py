"""Build SPANet training, validation, and test files from extracted samples.

Inputs are nano-spanet-extract outputs for the signal (TTtoLNuCB) and
background (TTtoLNu2Q) samples, possibly in several parts each. Each source
ROOT file goes entirely to one split. Training and validation keep fully
matched events only, with equal numbers from both samples; the test split keeps
every event of its files. Every output adds sin_phi and cos_phi to each input
with phi, and META/sample_id (0 background, 1 signal). Rows are shuffled with a
fixed seed, so the two samples are interleaved.
"""

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from spanet_reco.features import COS_FEATURE, PHI_FEATURE, SIN_FEATURE, sin_cos

SPLITS = ("train", "validation", "test")
# Splits restricted to fully matched events and balanced between samples.
TRAINING_SPLITS = ("train", "validation")
SAMPLE_IDS = {"background": 0, "signal": 1}
# Extraction settings that must be identical in every part.
SHARED_PROVENANCE = ("config", "jet_selection", "tag_score_policy", "spanet_inputs_yaml")
# Rewritten for the combined output rather than copied.
REWRITTEN = {"META/source_file"}
CHUNK_ROWS = 4096


@dataclass
class Part:
    """One extracted file and the per-event information needed to select rows."""

    sample: str
    path: Path
    files: list[str]
    source_entries: np.ndarray
    source_file: np.ndarray
    fully_matched: np.ndarray
    versions: str


def _text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def _layout(hdf5_file: h5py.File) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Map each per-event dataset path to its dtype and per-event shape."""
    layout = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[0] in ("INPUTS", "TARGETS", "META"):
            layout[name] = (obj.dtype.str, obj.shape[1:])

    hdf5_file.visititems(visit)
    return layout


def _load_part(sample: str, path: Path) -> tuple[Part, dict, dict]:
    with h5py.File(path, "r") as hdf5_file:
        if not hdf5_file.attrs.get("complete", False):
            raise ValueError(f"{path}: extraction is not marked complete")
        if hdf5_file.attrs.get("target_mode") != "supervised":
            raise ValueError(f"{path}: expected supervised targets")
        particles = list(hdf5_file["TARGETS"])
        if not particles:
            raise ValueError(f"{path}: no TARGETS")
        fully_matched = np.logical_and.reduce(
            [hdf5_file[f"TARGETS/{particle}/MASK"][:] for particle in particles]
        )
        files = [name.decode() for name in hdf5_file["PROVENANCE/files"][:]]
        source_file = hdf5_file["META/source_file"][:]
        if len(source_file) and (source_file.min() < 0 or source_file.max() >= len(files)):
            raise ValueError(f"{path}: META/source_file outside PROVENANCE/files")
        shared = {name: _text(hdf5_file[f"PROVENANCE/{name}"]) for name in SHARED_PROVENANCE}
        shared["extractor_version"] = str(hdf5_file.attrs["extractor_version"])
        part = Part(
            sample=sample,
            path=path.resolve(),
            files=files,
            source_entries=hdf5_file["PROVENANCE/source_entries"][:],
            source_file=source_file,
            fully_matched=fully_matched,
            versions=_text(hdf5_file["PROVENANCE/versions"]),
        )
        return part, shared, _layout(hdf5_file)


def _split_counts(n_files: int, fractions: tuple[float, float, float]) -> list[int]:
    validation, test = (max(1, round(n_files * fraction)) for fraction in fractions[1:])
    train = n_files - validation - test
    if train < 1:
        raise ValueError(f"{n_files} source files are too few for three splits")
    return [train, validation, test]


def _assign_files(
    parts: list[Part], fractions: tuple[float, float, float], seed: int
) -> dict[str, dict[str, str]]:
    """Assign every source file of each sample to one split, reproducibly."""
    assignment = {}
    for sample, sample_id in SAMPLE_IDS.items():
        files = sorted({name for part in parts if part.sample == sample for name in part.files})
        order = np.random.default_rng([seed, sample_id]).permutation(len(files))
        counts = _split_counts(len(files), fractions)
        bounds = np.cumsum([0, *counts])
        assignment[sample] = {
            files[index]: split
            for split, start, stop in zip(SPLITS, bounds[:-1], bounds[1:], strict=True)
            for index in order[start:stop]
        }
    return assignment


def _select_rows(
    parts: list[Part],
    assignment: dict[str, dict[str, str]],
    seed: int,
    train_limit: int | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Choose (part, row) pairs for each split in output order, and count them."""
    rng = np.random.default_rng([seed, len(SAMPLE_IDS)])
    selected, counts = {}, {}
    for split in SPLITS:
        per_sample = {}
        for sample in SAMPLE_IDS:
            part_index, rows = [], []
            for index, part in enumerate(parts):
                if part.sample != sample:
                    continue
                in_split = np.array(
                    [assignment[sample][name] == split for name in part.files], dtype=bool
                )
                keep = in_split[part.source_file]
                if split in TRAINING_SPLITS:
                    keep = keep & part.fully_matched
                found = np.flatnonzero(keep)
                part_index.append(np.full(len(found), index))
                rows.append(found)
            per_sample[sample] = (np.concatenate(part_index), np.concatenate(rows))

        available = {sample: len(rows) for sample, (_, rows) in per_sample.items()}
        if split in TRAINING_SPLITS:
            target = min(available.values())
            if split == "train" and train_limit is not None:
                target = min(target, train_limit)
            if target == 0:
                raise ValueError(f"{split}: a sample has no fully matched events")
            for sample, (part_index, rows) in per_sample.items():
                chosen = np.sort(rng.choice(len(rows), size=target, replace=False))
                per_sample[sample] = (part_index[chosen], rows[chosen])

        part_index = np.concatenate([values[0] for values in per_sample.values()])
        rows = np.concatenate([values[1] for values in per_sample.values()])
        order = rng.permutation(len(rows))
        selected[split] = (part_index[order], rows[order])
        matched = [part.fully_matched for part in parts]
        counts[split] = {
            sample: {
                "available": available[sample],
                "events": len(per_sample[sample][1]),
                "fully_matched": int(np.count_nonzero(_gather(matched, *per_sample[sample]))),
            }
            for sample in SAMPLE_IDS
        }
    return selected, counts


def _gather(columns: list[np.ndarray], part_index: np.ndarray, rows: np.ndarray) -> np.ndarray:
    output = np.empty((len(rows), *columns[0].shape[1:]), dtype=columns[0].dtype)
    for index, column in enumerate(columns):
        mask = part_index == index
        output[mask] = column[rows[mask]]
    return output


def _write(group: h5py.File, name: str, values: np.ndarray) -> None:
    group.create_dataset(
        name,
        data=values,
        chunks=(CHUNK_ROWS, *values.shape[1:]),
        maxshape=(None, *values.shape[1:]),
        compression="lzf",
    )


def build(
    samples: dict[str, list[Path]],
    output_dir: str | Path,
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20260922,
    train_limit: int | None = None,
) -> dict:
    """Write train.h5, validation.h5, test.h5, and summary.json into a new directory.

    samples maps "signal" and "background" to their extracted parts. The output
    directory must not exist; it appears only after every file is written.
    Returns the summary also saved as summary.json.
    """
    output_dir = Path(output_dir)
    if set(samples) != set(SAMPLE_IDS) or not all(samples.values()):
        raise ValueError(f"need extracted parts for each of {', '.join(SAMPLE_IDS)}")
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must be three positive numbers summing to 1")
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    parts, reference = [], None
    for sample, paths in samples.items():
        for path in paths:
            part, shared, layout = _load_part(sample, Path(path))
            if reference is None:
                reference = (part.path, shared, layout)
            elif (shared, layout) != reference[1:]:
                raise ValueError(
                    f"{part.path}: extraction settings or layout differ from {reference[0]}"
                )
            parts.append(part)
    _, shared, layout = reference
    all_files = [name for part in parts for name in part.files]
    if len(set(all_files)) != len(all_files):
        raise ValueError("a source file appears in more than one extracted part")

    combined_files = sorted(all_files)
    file_index = {name: index for index, name in enumerate(combined_files)}
    source_entries = np.zeros(len(combined_files), dtype=np.int64)
    for part in parts:
        source_entries[[file_index[name] for name in part.files]] = part.source_entries
    assignment = _assign_files(parts, fractions, seed)
    selected, counts = _select_rows(parts, assignment, seed, train_limit)

    for split in SPLITS:
        for sample in SAMPLE_IDS:
            split_files = [name for name, where in assignment[sample].items() if where == split]
            counts[split][sample]["files"] = len(split_files)
            counts[split][sample]["input_events"] = int(
                sum(source_entries[file_index[name]] for name in split_files)
            )
    record = {
        "tool": "spanet_reco.build_dataset",
        "spanet_reco_version": version("spanet-reco"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "fractions": dict(zip(SPLITS, fractions, strict=True)),
        "train_limit": train_limit,
        "sample_ids": SAMPLE_IDS,
        "selection": {
            "train": "fully matched, balanced between samples",
            "validation": "fully matched, balanced between samples",
            "test": "all events of its files",
        },
        "extractor_version": shared["extractor_version"],
        "parts": [
            {"sample": part.sample, "path": str(part.path), "versions": part.versions}
            for part in parts
        ],
        "split_files": {
            split: sorted(
                name
                for sample in SAMPLE_IDS
                for name, where in assignment[sample].items()
                if where == split
            )
            for split in SPLITS
        },
        "counts": counts,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        staging.mkdir()
        outputs = {split: h5py.File(staging / f"{split}.h5", "w") for split in SPLITS}
        sources = [h5py.File(part.path, "r") for part in parts]
        try:
            # One dataset at a time bounds memory by the largest single column.
            for name in sorted(set(layout) - REWRITTEN):
                columns = [source[name][:] for source in sources]
                for split, (part_index, rows) in selected.items():
                    _write(outputs[split], name, _gather(columns, part_index, rows))
            groups = sorted(
                name for name in sources[0]["INPUTS"] if PHI_FEATURE in sources[0]["INPUTS"][name]
            )
            for group in groups:
                phi = [source[f"INPUTS/{group}/{PHI_FEATURE}"][:] for source in sources]
                has_mask = "MASK" in sources[0]["INPUTS"][group]
                masks = (
                    [source[f"INPUTS/{group}/MASK"][:] for source in sources] if has_mask else None
                )
                for split, (part_index, rows) in selected.items():
                    real = _gather(masks, part_index, rows) if has_mask else None
                    sin, cos = sin_cos(_gather(phi, part_index, rows), real, group)
                    _write(outputs[split], f"INPUTS/{group}/{SIN_FEATURE}", sin)
                    _write(outputs[split], f"INPUTS/{group}/{COS_FEATURE}", cos)

            global_file = [
                np.array([file_index[name] for name in part.files], dtype=np.int64)
                for part in parts
            ]
            for split, (part_index, rows) in selected.items():
                output = outputs[split]
                local = _gather([part.source_file for part in parts], part_index, rows)
                source_file = np.empty(len(rows), dtype=np.int64)
                for index, mapping in enumerate(global_file):
                    mask = part_index == index
                    source_file[mask] = mapping[local[mask]]
                _write(output, "META/source_file", source_file)
                sample_id = np.array([SAMPLE_IDS[part.sample] for part in parts], dtype=np.uint8)
                _write(output, "META/sample_id", sample_id[part_index])

                provenance = output.create_group("PROVENANCE")
                provenance.create_dataset("files", data=combined_files, dtype=h5py.string_dtype())
                provenance.create_dataset("source_entries", data=source_entries)
                for name in SHARED_PROVENANCE:
                    provenance.create_dataset(name, data=shared[name], dtype=h5py.string_dtype())
                provenance.create_dataset(
                    "build", data=json.dumps({**record, "split": split}), dtype=h5py.string_dtype()
                )
                output.attrs.update(
                    {
                        "format": "SPANet v2",
                        "target_mode": "supervised",
                        "split": split,
                        "num_events": len(rows),
                        "extractor_version": shared["extractor_version"],
                        "spanet_reco_version": record["spanet_reco_version"],
                        "complete": True,
                    }
                )
        finally:
            for hdf5_file in (*outputs.values(), *sources):
                hdf5_file.close()
        (staging / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        shutil.move(staging, output_dir)
    return record


def main() -> None:
    """Build the datasets and print the event counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", nargs="+", type=Path, required=True, help="TTtoLNuCB parts")
    parser.add_argument("--background", nargs="+", type=Path, required=True, help="TTtoLNu2Q parts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="fractions of source files per split",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--train-limit", type=int, help="maximum training events per sample, after balancing"
    )
    args = parser.parse_args()
    record = build(
        {"signal": args.signal, "background": args.background},
        args.output_dir,
        fractions=tuple(args.fractions),
        seed=args.seed,
        train_limit=args.train_limit,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "counts": record["counts"]}, indent=2))


if __name__ == "__main__":
    main()
