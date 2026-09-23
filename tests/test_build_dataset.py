import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from spanet_reco.build_dataset import SAMPLE_IDS, build

SHARED = {
    "config": '{"version": 2}',
    "jet_selection": "{}",
    "tag_score_policy": "{}",
    "spanet_inputs_yaml": "INPUTS: {}",
}


def write_part(path, sample, file_numbers, events_per_file=20, config='{"version": 2}'):
    """Write a small extractor-format part; every event has a unique identity."""
    rng = np.random.default_rng(sum(file_numbers) + len(sample))
    files = [f"/mc/{sample}/batch_{number:03d}.root" for number in file_numbers]
    source_file = np.repeat(np.arange(len(files)), events_per_file)
    source_entry = np.tile(np.arange(events_per_file), len(files))
    n = len(source_file)
    mask = np.arange(4)[None, :] < rng.integers(1, 5, n)[:, None]
    matched = rng.random(n) < 0.5
    with h5py.File(path, "w") as output:
        output.attrs.update(
            complete=True, target_mode="supervised", extractor_version="0.2.0", num_events=n
        )
        output["INPUTS/Jets/MASK"] = mask
        output["INPUTS/Jets/pt"] = np.where(mask, rng.uniform(25, 200, (n, 4)), 0).astype("f4")
        output["INPUTS/Jets/phi"] = np.where(mask, rng.uniform(-3, 3, (n, 4)), 0).astype("f4")
        output["INPUTS/Met/phi"] = rng.uniform(-3, 3, n).astype("f4")
        output["INPUTS/Met/pt"] = rng.uniform(0, 100, n).astype("f4")
        output["TARGETS/had_top/b"] = np.where(matched, 0, -1)
        output["TARGETS/had_top/MASK"] = matched
        output["TARGETS/lep_top/b"] = np.ones(n, dtype=np.int64)
        output["TARGETS/lep_top/MASK"] = np.ones(n, dtype=bool)
        output["META/event"] = (np.array(file_numbers)[source_file] * 1000 + source_entry).astype(
            np.uint64
        )
        output["META/source_file"] = source_file
        output["META/source_entry"] = source_entry
        output["META/jet_selection/Jets/source_slot"] = np.where(mask, np.arange(4), -1)
        output["PROVENANCE/files"] = np.array(files, dtype=object)
        output["PROVENANCE/source_entries"] = np.full(len(files), events_per_file + 1)
        output["PROVENANCE/versions"] = "{}"
        for name, value in {**SHARED, "config": config}.items():
            output.create_dataset(f"PROVENANCE/{name}", data=value, dtype=h5py.string_dtype())
    return path


@pytest.fixture
def parts(tmp_path):
    return {
        "signal": [
            write_part(tmp_path / "signal-0.h5", "signal", range(0, 12)),
            write_part(tmp_path / "signal-1.h5", "signal", range(12, 20)),
        ],
        "background": [write_part(tmp_path / "background.h5", "background", range(0, 24))],
    }


def read_split(path):
    with h5py.File(path, "r") as hdf5_file:
        data = {}
        hdf5_file.visititems(
            lambda name, obj: (
                data.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None
            )
        )
        return data, dict(hdf5_file.attrs)


def source_rows(parts):
    """Map (sample, source file path, entry) to that event's datasets in its part."""
    rows = {}
    for sample, paths in parts.items():
        for path in paths:
            data, _ = read_split(path)
            files = [name.decode() for name in data["PROVENANCE/files"]]
            for row, (index, entry) in enumerate(
                zip(data["META/source_file"], data["META/source_entry"], strict=True)
            ):
                rows[(sample, files[index], entry)] = {
                    name: values[row]
                    for name, values in data.items()
                    if name.split("/")[0] in ("INPUTS", "TARGETS", "META")
                }
    return rows


def test_splits_rows_and_features(parts, tmp_path):
    output = tmp_path / "datasets"
    record = build(parts, output, fractions=(0.8, 0.1, 0.1), seed=7)
    sources = source_rows(parts)
    ids = {value: name for name, value in SAMPLE_IDS.items()}
    seen_files = {}

    for split in ("train", "validation", "test"):
        data, attrs = read_split(output / f"{split}.h5")
        assert attrs["split"] == split and attrs["complete"]
        files = [name.decode() for name in data["PROVENANCE/files"]]
        n = attrs["num_events"]
        full = data["TARGETS/had_top/MASK"] & data["TARGETS/lep_top/MASK"]
        samples = data["META/sample_id"]
        for row in range(n):
            sample = ids[samples[row]]
            name = files[data["META/source_file"][row]]
            seen_files.setdefault(name, set()).add(split)
            source = sources[(sample, name, data["META/source_entry"][row])]
            # Every copied dataset matches the source event exactly.
            for key, value in source.items():
                if key != "META/source_file":
                    np.testing.assert_array_equal(data[key][row], value, err_msg=key)
            for group, real in (("Jets", data["INPUTS/Jets/MASK"][row]), ("Met", True)):
                phi = data[f"INPUTS/{group}/phi"][row]
                np.testing.assert_array_equal(
                    data[f"INPUTS/{group}/cos_phi"][row], np.where(real, np.cos(phi), 0)
                )
        if split == "test":
            # Every event of the test files is kept, matched or not.
            assert not full.all()
            assert n == sum(20 for name, where in seen_files.items() if "test" in where)
        else:
            assert full.all()
            assert np.count_nonzero(samples == 0) == np.count_nonzero(samples == 1)
        for sample, sample_id in SAMPLE_IDS.items():
            assert record["counts"][split][sample]["events"] == np.count_nonzero(
                samples == sample_id
            )

    # Each source file belongs to exactly one split, and all files are used.
    assert all(len(where) == 1 for where in seen_files.values())
    assert json.loads((output / "summary.json").read_text())["seed"] == 7
    assert record["counts"]["test"]["signal"]["files"] == 2
    assert record["counts"]["test"]["signal"]["input_events"] == 2 * 21


def test_same_seed_same_output(parts, tmp_path):
    build(parts, tmp_path / "a", seed=3)
    build(parts, tmp_path / "b", seed=3)
    build(parts, tmp_path / "c", seed=4)
    for split in ("train", "validation", "test"):
        a, _ = read_split(tmp_path / "a" / f"{split}.h5")
        b, _ = read_split(tmp_path / "b" / f"{split}.h5")
        c, _ = read_split(tmp_path / "c" / f"{split}.h5")
        np.testing.assert_array_equal(a["META/event"], b["META/event"])
        assert not np.array_equal(a["META/event"], c["META/event"]) or split != "train"


def test_train_limit(parts, tmp_path):
    record = build(parts, tmp_path / "out", train_limit=5)
    assert {record["counts"]["train"][s]["events"] for s in SAMPLE_IDS} == {5}


def test_inconsistent_extraction_is_rejected(parts, tmp_path):
    parts["background"].append(
        write_part(tmp_path / "other.h5", "background", [99], config='{"version": 3}')
    )
    with pytest.raises(ValueError, match="extraction settings or layout differ"):
        build(parts, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_source_file_is_rejected(parts, tmp_path):
    parts["background"].append(write_part(tmp_path / "again.h5", "background", [0]))
    with pytest.raises(ValueError, match="more than one extracted part"):
        build(parts, tmp_path / "out")


def test_existing_output_is_refused(parts, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        build(parts, tmp_path / "out")


def test_cli(parts, tmp_path):
    output = tmp_path / "out"
    command = [sys.executable, "-m", "spanet_reco.build_dataset", "--output-dir", str(output)]
    command += [
        "--signal",
        *map(str, parts["signal"]),
        "--background",
        *map(str, parts["background"]),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    counts = json.loads(result.stdout)["counts"]
    assert set(counts) == {"train", "validation", "test"}
    assert sorted(path.name for path in output.iterdir()) == [
        "summary.json",
        "test.h5",
        "train.h5",
        "validation.h5",
    ]
