"""Exercise fixture recovery through its CLI without requiring private MC files."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import uproot

from spanet_reco.root_io import read_target_batch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pilot_fixture.py"


@pytest.fixture
def pilot_reference(tmp_path):
    sources = {}
    metadata = {
        name: []
        for name in (
            "sample_id",
            "source_file_id",
            "source_entry",
            "n_production_jets",
            "run",
            "luminosityBlock",
            "event",
        )
    }
    for sample_id, name in enumerate(("background", "signal")):
        source = tmp_path / f"{name}.root"
        columns = {
            "nJets": np.full(4, 8, dtype=np.int32),
            "GenHadBJetIdx": np.array([6, 0, 4, 0], dtype=np.int32),
            "GenHadQ1JetIdx": np.array([2, 1, 1, 2], dtype=np.int32),
            "GenHadQ2JetIdx": np.array([1, 2, 2, 1], dtype=np.int32),
            "GenLepBJetIdx": np.array([-1, 7, 0, 3], dtype=np.int32),
            "run": np.full(4, 1, dtype=np.uint32),
            "luminosityBlock": np.full(4, sample_id + 1, dtype=np.uint32),
            # Keep identities above the exact-integer range of float64.
            "event": np.arange(4, dtype=np.uint64) + np.uint64(2**54),
        }
        for field in (
            "METPt",
            "METPhi",
            "TriggerLeptonPt",
            "TriggerLeptonEta",
            "TriggerLeptonPhi",
            "TriggerLeptonCharge",
        ):
            columns[field] = np.arange(4, dtype=np.float32) + 0.25
        for field in ("Mass", "Pt", "Eta", "Phi", "btagPNetB", "ParTPosvsNeg", "NanoIdx"):
            for slot in range(10):
                columns[f"ak4Jet{field}{slot}"] = (
                    np.arange(4, dtype=np.float32) + slot / 10
                    if slot < 8
                    else np.full(4, -99999, dtype=np.float32)
                )
        with uproot.recreate(source) as root_file:
            root_file.mktree("Events", {key: value.dtype for key, value in columns.items()})
            root_file["Events"].extend(columns)
        with uproot.open(source) as root_file:
            file_id = int.from_bytes(
                hashlib.blake2b(str(root_file.file.uuid).encode(), digest_size=8).digest(),
                "little",
            )
        sources[str(file_id)] = str(source)
        entries = np.array([3, 0, 1, 2])
        metadata["sample_id"].extend([sample_id] * 4)
        metadata["source_file_id"].extend([file_id] * 4)
        metadata["source_entry"].extend(entries)
        metadata["n_production_jets"].extend(columns["nJets"][entries])
        for key in ("run", "luminosityBlock", "event"):
            metadata[key].extend(columns[key][entries])

    reference = tmp_path / "train.h5"
    with h5py.File(reference, "w") as handle:
        handle.attrs.update(
            complete=True,
            split="train",
            report=json.dumps({"source_files": sources}),
            selection=json.dumps({"production_commit": "synthetic-test"}),
            feature_config="{}",
        )
        for key, values in metadata.items():
            dtype = np.uint64 if key in ("source_file_id", "event") else np.int64
            handle[f"META/{key}"] = np.array(values, dtype=dtype)
        for top in ("ht", "lt"):
            handle[f"TARGETS/{top}/MASK"] = np.ones(8, dtype=bool)
        # Pilot targets intentionally disagree with the upstream assignments.
        for role, value in (("ht/b", 0), ("ht/q1", 1), ("ht/q2", 2), ("lt/b", 3)):
            handle[f"TARGETS/{role}"] = np.full(8, value, dtype=np.int64)
    return reference


def run_recipe(reference, output):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--output-dir",
            str(output),
            "--events-per-sample",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_fixture_recovers_original_rows_and_reselects_upstream_truth(pilot_reference, tmp_path):
    output = tmp_path / "fixture"
    result = run_recipe(pilot_reference, output)
    assert result.returncode == 0, result.stderr
    assert (output / "reference-train.h5").read_bytes() == pilot_reference.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["samples"]) == {"background", "signal"}
    for name, details in manifest["samples"].items():
        # Entry 0 is unmatched; entry 1, with a lepton-side b in saved slot 7, is eligible.
        assert details["source_entries"] == [3, 1]
        assert details["source_reference_candidates"] == {
            "events": 4,
            "unmatched": 1,
            "outside_retained": 0,
            "excluded": 1,
            "fully_matched": 3,
        }
        source = Path(details["source_root"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == details["source_root_sha256"]
        with uproot.open(source) as original, uproot.open(output / f"{name}.root") as copied:
            assert copied["Events"].num_entries == 2
            assert set(copied["Events"].keys()) == set(original["Events"].keys())
            for branch in original["Events"].keys():
                expected = original["Events"][branch].array(library="np")[[3, 1]]
                np.testing.assert_array_equal(
                    copied["Events"][branch].array(library="np"), expected, strict=True
                )
        np.testing.assert_array_equal(
            read_target_batch(output / f"{name}.root").targets,
            [[0, 2, 1, 3], [0, 1, 2, 7]],
        )
    before = (output / "manifest.json").read_bytes()
    rerun = run_recipe(pilot_reference, output)
    assert rerun.returncode != 0
    assert "Output already exists" in rerun.stderr
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "corruption,message",
    [
        ("identity", "Reference event identity disagrees"),
        ("split", "completed pilot training split"),
        ("uuid", "Source ROOT UUID disagrees"),
    ],
)
def test_stale_or_held_out_reference_is_rejected(pilot_reference, tmp_path, corruption, message):
    with h5py.File(pilot_reference, "r+") as handle:
        if corruption == "identity":
            handle["META/event"][0] = np.uint64(99)
        elif corruption == "split":
            handle.attrs["split"] = "test"
        else:
            old_id = int(handle["META/source_file_id"][0])
            handle["META/source_file_id"][:4] = np.uint64(1)
            report = json.loads(handle.attrs["report"])
            report["source_files"]["1"] = report["source_files"].pop(str(old_id))
            handle.attrs["report"] = json.dumps(report)
    output = tmp_path / "fixture"
    result = run_recipe(pilot_reference, output)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()
