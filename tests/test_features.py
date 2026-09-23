import json
import math
import subprocess
import sys

import h5py
import numpy as np
import pytest

from spanet_reco.features import add_angular_features

JET_PHI = np.array(
    [
        [0.5, -math.pi, 99.0, 0.0],  # Last two slots are padding; 99 must be ignored.
        [np.float32(math.pi), 1.0, -2.0, 3.0],  # float32(pi) is slightly above pi.
        [0.0, 0.0, 0.0, 0.0],  # No real jets.
    ],
    dtype=np.float32,
)
JET_MASK = np.array(
    [[True, True, False, False], [True, True, True, True], [False, False, False, False]]
)


def write_extracted(path, jet_phi=JET_PHI, lepton_phi=(0.1, -1.2, 2.9)):
    """Write the parts of the extractor layout that the feature step reads or copies."""
    events = len(jet_phi)
    with h5py.File(path, "w") as output:
        output.attrs["complete"] = True
        output.attrs["num_events"] = events
        jets = output.create_group("INPUTS/Jets")
        resizable = {"chunks": (2, 4), "maxshape": (None, 4), "compression": "lzf"}
        jets.create_dataset("MASK", data=JET_MASK[:events], **resizable)
        jets.create_dataset("phi", data=jet_phi, **resizable)
        jets.create_dataset("pt", data=np.full(jet_phi.shape, 40, dtype=np.float32))
        output.create_dataset("INPUTS/Lepton/phi", data=np.array(lepton_phi, dtype=np.float32))
        output.create_dataset("INPUTS/Met/phi", data=np.array([-3.0, 0.0, 1.5], dtype=np.float32))
        output.create_dataset("INPUTS/Met/pt", data=np.array([10, 20, 30], dtype=np.float32))
        output.create_dataset("TARGETS/had_top/b", data=np.array([0, 1, -1]))
        output["TARGETS/had_top"].attrs["note"] = "attributes are copied"
        output.create_dataset("META/event", data=np.array([1, 2, 2**63 + 5], dtype=np.uint64))
        output.create_dataset("PROVENANCE/config", data='{"version": 2}', dtype=h5py.string_dtype())


@pytest.fixture
def extracted(tmp_path):
    path = tmp_path / "extracted.h5"
    write_extracted(path)
    return path


def datasets(hdf5_file):
    found = {}

    def collect(name, obj):
        # A non-None return value would stop the traversal.
        if isinstance(obj, h5py.Dataset):
            found[name] = obj

    hdf5_file.visititems(collect)
    return found


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_angles_follow_phi_and_padding(extracted, tmp_path, chunk_size):
    output = tmp_path / "angles.h5"
    groups = add_angular_features(extracted, output, chunk_size=chunk_size)
    assert groups == ["Jets", "Lepton", "Met"]

    with h5py.File(output, "r") as result:
        for name in ("sin_phi", "cos_phi"):
            dataset = result[f"INPUTS/Jets/{name}"]
            assert dataset.dtype == np.float32 and dataset.shape == JET_PHI.shape
            assert dataset.compression == "lzf" and dataset.chunks == (2, 4)
        function = {"sin_phi": np.sin, "cos_phi": np.cos}
        for name, apply in function.items():
            expected = np.where(JET_MASK, apply(np.where(JET_MASK, JET_PHI, 0)), 0)
            np.testing.assert_array_equal(
                result[f"INPUTS/Jets/{name}"][:], expected.astype(np.float32)
            )
            # Global inputs have no mask: every event has one real object.
            np.testing.assert_array_equal(
                result[f"INPUTS/Met/{name}"][:],
                apply(np.array([-3.0, 0.0, 1.5], dtype=np.float32)),
            )
        # Directions on either side of the +-pi boundary agree after the transform.
        np.testing.assert_allclose(result["INPUTS/Jets/cos_phi"][0, 1], -1, atol=1e-6)
        np.testing.assert_allclose(result["INPUTS/Jets/cos_phi"][1, 0], -1, atol=1e-6)


def test_source_content_is_copied_unchanged(extracted, tmp_path):
    output = tmp_path / "angles.h5"
    add_angular_features(extracted, output)

    with h5py.File(extracted, "r") as source, h5py.File(output, "r") as result:
        assert dict(result.attrs) == dict(source.attrs)
        assert result["TARGETS/had_top"].attrs["note"] == "attributes are copied"
        source_datasets, result_datasets = datasets(source), datasets(result)
        added = {
            f"INPUTS/{group}/{name}"
            for group in ("Jets", "Lepton", "Met")
            for name in ("sin_phi", "cos_phi")
        }
        assert set(result_datasets) == set(source_datasets) | added | {
            "PROVENANCE/angular_features"
        }
        for name, dataset in source_datasets.items():
            assert result_datasets[name].dtype == dataset.dtype, name
            np.testing.assert_array_equal(result_datasets[name][()], dataset[()], strict=True)

        record = json.loads(result["PROVENANCE/angular_features"][()])
        assert record["groups"] == ["Jets", "Lepton", "Met"]
        assert record["source"] == str(extracted.resolve())
        assert record["features"] == {"sin_phi": "sin(phi)", "cos_phi": "cos(phi)"}


def test_empty_file(tmp_path):
    source = tmp_path / "empty.h5"
    write_extracted(source, jet_phi=np.zeros((0, 4), dtype=np.float32), lepton_phi=())
    with h5py.File(source, "a") as hdf5_file:
        del hdf5_file["INPUTS/Met"]
    output = tmp_path / "angles.h5"
    assert add_angular_features(source, output) == ["Jets", "Lepton"]
    with h5py.File(output, "r") as result:
        assert result["INPUTS/Jets/sin_phi"].shape == (0, 4)
        assert result["INPUTS/Lepton/cos_phi"].shape == (0,)


@pytest.mark.parametrize(
    ("group", "position", "value", "message"),
    [
        ("Jets", (1, 2), np.nan, "event 1, slot 2"),
        ("Jets", (0, 0), 3.5, "event 0, slot 0"),
        ("Lepton", 2, -np.inf, "event 2"),
    ],
)
def test_invalid_phi_is_rejected_without_output(tmp_path, group, position, value, message):
    jet_phi, lepton_phi = JET_PHI.copy(), np.array([0.1, -1.2, 2.9], dtype=np.float32)
    (jet_phi if group == "Jets" else lepton_phi)[position] = value
    source = tmp_path / "extracted.h5"
    write_extracted(source, jet_phi=jet_phi, lepton_phi=lepton_phi)
    output = tmp_path / "angles.h5"

    with pytest.raises(ValueError, match=f"INPUTS/{group}/phi.*{message}"):
        add_angular_features(source, output)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["extracted.h5"]


def test_existing_output_and_source_are_protected(extracted, tmp_path):
    output = tmp_path / "angles.h5"
    output.write_text("keep me")
    with pytest.raises(FileExistsError):
        add_angular_features(extracted, output)
    assert output.read_text() == "keep me"

    add_angular_features(extracted, output, overwrite=True)
    with h5py.File(output, "r") as result:
        assert "sin_phi" in result["INPUTS/Jets"]

    with pytest.raises(ValueError, match="differ from the source"):
        add_angular_features(extracted, extracted, overwrite=True)


def test_already_derived_input_is_rejected(tmp_path):
    source = tmp_path / "angles.h5"
    write_extracted(tmp_path / "extracted.h5")
    add_angular_features(tmp_path / "extracted.h5", source)
    with pytest.raises(ValueError, match="already has cos_phi, sin_phi"):
        add_angular_features(source, tmp_path / "twice.h5")


def test_file_without_phi_is_rejected(tmp_path):
    source = tmp_path / "no-phi.h5"
    with h5py.File(source, "w") as hdf5_file:
        hdf5_file.create_dataset("INPUTS/Met/pt", data=np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="no INPUTS group has a phi feature"):
        add_angular_features(source, tmp_path / "angles.h5")


def test_cli(extracted, tmp_path):
    output = tmp_path / "angles.h5"
    result = subprocess.run(
        [sys.executable, "-m", "spanet_reco.features", str(extracted), str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "output": str(output),
        "events": 3,
        "groups": ["Jets", "Lepton", "Met"],
    }
