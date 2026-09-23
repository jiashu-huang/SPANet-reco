import numpy as np
import pytest
import uproot

from spanet_reco.root_io import read_target_batch


@pytest.fixture
def source_arrays():
    # Deliberately use different columns to catch accidental role reordering.
    return {
        "nJets": np.array([7, 8, 4, 8], dtype=np.int32),
        "GenHadBJetIdx": np.array([6, 0, -1, -1], dtype=np.int32),
        "GenHadQ1JetIdx": np.array([2, 1, 1, 1], dtype=np.int32),
        "GenHadQ2JetIdx": np.array([1, 2, 2, 2], dtype=np.int32),
        "GenLepBJetIdx": np.array([0, 7, 3, 7], dtype=np.int32),
    }


def write_tree(path, arrays):
    with uproot.recreate(path) as root_file:
        tree = root_file.mktree("Events", {name: values.dtype for name, values in arrays.items()})
        tree.extend(arrays)


@pytest.fixture
def root_path(tmp_path, source_arrays):
    path = tmp_path / "events.root"
    write_tree(path, source_arrays)
    return path


def test_root_roundtrip_preserves_roles_and_counts(root_path):
    batch = read_target_batch(root_path)
    np.testing.assert_array_equal(
        batch.targets,
        np.array([[6, 2, 1, 0], [0, 1, 2, 7], [-1, 1, 2, 3], [-1, 1, 2, 7]], dtype=np.int32),
        strict=True,
    )
    # Saved slot 7 is a candidate: compaction can retain it.
    np.testing.assert_array_equal(batch.fully_matched, [True, True, False, False])
    assert batch.counts() == {
        "events": 4,
        "unmatched": 2,
        "outside_retained": 0,
        "excluded": 2,
        "fully_matched": 2,
    }


def test_entry_ranges_are_half_open(root_path):
    batch = read_target_batch(root_path, entry_start=1, entry_stop=3)
    np.testing.assert_array_equal(batch.targets, [[0, 1, 2, 7], [-1, 1, 2, 3]])
    assert batch.counts()["events"] == 2
    assert batch.counts()["excluded"] == 1


def test_empty_range(root_path):
    batch = read_target_batch(root_path, entry_start=4)
    assert batch.targets.shape == (0, 4)
    assert batch.fully_matched.shape == (0,)
    assert all(value == 0 for value in batch.counts().values())


@pytest.mark.parametrize("start,stop", [(-1, None), (0, -1), (3, 2), (0, 5), (5, None)])
def test_invalid_ranges_are_not_clipped(root_path, start, stop):
    with pytest.raises(ValueError):
        read_target_batch(root_path, entry_start=start, entry_stop=stop)


@pytest.mark.parametrize("start,stop", [(0.0, None), (False, None), (0, 2.0), (0, True)])
def test_entry_bounds_require_integers(root_path, start, stop):
    with pytest.raises(TypeError):
        read_target_batch(root_path, entry_start=start, entry_stop=stop)


def test_missing_branch_is_reported(tmp_path, source_arrays):
    source_arrays.pop("GenLepBJetIdx")
    path = tmp_path / "missing_branch.root"
    write_tree(path, source_arrays)
    with pytest.raises(ValueError, match="missing required branches: GenLepBJetIdx"):
        read_target_batch(path)


def test_missing_tree_is_reported(tmp_path):
    path = tmp_path / "missing_tree.root"
    with uproot.recreate(path):
        pass
    with pytest.raises(ValueError, match="missing Events TTree"):
        read_target_batch(path)


def test_wrong_events_object_is_reported(tmp_path):
    path = tmp_path / "wrong_object.root"
    with uproot.recreate(path) as root_file:
        root_file["Events"] = "not a tree"
    with pytest.raises(ValueError, match="Events must be a TTree"):
        read_target_batch(path)


def test_bad_labels_report_file_and_range(tmp_path, source_arrays):
    source_arrays["GenLepBJetIdx"][2] = 1
    path = tmp_path / "invalid.root"
    write_tree(path, source_arrays)
    with pytest.raises(ValueError, match=r"invalid.root, Events entries \[2, 3\).*event 0"):
        read_target_batch(path, entry_start=2, entry_stop=3)


@pytest.mark.parametrize("dtype", [np.float64, np.bool_])
def test_noninteger_target_branch_is_not_coerced(tmp_path, source_arrays, dtype):
    source_arrays["GenHadBJetIdx"] = source_arrays["GenHadBJetIdx"].astype(dtype)
    path = tmp_path / "noninteger_targets.root"
    write_tree(path, source_arrays)
    with pytest.raises(TypeError, match="integer dtype"):
        read_target_batch(path)
