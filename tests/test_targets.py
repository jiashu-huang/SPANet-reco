import numpy as np
import pytest

from spanet_reco.targets import fully_matched_mask


def test_selection_and_input_preservation():
    targets = np.array(
        [
            [0, 1, 2, 3],  # Four real jets, all matched.
            [0, 1, 2, 6],  # Slot 6 is valid.
            [0, 1, 2, 7],  # Saved slots 7 to 9 can be retained after compaction.
            [0, 1, -1, 2],  # One unmatched parton.
            [-1, -1, -1, -1],  # Repeated missing markers are allowed.
            [0, 1, 2, 9],  # Last saved slot in an event with 12 jets.
            [-1, 1, 2, 7],  # Unmatched, even though slot 7 is a candidate.
            [6, 2, 1, 0],  # Extra upstream jets do not force exclusion.
        ],
        dtype=np.int64,
    )
    n_jets = np.array([4, 7, 8, 3, 0, 12, 8, 12], dtype=np.int64)
    expected = np.array([True, True, True, False, False, True, False, True])

    original_targets = targets.copy()
    original_counts = n_jets.copy()

    result = fully_matched_mask(targets, n_jets)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected, strict=True)
    np.testing.assert_array_equal(targets, original_targets, strict=True)
    np.testing.assert_array_equal(n_jets, original_counts, strict=True)


@pytest.mark.parametrize(
    ("row", "count"),
    [
        ([0, 1, 1, 3], 4),  # Duplicate real-jet assignment.
        ([-1, 1, 1, 3], 4),  # Invalid even though already incomplete.
        ([0, 1, 8, 8], 9),  # Duplicates in a high saved slot.
        ([0, 1, 2, -2], 4),  # Unsupported negative index.
        ([0, 1, 2, 10], 12),  # Outside the ten saved slots.
        ([0, 1, 2, 4], 4),  # Assignment to an empty upstream slot.
        ([0, 1, 2, 8], 8),  # Empty high slot.
        ([-1, -1, -1, -1], -1),  # Invalid jet count.
    ],
)
def test_invalid_data_raise_value_error(row, count):
    targets = np.array([row], dtype=np.int64)
    n_jets = np.array([count], dtype=np.int64)

    with pytest.raises(ValueError):
        fully_matched_mask(targets, n_jets)


@pytest.mark.parametrize(
    ("target_dtype", "count_dtype"),
    [
        (np.float64, np.int64),
        (np.int64, np.float64),
        (np.bool_, np.int64),
        (np.int64, np.bool_),
    ],
)
def test_non_integer_dtypes_raise_type_error(target_dtype, count_dtype):
    targets = np.array([[0, 1, 2, 3]], dtype=target_dtype)
    n_jets = np.array([4], dtype=count_dtype)

    with pytest.raises(TypeError):
        fully_matched_mask(targets, n_jets)


@pytest.mark.parametrize(
    ("targets", "n_jets"),
    [
        pytest.param(np.array(0), np.array([4]), id="scalar-targets"),
        pytest.param(np.array([0, 1, 2, 3]), np.array([4]), id="flat-targets"),
        pytest.param(np.array([[0, 1, 2]]), np.array([4]), id="missing-role"),
        pytest.param(np.array([[[0, 1, 2, 3]]]), np.array([4]), id="extra-target-axis"),
        pytest.param(np.array([[0, 1, 2, 3]]), np.array([4, 5]), id="different-event-counts"),
        pytest.param(np.array([[0, 1, 2, 3]]), np.array(4), id="scalar-count"),
        pytest.param(np.array([[0, 1, 2, 3]]), np.array([[4]]), id="column-counts"),
    ],
)
def test_incompatible_shapes_raise_value_error(targets, n_jets):
    with pytest.raises(ValueError, match="shape"):
        fully_matched_mask(targets, n_jets)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_read_only_integer_arrays(dtype):
    targets = np.array([[6, 2, 1, 0], [9, 2, 1, 0]], dtype=dtype)
    n_jets = np.array([7, 12], dtype=dtype)
    targets.setflags(write=False)
    n_jets.setflags(write=False)

    np.testing.assert_array_equal(
        fully_matched_mask(targets, n_jets),
        np.array([True, True]),
        strict=True,
    )


@pytest.mark.parametrize(
    ("row", "count", "location"),
    [
        ([0, 1, 2, 10], 12, "event 1, column 3"),
        ([0, 1, 2, 4], 4, "event 1, column 3"),
        ([-1, 1, 1, 3], 4, "event 1"),
        ([-1, -1, -1, -1], -1, "event 1"),
    ],
)
def test_errors_identify_the_invalid_event(row, count, location):
    targets = np.array([[0, 1, 2, 3], row], dtype=np.int64)
    n_jets = np.array([4, count], dtype=np.int64)

    with pytest.raises(ValueError, match=location):
        fully_matched_mask(targets, n_jets)


def test_empty_batch():
    result = fully_matched_mask(
        np.empty((0, 4), dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )

    np.testing.assert_array_equal(
        result,
        np.empty(0, dtype=bool),
        strict=True,
    )
