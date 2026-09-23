"""Validation and selection of upstream jet-assignment targets."""

import numpy as np

TARGET_BRANCHES = (
    "GenHadBJetIdx",
    "GenHadQ1JetIdx",
    "GenHadQ2JetIdx",
    "GenLepBJetIdx",
)
N_TARGETS = len(TARGET_BRANCHES)
N_SAVED_JETS = 10
# Every saved slot may reach the model: the extractor removes jets failing its cuts and
# compacts the seven leading passing jets, so saved slots 7 to 9 can be retained. This
# pre-cut check cannot know which are; final eligibility comes from extracted target masks.
N_INPUT_JETS = 10


def fully_matched_mask(targets: np.ndarray, n_jets: np.ndarray) -> np.ndarray:
    """Validate upstream assignments and identify fully matched events.

    Parameters
    ----------
    targets:
        Integer array with shape (n_events, 4). Columns are ordered as
        had_top/b, had_top/q1, had_top/q2, lep_top/b.
    n_jets:
        Nonnegative integer array with shape (n_events,). Counts describe
        the upstream selected jets before truncation to ten saved slots.

    Returns
    -------
    np.ndarray
        Boolean array with shape (n_events,), identifying events whose
        four assignments refer to distinct real jets in the first
        N_INPUT_JETS saved slots (currently all ten).

    Raises
    ------
    TypeError
        If either input is not a NumPy array with an integer dtype.
    ValueError
        If shapes, jet counts, or upstream assignments are invalid.

    Neither input array is modified.
    """
    for name, values in (("targets", targets), ("n_jets", n_jets)):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must have an integer dtype; got {values.dtype}")

    if targets.ndim != 2 or targets.shape[1] != N_TARGETS:
        raise ValueError(f"targets must have shape (n_events, {N_TARGETS}); got {targets.shape}")
    if n_jets.shape != (targets.shape[0],):
        raise ValueError(f"n_jets must have shape ({targets.shape[0]},); got {n_jets.shape}")

    negative_counts = n_jets < 0
    if np.any(negative_counts):
        event = np.argmax(negative_counts)
        raise ValueError(f"Negative jet count at event {event}: {n_jets[event]}")

    invalid_indices = (targets < -1) | (targets >= N_SAVED_JETS)
    if np.any(invalid_indices):
        event, column = np.unravel_index(np.argmax(invalid_indices), invalid_indices.shape)
        raise ValueError(
            f"Invalid saved-jet index at event {event}, column {column}: "
            f"{targets[event, column]}; expected -1 or 0 through {N_SAVED_JETS - 1}"
        )

    assigned = targets >= 0
    empty_slots = assigned & (targets >= n_jets[:, None])
    if np.any(empty_slots):
        event, column = np.unravel_index(np.argmax(empty_slots), empty_slots.shape)
        raise ValueError(
            f"Target at event {event}, column {column} refers to empty jet slot "
            f"{targets[event, column]} (n_jets={n_jets[event]})"
        )

    # Sorting a copy exposes duplicates while preserving the physical role order.
    ordered = np.sort(targets, axis=1)
    duplicates = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
    if np.any(duplicates):
        event, pair = np.unravel_index(np.argmax(duplicates), duplicates.shape)
        raise ValueError(
            f"Duplicate assignment at event {event}: jet {ordered[event, pair + 1]} "
            "is assigned to more than one role"
        )

    # Validate the entire batch before excluding incomplete or truncated events.
    return np.all(assigned & (targets < N_INPUT_JETS), axis=1)
