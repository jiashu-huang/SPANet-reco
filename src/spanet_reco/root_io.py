"""Read upstream truth assignments without altering their saved jet indices."""

import argparse
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import uproot

from spanet_reco.targets import N_INPUT_JETS, TARGET_BRANCHES, fully_matched_mask

TARGET_INPUT_BRANCHES = ("nJets", *TARGET_BRANCHES)


@dataclass(frozen=True)
class TargetBatch:
    """Validated assignments and their eligibility for the first baseline."""

    targets: np.ndarray
    n_jets: np.ndarray
    fully_matched: np.ndarray

    def counts(self) -> dict[str, int]:
        """Count exclusions independently, allowing the reasons to overlap."""
        return {
            "events": len(self.n_jets),
            "unmatched": int(np.count_nonzero(np.any(self.targets == -1, axis=1))),
            "outside_retained": int(np.count_nonzero(np.any(self.targets >= N_INPUT_JETS, axis=1))),
            "excluded": int(np.count_nonzero(~self.fully_matched)),
            "fully_matched": int(np.count_nonzero(self.fully_matched)),
        }


def read_target_batch(
    path: str | Path,
    *,
    entry_start: int = 0,
    entry_stop: int | None = None,
) -> TargetBatch:
    """Read and validate [entry_start, entry_stop) from an Events TTree.

    Only the four target branches and nJets are read. None for entry_stop
    means the end of the tree. Bounds must lie inside the tree; invalid
    ranges are rejected rather than silently clipped. Validation errors
    include the file and entry range; event indices in them are batch-relative.
    """
    for name, bound in (("entry_start", entry_start), ("entry_stop", entry_stop)):
        if bound is None and name == "entry_stop":
            continue
        if isinstance(bound, (bool, np.bool_)) or not isinstance(bound, Integral):
            raise TypeError(f"{name} must be an integer")
        if bound < 0:
            raise ValueError(f"{name} must be nonnegative")

    with uproot.open(path) as root_file:
        if "Events" not in root_file:
            raise ValueError(f"{path}: missing Events TTree")
        tree = root_file["Events"]
        if not isinstance(tree, uproot.behaviors.TTree.TTree):
            raise ValueError(f"{path}: Events must be a TTree")
        missing = sorted(set(TARGET_INPUT_BRANCHES) - set(tree.keys()))
        if missing:
            raise ValueError(f"{path}: missing required branches: {', '.join(missing)}")

        stop = tree.num_entries if entry_stop is None else int(entry_stop)
        start = int(entry_start)
        if not 0 <= start <= stop <= tree.num_entries:
            raise ValueError(
                f"{path}: invalid entry range [{start}, {stop}) for {tree.num_entries} events"
            )
        arrays = tree.arrays(
            list(TARGET_INPUT_BRANCHES), entry_start=start, entry_stop=stop, library="np"
        )

    try:
        # Check each branch before stacking can promote a boolean or float column.
        for name in TARGET_BRANCHES:
            if not np.issubdtype(arrays[name].dtype, np.integer):
                raise TypeError(f"{name} must have an integer dtype; got {arrays[name].dtype}")
        targets = np.column_stack([arrays[name] for name in TARGET_BRANCHES])
        n_jets = arrays["nJets"]
        mask = fully_matched_mask(targets, n_jets)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}, Events entries [{start}, {stop}): {error}") from error
    return TargetBatch(targets=targets, n_jets=n_jets, fully_matched=mask)


def main() -> None:
    """Print target-selection counts for one ROOT file or entry range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--entry-stop", type=int)
    args = parser.parse_args()
    batch = read_target_batch(args.path, entry_start=args.entry_start, entry_stop=args.entry_stop)
    print(json.dumps(batch.counts(), indent=2))


if __name__ == "__main__":
    main()
