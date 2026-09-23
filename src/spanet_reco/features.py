"""Add sin(phi) and cos(phi) inputs to an extracted SPANet HDF5 file.

Raw phi jumps from +pi to -pi although both are the same direction. The derived
features place each direction on the unit circle instead. The source file is
copied unchanged, including raw phi, and the new datasets are added beside it.
"""

import argparse
import json
import math
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

PHI_FEATURE = "phi"
SIN_FEATURE = "sin_phi"
COS_FEATURE = "cos_phi"
# Stored phi lies in [-pi, pi]; float32 rounding can exceed pi slightly.
PHI_TOLERANCE = 1e-5
PROVENANCE_NAME = "angular_features"


def angle_groups(hdf5_file: h5py.File) -> list[str]:
    """Return the INPUTS groups that have a phi feature, in file order."""
    if "INPUTS" not in hdf5_file:
        raise ValueError("missing INPUTS group")
    return [
        name
        for name, group in hdf5_file["INPUTS"].items()
        if isinstance(group, h5py.Group) and PHI_FEATURE in group
    ]


def _check_phi(values: np.ndarray, real: np.ndarray, group: str, start: int) -> None:
    """Reject nonfinite or out-of-range phi on real objects; padding is ignored."""
    invalid = real & ~(np.isfinite(values) & (np.abs(values) <= math.pi + PHI_TOLERANCE))
    if np.any(invalid):
        position = np.unravel_index(np.argmax(invalid), invalid.shape)
        where = f"event {start + position[0]}"
        if len(position) > 1:
            where += f", slot {position[1]}"
        raise ValueError(
            f"INPUTS/{group}/{PHI_FEATURE}: invalid phi {values[position]} at {where}; "
            f"expected a finite value within [-pi, pi]"
        )


def _write_angles(source: h5py.Group, target: h5py.Group, group: str, chunk_size: int) -> None:
    phi = source[PHI_FEATURE]
    mask = source["MASK"] if "MASK" in source else None
    outputs = {
        name: target.create_dataset(
            name,
            shape=phi.shape,
            dtype=phi.dtype,
            chunks=phi.chunks,
            maxshape=phi.maxshape if phi.chunks else None,
            compression=phi.compression,
            compression_opts=phi.compression_opts,
            shuffle=phi.shuffle,
        )
        for name in (SIN_FEATURE, COS_FEATURE)
    }
    for start in range(0, phi.shape[0], chunk_size):
        stop = min(start + chunk_size, phi.shape[0])
        values = phi[start:stop]
        # Sequential inputs mark real objects with MASK; global inputs have one per event.
        real = mask[start:stop] if mask is not None else np.ones(values.shape, dtype=bool)
        if real.shape != values.shape:
            raise ValueError(f"INPUTS/{group}: MASK shape {real.shape} differs from phi")
        _check_phi(values, real, group, start)
        # Padded slots stay zero for every feature, as in the extractor output.
        angles = np.where(real, values, 0)
        outputs[SIN_FEATURE][start:stop] = np.where(real, np.sin(angles), 0).astype(phi.dtype)
        outputs[COS_FEATURE][start:stop] = np.where(real, np.cos(angles), 0).astype(phi.dtype)


def add_angular_features(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 100_000,
) -> list[str]:
    """Copy an extracted file to output, adding sin_phi and cos_phi to every input with phi.

    Every dataset and attribute of the source is copied unchanged. For each
    INPUTS/<group>/phi, INPUTS/<group>/sin_phi and INPUTS/<group>/cos_phi are
    written with the dtype, shape, and storage of phi, and zero where MASK is
    false. PROVENANCE/angular_features records the settings.

    The output appears only after success. An existing output is refused unless
    overwrite is true; the source itself can never be the output.

    Returns
    -------
    list[str]
        Names of the INPUTS groups that received the new features.
    """
    source, output = Path(source), Path(output)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if output.exists():
        if output.resolve() == source.resolve():
            raise ValueError("output must differ from the source file")
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    os.close(handle)
    try:
        with h5py.File(source, "r") as source_file, h5py.File(temporary, "w") as output_file:
            groups = angle_groups(source_file)
            if not groups:
                raise ValueError(f"{source}: no INPUTS group has a {PHI_FEATURE} feature")
            for group in groups:
                existing = {SIN_FEATURE, COS_FEATURE} & set(source_file["INPUTS"][group])
                if existing:
                    raise ValueError(f"INPUTS/{group} already has {', '.join(sorted(existing))}")
            if f"PROVENANCE/{PROVENANCE_NAME}" in source_file:
                raise ValueError(f"PROVENANCE/{PROVENANCE_NAME} already exists")

            output_file.attrs.update(source_file.attrs)
            for name in source_file:
                source_file.copy(source_file[name], output_file, name=name)
            for group in groups:
                _write_angles(
                    source_file["INPUTS"][group], output_file["INPUTS"][group], group, chunk_size
                )

            record = {
                "tool": "spanet_reco.features",
                "spanet_reco_version": version("spanet-reco"),
                "source": str(source.resolve()),
                "groups": groups,
                "features": {SIN_FEATURE: "sin(phi)", COS_FEATURE: "cos(phi)"},
                "padding": "0 where MASK is false",
                "phi_range": f"abs(phi) <= pi + {PHI_TOLERANCE}",
            }
            output_file.require_group("PROVENANCE").create_dataset(
                PROVENANCE_NAME, data=json.dumps(record), dtype=h5py.string_dtype()
            )
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return groups


def main() -> None:
    """Write a copy of an extracted file with sin_phi and cos_phi inputs added."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HDF5 file written by nano-spanet-extract")
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="events per chunk")
    args = parser.parse_args()
    groups = add_angular_features(
        args.source, args.output, overwrite=args.overwrite, chunk_size=args.chunk_size
    )
    with h5py.File(args.output, "r") as output_file:
        events = output_file["INPUTS"][groups[0]][PHI_FEATURE].shape[0]
    print(json.dumps({"output": str(args.output), "events": events, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
