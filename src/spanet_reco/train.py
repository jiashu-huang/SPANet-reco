"""Train SPANet with the loss alpha * L_SPANet + (1 - alpha) * <chi2>.

Accepts every option of `python -m spanet.train`, plus the mass chi-square
settings below (see spanet_reco.model) and a random seed. The default alpha = 1
is plain SPANet. The settings and seed are saved as mass_chi2.json in the run's
version directory.

Example:
    python -m spanet_reco.train -ef configs/event-vcb.yaml -of configs/options-vcb.json \\
        -tf train.h5 -vf validation.h5 -l outputs -n vcb --alpha 0.5
"""

import argparse
import runpy
import sys

MASS_OPTIONS = {
    "--alpha": ("alpha", "weight of SPANet's loss; <chi2> gets 1 - alpha (default 1)"),
    "--top-mass": ("top_mass", "hadronic top mass in GeV (default 172.5)"),
    "--top-width": ("top_width", "width of the top term in GeV (default 20)"),
    "--w-mass": ("w_mass", "hadronic W mass in GeV (default 80.4)"),
    "--w-width": ("w_width", "width of the W term in GeV (default 15)"),
}


def parse(argv: list[str]) -> tuple[dict[str, float], int | None, list[str]]:
    """Split the mass chi-square settings and seed from the options for spanet.train."""
    parser = argparse.ArgumentParser(
        prog="python -m spanet_reco.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    for flag, (dest, text) in MASS_OPTIONS.items():
        parser.add_argument(flag, dest=dest, type=float, help=text)
    parser.add_argument(
        "--seed",
        type=int,
        help="seed Python, NumPy, and PyTorch before the model is built; "
        "spanet.train sets no seed, and its -r only shuffles the data",
    )
    if {"-h", "--help"} & set(argv):
        parser.print_help()
        print("\nOptions of spanet.train follow.\n")
    known, rest = parser.parse_known_args(argv)
    values = vars(known)
    seed = values.pop("seed")
    return {key: value for key, value in values.items() if value is not None}, seed, rest


def main(argv: list[str] | None = None) -> None:
    settings, seed, spanet_argv = parse(sys.argv[1:] if argv is None else argv)

    # Import here so that --help and argument errors do not need PyTorch.
    import pytorch_lightning as pl
    import spanet

    from spanet_reco.model import MassChi2Model, MassChi2Settings

    MassChi2Model.settings = MassChi2Settings(**settings)
    MassChi2Model.seed = seed
    print(f"Mass chi-square settings: {MassChi2Model.settings}, seed {seed}", flush=True)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # spanet.train builds `JetReconstructionModel(options)` from the name it imports
    # from the spanet package. Replacing that name before running the unmodified
    # program trains MassChi2Model with every spanet.train option available.
    spanet.JetReconstructionModel = MassChi2Model
    sys.argv = ["spanet.train", *spanet_argv]
    runpy.run_module("spanet.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
