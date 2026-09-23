"""Tests of the mass chi-square training entry point.

The model checks need PyTorch and SPANet, which are not in this environment.
Run them with SPANET_PYTHON=/path/to/spanet/environment/bin/python pytest.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spanet_reco.train import parse

REPO = Path(__file__).resolve().parents[1]


def test_mass_options_are_separated_from_spanet_options():
    settings, seed, rest = parse(
        ["-ef", "event.yaml", "--alpha", "0.5", "-b", "256", "--top-mass", "173", "--seed", "7"]
    )
    assert settings == {"alpha": 0.5, "top_mass": 173.0} and seed == 7
    assert rest == ["-ef", "event.yaml", "-b", "256"]


def test_defaults_leave_settings_empty():
    assert parse(["-tf", "train.h5", "-r", "3"]) == ({}, None, ["-tf", "train.h5", "-r", "3"])


def test_abbreviations_are_not_taken_as_mass_options():
    settings, seed, rest = parse(["--alph", "0.5"])
    assert settings == {} and seed is None and rest == ["--alph", "0.5"]


def test_help_lists_mass_options(capsys):
    parse(["--help"])
    assert "--alpha" in capsys.readouterr().out


def test_help_without_spanet():
    result = subprocess.run(
        [sys.executable, "-c", "from spanet_reco.train import parse; parse(['-h'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--w-width" in result.stdout


@pytest.mark.skipif(
    not os.environ.get("SPANET_PYTHON"), reason="set SPANET_PYTHON for model checks"
)
def test_model_with_spanet():
    result = subprocess.run(
        [os.environ["SPANET_PYTHON"], str(REPO / "tests" / "spanet_model_checks.py")],
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "checks passed" in result.stdout
