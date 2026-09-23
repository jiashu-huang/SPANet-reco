"""Keep the SPANet event and options files consistent with the extraction mapping."""

import json
from pathlib import Path

import pytest
import yaml

from spanet_reco.features import COS_FEATURE, PHI_FEATURE, SIN_FEATURE

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def mapping():
    return yaml.safe_load((CONFIGS / "extract-vcb.yaml").read_text())


@pytest.fixture(scope="module")
def event():
    return yaml.safe_load((CONFIGS / "event-vcb.yaml").read_text())


def model_inputs(mapping):
    """Features in the extracted file after the angular step, by input type."""
    sequential = {name: list(spec["features"]) for name, spec in mapping["sequential"].items()}
    indicator = mapping["tag_selection"]["indicator"]
    if indicator is not None:
        sequential[mapping["tag_selection"]["input"]].append(indicator)
    inputs = {
        "SEQUENTIAL": sequential,
        "GLOBAL": {k: list(v) for k, v in mapping["global"].items()},
    }
    # Raw phi stays in the file for traceability but is replaced by its sin and cos.
    for groups in inputs.values():
        for features in groups.values():
            if PHI_FEATURE in features:
                features.remove(PHI_FEATURE)
                features.extend([SIN_FEATURE, COS_FEATURE])
    return inputs


def test_event_inputs_match_extracted_features(mapping, event):
    expected = model_inputs(mapping)
    # This SPANet version indexes assignment sources assuming sequential inputs come first.
    assert list(event["INPUTS"]) == ["SEQUENTIAL", "GLOBAL"]
    for kind, groups in expected.items():
        assert set(event["INPUTS"][kind]) == set(groups), kind
        for group, features in groups.items():
            assert sorted(event["INPUTS"][kind][group]) == sorted(features), group


def test_every_phi_is_replaced_by_sin_and_cos(event):
    for groups in event["INPUTS"].values():
        for group, features in groups.items():
            assert PHI_FEATURE not in features, group
            assert {SIN_FEATURE, COS_FEATURE} <= set(features), group


def test_event_targets_match_extraction_targets(mapping, event):
    targets = {
        particle: [(role, spec["input"]) for role, spec in roles.items()]
        for particle, roles in mapping["targets"].items()
    }
    declared = {
        particle: [next(iter(daughter.items())) for daughter in daughters]
        for particle, daughters in event["EVENT"].items()
    }
    # Daughter order defines the axes of each assignment tensor.
    assert declared == targets


def test_w_daughters_are_an_unordered_pair(event):
    assert event["PERMUTATIONS"] == {"had_top": [["q1", "q2"]]}


def test_options_leave_dataset_paths_to_the_command_line():
    options = json.loads((CONFIGS / "options-vcb.json").read_text())
    # spanet.train applies the JSON after -ef/-tf/-vf, so paths here would override them.
    assert not {"event_info_file", "training_file", "validation_file", "testing_file"} & set(
        options
    )
    # The first baseline trains on fully matched events only.
    assert options["partial_events"] is False
