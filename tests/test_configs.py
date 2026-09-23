"""Keep the SPANet event and options files consistent with the extraction mappings.

Every configs/event-<name>.yaml is checked against configs/extract-<name>.yaml
when that file exists, and against configs/extract-vcb.yaml otherwise. The
default event file must read every extracted feature; variants may read a subset.
"""

import json
from pathlib import Path

import pytest
import yaml

from spanet_reco.features import COS_FEATURE, PHI_FEATURE, SIN_FEATURE

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_EVENT = CONFIGS / "event-vcb.yaml"
EVENT_FILES = sorted(CONFIGS.glob("event-*.yaml"))
# Jet features that spanet_reco.model needs for the mass chi-square; it checks them
# when the model is built, whatever alpha is (see KINEMATIC_FEATURES there).
MASS_FEATURES = {"mass", "pt", "eta", SIN_FEATURE, COS_FEATURE}


def load(path):
    return yaml.safe_load(path.read_text())


def mapping_for(event_path):
    name = event_path.stem.removeprefix("event-")
    paired = CONFIGS / f"extract-{name}.yaml"
    return load(paired if paired.exists() else CONFIGS / "extract-vcb.yaml")


def extracted_features(mapping):
    """Model features in a built dataset, by input type: the mapping's features, the
    tag indicator, and sin_phi and cos_phi for every input with phi."""
    sequential = {name: list(spec["features"]) for name, spec in mapping["sequential"].items()}
    selection = mapping.get("tag_selection")
    if selection and selection.get("indicator"):
        sequential[selection["input"]].append(selection["indicator"])
    features = {
        "SEQUENTIAL": sequential,
        "GLOBAL": {name: list(spec) for name, spec in mapping.get("global", {}).items()},
    }
    for groups in features.values():
        for names in groups.values():
            if PHI_FEATURE in names:
                names.extend([SIN_FEATURE, COS_FEATURE])
    return features


@pytest.fixture(params=EVENT_FILES, ids=lambda path: path.name)
def event_path(request):
    return request.param


def test_default_event_file_exists():
    assert DEFAULT_EVENT in EVENT_FILES


def test_event_file_reads_only_extracted_features(event_path):
    event, available = load(event_path), extracted_features(mapping_for(event_path))
    # This SPANet version indexes assignment sources assuming sequential inputs come first.
    assert list(event["INPUTS"]) == ["SEQUENTIAL", "GLOBAL"]
    for kind, groups in event["INPUTS"].items():
        for group, features in groups.items():
            assert group in available[kind], f"{kind}/{group} is not extracted"
            missing = set(features) - set(available[kind][group])
            assert not missing, f"{kind}/{group} reads features not extracted: {sorted(missing)}"


def test_default_event_file_reads_every_extracted_feature():
    event, available = load(DEFAULT_EVENT), extracted_features(mapping_for(DEFAULT_EVENT))
    for kind, groups in available.items():
        assert set(event["INPUTS"][kind]) == set(groups), kind
        for group, features in groups.items():
            expected = set(features) - {PHI_FEATURE}
            assert set(event["INPUTS"][kind][group]) == expected, group


def test_phi_enters_as_sin_and_cos(event_path):
    for groups in load(event_path)["INPUTS"].values():
        for group, features in groups.items():
            assert PHI_FEATURE not in features, group
            angles = {SIN_FEATURE, COS_FEATURE} & set(features)
            assert angles in (set(), {SIN_FEATURE, COS_FEATURE}), group


def test_jets_keep_the_mass_chi2_inputs(event_path):
    jets = load(event_path)["INPUTS"]["SEQUENTIAL"]["Jets"]
    assert MASS_FEATURES <= set(jets)


def test_event_targets_match_extraction_targets(event_path):
    targets = {
        particle: [(role, spec["input"]) for role, spec in roles.items()]
        for particle, roles in mapping_for(event_path)["targets"].items()
    }
    declared = {
        particle: [next(iter(daughter.items())) for daughter in daughters]
        for particle, daughters in load(event_path)["EVENT"].items()
    }
    # Daughter order defines the axes of each assignment tensor.
    assert declared == targets


def test_w_daughters_are_an_unordered_pair(event_path):
    assert load(event_path)["PERMUTATIONS"] == {"had_top": [["q1", "q2"]]}


def test_options_leave_dataset_paths_to_the_command_line():
    options = json.loads((CONFIGS / "options-vcb.json").read_text())
    # spanet.train applies the JSON after -ef/-tf/-vf, so paths here would override them.
    assert not {"event_info_file", "training_file", "validation_file", "testing_file"} & set(
        options
    )
    # The first baseline trains on fully matched events only.
    assert options["partial_events"] is False
