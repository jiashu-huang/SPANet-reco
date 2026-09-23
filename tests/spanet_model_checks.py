"""Checks of spanet_reco.model that need PyTorch and SPANet.

Run by tests/test_model.py with the Python named by SPANET_PYTHON, and with
src/ on PYTHONPATH. Every check raises AssertionError on failure.
"""

import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from spanet import JetReconstructionModel, Options

from spanet_reco.model import (
    MassChi2Model,
    MassChi2Settings,
    hadronic_chi2,
    jet_four_vectors,
)

REPO = Path(__file__).resolve().parents[1]
EVENT_FILE = REPO / "configs" / "event-vcb.yaml"
OPTIONS_FILE = REPO / "configs" / "options-vcb.json"
FEATURES = {"mass": (0, True), "pt": (1, True), "eta": (2, False)}
FEATURES |= {"sin_phi": (3, False), "cos_phi": (4, False)}


def encoded_jets(rng, events, jets):
    """Random jets, as stored for SPANet: log(x + 1) for mass and pt."""
    pt = rng.uniform(25, 300, (events, jets))
    eta = rng.uniform(-2.4, 2.4, (events, jets))
    phi = rng.uniform(-math.pi, math.pi, (events, jets))
    mass = rng.uniform(2, 40, (events, jets))
    data = np.stack([np.log1p(mass), np.log1p(pt), eta, np.sin(phi), np.cos(phi)], -1)
    return data, (pt, eta, phi, mass)


def reference_mass(pt, eta, phi, mass, jets):
    """Invariant mass of a set of jets with float64 NumPy."""
    px, py = pt[jets] * np.cos(phi[jets]), pt[jets] * np.sin(phi[jets])
    pz = pt[jets] * np.sinh(eta[jets])
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass[jets] ** 2)
    return math.sqrt(energy.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2)


def check_masses_and_chi2():
    rng = np.random.default_rng(1)
    data, (pt, eta, phi, mass) = encoded_jets(rng, 4, 7)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[1, 5:] = False  # Event 1 has five real jets.
    data[1, 5:] = 0
    settings = MassChi2Settings(top_mass=172.5, top_width=20, w_mass=80.4, w_width=15)
    p4 = jet_four_vectors(torch.tensor(data, dtype=torch.float32), FEATURES)
    top, w, valid = hadronic_chi2(p4, mask, settings)

    for event, b, q1, q2 in [(0, 0, 1, 2), (0, 6, 3, 1), (2, 2, 4, 5), (3, 1, 0, 6)]:
        m_top = reference_mass(pt[event], eta[event], phi[event], mass[event], [b, q1, q2])
        m_w = reference_mass(pt[event], eta[event], phi[event], mass[event], [q1, q2])
        expected_top = ((m_top - 172.5) / 20) ** 2
        expected_w = ((m_w - 80.4) / 15) ** 2
        assert math.isclose(top[event, b, q1, q2].item(), expected_top, rel_tol=1e-3, abs_tol=1e-3)
        assert math.isclose(w[event, b, q1, q2].item(), expected_w, rel_tol=1e-3, abs_tol=1e-3)

    # The W pair is unordered, and so is every term.
    assert torch.equal(top, top.transpose(2, 3)) and torch.equal(w, w.transpose(2, 3))
    # Valid triplets: distinct real jets only.
    assert valid[0].sum().item() == 7 * 6 * 5
    assert valid[1].sum().item() == 5 * 4 * 3
    assert not valid[0, 1, 1, 2] and not valid[1, 0, 1, 5]
    assert torch.isfinite(top).all() and torch.isfinite(w).all()


def write_dataset(path, events=64, seed=2):
    """A small fully matched dataset with exactly the inputs of the event file."""
    event = yaml.safe_load(EVENT_FILE.read_text())
    rng = np.random.default_rng(seed)
    data, _ = encoded_jets(rng, events, 7)
    n_jets = rng.integers(4, 8, events)
    mask = np.arange(7)[None, :] < n_jets[:, None]
    raw = {  # Stored values: SPANet applies log(x + 1) itself when it loads the file.
        "mass": np.expm1(data[..., 0]),
        "pt": np.expm1(data[..., 1]),
        "eta": data[..., 2],
        "sin_phi": data[..., 3],
        "cos_phi": data[..., 4],
    }
    with h5py.File(path, "w") as output:
        output["INPUTS/Jets/MASK"] = mask
        for name in event["INPUTS"]["SEQUENTIAL"]["Jets"]:
            values = raw.get(name, rng.uniform(0, 1, (events, 7)))
            output[f"INPUTS/Jets/{name}"] = np.where(mask, values, 0).astype(np.float32)
        for group, features in event["INPUTS"]["GLOBAL"].items():
            for name in features:
                values = (
                    rng.uniform(0.1, 1, events) if name != "charge" else rng.choice([-1, 1], events)
                )
                output[f"INPUTS/{group}/{name}"] = values.astype(np.float32)
        slots = np.array([rng.permutation(n)[:4] for n in n_jets])
        for particle, daughters in event["EVENT"].items():
            for daughter in daughters:
                (role,) = daughter
                column = {"b": 0, "q1": 1, "q2": 2}[role] if particle == "had_top" else 3
                output[f"TARGETS/{particle}/{role}"] = slots[:, column].astype(np.int64)
            output[f"TARGETS/{particle}/MASK"] = np.ones(events, dtype=bool)


def make_model(path):
    options = Options(str(EVENT_FILE), str(path), str(path))
    options.update_options(json.loads(OPTIONS_FILE.read_text()))
    options.batch_size, options.num_dataloader_workers = 16, 0
    torch.manual_seed(3)
    model = MassChi2Model(options)
    model.eval()  # No dropout, so repeated evaluations agree.
    return model


def check_loss(path):
    model = make_model(path)
    batch = next(iter(model.train_dataloader()))
    assert batch.sources[model.jet_source_index].data.shape[1:] == (7, 8)

    MassChi2Model.settings = MassChi2Settings(alpha=1.0)
    plain = JetReconstructionModel.training_step(model, batch, 0)
    ours = model.training_step(batch, 0)
    assert torch.equal(plain, ours), (plain, ours)

    MassChi2Model.settings = MassChi2Settings(alpha=0.5)
    outputs = model.forward(batch.sources)
    top, w = model.expected_chi2(outputs, batch)
    expected = 0.5 * model.spanet_loss(outputs, batch) + 0.5 * (top + w).mean()
    loss = model.training_step(batch, 0)
    assert torch.allclose(loss, expected), (loss, expected)
    assert (top > 0).all() and (w > 0).all()

    # <chi2> equals the probability-weighted sum over valid triplets.
    jets = batch.sources[model.jet_source_index]
    p4 = jet_four_vectors(jets.data, model.kinematic_features)
    chi2_top, _, valid = hadronic_chi2(p4, jets.mask, model.settings)
    probability = outputs.assignments[model.had_top_index].exp()
    assert torch.allclose(probability.flatten(1).sum(1), torch.ones(len(top)), atol=1e-5)
    assert torch.equal(probability[~valid], torch.zeros_like(probability[~valid]))
    assert torch.allclose(top, (probability * chi2_top * valid).flatten(1).sum(1))

    # Gradients reach the network through the assignment probabilities alone.
    MassChi2Model.settings = MassChi2Settings(alpha=0.0)
    model.zero_grad()
    model.training_step(batch, 0).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def main():
    warnings.filterwarnings("ignore")
    check_masses_and_chi2()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.h5"
        write_dataset(path)
        check_loss(path)
    print("spanet_reco.model checks passed")


if __name__ == "__main__":
    sys.exit(main())
