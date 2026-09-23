"""SPANet model whose training loss adds a hadronic top and W mass chi-square.

The loss is alpha * L_SPANet + (1 - alpha) * <chi2>, where

    chi2(b, q1, q2) = ((m(b q1 q2) - m_top) / sigma_top)^2 + ((m(q1 q2) - m_W) / sigma_W)^2

is evaluated for every jet triplet, and <chi2> is its average under the network's
had_top assignment probabilities P(b, q1, q2), then over the events of a batch.
alpha = 1 reproduces SPANet's loss exactly. The network itself is unchanged, so
checkpoints load in SPANet's own test and predict tools.

This module needs PyTorch and SPANet; run it in the SPANet environment.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from spanet import JetReconstructionModel
from spanet.dataset.types import Batch, Outputs
from torch import Tensor

HAD_TOP = "had_top"
# Daughter order of had_top in the event file, which is also the axis order of its
# assignment tensor.
HAD_TOP_DAUGHTERS = ("b", "q1", "q2")
JET_INPUT = "Jets"
KINEMATIC_FEATURES = ("mass", "pt", "eta", "sin_phi", "cos_phi")


@dataclass(frozen=True)
class MassChi2Settings:
    """Loss weight and mass constraints, in GeV."""

    alpha: float = 1.0
    top_mass: float = 172.5
    top_width: float = 20.0
    w_mass: float = 80.4
    w_width: float = 15.0

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be within [0, 1]; got {self.alpha}")
        for name in ("top_mass", "top_width", "w_mass", "w_width"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")


def jet_four_vectors(data: Tensor, features: dict[str, tuple[int, bool]]) -> Tensor:
    """Return (E, px, py, pz) in GeV for every jet slot, shape (events, jets, 4).

    data holds the jet features as SPANet loads them: features marked log_normalize
    or log in the event file are stored as log(x + 1), and no standardization has been
    applied yet. features maps each kinematic name to (column, log-transformed).
    """

    def column(name: str) -> Tensor:
        index, logged = features[name]
        values = data[..., index]
        return torch.expm1(values) if logged else values

    mass, pt, eta = column("mass"), column("pt"), column("eta")
    px, py, pz = pt * column("cos_phi"), pt * column("sin_phi"), pt * torch.sinh(eta)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return torch.stack((energy, px, py, pz), dim=-1)


def invariant_mass(p4: Tensor) -> Tensor:
    """Invariant mass of (E, px, py, pz) along the last axis; unphysical m^2 < 0 gives 0."""
    m2 = p4[..., 0] ** 2 - (p4[..., 1:] ** 2).sum(-1)
    return torch.sqrt(torch.clamp(m2, min=0))


def hadronic_chi2(
    p4: Tensor, mask: Tensor, settings: MassChi2Settings
) -> tuple[Tensor, Tensor, Tensor]:
    """Chi-square terms for every (b, q1, q2) jet triplet.

    Returns the top term, the W term, and a validity mask, each of shape
    (events, jets, jets, jets) with axes (b, q1, q2). A triplet is valid when its
    three jets are real and distinct. The W term depends on q1 and q2 only.
    """
    pair = p4[:, :, None, :] + p4[:, None, :, :]
    triplet = p4[:, :, None, None, :] + pair[:, None, :, :, :]
    top = ((invariant_mass(triplet) - settings.top_mass) / settings.top_width) ** 2
    w = ((invariant_mass(pair) - settings.w_mass) / settings.w_width) ** 2
    w = w[:, None, :, :].expand_as(top)

    jets = mask.shape[1]
    index = torch.arange(jets, device=mask.device)
    distinct = (
        (index[:, None, None] != index[None, :, None])
        & (index[:, None, None] != index[None, None, :])
        & (index[None, :, None] != index[None, None, :])
    )
    real = mask[:, :, None, None] & mask[:, None, :, None] & mask[:, None, None, :]
    return top, w, real & distinct


class MassChi2Model(JetReconstructionModel):
    """JetReconstructionModel trained with alpha * L_SPANet + (1 - alpha) * <chi2>.

    SPANet's training program constructs the model from its options alone, so the
    settings are a class attribute, set before training starts.
    """

    settings = MassChi2Settings()
    seed: int | None = None

    def __init__(self, options, torch_script: bool = False):
        super().__init__(options, torch_script)
        names = list(self.event_particle_names)
        if HAD_TOP not in names:
            raise ValueError(f"event file has no {HAD_TOP} particle")
        self.had_top_index = names.index(HAD_TOP)
        daughters = tuple(self.event_info.product_particles[HAD_TOP].names)
        if daughters != HAD_TOP_DAUGHTERS:
            raise ValueError(f"{HAD_TOP} daughters must be {HAD_TOP_DAUGHTERS}; got {daughters}")

        self.jet_source_index = list(self.event_info.input_names).index(JET_INPUT)
        features = {
            feature.name: (column, bool(feature.log_scale))
            for column, feature in enumerate(self.event_info.input_features[JET_INPUT])
        }
        missing = [name for name in KINEMATIC_FEATURES if name not in features]
        if missing:
            raise ValueError(f"{JET_INPUT} lacks features needed for masses: {missing}")
        self.kinematic_features = {name: features[name] for name in KINEMATIC_FEATURES}

    def on_fit_start(self):
        super().on_fit_start()
        if self.trainer.is_global_zero and self.logger is not None and self.logger.log_dir:
            path = Path(self.logger.log_dir) / "mass_chi2.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {**asdict(self.settings), "seed": self.seed}
            path.write_text(json.dumps(record, indent=2) + "\n")

    def expected_chi2(self, outputs: Outputs, batch: Batch) -> tuple[Tensor, Tensor]:
        """Per-event <chi2_top> and <chi2_W> under the had_top assignment probabilities."""
        jets = batch.sources[self.jet_source_index]
        with torch.no_grad():
            p4 = jet_four_vectors(jets.data.float(), self.kinematic_features)
            top, w, valid = hadronic_chi2(p4, jets.mask, self.settings)
        probability = outputs.assignments[self.had_top_index].exp()
        top = torch.where(valid, probability * top, 0).flatten(1).sum(1)
        w = torch.where(valid, probability * w, 0).flatten(1).sum(1)
        return top, w

    def training_step(self, batch: Batch, batch_nb: int) -> Tensor:
        outputs = self.forward(batch.sources)
        spanet_loss = self.spanet_loss(outputs, batch)
        chi2_top, chi2_w = self.expected_chi2(outputs, batch)
        chi2 = (chi2_top + chi2_w).mean()
        alpha = self.settings.alpha
        loss = alpha * spanet_loss + (1 - alpha) * chi2

        with torch.no_grad():
            self.log("loss/spanet_loss", spanet_loss, sync_dist=True)
            self.log("loss/mass_chi2", chi2, sync_dist=True)
            self.log("loss/mass_chi2_top", chi2_top.mean(), sync_dist=True)
            self.log("loss/mass_chi2_w", chi2_w.mean(), sync_dist=True)
            self.log("loss/combined_loss", loss, sync_dist=True)
            if not torch.isfinite(loss):
                raise ValueError("Combined loss is not finite")
        return loss

    def spanet_loss(self, outputs: Outputs, batch: Batch) -> Tensor:
        """SPANet's training loss for already computed outputs.

        This is the body of JetReconstructionTraining.training_step from SPANet commit
        46c6805 (spanet/network/jet_reconstruction/jet_reconstruction_training.py,
        lines 211-290), after its forward pass, so that the outputs are shared with
        the chi-square term. Keep it in sync if the pinned SPANet version changes.
        """
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets,
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        weights = torch.ones_like(symmetric_losses)
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(
            masks.sum(-1), 1, None
        )
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        with torch.no_grad():
            for name, value in zip(self.training_dataset.assignments, assignment_loss, strict=True):
                self.log(f"loss/{name}/assignment_loss", value, sync_dist=True)
            for name, value in zip(self.training_dataset.assignments, detection_loss, strict=True):
                self.log(f"loss/{name}/detection_loss", value, sync_dist=True)
            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")
            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        total_loss = []
        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)
        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)
        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(
                total_loss, outputs.regressions, batch.regression_targets
            )
        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(
                total_loss, outputs.classifications, batch.classification_targets
            )

        total_loss = torch.cat([loss.view(-1) for loss in total_loss])
        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)
        return total_loss.mean()
