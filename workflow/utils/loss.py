"""Functionality for calculating the loss."""

from typing import Protocol, TypedDict

import datasets
import descent.targets.dimers
import descent.train
import smee
import torch
from typing_extensions import Unpack

from .convert import to_vdw_only_ff


class LossFnArgs(TypedDict):
    x: torch.Tensor
    trainable: descent.train.Trainable
    batch: datasets.Dataset
    topologies: dict[str, smee.TensorTopology]
    vdw_only: bool


# Using Protocol rather than Callable to preserve names of arguments
class LossFnProto(Protocol):
    def __call__(self, **kwargs: Unpack[LossFnArgs]) -> torch.Tensor: ...


def get_loss_dimer_boltz_ref_0(
    x: torch.Tensor,
    trainable: descent.train.Trainable,
    batch: datasets.Dataset,
    topologies: dict[str, smee.TensorTopology],
    vdw_only: bool,
) -> torch.Tensor:
    """
    Get the loss for dimers by Boltzmann weighting energies above 0,
    and energies below 0 by 1.
    """
    ff = trainable.to_force_field(x)
    if vdw_only:
        ff = to_vdw_only_ff(ff)
    y_ref, y_pred = descent.targets.dimers.predict(batch, ff, topologies)
    # Weight energies above 0 by the Boltzmann factor
    weights = torch.exp(-(y_ref) / 0.59)  # kT at 300K
    # Weight energies below 0 by 1
    weights[y_ref < 0] = 1.0
    # Normalize weights
    weights /= weights.sum()

    weighted_sq_diff = torch.square(y_pred - y_ref) * weights

    # Use sum rather than mean to avoid scaling by batch size.
    # The weights are already normalized
    return torch.sqrt(weighted_sq_diff.sum())


def get_loss_dimer_cutoff_10(
    x: torch.Tensor,
    trainable: descent.train.Trainable,
    batch: datasets.Dataset,
    topologies: dict[str, smee.TensorTopology],
    vdw_only: bool,
) -> torch.Tensor:
    """
    Get the loss for dimers by Boltzmann weighting energies above 0,
    and energies below 0 by 1.
    """
    ff = trainable.to_force_field(x)
    if vdw_only:
        ff = to_vdw_only_ff(ff)
    y_ref, y_pred = descent.targets.dimers.predict(batch, ff, topologies)
    # Set weight to 0 if greater than 10 kcal mol^-1, otherwise
    # 1
    weights = torch.where(
        y_ref > 10.0,
        torch.tensor(0, device=x.device),
        torch.tensor(1.0, device=x.device),
    )
    # Normalize weights
    weights /= weights.sum()

    weighted_sq_diff = torch.square(y_pred - y_ref) * weights

    # Use sum rather than mean to avoid scaling by batch size.
    # The weights are already normalized
    return torch.sqrt(weighted_sq_diff.sum())

    # def loss_fn(_x):
    #     ff_vdw = to_vdw_only_ff(trainable.to_force_field(_x))
    #     y_ref, y_pred = descent.targets.dimers.predict(
    #         batch, ff_vdw, topologies
    #     )
    #     return torch.sqrt(((y_pred - y_ref) ** 2).mean())

    # def loss_fn(_x):
    #     ff_vdw = to_vdw_only_ff(trainable.to_force_field(_x))
    #     y_ref, y_pred = descent.targets.dimers.predict(
    #         batch, ff_vdw, topologies
    #     )
    #     # Weight the loss by the Boltzmann factor of the QM energy
    #     qm_0 = min(y_ref)
    #     weights = torch.exp(-(y_ref - qm_0) / 0.59)  # kT at 300K
    #     weights /= weights.sum()
    #     weighted_sq_diff = torch.square(y_pred - y_ref) * weights

    #     return torch.sqrt(weighted_sq_diff.mean())
