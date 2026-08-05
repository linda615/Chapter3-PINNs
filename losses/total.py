"""Weighted total loss aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Optional

import tensorflow as tf

from boundary import (
    BoundaryBatch,
    BoundaryConditionBatch,
    BoundaryLossConfig,
    BoundaryType,
    compute_boundary_loss,
    compute_grouped_boundary_loss,
)
from physics.plate_residuals import PlateParameters

from .consistency import MxyConsistencyConfig
from .normalization import PhysicsLossConfig
from .physics_loss import compute_physics_loss
from .weights import LossWeights


def compute_total_loss(
    model: tf.keras.Model,
    interior_xy: tf.Tensor,
    boundary_batch: BoundaryBatch,
    boundary_type: BoundaryType | str,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    weights: LossWeights = LossWeights(),
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    physics_loss_config: Optional[PhysicsLossConfig] = None,
    mxy_consistency_config: Optional[MxyConsistencyConfig] = None,
    training: bool = False,
    boundary_loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Compute weighted total loss from existing physics and boundary modules."""
    physics_losses = compute_physics_loss(
        model=model,
        xy=interior_xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        loss_config=physics_loss_config,
        mxy_consistency_config=mxy_consistency_config,
        training=training,
    )
    boundary_losses = compute_boundary_loss(
        model=model,
        batch=boundary_batch,
        boundary_type=boundary_type,
        training=training,
        loss_config=boundary_loss_config,
    )
    return _assemble_total_loss(physics_losses, boundary_losses, weights)


def compute_grouped_boundary_total_loss(
    model: tf.keras.Model,
    interior_xy: tf.Tensor,
    boundary_groups: Sequence[BoundaryConditionBatch],
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    weights: LossWeights = LossWeights(),
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    physics_loss_config: Optional[PhysicsLossConfig] = None,
    mxy_consistency_config: Optional[MxyConsistencyConfig] = None,
    training: bool = False,
    boundary_loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Compute total loss for boundary batches with different conditions."""
    physics_losses = compute_physics_loss(
        model=model,
        xy=interior_xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        loss_config=physics_loss_config,
        mxy_consistency_config=mxy_consistency_config,
        training=training,
    )
    boundary_losses = compute_grouped_boundary_loss(
        model=model,
        groups=boundary_groups,
        training=training,
        loss_config=boundary_loss_config,
    )
    return _assemble_total_loss(physics_losses, boundary_losses, weights)


def _assemble_total_loss(
    physics_losses: dict[str, tf.Tensor],
    boundary_losses: dict[str, tf.Tensor],
    weights: LossWeights,
) -> dict[str, tf.Tensor]:
    """Combine precomputed physics and boundary loss dictionaries."""

    moment_x_weight = physics_losses["moment_x_weight"]
    moment_y_weight = physics_losses["moment_y_weight"]
    twisting_moment_weight = physics_losses["twisting_moment_weight"]
    shear_x_weight = physics_losses["shear_x_weight"]
    shear_y_weight = physics_losses["shear_y_weight"]
    moment_x_loss = physics_losses["moment_x_loss"]
    moment_y_loss = physics_losses["moment_y_loss"]
    twisting_moment_loss = physics_losses["twisting_moment_loss"]
    weighted_moment_x_loss = physics_losses["weighted_moment_x_loss"]
    weighted_moment_y_loss = physics_losses["weighted_moment_y_loss"]
    weighted_twisting_moment_loss = physics_losses[
        "weighted_twisting_moment_loss"
    ]
    moment_loss = physics_losses["moment_loss"]
    shear_x_loss = physics_losses["shear_x_loss"]
    shear_y_loss = physics_losses["shear_y_loss"]
    weighted_shear_x_loss = physics_losses["weighted_shear_x_loss"]
    weighted_shear_y_loss = physics_losses["weighted_shear_y_loss"]
    shear_loss = physics_losses["shear_loss"]
    equilibrium_loss = physics_losses["equilibrium_loss"]
    physics_loss = moment_loss + shear_loss + equilibrium_loss
    mxy_consistency_loss = physics_losses["mxy_consistency_loss"]
    mxy_consistency_scale = physics_losses["mxy_consistency_scale"]
    boundary_loss = boundary_losses["boundary_loss"]

    weighted_moment_loss = _weighted(moment_loss, weights.moment)
    weighted_shear_loss = _weighted(shear_loss, weights.shear)
    weighted_equilibrium_loss = _weighted(equilibrium_loss, weights.equilibrium)
    weighted_boundary_loss = _weighted(boundary_loss, weights.boundary)
    weighted_mxy_consistency_loss = _weighted(
        mxy_consistency_loss,
        weights.mxy_consistency,
    )
    total_loss = (
        weighted_moment_loss
        + weighted_shear_loss
        + weighted_equilibrium_loss
        + weighted_boundary_loss
        + weighted_mxy_consistency_loss
    )

    return {
        "moment_x_weight": moment_x_weight,
        "moment_y_weight": moment_y_weight,
        "twisting_moment_weight": twisting_moment_weight,
        "shear_x_weight": shear_x_weight,
        "shear_y_weight": shear_y_weight,
        "moment_x_loss": moment_x_loss,
        "moment_y_loss": moment_y_loss,
        "twisting_moment_loss": twisting_moment_loss,
        "weighted_moment_x_loss": weighted_moment_x_loss,
        "weighted_moment_y_loss": weighted_moment_y_loss,
        "weighted_twisting_moment_loss": weighted_twisting_moment_loss,
        "moment_loss": moment_loss,
        "shear_x_loss": shear_x_loss,
        "shear_y_loss": shear_y_loss,
        "weighted_shear_x_loss": weighted_shear_x_loss,
        "weighted_shear_y_loss": weighted_shear_y_loss,
        "shear_loss": shear_loss,
        "equilibrium_loss": equilibrium_loss,
        "physics_loss": physics_loss,
        "mxy_consistency_loss": mxy_consistency_loss,
        "mxy_consistency_scale": mxy_consistency_scale,
        "deflection_bc_loss": boundary_losses["deflection_bc_loss"],
        "slope_bc_loss": boundary_losses["slope_bc_loss"],
        "moment_bc_loss": boundary_losses["moment_bc_loss"],
        "shear_bc_loss": boundary_losses["shear_bc_loss"],
        "boundary_loss": boundary_loss,
        "weighted_moment_loss": weighted_moment_loss,
        "weighted_shear_loss": weighted_shear_loss,
        "weighted_equilibrium_loss": weighted_equilibrium_loss,
        "weighted_boundary_loss": weighted_boundary_loss,
        "weighted_mxy_consistency_loss": weighted_mxy_consistency_loss,
        "total_loss": total_loss,
    }


def _weighted(loss: tf.Tensor, weight: float) -> tf.Tensor:
    """Multiply a scalar Tensor loss by a Python weight."""
    loss = tf.convert_to_tensor(loss)
    return loss * tf.cast(weight, loss.dtype)
