"""Loss construction for the multi-output fourth-order PINN."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from boundary import BoundaryBatch, BoundaryType, compute_boundary_loss
from physics import (
    PlateParameters,
    compute_multi_output_fourth_order_residuals,
)

from .weights import LossWeights


def compute_mo4_physics_loss(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute moment, shear, and fourth-order equilibrium losses."""
    residuals = compute_multi_output_fourth_order_residuals(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    moment_x_loss = _mean_square(residuals["moment_x"])
    moment_y_loss = _mean_square(residuals["moment_y"])
    twisting_moment_loss = _mean_square(residuals["twisting_moment"])
    moment_loss = (
        moment_x_loss
        + moment_y_loss
        + twisting_moment_loss
    ) / 3.0
    shear_x_loss = _mean_square(residuals["shear_x"])
    shear_y_loss = _mean_square(residuals["shear_y"])
    shear_loss = shear_x_loss + shear_y_loss
    fourth_order_loss = _mean_square(residuals["equilibrium"])
    physics_loss = moment_loss + shear_loss + fourth_order_loss

    return {
        "moment_x_loss": moment_x_loss,
        "moment_y_loss": moment_y_loss,
        "twisting_moment_loss": twisting_moment_loss,
        "moment_loss": moment_loss,
        "shear_x_loss": shear_x_loss,
        "shear_y_loss": shear_y_loss,
        "shear_loss": shear_loss,
        "fourth_order_loss": fourth_order_loss,
        # Keep the common key so existing logging and weighting remain usable.
        "equilibrium_loss": fourth_order_loss,
        "physics_loss": physics_loss,
    }


def compute_mo4_total_loss(
    model: tf.keras.Model,
    interior_xy: tf.Tensor,
    boundary_batch: BoundaryBatch,
    boundary_type: BoundaryType | str,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    weights: LossWeights = LossWeights(),
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Combine MO4 physics and the existing mixed-field boundary loss."""
    physics_losses = compute_mo4_physics_loss(
        model=model,
        xy=interior_xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    boundary_losses = compute_boundary_loss(
        model=model,
        batch=boundary_batch,
        boundary_type=boundary_type,
        training=training,
    )

    moment_loss = physics_losses["moment_loss"]
    shear_loss = physics_losses["shear_loss"]
    fourth_order_loss = physics_losses["fourth_order_loss"]
    boundary_loss = boundary_losses["boundary_loss"]
    weighted_moment_loss = _weighted(moment_loss, weights.moment)
    weighted_shear_loss = _weighted(shear_loss, weights.shear)
    weighted_equilibrium_loss = _weighted(
        fourth_order_loss,
        weights.equilibrium,
    )
    weighted_boundary_loss = _weighted(boundary_loss, weights.boundary)
    total_loss = (
        weighted_moment_loss
        + weighted_shear_loss
        + weighted_equilibrium_loss
        + weighted_boundary_loss
    )

    return {
        "moment_x_loss": physics_losses["moment_x_loss"],
        "moment_y_loss": physics_losses["moment_y_loss"],
        "twisting_moment_loss": physics_losses["twisting_moment_loss"],
        "moment_loss": moment_loss,
        "shear_x_loss": physics_losses["shear_x_loss"],
        "shear_y_loss": physics_losses["shear_y_loss"],
        "shear_loss": shear_loss,
        "fourth_order_loss": fourth_order_loss,
        "equilibrium_loss": fourth_order_loss,
        "physics_loss": physics_losses["physics_loss"],
        "deflection_bc_loss": boundary_losses["deflection_bc_loss"],
        "slope_bc_loss": boundary_losses["slope_bc_loss"],
        "moment_bc_loss": boundary_losses["moment_bc_loss"],
        "shear_bc_loss": boundary_losses["shear_bc_loss"],
        "boundary_loss": boundary_loss,
        "weighted_moment_loss": weighted_moment_loss,
        "weighted_shear_loss": weighted_shear_loss,
        "weighted_equilibrium_loss": weighted_equilibrium_loss,
        "weighted_boundary_loss": weighted_boundary_loss,
        "total_loss": total_loss,
    }


def _mean_square(value: tf.Tensor) -> tf.Tensor:
    """Return the scalar mean-square residual."""
    return tf.reduce_mean(tf.square(value))


def _weighted(loss: tf.Tensor, weight: float) -> tf.Tensor:
    """Apply a scalar Python loss weight without leaving TensorFlow."""
    return loss * tf.cast(weight, loss.dtype)
