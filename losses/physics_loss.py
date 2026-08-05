"""Physics loss terms for Kirchhoff thin plate PINNs."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from physics.plate_residuals import (
    PlateParameters,
    compute_plate_residual_state,
)

from .consistency import MxyConsistencyConfig, compute_mxy_consistency_loss
from .normalization import PhysicsLossConfig


def compute_physics_loss(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    loss_config: Optional[PhysicsLossConfig] = None,
    mxy_consistency_config: Optional[MxyConsistencyConfig] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute residual-based physics losses.

    This function intentionally delegates all automatic differentiation and
    Kirchhoff residual construction to ``compute_plate_residuals``. It only
    aggregates residual mean-square terms for moment, shear, and equilibrium
    equations.
    """
    state = compute_plate_residual_state(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    config = PhysicsLossConfig() if loss_config is None else loss_config
    residuals = _maybe_normalize_residuals(state.residuals, config)
    consistency_losses = compute_mxy_consistency_loss(
        state=state,
        parameters=parameters,
        config=mxy_consistency_config,
    )

    moment_x_loss = _mean_square(residuals["moment_x"])
    moment_y_loss = _mean_square(residuals["moment_y"])
    twisting_moment_loss = _mean_square(residuals["twisting_moment"])
    component_weights = config.component_weights
    weighted_moment_x_loss = _weighted(
        moment_x_loss,
        component_weights.moment_x,
    )
    weighted_moment_y_loss = _weighted(
        moment_y_loss,
        component_weights.moment_y,
    )
    weighted_twisting_moment_loss = _weighted(
        twisting_moment_loss,
        component_weights.twisting_moment,
    )
    moment_loss = (
        weighted_moment_x_loss
        + weighted_moment_y_loss
        + weighted_twisting_moment_loss
    )
    shear_x_loss = _mean_square(residuals["shear_x"])
    shear_y_loss = _mean_square(residuals["shear_y"])
    weighted_shear_x_loss = _weighted(
        shear_x_loss,
        component_weights.shear_x,
    )
    weighted_shear_y_loss = _weighted(
        shear_y_loss,
        component_weights.shear_y,
    )
    shear_loss = weighted_shear_x_loss + weighted_shear_y_loss
    equilibrium_loss = _mean_square(residuals["equilibrium"])
    physics_loss = moment_loss + shear_loss + equilibrium_loss

    return {
        "moment_x_weight": _weight_tensor(
            component_weights.moment_x,
            moment_x_loss,
        ),
        "moment_y_weight": _weight_tensor(
            component_weights.moment_y,
            moment_y_loss,
        ),
        "twisting_moment_weight": _weight_tensor(
            component_weights.twisting_moment,
            twisting_moment_loss,
        ),
        "shear_x_weight": _weight_tensor(
            component_weights.shear_x,
            shear_x_loss,
        ),
        "shear_y_weight": _weight_tensor(
            component_weights.shear_y,
            shear_y_loss,
        ),
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
        "mxy_consistency_loss": consistency_losses["mxy_consistency_loss"],
        "mxy_consistency_scale": consistency_losses["mxy_consistency_scale"],
    }


def _mean_square(value: tf.Tensor) -> tf.Tensor:
    """Return the graph-compatible mean square of a residual tensor."""
    return tf.reduce_mean(tf.square(value))


def _maybe_normalize_residuals(
    residuals: dict[str, tf.Tensor],
    loss_config: PhysicsLossConfig | None,
) -> dict[str, tf.Tensor]:
    """Return raw or scale-normalized residuals according to ``loss_config``."""
    config = PhysicsLossConfig() if loss_config is None else loss_config
    if not config.normalize_residuals:
        return residuals

    scales = config.residual_scales
    return {
        "moment_x": _divide_by_scale(residuals["moment_x"], scales.moment_x),
        "moment_y": _divide_by_scale(residuals["moment_y"], scales.moment_y),
        "twisting_moment": _divide_by_scale(
            residuals["twisting_moment"],
            scales.twisting_moment,
        ),
        "shear_x": _divide_by_scale(residuals["shear_x"], scales.shear_x),
        "shear_y": _divide_by_scale(residuals["shear_y"], scales.shear_y),
        "equilibrium": _divide_by_scale(residuals["equilibrium"], scales.equilibrium),
    }


def _divide_by_scale(value: tf.Tensor, scale: float) -> tf.Tensor:
    """Divide one residual tensor by a positive Python scale."""
    value = tf.convert_to_tensor(value)
    return value / tf.cast(scale, value.dtype)


def _weighted(loss: tf.Tensor, weight: float) -> tf.Tensor:
    """Apply one component weight without leaving the TensorFlow graph."""
    return loss * tf.cast(weight, loss.dtype)


def _weight_tensor(weight: float, reference: tf.Tensor) -> tf.Tensor:
    """Return a scalar component weight with the loss computation dtype."""
    return tf.cast(weight, reference.dtype)
