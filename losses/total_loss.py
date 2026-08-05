"""Backward-compatible total loss wrapper."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from physics.plate_residuals import PlateParameters

from boundary import BoundaryBatch, BoundaryType
from .normalization import PhysicsLossConfig
from .total import compute_total_loss as _compute_total_loss
from .weights import LossWeights


def compute_total_loss(
    model: tf.keras.Model,
    interior_xy: tf.Tensor,
    boundary_xy: tf.Tensor,
    boundary_normals: tf.Tensor,
    boundary_type: BoundaryType | str,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    weights: LossWeights = LossWeights(),
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    physics_loss_config: Optional[PhysicsLossConfig] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute total loss while preserving the previous tensor arguments."""
    return _compute_total_loss(
        model=model,
        interior_xy=interior_xy,
        boundary_batch=BoundaryBatch(xy=boundary_xy, normals=boundary_normals),
        boundary_type=boundary_type,
        load_fn=load_fn,
        parameters=parameters,
        weights=weights,
        foundation_fn=foundation_fn,
        physics_loss_config=physics_loss_config,
        training=training,
    )
