"""Loss assembly for the deflection-only W-PINN baseline."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from boundary import (
    BoundaryBatch,
    BoundaryLossConfig,
    BoundaryResiduals,
    BoundaryType,
    boundary_residual_loss,
    normal_moment,
    normal_slope,
)
from physics.plate_residuals import PlateParameters
from physics.w_pinn_fields import (
    compute_w_pinn_derivatives,
    fields_from_w_derivatives,
)
from physics.w_pinn_residuals import compute_w_pinn_physics_loss

from .weights import LossWeights


def compute_w_pinn_boundary_loss(
    model: tf.keras.Model,
    boundary_batch: BoundaryBatch,
    boundary_type: BoundaryType | str,
    parameters: PlateParameters,
    training: bool = False,
    loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Compute simple or clamped boundary loss from derivatives of ``w``."""
    batch = boundary_batch.validated()
    boundary_type = BoundaryType(boundary_type)
    derivatives = compute_w_pinn_derivatives(
        model=model,
        xy=batch.xy,
        training=training,
    )

    if boundary_type == BoundaryType.SIMPLE:
        fields = fields_from_w_derivatives(derivatives, parameters)
        residuals = BoundaryResiduals(
            deflection=fields["w"],
            moment=normal_moment(fields, batch.normals),
        )
    elif boundary_type == BoundaryType.CLAMPED:
        residuals = BoundaryResiduals(
            deflection=derivatives["w"],
            slope=normal_slope(derivatives, batch.normals),
        )
    else:
        raise ValueError(
            "W-PINN boundary loss currently supports only simple and clamped "
            f"boundaries, got {boundary_type.value!r}."
        )

    return boundary_residual_loss(
        residuals,
        weights=batch.weights,
        loss_config=loss_config,
    )


def compute_w_pinn_total_loss(
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
    """Combine fourth-order equilibrium and boundary losses."""
    physics_losses = compute_w_pinn_physics_loss(
        model=model,
        xy=interior_xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    boundary_losses = compute_w_pinn_boundary_loss(
        model=model,
        boundary_batch=boundary_batch,
        boundary_type=boundary_type,
        parameters=parameters,
        training=training,
    )

    equilibrium_loss = physics_losses["equilibrium_loss"]
    physics_loss = physics_losses["physics_loss"]
    boundary_loss = boundary_losses["boundary_loss"]
    weighted_equilibrium_loss = equilibrium_loss * tf.cast(
        weights.equilibrium,
        equilibrium_loss.dtype,
    )
    weighted_boundary_loss = boundary_loss * tf.cast(
        weights.boundary,
        boundary_loss.dtype,
    )
    total_loss = weighted_equilibrium_loss + weighted_boundary_loss

    return {
        "equilibrium_loss": equilibrium_loss,
        "physics_loss": physics_loss,
        "deflection_bc_loss": boundary_losses["deflection_bc_loss"],
        "slope_bc_loss": boundary_losses["slope_bc_loss"],
        "moment_bc_loss": boundary_losses["moment_bc_loss"],
        "shear_bc_loss": boundary_losses["shear_bc_loss"],
        "boundary_loss": boundary_loss,
        "weighted_equilibrium_loss": weighted_equilibrium_loss,
        "weighted_boundary_loss": weighted_boundary_loss,
        "total_loss": total_loss,
    }
