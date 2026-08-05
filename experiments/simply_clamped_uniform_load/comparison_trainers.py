"""Mixed-edge trainer adapters for W-PINN and MO4-PINN comparisons."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from boundary import (
    BoundaryConditionBatch,
    BoundaryType,
    compute_grouped_boundary_loss,
)
from losses import (
    LossWeights,
    PhysicsLossConfig,
    compute_w_pinn_boundary_loss,
    resolve_loss_weights,
)
from physics import (
    compute_multi_output_fourth_order_residuals,
    compute_w_pinn_physics_loss,
)
from trainer import PINNTrainer

from .config import ExperimentConfig
from .problem import sample_boundary_groups


class _MixedEdgeSamplingTrainer(PINNTrainer):
    """Base adapter that samples edge-specific simple and clamped groups."""

    def __init__(
        self,
        *args: Any,
        experiment_config: ExperimentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_config = experiment_config

    def sample_epoch(
        self,
        epoch: int,
    ) -> tuple[tf.Tensor, tuple[BoundaryConditionBatch, ...]]:
        """Sample reproducible interior points and both boundary groups."""
        seed = self.config.seed
        interior_seed = None if seed is None else seed + epoch
        boundary_seed = None if seed is None else seed + 100000 + epoch
        interior_xy = self.interior_sampler.sample(
            self.config.interior_points,
            seed=interior_seed,
        )
        groups = sample_boundary_groups(
            sampler=self.boundary_sampler,
            points_per_side=self.config.boundary_points_per_side,
            config=self.experiment_config,
            seed=boundary_seed,
        )
        return interior_xy, groups


class MixedBoundaryWPINNTrainer(_MixedEdgeSamplingTrainer):
    """Train W-PINN with fourth-order physics and mixed edge conditions."""

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: tuple[BoundaryConditionBatch, ...],
        loss_weights: LossWeights | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute fourth-order equilibrium and grouped W-derived BC losses."""
        weights = (
            resolve_loss_weights(self.loss_weights, 1)
            if loss_weights is None
            else loss_weights
        )
        physics = compute_w_pinn_physics_loss(
            model=self.model,
            xy=interior_xy,
            load_fn=self.load_fn,
            parameters=self.parameters,
            foundation_fn=self.foundation_fn,
            training=training,
        )
        grouped = _compute_grouped_w_boundary_loss(
            model=self.model,
            groups=boundary_batch,
            parameters=self.parameters,
            training=training,
            loss_config=self.boundary_loss_config,
        )
        equilibrium_loss = physics["equilibrium_loss"]
        if self.physics_loss_config is not None:
            config = self.physics_loss_config
            if config.normalize_residuals:
                scale = tf.cast(
                    config.residual_scales.equilibrium,
                    equilibrium_loss.dtype,
                )
                equilibrium_loss = equilibrium_loss / tf.square(scale)
        boundary_loss = grouped["boundary_loss"]
        weighted_equilibrium_loss = _weighted(
            equilibrium_loss,
            weights.equilibrium,
        )
        weighted_boundary_loss = _weighted(
            boundary_loss,
            weights.boundary,
        )
        return {
            "equilibrium_loss": equilibrium_loss,
            "physics_loss": equilibrium_loss,
            **grouped,
            "weighted_equilibrium_loss": weighted_equilibrium_loss,
            "weighted_boundary_loss": weighted_boundary_loss,
            "total_loss": weighted_equilibrium_loss + weighted_boundary_loss,
        }


class MixedBoundaryMO4PINNTrainer(_MixedEdgeSamplingTrainer):
    """Train six-output MO4-PINN with edge-specific boundary operators."""

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: tuple[BoundaryConditionBatch, ...],
        loss_weights: LossWeights | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute normalized MO4 physics and grouped mixed-edge BC losses."""
        weights = (
            resolve_loss_weights(self.loss_weights, 1)
            if loss_weights is None
            else loss_weights
        )
        physics = _compute_normalized_mo4_physics_loss(
            model=self.model,
            xy=interior_xy,
            load_fn=self.load_fn,
            parameters=self.parameters,
            foundation_fn=self.foundation_fn,
            loss_config=self.physics_loss_config,
            training=training,
        )
        boundary = compute_grouped_boundary_loss(
            model=self.model,
            groups=boundary_batch,
            training=training,
            loss_config=self.boundary_loss_config,
        )
        weighted_moment_loss = _weighted(physics["moment_loss"], weights.moment)
        weighted_shear_loss = _weighted(physics["shear_loss"], weights.shear)
        weighted_equilibrium_loss = _weighted(
            physics["equilibrium_loss"],
            weights.equilibrium,
        )
        weighted_boundary_loss = _weighted(
            boundary["boundary_loss"],
            weights.boundary,
        )
        return {
            **physics,
            **boundary,
            "weighted_moment_loss": weighted_moment_loss,
            "weighted_shear_loss": weighted_shear_loss,
            "weighted_equilibrium_loss": weighted_equilibrium_loss,
            "weighted_boundary_loss": weighted_boundary_loss,
            "total_loss": (
                weighted_moment_loss
                + weighted_shear_loss
                + weighted_equilibrium_loss
                + weighted_boundary_loss
            ),
        }


def _compute_grouped_w_boundary_loss(
    model: tf.keras.Model,
    groups: tuple[BoundaryConditionBatch, ...],
    parameters,
    training: bool,
    loss_config,
) -> dict[str, tf.Tensor]:
    """Aggregate W-derived BC components over only applicable edge groups."""
    component_values = {
        "deflection_bc_loss": [],
        "slope_bc_loss": [],
        "moment_bc_loss": [],
        "shear_bc_loss": [],
    }
    reference = None
    for group in groups:
        group = group.validated()
        losses = compute_w_pinn_boundary_loss(
            model=model,
            boundary_batch=group.batch,
            boundary_type=group.boundary_type,
            parameters=parameters,
            training=training,
            loss_config=loss_config,
        )
        if reference is None:
            reference = losses["boundary_loss"]
        component_values["deflection_bc_loss"].append(
            losses["deflection_bc_loss"]
        )
        if group.boundary_type == BoundaryType.CLAMPED:
            component_values["slope_bc_loss"].append(losses["slope_bc_loss"])
        if group.boundary_type == BoundaryType.SIMPLE:
            component_values["moment_bc_loss"].append(losses["moment_bc_loss"])

    if reference is None:
        raise ValueError("groups must contain at least one boundary batch.")
    zero = tf.zeros((), dtype=reference.dtype)
    aggregated = {
        name: (
            tf.add_n(values) / tf.cast(len(values), values[0].dtype)
            if values
            else zero
        )
        for name, values in component_values.items()
    }
    aggregated["boundary_loss"] = tf.add_n(list(aggregated.values()))
    return aggregated


def _compute_normalized_mo4_physics_loss(
    model,
    xy,
    load_fn,
    parameters,
    foundation_fn,
    loss_config: PhysicsLossConfig | None,
    training: bool,
) -> dict[str, tf.Tensor]:
    """Match Mixed PINN residual scaling and component aggregation for MO4."""
    residuals = compute_multi_output_fourth_order_residuals(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    config = PhysicsLossConfig() if loss_config is None else loss_config
    if config.normalize_residuals:
        scales = config.residual_scales
        residuals = {
            "moment_x": residuals["moment_x"] / tf.cast(scales.moment_x, xy.dtype),
            "moment_y": residuals["moment_y"] / tf.cast(scales.moment_y, xy.dtype),
            "twisting_moment": residuals["twisting_moment"] / tf.cast(scales.twisting_moment, xy.dtype),
            "shear_x": residuals["shear_x"] / tf.cast(scales.shear_x, xy.dtype),
            "shear_y": residuals["shear_y"] / tf.cast(scales.shear_y, xy.dtype),
            "equilibrium": residuals["equilibrium"] / tf.cast(scales.equilibrium, xy.dtype),
        }
    component_weights = config.component_weights
    moment_x_loss = _mean_square(residuals["moment_x"])
    moment_y_loss = _mean_square(residuals["moment_y"])
    twisting_moment_loss = _mean_square(residuals["twisting_moment"])
    shear_x_loss = _mean_square(residuals["shear_x"])
    shear_y_loss = _mean_square(residuals["shear_y"])
    weighted_moment_x_loss = _weighted(moment_x_loss, component_weights.moment_x)
    weighted_moment_y_loss = _weighted(moment_y_loss, component_weights.moment_y)
    weighted_twisting_moment_loss = _weighted(
        twisting_moment_loss,
        component_weights.twisting_moment,
    )
    weighted_shear_x_loss = _weighted(shear_x_loss, component_weights.shear_x)
    weighted_shear_y_loss = _weighted(shear_y_loss, component_weights.shear_y)
    moment_loss = (
        weighted_moment_x_loss
        + weighted_moment_y_loss
        + weighted_twisting_moment_loss
    )
    shear_loss = weighted_shear_x_loss + weighted_shear_y_loss
    equilibrium_loss = _mean_square(residuals["equilibrium"])
    return {
        "moment_x_weight": tf.cast(component_weights.moment_x, xy.dtype),
        "moment_y_weight": tf.cast(component_weights.moment_y, xy.dtype),
        "twisting_moment_weight": tf.cast(component_weights.twisting_moment, xy.dtype),
        "shear_x_weight": tf.cast(component_weights.shear_x, xy.dtype),
        "shear_y_weight": tf.cast(component_weights.shear_y, xy.dtype),
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
        "fourth_order_loss": equilibrium_loss,
        "equilibrium_loss": equilibrium_loss,
        "physics_loss": moment_loss + shear_loss + equilibrium_loss,
    }


def _mean_square(value: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(value))


def _weighted(value: tf.Tensor, weight: float) -> tf.Tensor:
    return value * tf.cast(weight, value.dtype)


__all__ = ["MixedBoundaryMO4PINNTrainer", "MixedBoundaryWPINNTrainer"]
