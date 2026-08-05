"""Boundary condition dispatch and loss aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

from .conditions import (
    BoundaryCondition,
    ClampedBoundary,
    FreeBoundary,
    SimplySupportedBoundary,
)
from .data import BoundaryBatch, BoundaryConditionBatch, BoundaryResiduals
from .types import BoundaryType
from .scaling import BoundaryLossConfig


def get_boundary_condition(boundary_type: BoundaryType | str) -> BoundaryCondition:
    """Create a boundary condition implementation for ``boundary_type``."""
    boundary_type = BoundaryType(boundary_type)
    if boundary_type == BoundaryType.SIMPLE:
        return SimplySupportedBoundary()
    if boundary_type == BoundaryType.CLAMPED:
        return ClampedBoundary()
    if boundary_type == BoundaryType.FREE:
        return FreeBoundary()
    raise ValueError(f"Unsupported boundary_type: {boundary_type!r}.")


def compute_boundary_loss(
    model: tf.keras.Model,
    batch: BoundaryBatch,
    boundary_type: BoundaryType | str,
    training: bool = False,
    loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Compute standard boundary loss terms for a boundary batch."""
    condition = get_boundary_condition(boundary_type)
    batch = batch.validated()
    residuals = condition.compute_residuals(model, batch, training=training)
    return boundary_residual_loss(
        residuals,
        weights=batch.weights,
        loss_config=loss_config,
    )


def compute_grouped_boundary_loss(
    model: tf.keras.Model,
    groups: Sequence[BoundaryConditionBatch],
    training: bool = False,
    loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Compute one boundary loss from batches with different conditions.

    Residuals are combined component by component. Each component is averaged
    over only the groups where that condition is defined. This prevents, for
    example, clamped edges from diluting the moment loss that applies only to
    simply supported edges.
    """
    groups = tuple(group.validated() for group in groups)
    if not groups:
        raise ValueError("groups must contain at least one boundary batch.")

    residual_groups = []
    for group in groups:
        condition = get_boundary_condition(group.boundary_type)
        residuals = condition.compute_residuals(
            model,
            group.batch,
            training=training,
        )
        residual_groups.append(
            (
                _maybe_normalize_residuals(residuals, loss_config),
                group.batch.weights,
            )
        )

    dtype = _infer_grouped_dtype(residual_groups)
    deflection_bc_loss = _grouped_component_loss(
        residual_groups,
        "deflection",
        dtype,
    )
    slope_bc_loss = _grouped_component_loss(
        residual_groups,
        "slope",
        dtype,
    )
    moment_bc_loss = _grouped_component_loss(
        residual_groups,
        "moment",
        dtype,
    )
    shear_bc_loss = _grouped_component_loss(
        residual_groups,
        "shear",
        dtype,
    )
    boundary_loss = (
        deflection_bc_loss
        + slope_bc_loss
        + moment_bc_loss
        + shear_bc_loss
    )
    return {
        "deflection_bc_loss": deflection_bc_loss,
        "slope_bc_loss": slope_bc_loss,
        "moment_bc_loss": moment_bc_loss,
        "shear_bc_loss": shear_bc_loss,
        "boundary_loss": boundary_loss,
    }


def compute_ss_boundary(
    model: tf.keras.Model,
    batch: BoundaryBatch,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute simply supported boundary loss terms."""
    batch = batch.validated()
    residuals = SimplySupportedBoundary().compute_residuals(model, batch, training=training)
    return boundary_residual_loss(residuals, weights=batch.weights)


def compute_clamped_boundary(
    model: tf.keras.Model,
    batch: BoundaryBatch,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute clamped boundary loss terms."""
    batch = batch.validated()
    residuals = ClampedBoundary().compute_residuals(model, batch, training=training)
    return boundary_residual_loss(residuals, weights=batch.weights)


def compute_free_boundary(
    model: tf.keras.Model,
    batch: BoundaryBatch,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute free boundary loss terms."""
    batch = batch.validated()
    residuals = FreeBoundary().compute_residuals(model, batch, training=training)
    return boundary_residual_loss(residuals, weights=batch.weights)


def boundary_residual_loss(
    residuals: BoundaryResiduals,
    weights: tf.Tensor | None = None,
    loss_config: BoundaryLossConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Aggregate residual tensors into the standard boundary loss dictionary."""
    residuals = _maybe_normalize_residuals(residuals, loss_config)
    dtype = _infer_dtype(residuals, weights)
    deflection_bc_loss = _component_loss(residuals.deflection, weights, dtype)
    slope_bc_loss = _component_loss(residuals.slope, weights, dtype)
    moment_bc_loss = _component_loss(residuals.moment, weights, dtype)
    shear_bc_loss = _component_loss(residuals.shear, weights, dtype)
    boundary_loss = deflection_bc_loss + moment_bc_loss
    boundary_loss = boundary_loss + slope_bc_loss + shear_bc_loss
    return {
        "deflection_bc_loss": deflection_bc_loss,
        "slope_bc_loss": slope_bc_loss,
        "moment_bc_loss": moment_bc_loss,
        "shear_bc_loss": shear_bc_loss,
        "boundary_loss": boundary_loss,
    }


def _maybe_normalize_residuals(
    residuals: BoundaryResiduals,
    config: BoundaryLossConfig | None,
) -> BoundaryResiduals:
    """Return raw or characteristic-scale-normalized residuals."""
    if config is None or not config.normalize_residuals:
        return residuals
    scales = config.residual_scales
    return BoundaryResiduals(
        deflection=_divide_optional(residuals.deflection, scales.deflection),
        slope=_divide_optional(residuals.slope, scales.slope),
        moment=_divide_optional(residuals.moment, scales.moment),
        shear=_divide_optional(residuals.shear, scales.shear),
    )


def _divide_optional(
    value: tf.Tensor | None,
    scale: float,
) -> tf.Tensor | None:
    """Divide an optional residual by a positive Python scale."""
    if value is None:
        return None
    value = tf.convert_to_tensor(value)
    return value / tf.cast(scale, value.dtype)


def _component_loss(
    value: tf.Tensor | None,
    weights: tf.Tensor | None,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return a scalar loss for one residual component."""
    if value is None:
        return tf.zeros((), dtype=dtype)
    return _mean_square(value, weights)


def _mean_square(value: tf.Tensor, weights: tf.Tensor | None) -> tf.Tensor:
    """Return weighted or unweighted mean square."""
    value = tf.convert_to_tensor(value)
    squared = tf.square(value)
    if weights is not None:
        weights = tf.cast(weights, squared.dtype)
        return tf.reduce_sum(weights * squared) / tf.reduce_sum(weights)
    return tf.reduce_mean(squared)


def _infer_dtype(
    residuals: BoundaryResiduals,
    weights: tf.Tensor | None,
) -> tf.dtypes.DType:
    """Infer a TensorFlow dtype for zero-valued missing component losses."""
    for value in (
        residuals.deflection,
        residuals.slope,
        residuals.moment,
        residuals.shear,
    ):
        if value is not None:
            return tf.convert_to_tensor(value).dtype
    if weights is not None:
        return tf.convert_to_tensor(weights).dtype
    return tf.float32


def _infer_grouped_dtype(
    residual_groups: Sequence[
        tuple[BoundaryResiduals, tf.Tensor | None]
    ],
) -> tf.dtypes.DType:
    """Infer the common dtype used by grouped boundary losses."""
    for residuals, weights in residual_groups:
        dtype = _infer_dtype(residuals, weights)
        if dtype is not None:
            return dtype
    return tf.float32


def _grouped_component_loss(
    residual_groups: Sequence[
        tuple[BoundaryResiduals, tf.Tensor | None]
    ],
    component: str,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return a point-count-aware MSE for one grouped residual component."""
    numerators = []
    denominators = []
    for residuals, weights in residual_groups:
        value = getattr(residuals, component)
        if value is None:
            continue
        value = tf.convert_to_tensor(value)
        squared = tf.square(value)
        if weights is None:
            numerators.append(tf.reduce_sum(squared))
            denominators.append(tf.cast(tf.size(squared), squared.dtype))
        else:
            component_weights = tf.cast(weights, squared.dtype)
            numerators.append(tf.reduce_sum(component_weights * squared))
            denominators.append(tf.reduce_sum(component_weights))

    if not numerators:
        return tf.zeros((), dtype=dtype)
    numerator = tf.add_n(numerators)
    denominator = tf.add_n(denominators)
    tf.debugging.assert_positive(
        denominator,
        message=f"Grouped {component} boundary weights must sum to positive.",
    )
    return numerator / denominator
