"""Problem definition for the nonuniform-support clamped rock-roof example."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from losses import BoundaryType
from models import (
    DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    MultiSubNetPINN,
)
from sampling import RectangularBoundarySampler, RectangularInteriorSampler

from .config import DEFAULT_CONFIG, ExperimentConfig


def create_model(config: ExperimentConfig = DEFAULT_CONFIG) -> MultiSubNetPINN:
    """Create the scale-decoupled PINN with configured hard constraints."""
    transform_name = (
        DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE
        if config.hard_clamped_slope
        else DEFLECTION_TRANSFORM_UNIT_SQUARE
    )
    return MultiSubNetPINN(
        grouping=config.model_grouping,
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
        activation=config.activation,
        use_shared_trunk=config.use_shared_trunk,
        shared_width=config.shared_width,
        shared_depth=config.shared_depth,
        apply_deflection_transform=True,
        deflection_transform_name=transform_name,
        field_output_scales=config.field_output_scales,
        output_activation=config.output_activation,
    )


def clamped_deflection_transform(xy: tf.Tensor, raw_w: tf.Tensor) -> tf.Tensor:
    """Force ``w = 0`` and ``dw/dn = 0`` on every square edge.

    This transform is applied only to the deflection field. It must not be
    multiplied into ``Mx``, ``My``, ``Mxy``, ``Qx``, or ``Qy`` because those
    fields are independent mixed variables in the current PINN formulation.
    The squared coordinate factors make both the value and first normal
    derivative vanish on every edge.
    """
    xy = tf.convert_to_tensor(xy)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    envelope = tf.square(x * (1.0 - x))
    envelope = envelope * tf.square(y * (1.0 - y))
    return envelope * raw_w


def rock_roof_load(
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tf.Tensor:
    """Return the smooth nonuniform transverse load ``q(x, y)``."""
    xy = _validate_xy(xy)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    dtype = xy.dtype

    local_concentration = tf.cast(config.load_local_amplitude, dtype) * tf.exp(
        -tf.square((x - tf.cast(config.load_center_x, dtype)) / tf.cast(config.load_width_x, dtype))
        -tf.square((y - tf.cast(config.load_center_y, dtype)) / tf.cast(config.load_width_y, dtype))
    )

    return (
        tf.ones_like(x) * tf.cast(config.load_background, dtype)
        + tf.cast(config.load_gradient_x, dtype) * x
        + local_concentration
    )


def foundation_stiffness(
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tf.Tensor:
    """Return the smooth Winkler foundation stiffness ``k(x, y)``."""
    xy = _validate_xy(xy)
    x = xy[:, 0:1]
    dtype = xy.dtype
    stiffness_range = tf.cast(config.foundation_max - config.foundation_min, dtype)
    return (
        tf.cast(config.foundation_min, dtype)
        + stiffness_range
        / (
            tf.ones_like(x)
            + tf.exp(
                (x - tf.cast(config.foundation_transition_x, dtype))
                / tf.cast(config.foundation_transition_width, dtype)
            )
        )
    )


def active_foundation_stiffness(
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tf.Tensor:
    """Return active foundation stiffness, or zeros when it is disabled."""
    xy = _validate_xy(xy)
    if not config.foundation_enabled:
        return tf.zeros((tf.shape(xy)[0], 1), dtype=xy.dtype)
    return foundation_stiffness(xy, config=config)


def create_load_fn(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create the configured load function."""

    def load_fn(xy: tf.Tensor) -> tf.Tensor:
        return rock_roof_load(xy, config=config)

    return load_fn


def create_foundation_fn(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
    """Create the configured foundation function, or ``None`` when disabled."""
    if not config.foundation_enabled:
        return None

    def foundation_fn(xy: tf.Tensor) -> tf.Tensor:
        return active_foundation_stiffness(xy, config=config)

    return foundation_fn


def create_interior_sampler(
    dtype: tf.dtypes.DType = tf.float32,
) -> RectangularInteriorSampler:
    """Create the unit-square interior sampler."""
    return RectangularInteriorSampler(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        dtype=dtype,
    )


def create_boundary_sampler(
    dtype: tf.dtypes.DType = tf.float32,
) -> RectangularBoundarySampler:
    """Create the unit-square boundary sampler."""
    return RectangularBoundarySampler(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        dtype=dtype,
    )


def get_boundary_type(config: ExperimentConfig = DEFAULT_CONFIG) -> BoundaryType:
    """Return the boundary type used by all four edges."""
    return config.boundary_type


def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate coordinate tensor shape and dtype."""
    xy = tf.convert_to_tensor(xy)
    if not xy.dtype.is_floating:
        raise ValueError(f"xy must be a floating Tensor, got {xy.dtype}.")
    if xy.shape.rank is not None:
        if xy.shape.rank != 2:
            raise ValueError(f"xy must be rank 2, got rank {xy.shape.rank}.")
        if xy.shape[-1] != 2:
            raise ValueError(f"xy must have shape (batch_size, 2), got {xy.shape}.")
    else:
        tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be rank 2.")
        tf.debugging.assert_equal(tf.shape(xy)[-1], 2, message="xy must have shape (N, 2).")
    return xy


def main() -> None:
    """Run the problem diagnostic command."""
    from .diagnose import main as diagnose_main

    diagnose_main()


if __name__ == "__main__":
    main()
