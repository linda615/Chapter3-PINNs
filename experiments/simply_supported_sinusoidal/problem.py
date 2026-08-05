"""Problem setup for the simply supported sinusoidal plate example."""

from __future__ import annotations

import tensorflow as tf

from losses import BoundaryType
from models import DEFLECTION_TRANSFORM_UNIT_SQUARE, MultiSubNetPINN
from sampling import RectangularBoundarySampler, RectangularInteriorSampler, sinusoidal_load

from .config import DEFAULT_CONFIG, ExperimentConfig


def create_model(config: ExperimentConfig = DEFAULT_CONFIG) -> MultiSubNetPINN:
    """Create the configured PINN with hard deflection boundary transform."""
    return MultiSubNetPINN(
        grouping=config.model_grouping,
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
        activation=config.activation,
        use_shared_trunk=config.use_shared_trunk,
        shared_width=config.shared_width,
        shared_depth=config.shared_depth,
        apply_deflection_transform=True,
        deflection_transform_name=DEFLECTION_TRANSFORM_UNIT_SQUARE,
    )


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


def create_load_fn():
    """Create ``q(x, y) = sin(pi*x) sin(pi*y)`` on the unit square."""
    return sinusoidal_load(
        amplitude=1.0,
        mode_x=1,
        mode_y=1,
        length_x=1.0,
        length_y=1.0,
    )


def get_boundary_type(config: ExperimentConfig = DEFAULT_CONFIG) -> BoundaryType:
    """Return the boundary type used by all four edges."""
    return config.boundary_type
