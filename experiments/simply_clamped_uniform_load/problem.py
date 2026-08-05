"""Problem construction for the mixed-edge uniform-load plate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import tensorflow as tf

from boundary import BoundaryBatch, BoundaryConditionBatch, BoundaryType
from models import (
    DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
    DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    MOMENT_TRANSFORM_NONE,
    MOMENT_TRANSFORM_SIMPLE_X_EDGES,
    MOMENT_TRANSFORM_SIMPLE_Y_EDGES,
    MultiSubNetPINN,
)
from sampling import (
    BoundarySide,
    RectangularBoundarySampler,
    RectangularInteriorSampler,
    constant_load,
)

from .analytical import levy_series_load
from .config import (
    DEFAULT_CONFIG,
    LOAD_MODE_MATCHED_LEVY,
    LOAD_MODE_UNIFORM,
    ExperimentConfig,
)


def create_model(config: ExperimentConfig = DEFAULT_CONFIG) -> MultiSubNetPINN:
    """Create a mixed PINN with the configured hard edge constraints."""
    moment_transform_name = _simple_moment_transform_name(config)
    deflection_transform_name = _deflection_transform_name(config)
    return MultiSubNetPINN(
        grouping=config.model_grouping,
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
        activation=config.activation,
        use_shared_trunk=config.use_shared_trunk,
        shared_width=config.shared_width,
        shared_depth=config.shared_depth,
        apply_deflection_transform=True,
        deflection_transform_name=deflection_transform_name,
        apply_moment_transform=config.hard_simple_moment,
        moment_transform_name=moment_transform_name,
        field_output_scales=config.field_output_scales,
        output_activation=config.output_activation,
    )


def create_interior_sampler(
    dtype: tf.dtypes.DType = tf.float32,
) -> RectangularInteriorSampler:
    """Create the unit-square interior sampler."""
    return RectangularInteriorSampler(dtype=dtype)


def create_boundary_sampler(
    dtype: tf.dtypes.DType = tf.float32,
) -> RectangularBoundarySampler:
    """Create the unit-square boundary sampler."""
    return RectangularBoundarySampler(dtype=dtype)


def create_load_fn(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the configured constant or Levy-matched transverse load."""
    if config.load_mode == LOAD_MODE_UNIFORM:
        return constant_load(config.q0)
    if config.load_mode == LOAD_MODE_MATCHED_LEVY:
        return lambda xy: levy_series_load(
            xy,
            q0=config.q0,
            mode_count=config.analytical_modes,
        )
    raise ValueError(f"Unsupported load_mode {config.load_mode!r}.")


def sample_boundary_groups(
    sampler: RectangularBoundarySampler,
    points_per_side: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
    seed: int | None = None,
) -> tuple[BoundaryConditionBatch, BoundaryConditionBatch]:
    """Sample simple and clamped opposite-edge batches independently."""
    simple_batch = _sample_sides(
        sampler,
        config.simple_sides,
        points_per_side,
        seed,
    )
    clamped_batch = _sample_sides(
        sampler,
        config.clamped_sides,
        points_per_side,
        None if seed is None else seed + 1000,
    )
    return (
        BoundaryConditionBatch(BoundaryType.SIMPLE, simple_batch),
        BoundaryConditionBatch(BoundaryType.CLAMPED, clamped_batch),
    )


def _sample_sides(
    sampler: RectangularBoundarySampler,
    sides: Sequence[BoundarySide],
    points_per_side: int,
    seed: int | None,
) -> BoundaryBatch:
    """Concatenate sampled coordinates and normals for selected sides."""
    samples = []
    for index, side in enumerate(sides):
        side_seed = None if seed is None else seed + index
        samples.append(
            sampler.sample_side(
                side,
                points_per_side,
                seed=side_seed,
            )
        )
    return BoundaryBatch(
        xy=tf.concat([sample[0] for sample in samples], axis=0),
        normals=tf.concat([sample[1] for sample in samples], axis=0),
    )


def _simple_moment_transform_name(config: ExperimentConfig) -> str:
    """Select the normal-moment transform for the configured simple edges."""
    if not config.hard_simple_moment:
        return MOMENT_TRANSFORM_NONE
    simple_sides = {BoundarySide(side) for side in config.simple_sides}
    if simple_sides == {BoundarySide.LEFT, BoundarySide.RIGHT}:
        return MOMENT_TRANSFORM_SIMPLE_X_EDGES
    if simple_sides == {BoundarySide.BOTTOM, BoundarySide.TOP}:
        return MOMENT_TRANSFORM_SIMPLE_Y_EDGES
    raise ValueError(
        "Hard simple-moment transforms require one pair of opposite, "
        "axis-aligned simply supported edges."
    )


def _deflection_transform_name(config: ExperimentConfig) -> str:
    """Select a hard w transform matching the configured clamped edges."""
    if not config.hard_clamped_slope:
        return DEFLECTION_TRANSFORM_UNIT_SQUARE
    clamped_sides = {BoundarySide(side) for side in config.clamped_sides}
    if clamped_sides == {BoundarySide.BOTTOM, BoundarySide.TOP}:
        return DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y
    if clamped_sides == {BoundarySide.LEFT, BoundarySide.RIGHT}:
        return DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y
    raise ValueError(
        "Hard clamped-slope transforms require one pair of opposite, "
        "axis-aligned clamped edges."
    )


__all__ = [
    "create_boundary_sampler",
    "create_interior_sampler",
    "create_load_fn",
    "create_model",
    "sample_boundary_groups",
]
