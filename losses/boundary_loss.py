"""Backward-compatible boundary loss wrappers.

Boundary logic lives in the top-level ``boundary`` package. This module keeps
the original loss API stable for existing training and tests.
"""

from __future__ import annotations

import tensorflow as tf

from boundary import (
    BoundaryBatch,
    BoundaryType,
    compute_boundary_loss as _compute_boundary_loss,
    compute_clamped_boundary as _compute_clamped_boundary,
    compute_free_boundary as _compute_free_boundary,
    compute_ss_boundary as _compute_ss_boundary,
)


def compute_boundary_loss(
    model: tf.keras.Model,
    boundary_xy: tf.Tensor,
    normals: tf.Tensor,
    boundary_type: BoundaryType | str,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Dispatch to the boundary package while preserving the old signature."""
    return _compute_boundary_loss(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        boundary_type=boundary_type,
        training=training,
    )


def compute_ss_boundary(
    model: tf.keras.Model,
    boundary_xy: tf.Tensor,
    normals: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute simply supported boundary losses with the old signature."""
    return _compute_ss_boundary(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        training=training,
    )


def compute_clamped_boundary(
    model: tf.keras.Model,
    boundary_xy: tf.Tensor,
    normals: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute clamped boundary losses with the old signature."""
    return _compute_clamped_boundary(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        training=training,
    )


def compute_free_boundary(
    model: tf.keras.Model,
    boundary_xy: tf.Tensor,
    normals: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute free boundary losses with the old signature."""
    return _compute_free_boundary(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        training=training,
    )
