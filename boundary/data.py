"""Boundary data containers and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .types import BoundaryType


@dataclass(frozen=True)
class BoundaryBatch:
    """Boundary coordinates, outward normals, and optional point weights."""

    xy: tf.Tensor
    normals: tf.Tensor
    weights: tf.Tensor | None = None

    def validated(self) -> "BoundaryBatch":
        """Return a validated boundary batch with TensorFlow tensors."""
        xy = validate_boundary_xy(self.xy)
        normals = validate_normals(self.normals, xy)
        weights = None if self.weights is None else validate_weights(self.weights, xy)
        return BoundaryBatch(xy=xy, normals=normals, weights=weights)


@dataclass(frozen=True)
class BoundaryConditionBatch:
    """One boundary point batch paired with its boundary-condition type."""

    boundary_type: BoundaryType | str
    batch: BoundaryBatch

    def validated(self) -> "BoundaryConditionBatch":
        """Return a batch with normalized boundary type and validated tensors."""
        return BoundaryConditionBatch(
            boundary_type=BoundaryType(self.boundary_type),
            batch=self.batch.validated(),
        )


@dataclass(frozen=True)
class BoundaryResiduals:
    """Residual tensors for boundary condition terms."""

    deflection: tf.Tensor | None = None
    slope: tf.Tensor | None = None
    moment: tf.Tensor | None = None
    shear: tf.Tensor | None = None


def validate_boundary_xy(boundary_xy: tf.Tensor) -> tf.Tensor:
    """Validate boundary coordinates."""
    boundary_xy = tf.convert_to_tensor(boundary_xy)
    if not boundary_xy.dtype.is_floating:
        raise TypeError(f"boundary_xy must be floating-point, got {boundary_xy.dtype}.")
    if boundary_xy.shape.rank is not None:
        if boundary_xy.shape.rank != 2:
            raise ValueError(f"boundary_xy must be 2D, got rank {boundary_xy.shape.rank}.")
        if boundary_xy.shape[-1] != 2:
            raise ValueError(f"boundary_xy must have shape (batch_size, 2), got {boundary_xy.shape}.")
        return boundary_xy
    tf.debugging.assert_equal(tf.rank(boundary_xy), 2, message="boundary_xy must be 2D.")
    tf.debugging.assert_equal(
        tf.shape(boundary_xy)[-1],
        2,
        message="boundary_xy must have shape (batch_size, 2).",
    )
    return boundary_xy


def validate_normals(normals: tf.Tensor, boundary_xy: tf.Tensor) -> tf.Tensor:
    """Validate boundary normal vectors."""
    normals = tf.convert_to_tensor(normals, dtype=boundary_xy.dtype)
    if normals.shape.rank is not None:
        if normals.shape.rank != 2:
            raise ValueError(f"normals must be 2D, got rank {normals.shape.rank}.")
        if normals.shape[-1] != 2:
            raise ValueError(f"normals must have shape (batch_size, 2), got {normals.shape}.")
    else:
        tf.debugging.assert_equal(tf.rank(normals), 2, message="normals must be 2D.")
        tf.debugging.assert_equal(
            tf.shape(normals)[-1],
            2,
            message="normals must have shape (batch_size, 2).",
        )
    tf.debugging.assert_equal(
        tf.shape(normals)[0],
        tf.shape(boundary_xy)[0],
        message="normals batch size must match boundary_xy batch size.",
    )
    return normals


def validate_weights(weights: tf.Tensor, boundary_xy: tf.Tensor) -> tf.Tensor:
    """Validate optional boundary quadrature or point weights."""
    weights = tf.convert_to_tensor(weights, dtype=boundary_xy.dtype)
    if weights.shape.rank is not None:
        if weights.shape.rank != 2:
            raise ValueError(f"weights must be 2D, got rank {weights.shape.rank}.")
        if weights.shape[-1] != 1:
            raise ValueError(f"weights must have shape (batch_size, 1), got {weights.shape}.")
    else:
        tf.debugging.assert_equal(tf.rank(weights), 2, message="weights must be 2D.")
        tf.debugging.assert_equal(
            tf.shape(weights)[-1],
            1,
            message="weights must have shape (batch_size, 1).",
        )
    tf.debugging.assert_equal(
        tf.shape(weights)[0],
        tf.shape(boundary_xy)[0],
        message="weights batch size must match boundary_xy batch size.",
    )
    return weights
