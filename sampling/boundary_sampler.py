"""Boundary collocation point samplers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import tensorflow as tf


class BoundarySide(str, Enum):
    """Sides of a rectangular plate boundary."""

    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"


@dataclass(frozen=True)
class RectangularBoundarySampler:
    """Uniform sampler for boundary coordinates and outward normal vectors."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    dtype: tf.dtypes.DType = tf.float32

    def __post_init__(self) -> None:
        """Validate rectangular domain bounds."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min must be smaller than x_max, got {self.x_min}, {self.x_max}.")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min must be smaller than y_max, got {self.y_min}, {self.y_max}.")

    def sample_side(
        self,
        side: BoundarySide | str,
        num_points: int,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample one boundary side and its outward normal vectors."""
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}.")

        side = BoundarySide(side)
        if seed is None:
            t = tf.random.uniform((num_points, 1), dtype=self.dtype)
        else:
            t = tf.random.stateless_uniform(
                (num_points, 1),
                seed=_stateless_seed(seed),
                dtype=self.dtype,
            )

        if side == BoundarySide.LEFT:
            x = tf.fill((num_points, 1), tf.cast(self.x_min, self.dtype))
            y = self.y_min + (self.y_max - self.y_min) * t
            normal = (-1.0, 0.0)
        elif side == BoundarySide.RIGHT:
            x = tf.fill((num_points, 1), tf.cast(self.x_max, self.dtype))
            y = self.y_min + (self.y_max - self.y_min) * t
            normal = (1.0, 0.0)
        elif side == BoundarySide.BOTTOM:
            x = self.x_min + (self.x_max - self.x_min) * t
            y = tf.fill((num_points, 1), tf.cast(self.y_min, self.dtype))
            normal = (0.0, -1.0)
        elif side == BoundarySide.TOP:
            x = self.x_min + (self.x_max - self.x_min) * t
            y = tf.fill((num_points, 1), tf.cast(self.y_max, self.dtype))
            normal = (0.0, 1.0)
        else:
            raise ValueError(f"Unsupported boundary side: {side!r}.")

        xy = tf.concat([x, y], axis=1)
        normals = tf.tile(tf.constant([normal], dtype=self.dtype), [num_points, 1])
        return xy, normals

    def sample_all_sides(
        self,
        points_per_side: int,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample all four sides and concatenate coordinates and normals."""
        if points_per_side <= 0:
            raise ValueError(f"points_per_side must be positive, got {points_per_side}.")

        seeds = [None, None, None, None]
        if seed is not None:
            seeds = [seed + offset for offset in range(4)]

        samples = [
            self.sample_side(BoundarySide.LEFT, points_per_side, seed=seeds[0]),
            self.sample_side(BoundarySide.RIGHT, points_per_side, seed=seeds[1]),
            self.sample_side(BoundarySide.BOTTOM, points_per_side, seed=seeds[2]),
            self.sample_side(BoundarySide.TOP, points_per_side, seed=seeds[3]),
        ]
        xy = tf.concat([item[0] for item in samples], axis=0)
        normals = tf.concat([item[1] for item in samples], axis=0)
        return xy, normals


def _stateless_seed(seed: int) -> tf.Tensor:
    """Convert a Python seed into a TensorFlow stateless RNG seed pair."""
    return tf.constant([int(seed), 0], dtype=tf.int32)
