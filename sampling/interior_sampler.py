"""Interior collocation point samplers."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class RectangularInteriorSampler:
    """Uniform sampler for points inside a rectangular plate domain."""

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

    def sample(self, num_points: int, seed: int | None = None) -> tf.Tensor:
        """Sample ``num_points`` interior coordinates with shape ``(N, 2)``."""
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}.")

        if seed is None:
            samples = tf.random.uniform((num_points, 2), dtype=self.dtype)
        else:
            samples = tf.random.stateless_uniform(
                (num_points, 2),
                seed=_stateless_seed(seed),
                dtype=self.dtype,
            )
        x = self.x_min + (self.x_max - self.x_min) * samples[:, 0:1]
        y = self.y_min + (self.y_max - self.y_min) * samples[:, 1:2]
        return tf.concat([x, y], axis=1)


def _stateless_seed(seed: int) -> tf.Tensor:
    """Convert a Python seed into a TensorFlow stateless RNG seed pair."""
    return tf.constant([int(seed), 0], dtype=tf.int32)
