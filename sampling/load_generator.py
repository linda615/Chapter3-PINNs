"""Load function generators for plate PINN training."""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf


def constant_load(value: float) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create a constant transverse load function ``q(x, y) = value``."""

    def load_fn(xy: tf.Tensor) -> tf.Tensor:
        xy = _validate_xy(xy)
        return tf.ones((tf.shape(xy)[0], 1), dtype=xy.dtype) * tf.cast(value, xy.dtype)

    return load_fn


def sinusoidal_load(
    amplitude: float = 1.0,
    mode_x: int = 1,
    mode_y: int = 1,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create ``q = A sin(mx*pi*x/Lx) sin(my*pi*y/Ly)``."""
    if mode_x <= 0:
        raise ValueError(f"mode_x must be positive, got {mode_x}.")
    if mode_y <= 0:
        raise ValueError(f"mode_y must be positive, got {mode_y}.")
    if length_x <= 0.0:
        raise ValueError(f"length_x must be positive, got {length_x}.")
    if length_y <= 0.0:
        raise ValueError(f"length_y must be positive, got {length_y}.")

    def load_fn(xy: tf.Tensor) -> tf.Tensor:
        xy = _validate_xy(xy)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        pi = tf.cast(tf.constant(3.141592653589793), xy.dtype)
        return (
            tf.cast(amplitude, xy.dtype)
            * tf.sin(tf.cast(mode_x, xy.dtype) * pi * x / tf.cast(length_x, xy.dtype))
            * tf.sin(tf.cast(mode_y, xy.dtype) * pi * y / tf.cast(length_y, xy.dtype))
        )

    return load_fn


def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate coordinate tensor for generated load functions."""
    xy = tf.convert_to_tensor(xy)
    if not xy.dtype.is_floating:
        raise TypeError(f"xy must be floating-point, got {xy.dtype}.")
    if xy.shape.rank is not None:
        if xy.shape.rank != 2:
            raise ValueError(f"xy must be 2D, got rank {xy.shape.rank}.")
        if xy.shape[-1] != 2:
            raise ValueError(f"xy must have shape (batch_size, 2), got {xy.shape}.")
        return xy
    tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be 2D.")
    tf.debugging.assert_equal(
        tf.shape(xy)[-1],
        2,
        message="xy must have shape (batch_size, 2).",
    )
    return xy
