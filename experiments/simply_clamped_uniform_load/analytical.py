"""Levy-series reference for a uniformly loaded mixed-edge plate."""

from __future__ import annotations

import math

import tensorflow as tf

from physics.plate_residuals import PlateParameters


FIELD_NAMES = ("w", "Mx", "My", "Mxy", "Qx", "Qy")


def analytical_fields(
    xy: tf.Tensor,
    parameters: PlateParameters,
    q0: float = 1.0,
    mode_count: int = 40,
) -> dict[str, tf.Tensor]:
    """Return the six Levy-series fields on the unit square.

    The edges ``x=0,1`` are simply supported and ``y=0,1`` are clamped.
    Uniform loading is expanded in odd sine modes along the x direction.
    """
    xy = _validated_xy(xy)
    if mode_count <= 0:
        raise ValueError(f"mode_count must be positive, got {mode_count}.")

    dtype = xy.dtype
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    pi = tf.cast(math.pi, dtype)
    D = tf.cast(parameters.D, dtype)
    nu = tf.cast(parameters.nu, dtype)
    q0_tensor = tf.cast(q0, dtype)

    odd_modes = tf.cast(
        tf.range(1, 2 * mode_count, delta=2),
        dtype,
    )[tf.newaxis, :]
    alpha = pi * odd_modes
    q_modes = 4.0 * q0_tensor / (pi * odd_modes)
    particular = q_modes / (D * tf.pow(alpha, 4))

    z = y - tf.cast(0.5, dtype)
    half_alpha = 0.5 * alpha
    exp_minus_two_half_alpha = tf.exp(-2.0 * half_alpha)
    sinh_half_scaled = 0.5 * (1.0 - exp_minus_two_half_alpha)
    cosh_half_scaled = 0.5 * (1.0 + exp_minus_two_half_alpha)
    denominator_scaled = (
        cosh_half_scaled * sinh_half_scaled
        + half_alpha * exp_minus_two_half_alpha
    )
    coefficient_a_scaled = (
        -particular
        * (
            sinh_half_scaled
            + half_alpha * cosh_half_scaled
        )
        / denominator_scaled
    )
    coefficient_b_scaled = (
        alpha
        * sinh_half_scaled
        * particular
        / denominator_scaled
    )

    alpha_z = alpha * z
    # Every exponent is non-positive because abs(alpha * z) <= alpha / 2.
    # This scaled form is algebraically identical to multiplying sinh/cosh
    # by exp(-alpha / 2), but remains finite for high Levy modes.
    exp_positive_z_scaled = tf.exp(alpha_z - half_alpha)
    exp_negative_z_scaled = tf.exp(-alpha_z - half_alpha)
    sinh_z_scaled = 0.5 * (
        exp_positive_z_scaled - exp_negative_z_scaled
    )
    cosh_z_scaled = 0.5 * (
        exp_positive_z_scaled + exp_negative_z_scaled
    )
    w_mode = (
        particular
        + coefficient_a_scaled * cosh_z_scaled
        + coefficient_b_scaled * z * sinh_z_scaled
    )
    w_y_mode = (
        coefficient_a_scaled * alpha * sinh_z_scaled
        + coefficient_b_scaled
        * (
            sinh_z_scaled
            + alpha * z * cosh_z_scaled
        )
    )
    w_yy_mode = (
        coefficient_a_scaled * tf.square(alpha) * cosh_z_scaled
        + coefficient_b_scaled
        * (
            2.0 * alpha * cosh_z_scaled
            + tf.square(alpha) * z * sinh_z_scaled
        )
    )
    w_yyy_mode = (
        coefficient_a_scaled * tf.pow(alpha, 3) * sinh_z_scaled
        + coefficient_b_scaled
        * (
            3.0 * tf.square(alpha) * sinh_z_scaled
            + tf.pow(alpha, 3) * z * cosh_z_scaled
        )
    )

    sin_x = tf.sin(alpha * x)
    cos_x = tf.cos(alpha * x)
    alpha_squared = tf.square(alpha)

    w = tf.reduce_sum(w_mode * sin_x, axis=1, keepdims=True)
    Mx = D * tf.reduce_sum(
        (alpha_squared * w_mode - nu * w_yy_mode) * sin_x,
        axis=1,
        keepdims=True,
    )
    My = D * tf.reduce_sum(
        (nu * alpha_squared * w_mode - w_yy_mode) * sin_x,
        axis=1,
        keepdims=True,
    )
    Mxy = -D * (1.0 - nu) * tf.reduce_sum(
        alpha * w_y_mode * cos_x,
        axis=1,
        keepdims=True,
    )
    Qx = D * tf.reduce_sum(
        alpha * (alpha_squared * w_mode - w_yy_mode) * cos_x,
        axis=1,
        keepdims=True,
    )
    Qy = D * tf.reduce_sum(
        (alpha_squared * w_y_mode - w_yyy_mode) * sin_x,
        axis=1,
        keepdims=True,
    )
    return {
        "w": w,
        "Mx": Mx,
        "My": My,
        "Mxy": Mxy,
        "Qx": Qx,
        "Qy": Qy,
    }


def levy_series_load(
    xy: tf.Tensor,
    q0: float = 1.0,
    mode_count: int = 20,
) -> tf.Tensor:
    """Return the truncated odd sine expansion used by the Levy solution."""
    xy = _validated_xy(xy)
    if mode_count <= 0:
        raise ValueError(f"mode_count must be positive, got {mode_count}.")
    dtype = xy.dtype
    x = xy[:, 0:1]
    pi = tf.cast(math.pi, dtype)
    odd_modes = tf.cast(
        tf.range(1, 2 * mode_count, delta=2),
        dtype,
    )[tf.newaxis, :]
    coefficients = 4.0 * tf.cast(q0, dtype) / (pi * odd_modes)
    return tf.reduce_sum(
        coefficients * tf.sin(pi * odd_modes * x),
        axis=1,
        keepdims=True,
    )


class AnalyticalPlateModel(tf.keras.Model):
    """TensorFlow model wrapper around the Levy-series reference."""

    def __init__(
        self,
        parameters: PlateParameters,
        q0: float = 1.0,
        mode_count: int = 20,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.q0 = q0
        self.mode_count = mode_count

    def call(
        self,
        xy: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Return analytical fields using graph-compatible operations."""
        del training
        return analytical_fields(
            xy=xy,
            parameters=self.parameters,
            q0=self.q0,
            mode_count=self.mode_count,
        )


def _validated_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate coordinates while preserving TensorFlow graph execution."""
    xy = tf.convert_to_tensor(xy)
    if not xy.dtype.is_floating:
        raise TypeError(f"xy must be floating-point, got {xy.dtype}.")
    if xy.shape.rank is not None:
        if xy.shape.rank != 2:
            raise ValueError(f"xy must be 2D, got rank {xy.shape.rank}.")
        if xy.shape[-1] != 2:
            raise ValueError(
                f"xy must have shape (batch_size, 2), got {xy.shape}."
            )
        return xy
    tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be 2D.")
    tf.debugging.assert_equal(
        tf.shape(xy)[1],
        2,
        message="xy must have shape (batch_size, 2).",
    )
    return xy


__all__ = [
    "AnalyticalPlateModel",
    "FIELD_NAMES",
    "analytical_fields",
    "levy_series_load",
]
