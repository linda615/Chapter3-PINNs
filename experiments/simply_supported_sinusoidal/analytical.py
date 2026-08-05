"""Analytical solution for a simply supported sinusoidal Kirchhoff plate."""

from __future__ import annotations

import tensorflow as tf

from physics.plate_residuals import PlateParameters


def amplitude(parameters: PlateParameters) -> tf.Tensor:
    """Return ``A`` in ``w = A sin(pi x) sin(pi y)`` for q = sin(pi x) sin(pi y)."""
    pi = tf.constant(3.141592653589793, dtype=tf.float32)
    return 1.0 / (4.0 * tf.cast(parameters.D, tf.float32) * pi**4)


def analytical_fields(
    xy: tf.Tensor,
    parameters: PlateParameters,
) -> dict[str, tf.Tensor]:
    """Return analytical ``w, Mx, My, Mxy, Qx, Qy`` matching project signs."""
    xy = tf.convert_to_tensor(xy)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    dtype = xy.dtype
    pi = tf.cast(tf.constant(3.141592653589793), dtype)
    D = tf.cast(parameters.D, dtype)
    nu = tf.cast(parameters.nu, dtype)
    a = tf.cast(amplitude(parameters), dtype)

    sin_x = tf.sin(pi * x)
    sin_y = tf.sin(pi * y)
    cos_x = tf.cos(pi * x)
    cos_y = tf.cos(pi * y)

    w = a * sin_x * sin_y
    mx = D * pi**2 * (1.0 + nu) * w
    my = D * pi**2 * (1.0 + nu) * w
    mxy = -D * (1.0 - nu) * a * pi**2 * cos_x * cos_y
    qx = 2.0 * D * a * pi**3 * cos_x * sin_y
    qy = 2.0 * D * a * pi**3 * sin_x * cos_y

    return {
        "w": w,
        "Mx": mx,
        "My": my,
        "Mxy": mxy,
        "Qx": qx,
        "Qy": qy,
    }


class AnalyticalPlateModel(tf.keras.Model):
    """TensorFlow model wrapper around the analytical solution."""

    def __init__(self, parameters: PlateParameters) -> None:
        super().__init__()
        self.parameters = parameters

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        """Return analytical fields in the same format as ``MultiSubNetPINN``."""
        return analytical_fields(xy, self.parameters)
