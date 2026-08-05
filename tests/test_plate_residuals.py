"""Tests for Kirchhoff thin plate residual construction."""

import tensorflow as tf

from physics.plate_residuals import PlateParameters, compute_plate_residuals


class ZeroResidualPlateModel(tf.keras.Model):
    """Analytic plate fields that satisfy the implemented residual equations."""

    def __init__(self, parameters: PlateParameters) -> None:
        super().__init__()
        self.parameters = parameters

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        D = tf.cast(self.parameters.D, xy.dtype)
        nu = tf.cast(self.parameters.nu, xy.dtype)

        w = x**2 + y**2
        mx = -2.0 * D * (1.0 + nu) + 0.0 * x
        my = -2.0 * D * (1.0 + nu) + 0.0 * x
        mxy = 0.0 * x * y
        qx = 0.0 * x
        qy = 0.0 * y

        return {
            "w": w,
            "Mx": mx,
            "My": my,
            "Mxy": mxy,
            "Qx": qx,
            "Qy": qy,
        }


def test_plate_residuals_are_zero_for_analytic_solution() -> None:
    """The residual helper should vanish for a consistent analytic model."""
    parameters = PlateParameters(D=2.5, nu=0.25)
    model = ZeroResidualPlateModel(parameters)
    xy = tf.constant(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.5, 0.5],
            [2.0, -3.0],
        ],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    residuals = compute_plate_residuals(model, xy, load_fn, parameters)

    assert tuple(residuals.keys()) == (
        "moment_x",
        "moment_y",
        "twisting_moment",
        "shear_x",
        "shear_y",
        "equilibrium",
    )
    for residual in residuals.values():
        assert residual.shape == (4, 1)
        tf.debugging.assert_near(residual, tf.zeros_like(residual))
