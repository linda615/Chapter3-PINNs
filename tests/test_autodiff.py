"""Tests for TensorFlow autodiff helpers."""

import tensorflow as tf

from physics.autodiff import (
    FIELD_NAMES,
    deflection_derivatives,
    deflection_third_derivatives,
    first_derivatives,
)


class AnalyticKirchhoffModel(tf.keras.Model):
    """Simple analytic model with known deflection derivatives."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        # Explicit products keep third derivatives finite at x=0 in TF 2.10.
        w = x * x * x + x * y * y
        zeros = 0.0 * w
        qy = x**2 * y**3
        return {
            "w": w,
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": qy,
        }


def test_deflection_derivatives_match_analytic_solution() -> None:
    """Autodiff derivatives for w should match the closed-form derivatives."""
    model = AnalyticKirchhoffModel()
    xy = tf.constant(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-0.5, 3.0],
            [2.0, -1.0],
        ],
        dtype=tf.float32,
    )
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    result = deflection_derivatives(model, xy)

    expected = {
        "w": x**3 + x * y**2,
        "w_x": 3.0 * x**2 + y**2,
        "w_y": 2.0 * x * y,
        "w_xx": 6.0 * x,
        "w_yy": 2.0 * x,
        "w_xy": 2.0 * y,
    }

    for key, expected_value in expected.items():
        assert result[key].shape == (4, 1)
        tf.debugging.assert_near(result[key], expected_value)


def test_first_derivatives_return_expected_shapes() -> None:
    """First-derivative helper should return all fields and x/y derivatives."""
    model = AnalyticKirchhoffModel()
    xy = tf.ones((5, 2), dtype=tf.float32)

    fields, derivatives = first_derivatives(model, xy)

    assert tuple(fields.keys()) == FIELD_NAMES
    assert tuple(derivatives.keys()) == FIELD_NAMES
    for field_name in FIELD_NAMES:
        assert fields[field_name].shape == (5, 1)
        assert derivatives[field_name]["x"].shape == (5, 1)
        assert derivatives[field_name]["y"].shape == (5, 1)


def test_deflection_third_derivatives_match_analytic_solution() -> None:
    """Third derivatives should match w=x^3+x*y^2 exactly."""
    model = AnalyticKirchhoffModel()
    xy = tf.constant(
        [[0.0, 0.0], [1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        dtype=tf.float32,
    )
    result = deflection_third_derivatives(model, xy)
    expected = {
        "w_xxx": tf.fill((4, 1), 6.0),
        "w_xxy": tf.zeros((4, 1), dtype=tf.float32),
        "w_xyy": tf.fill((4, 1), 2.0),
        "w_yyy": tf.zeros((4, 1), dtype=tf.float32),
    }

    for key, expected_value in expected.items():
        assert result[key].shape == (4, 1)
        tf.debugging.assert_near(result[key], expected_value)


def test_first_derivatives_compute_qy_y_correctly() -> None:
    """Autodiff should compute the y-derivative of the Qy field."""
    model = AnalyticKirchhoffModel()
    xy = tf.constant(
        [
            [1.0, 2.0],
            [-0.5, 3.0],
            [2.0, -1.0],
        ],
        dtype=tf.float32,
    )
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    _, derivatives = first_derivatives(model, xy)

    tf.debugging.assert_near(derivatives["Qy"]["y"], 3.0 * x**2 * y**2)
