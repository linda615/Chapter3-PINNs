"""Tests for optional Winkler foundation support in plate residuals."""

from __future__ import annotations

import tensorflow as tf

from physics.plate_residuals import PlateParameters, compute_plate_residuals


class LinearDeflectionModel(tf.keras.Model):
    """Simple deterministic model with nonzero deflection."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        del training
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        zeros = 0.0 * x * y
        return {
            "w": x * x + y * y + x * y,
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


def test_foundation_none_keeps_original_equilibrium_residual() -> None:
    """Without foundation, equilibrium remains Qx_x + Qy_y + q."""
    xy = tf.constant([[0.2, 0.3], [0.7, 0.4]], dtype=tf.float32)
    model = LinearDeflectionModel()
    parameters = PlateParameters(D=1.0, nu=0.25)

    residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=lambda points: tf.ones((tf.shape(points)[0], 1), dtype=points.dtype),
        parameters=parameters,
        foundation_fn=None,
    )

    tf.debugging.assert_near(residuals["equilibrium"], tf.ones((2, 1), dtype=tf.float32))


def test_zero_foundation_matches_no_foundation() -> None:
    """A zero foundation tensor must be equivalent to foundation_fn=None."""
    xy = tf.constant([[0.2, 0.3], [0.7, 0.4]], dtype=tf.float32)
    model = LinearDeflectionModel()
    parameters = PlateParameters(D=1.0, nu=0.25)

    load_fn = lambda points: tf.ones((tf.shape(points)[0], 1), dtype=points.dtype)
    old = compute_plate_residuals(model, xy, load_fn, parameters, foundation_fn=None)
    new = compute_plate_residuals(
        model,
        xy,
        load_fn,
        parameters,
        foundation_fn=lambda points: tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype),
    )

    tf.debugging.assert_near(new["equilibrium"], old["equilibrium"])


def test_foundation_equilibrium_adds_negative_k_times_w() -> None:
    """The foundation contribution must satisfy new_r_eq - old_r_eq = -k*w."""
    xy = tf.constant([[0.2, 0.3], [0.7, 0.4]], dtype=tf.float32)
    model = LinearDeflectionModel()
    parameters = PlateParameters(D=1.0, nu=0.25)

    load_fn = lambda points: tf.ones((tf.shape(points)[0], 1), dtype=points.dtype)
    foundation_fn = lambda points: 3.0 * tf.ones((tf.shape(points)[0], 1), dtype=points.dtype)
    old = compute_plate_residuals(model, xy, load_fn, parameters, foundation_fn=None)
    new = compute_plate_residuals(model, xy, load_fn, parameters, foundation_fn=foundation_fn)

    fields = model(xy)
    tf.debugging.assert_near(new["equilibrium"] - old["equilibrium"], -foundation_fn(xy) * fields["w"])


def test_foundation_rejects_negative_values() -> None:
    """Foundation stiffness must be non-negative."""
    xy = tf.constant([[0.2, 0.3]], dtype=tf.float32)
    model = LinearDeflectionModel()
    parameters = PlateParameters(D=1.0, nu=0.25)

    try:
        compute_plate_residuals(
            model=model,
            xy=xy,
            load_fn=lambda points: tf.ones((tf.shape(points)[0], 1), dtype=points.dtype),
            parameters=parameters,
            foundation_fn=lambda points: -tf.ones((tf.shape(points)[0], 1), dtype=points.dtype),
        )
    except tf.errors.InvalidArgumentError:
        return
    raise AssertionError("Negative foundation stiffness should raise InvalidArgumentError.")
