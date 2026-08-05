"""Tests for boundary loss dispatch and boundary condition families."""

import pytest
import tensorflow as tf

from losses.boundary_loss import (
    BoundaryType,
    compute_boundary_loss,
    compute_clamped_boundary,
    compute_free_boundary,
    compute_ss_boundary,
)


class SimpleBoundaryModel(tf.keras.Model):
    """Analytic fields satisfying w = 0 and M_n = 0 on x = 0."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        zeros = 0.0 * x
        return {
            "w": x * (1.0 + y),
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


class ClampedBoundaryModel(tf.keras.Model):
    """Analytic fields satisfying w = 0 and dw/dn = 0 on x = 0."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        w = x**2 * (1.0 + y)
        zeros = 0.0 * w
        return {
            "w": w,
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


class FreeBoundaryModel(tf.keras.Model):
    """Analytic fields satisfying M_n = 0 and Q_n = 0 on x = 0."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        zeros = 0.0 * x
        return {
            "w": 1.0 + x + y,
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


def _boundary_points() -> tuple[tf.Tensor, tf.Tensor]:
    xy = tf.constant([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=tf.float32)
    return xy, normals


@pytest.mark.parametrize(
    ("boundary_type", "model", "direct_fn"),
    [
        (BoundaryType.SIMPLE, SimpleBoundaryModel(), compute_ss_boundary),
        (BoundaryType.CLAMPED, ClampedBoundaryModel(), compute_clamped_boundary),
        (BoundaryType.FREE, FreeBoundaryModel(), compute_free_boundary),
    ],
)
def test_boundary_dispatch_matches_direct_function(boundary_type, model, direct_fn) -> None:
    """The dispatcher should call the same implementation as the direct helper."""
    xy, normals = _boundary_points()

    dispatched = compute_boundary_loss(model, xy, normals, boundary_type)
    direct = direct_fn(model, xy, normals)

    assert tuple(dispatched.keys()) == (
        "deflection_bc_loss",
        "slope_bc_loss",
        "moment_bc_loss",
        "shear_bc_loss",
        "boundary_loss",
    )
    for key in dispatched:
        tf.debugging.assert_near(dispatched[key], direct[key])
        tf.debugging.assert_near(dispatched[key], tf.constant(0.0, dtype=dispatched[key].dtype))
