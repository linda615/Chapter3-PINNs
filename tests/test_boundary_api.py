"""Tests for the standalone boundary API."""

import tensorflow as tf

from boundary import (
    BoundaryBatch,
    BoundaryType,
    ClampedBoundary,
    FreeBoundary,
    SimplySupportedBoundary,
    compute_boundary_loss,
    normal_moment,
    normal_shear,
    normal_slope,
)
from losses.boundary_loss import compute_boundary_loss as compute_legacy_boundary_loss


class BoundaryApiModel(tf.keras.Model):
    """Simple model with zero boundary resultants on x = 0."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        zeros = 0.0 * x
        return {
            "w": x**2 * (1.0 + y),
            "Mx": zeros,
            "My": zeros,
            "Mxy": zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


def test_boundary_condition_classes_return_residuals() -> None:
    """Boundary condition classes should expose residual tensors directly."""
    model = BoundaryApiModel()
    batch = BoundaryBatch(
        xy=tf.constant([[0.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
        normals=tf.constant([[1.0, 0.0], [1.0, 0.0]], dtype=tf.float32),
    )

    simple = SimplySupportedBoundary().compute_residuals(model, batch)
    clamped = ClampedBoundary().compute_residuals(model, batch)
    free = FreeBoundary().compute_residuals(model, batch)

    tf.debugging.assert_near(simple.deflection, tf.zeros_like(simple.deflection))
    tf.debugging.assert_near(simple.moment, tf.zeros_like(simple.moment))
    tf.debugging.assert_near(clamped.deflection, tf.zeros_like(clamped.deflection))
    tf.debugging.assert_near(clamped.slope, tf.zeros_like(clamped.slope))
    tf.debugging.assert_near(free.moment, tf.zeros_like(free.moment))
    tf.debugging.assert_near(free.shear, tf.zeros_like(free.shear))


def test_boundary_api_and_legacy_loss_wrapper_match() -> None:
    """The new BoundaryBatch API and legacy loss wrapper should agree."""
    model = BoundaryApiModel()
    xy = tf.constant([[0.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 0.0], [1.0, 0.0]], dtype=tf.float32)
    batch = BoundaryBatch(xy=xy, normals=normals)

    new_loss = compute_boundary_loss(model, batch, BoundaryType.SIMPLE)
    old_loss = compute_legacy_boundary_loss(model, xy, normals, BoundaryType.SIMPLE)

    for key in new_loss:
        tf.debugging.assert_near(new_loss[key], old_loss[key])


def test_boundary_operator_normal_moment() -> None:
    """Normal moment should combine Mx, My, and Mxy by normal direction."""
    fields = {
        "Mx": tf.constant([[2.0]], dtype=tf.float32),
        "My": tf.constant([[4.0]], dtype=tf.float32),
        "Mxy": tf.constant([[3.0]], dtype=tf.float32),
    }
    normals = tf.constant([[1.0, 0.0]], dtype=tf.float32)

    value = normal_moment(fields, normals)

    tf.debugging.assert_near(value, tf.constant([[2.0]], dtype=tf.float32))


def test_boundary_operators_select_expected_rectangular_edge_components() -> None:
    """Axis-aligned normals should select the matching moment, shear, and slope terms."""
    fields = {
        "Mx": tf.constant([[2.0], [2.0], [2.0], [2.0]], dtype=tf.float32),
        "My": tf.constant([[4.0], [4.0], [4.0], [4.0]], dtype=tf.float32),
        "Mxy": tf.constant([[3.0], [3.0], [3.0], [3.0]], dtype=tf.float32),
        "Qx": tf.constant([[5.0], [5.0], [5.0], [5.0]], dtype=tf.float32),
        "Qy": tf.constant([[6.0], [6.0], [6.0], [6.0]], dtype=tf.float32),
    }
    derivatives = {
        "w_x": tf.constant([[7.0], [7.0], [7.0], [7.0]], dtype=tf.float32),
        "w_y": tf.constant([[8.0], [8.0], [8.0], [8.0]], dtype=tf.float32),
    }
    normals = tf.constant(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=tf.float32,
    )

    tf.debugging.assert_near(
        normal_moment(fields, normals),
        tf.constant([[2.0], [2.0], [4.0], [4.0]], dtype=tf.float32),
    )
    tf.debugging.assert_near(
        normal_shear(fields, normals),
        tf.constant([[-5.0], [5.0], [-6.0], [6.0]], dtype=tf.float32),
    )
    tf.debugging.assert_near(
        normal_slope(derivatives, normals),
        tf.constant([[-7.0], [7.0], [-8.0], [8.0]], dtype=tf.float32),
    )
