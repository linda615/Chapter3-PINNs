"""Tests for optional boundary-residual normalization."""

import tensorflow as tf

from boundary import (
    BoundaryLossConfig,
    BoundaryResidualScales,
    BoundaryResiduals,
    boundary_residual_loss,
)


def test_boundary_residual_scales_normalize_each_component() -> None:
    """Each available boundary residual uses its own characteristic scale."""
    residuals = BoundaryResiduals(
        deflection=tf.constant(((2.0,), (4.0,))),
        slope=tf.constant(((3.0,), (6.0,))),
        moment=tf.constant(((5.0,), (10.0,))),
        shear=tf.constant(((7.0,), (14.0,))),
    )
    config = BoundaryLossConfig(
        normalize_residuals=True,
        residual_scales=BoundaryResidualScales(
            deflection=2.0,
            slope=3.0,
            moment=5.0,
            shear=7.0,
        ),
    )

    losses = boundary_residual_loss(residuals, loss_config=config)

    expected = tf.constant(2.5)
    for name in (
        "deflection_bc_loss",
        "slope_bc_loss",
        "moment_bc_loss",
        "shear_bc_loss",
    ):
        tf.debugging.assert_near(losses[name], expected)
    tf.debugging.assert_near(losses["boundary_loss"], 4.0 * expected)


def test_boundary_scaling_is_disabled_by_default() -> None:
    """The legacy raw-MSE behavior remains the default."""
    residuals = BoundaryResiduals(slope=tf.constant(((2.0,), (4.0,))))

    losses = boundary_residual_loss(residuals)

    tf.debugging.assert_near(losses["slope_bc_loss"], tf.constant(10.0))
