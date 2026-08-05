"""Tests for residual-based physics loss aggregation."""

import pytest
import tensorflow as tf

from losses.normalization import (
    PhysicsComponentWeights,
    PhysicsLossConfig,
    ResidualScales,
)
from losses.physics_loss import compute_physics_loss
from physics.plate_residuals import PlateParameters


class ZeroResidualPlateModel(tf.keras.Model):
    """Analytic plate fields that make all implemented residuals vanish."""

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


class QyBiasedPlateModel(ZeroResidualPlateModel):
    """Analytic plate fields with a constant Qy residual."""

    def __init__(self, parameters: PlateParameters, qy_bias: float) -> None:
        super().__init__(parameters)
        self.qy_bias = qy_bias

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        fields = super().call(xy, training=training)
        fields["Qy"] = fields["Qy"] + tf.cast(self.qy_bias, xy.dtype)
        return fields


class MomentBiasedPlateModel(ZeroResidualPlateModel):
    """Analytic plate fields with independent constant moment residuals."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        fields = super().call(xy, training=training)
        fields["Mx"] = fields["Mx"] + tf.cast(1.0, xy.dtype)
        fields["My"] = fields["My"] + tf.cast(2.0, xy.dtype)
        fields["Mxy"] = fields["Mxy"] + tf.cast(3.0, xy.dtype)
        return fields


def test_physics_loss_is_zero_for_zero_residual_model() -> None:
    """All loss components should vanish when all residuals vanish."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    model = ZeroResidualPlateModel(parameters)
    xy = tf.constant(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 0.5],
            [-0.5, 3.0],
        ],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    losses = compute_physics_loss(model, xy, load_fn, parameters)

    assert tuple(losses.keys()) == (
        "moment_x_weight",
        "moment_y_weight",
        "twisting_moment_weight",
        "shear_x_weight",
        "shear_y_weight",
        "moment_x_loss",
        "moment_y_loss",
        "twisting_moment_loss",
        "weighted_moment_x_loss",
        "weighted_moment_y_loss",
        "weighted_twisting_moment_loss",
        "moment_loss",
        "shear_x_loss",
        "shear_y_loss",
        "weighted_shear_x_loss",
        "weighted_shear_y_loss",
        "shear_loss",
        "equilibrium_loss",
        "physics_loss",
        "mxy_consistency_loss",
        "mxy_consistency_scale",
    )
    weight_keys = (
        "moment_x_weight",
        "moment_y_weight",
        "twisting_moment_weight",
        "shear_x_weight",
        "shear_y_weight",
    )
    for key in weight_keys:
        tf.debugging.assert_near(losses[key], tf.constant(1.0))
    for key, loss_value in losses.items():
        assert loss_value.shape == ()
        if key in weight_keys:
            continue
        tf.debugging.assert_near(loss_value, tf.constant(0.0, dtype=loss_value.dtype))


def test_qy_residual_contributes_to_shear_loss() -> None:
    """A pure Qy offset should enter shear_loss through the shear_y residual."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    qy_bias = 0.25
    model = QyBiasedPlateModel(parameters, qy_bias=qy_bias)
    xy = tf.constant(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 0.5],
            [-0.5, 3.0],
        ],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    losses = compute_physics_loss(model, xy, load_fn, parameters)

    expected_shear_loss = tf.constant(qy_bias**2, dtype=xy.dtype)
    tf.debugging.assert_near(losses["moment_loss"], tf.constant(0.0, dtype=xy.dtype))
    tf.debugging.assert_near(losses["moment_x_loss"], tf.constant(0.0, dtype=xy.dtype))
    tf.debugging.assert_near(losses["moment_y_loss"], tf.constant(0.0, dtype=xy.dtype))
    tf.debugging.assert_near(
        losses["twisting_moment_loss"],
        tf.constant(0.0, dtype=xy.dtype),
    )
    tf.debugging.assert_near(losses["shear_x_loss"], tf.constant(0.0, dtype=xy.dtype))
    tf.debugging.assert_near(losses["shear_y_loss"], expected_shear_loss)
    tf.debugging.assert_near(losses["shear_loss"], expected_shear_loss)
    tf.debugging.assert_near(losses["equilibrium_loss"], tf.constant(0.0, dtype=xy.dtype))


def test_moment_residual_components_are_reported_separately() -> None:
    """Each moment residual MSE should be exposed before aggregation."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    model = MomentBiasedPlateModel(parameters)
    xy = tf.constant(
        [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5], [-0.5, 3.0]],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    losses = compute_physics_loss(model, xy, load_fn, parameters)

    tf.debugging.assert_near(losses["moment_x_loss"], tf.constant(1.0))
    tf.debugging.assert_near(losses["moment_y_loss"], tf.constant(4.0))
    tf.debugging.assert_near(losses["twisting_moment_loss"], tf.constant(9.0))
    tf.debugging.assert_near(losses["moment_loss"], tf.constant(14.0))


def test_component_weights_default_to_one() -> None:
    """Moment and shear residual component weights should default to one."""
    weights = PhysicsComponentWeights()

    assert weights.moment_x == 1.0
    assert weights.moment_y == 1.0
    assert weights.twisting_moment == 1.0
    assert weights.shear_x == 1.0
    assert weights.shear_y == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moment_x": -1.0},
        {"moment_y": float("nan")},
        {"twisting_moment": float("inf")},
        {"shear_x": float("-inf")},
        {"shear_y": -0.1},
    ],
)
def test_component_weights_reject_invalid_values(kwargs) -> None:
    """Component weights must be finite and non-negative."""
    with pytest.raises(ValueError):
        PhysicsComponentWeights(**kwargs)


def test_component_weights_change_moment_aggregation() -> None:
    """Configured component weights should scale each raw moment MSE."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    model = MomentBiasedPlateModel(parameters)
    xy = tf.constant(
        [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5], [-0.5, 3.0]],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    losses = compute_physics_loss(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        loss_config=PhysicsLossConfig(
            component_weights=PhysicsComponentWeights(
                moment_x=2.0,
                moment_y=3.0,
                twisting_moment=4.0,
            )
        ),
    )

    tf.debugging.assert_near(losses["weighted_moment_x_loss"], tf.constant(2.0))
    tf.debugging.assert_near(losses["weighted_moment_y_loss"], tf.constant(12.0))
    tf.debugging.assert_near(
        losses["weighted_twisting_moment_loss"],
        tf.constant(36.0),
    )
    tf.debugging.assert_near(losses["moment_loss"], tf.constant(50.0))


def test_component_weights_change_shear_aggregation() -> None:
    """Configured Qy weight should scale its contribution to shear loss."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    qy_bias = 0.25
    model = QyBiasedPlateModel(parameters, qy_bias=qy_bias)
    xy = tf.constant(
        [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5], [-0.5, 3.0]],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    losses = compute_physics_loss(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        loss_config=PhysicsLossConfig(
            component_weights=PhysicsComponentWeights(shear_y=4.0)
        ),
    )

    tf.debugging.assert_near(losses["shear_y_loss"], tf.constant(qy_bias**2))
    tf.debugging.assert_near(
        losses["weighted_shear_y_loss"],
        tf.constant(4.0 * qy_bias**2),
    )
    tf.debugging.assert_near(
        losses["shear_loss"],
        tf.constant(4.0 * qy_bias**2),
    )


def test_normalized_physics_loss_scales_residual_components() -> None:
    """Normalized mode should divide residuals by configured component scales."""
    parameters = PlateParameters(D=3.0, nu=0.2)
    qy_bias = 0.25
    model = QyBiasedPlateModel(parameters, qy_bias=qy_bias)
    xy = tf.constant(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 0.5],
            [-0.5, 3.0],
        ],
        dtype=tf.float32,
    )

    def load_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros((tf.shape(points)[0], 1), dtype=points.dtype)

    loss_config = PhysicsLossConfig(
        normalize_residuals=True,
        residual_scales=ResidualScales(shear_y=0.5),
    )
    losses = compute_physics_loss(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        loss_config=loss_config,
    )

    expected_shear_y_loss = tf.constant((qy_bias / 0.5) ** 2, dtype=xy.dtype)
    tf.debugging.assert_near(losses["shear_x_loss"], tf.constant(0.0, dtype=xy.dtype))
    tf.debugging.assert_near(losses["shear_y_loss"], expected_shear_y_loss)
    tf.debugging.assert_near(losses["shear_loss"], expected_shear_y_loss)
