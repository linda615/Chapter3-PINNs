"""Tests for one-way normalized Mxy consistency."""

import pytest
import tensorflow as tf

from boundary import BoundaryBatch, BoundaryType
from losses import (
    LossWeights,
    MxyConsistencyConfig,
    compute_mxy_consistency_loss,
    compute_total_loss,
)
from physics.plate_residuals import PlateParameters, PlateResidualState
from sampling import constant_load


def _state(mxy_pred: tf.Tensor, w_xy: tf.Tensor) -> PlateResidualState:
    """Build the minimal state required by the consistency loss."""
    return PlateResidualState(
        fields={"Mxy": mxy_pred},
        field_derivatives={},
        deflection_derivatives={"w_xy": w_xy},
        residuals={},
    )


def test_mxy_consistency_uses_detached_target_and_rms_scale() -> None:
    """The target and its RMS scale must not receive consistency gradients."""
    mxy_pred = tf.Variable([[0.5], [0.5]], dtype=tf.float32)
    w_xy = tf.Variable([[2.0], [2.0]], dtype=tf.float32)
    config = MxyConsistencyConfig(enabled=True, minimum_scale=1e-3)

    with tf.GradientTape() as tape:
        losses = compute_mxy_consistency_loss(
            state=_state(mxy_pred, w_xy),
            parameters=PlateParameters(D=1.0, nu=0.25),
            config=config,
        )
    pred_gradient, deflection_gradient = tape.gradient(
        losses["mxy_consistency_loss"],
        (mxy_pred, w_xy),
    )

    tf.debugging.assert_near(
        losses["mxy_consistency_scale"],
        tf.constant(1.5),
    )
    tf.debugging.assert_near(
        losses["mxy_consistency_loss"],
        tf.constant((2.0 / 1.5) ** 2),
    )
    assert pred_gradient is not None
    tf.debugging.assert_none_equal(pred_gradient, tf.zeros_like(pred_gradient))
    assert deflection_gradient is None


def test_mxy_consistency_uses_minimum_scale_for_zero_target() -> None:
    """A zero deflection-derived target must not cause division by zero."""
    losses = compute_mxy_consistency_loss(
        state=_state(
            tf.constant([[2e-3]], dtype=tf.float32),
            tf.zeros((1, 1), dtype=tf.float32),
        ),
        parameters=PlateParameters(D=1.0, nu=0.25),
        config=MxyConsistencyConfig(enabled=True, minimum_scale=1e-3),
    )

    tf.debugging.assert_near(
        losses["mxy_consistency_scale"],
        tf.constant(1e-3),
    )
    tf.debugging.assert_near(
        losses["mxy_consistency_loss"],
        tf.constant(4.0),
    )


def test_disabled_mxy_consistency_returns_scalar_zeros() -> None:
    """Disabled consistency must preserve the existing training objective."""
    losses = compute_mxy_consistency_loss(
        state=_state(
            tf.ones((2, 1), dtype=tf.float32),
            tf.ones((2, 1), dtype=tf.float32),
        ),
        parameters=PlateParameters(D=1.0, nu=0.25),
    )

    for value in losses.values():
        assert value.shape == ()
        tf.debugging.assert_near(value, tf.constant(0.0))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"minimum_scale": 0.0},
        {"minimum_scale": -1.0},
        {"minimum_scale": float("nan")},
        {"minimum_scale": float("inf")},
    ),
)
def test_mxy_consistency_config_rejects_invalid_scale(kwargs) -> None:
    """The scale floor must remain finite and positive."""
    with pytest.raises(ValueError):
        MxyConsistencyConfig(**kwargs)


def test_mxy_consistency_runs_inside_tf_function() -> None:
    """The normalized consistency calculation must remain graph compatible."""
    config = MxyConsistencyConfig(enabled=True, minimum_scale=1e-3)
    parameters = PlateParameters(D=1.0, nu=0.25)

    @tf.function
    def loss_fn(mxy_pred: tf.Tensor, w_xy: tf.Tensor) -> tf.Tensor:
        return compute_mxy_consistency_loss(
            state=_state(mxy_pred, w_xy),
            parameters=parameters,
            config=config,
        )["mxy_consistency_loss"]

    value = loss_fn(
        tf.constant([[0.5], [0.5]], dtype=tf.float32),
        tf.constant([[2.0], [2.0]], dtype=tf.float32),
    )

    assert value.shape == ()
    tf.debugging.assert_all_finite(value, "consistency loss must be finite")


class IndependentChainModel(tf.keras.Model):
    """Small analytic model with independently trainable w and Mxy."""

    def __init__(self) -> None:
        super().__init__()
        self.w_coefficient = self.add_weight(
            name="w_coefficient",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
        )
        self.mxy_value = self.add_weight(
            name="mxy_value",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
        )

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        del training
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        zeros = 0.0 * x
        return {
            "w": self.w_coefficient * x * y,
            "Mx": zeros,
            "My": zeros,
            "Mxy": self.mxy_value + zeros,
            "Qx": zeros,
            "Qy": zeros,
        }


def test_total_loss_weights_one_way_mxy_consistency() -> None:
    """The auxiliary term should enter total loss and update only direct Mxy."""
    model = IndependentChainModel()
    interior_xy = tf.constant(
        [[0.2, 0.3], [0.7, 0.8]],
        dtype=tf.float32,
    )
    boundary_batch = BoundaryBatch(
        xy=tf.constant([[0.0, 0.25], [1.0, 0.75]], dtype=tf.float32),
        normals=tf.constant([[-1.0, 0.0], [1.0, 0.0]], dtype=tf.float32),
    )

    with tf.GradientTape() as tape:
        losses = compute_total_loss(
            model=model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=BoundaryType.CLAMPED,
            load_fn=constant_load(0.0),
            parameters=PlateParameters(D=1.0, nu=0.25),
            weights=LossWeights(
                moment=0.0,
                shear=0.0,
                equilibrium=0.0,
                boundary=0.0,
                mxy_consistency=0.01,
            ),
            mxy_consistency_config=MxyConsistencyConfig(enabled=True),
        )
    w_gradient, mxy_gradient = tape.gradient(
        losses["total_loss"],
        (model.w_coefficient, model.mxy_value),
    )

    tf.debugging.assert_near(
        losses["weighted_mxy_consistency_loss"],
        0.01 * losses["mxy_consistency_loss"],
    )
    tf.debugging.assert_near(
        losses["total_loss"],
        losses["weighted_mxy_consistency_loss"],
    )
    if w_gradient is not None:
        tf.debugging.assert_near(w_gradient, tf.zeros_like(w_gradient))
    assert mxy_gradient is not None
    tf.debugging.assert_none_equal(mxy_gradient, tf.zeros_like(mxy_gradient))
