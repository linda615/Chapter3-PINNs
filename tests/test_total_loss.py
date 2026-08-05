"""Tests for weighted total loss aggregation."""

import pytest
import tensorflow as tf

import losses.total as total_module
from boundary import BoundaryBatch, BoundaryType
from losses import LossWeights, compute_total_loss
from losses.consistency import MxyConsistencyConfig
from losses.normalization import PhysicsLossConfig, ResidualScales
from models import DEFLECTION_TRANSFORM_UNIT_SQUARE, MultiSubNetPINN
from physics.plate_residuals import PlateParameters
from sampling import constant_load


EXPECTED_TOTAL_KEYS = (
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
    "deflection_bc_loss",
    "slope_bc_loss",
    "moment_bc_loss",
    "shear_bc_loss",
    "boundary_loss",
    "weighted_moment_loss",
    "weighted_shear_loss",
    "weighted_equilibrium_loss",
    "weighted_boundary_loss",
    "weighted_mxy_consistency_loss",
    "total_loss",
)


class ZeroResidualModel(tf.keras.Model):
    """Analytic model with zero physics residuals and simple boundary terms."""

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
        zeros = 0.0 * x * y
        return {
            "w": w,
            "Mx": mx,
            "My": my,
            "Mxy": zeros,
            "Qx": 0.0 * x,
            "Qy": 0.0 * y,
        }


class QyBiasedZeroResidualModel(ZeroResidualModel):
    """Zero-residual model with a controllable Qy offset."""

    def __init__(self, parameters: PlateParameters, qy_bias: float) -> None:
        super().__init__(parameters)
        self.qy_bias = qy_bias

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        fields = super().call(xy, training=training)
        fields["Qy"] = fields["Qy"] + tf.cast(self.qy_bias, xy.dtype)
        return fields


def _interior_xy() -> tf.Tensor:
    return tf.constant([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0]], dtype=tf.float32)


def _boundary_batch() -> BoundaryBatch:
    return BoundaryBatch(
        xy=tf.constant([[0.0, 0.0], [0.0, 0.5]], dtype=tf.float32),
        normals=tf.constant([[1.0, 0.0], [1.0, 0.0]], dtype=tf.float32),
    )


def test_total_loss_uses_weighted_sum_with_controllable_components(monkeypatch) -> None:
    """Fake component losses should produce the expected weighted total."""

    def fake_physics_loss(**kwargs):
        del kwargs
        return {
            "moment_x_weight": tf.constant(1.0),
            "moment_y_weight": tf.constant(1.0),
            "twisting_moment_weight": tf.constant(1.0),
            "shear_x_weight": tf.constant(1.0),
            "shear_y_weight": tf.constant(1.0),
            "moment_x_loss": tf.constant(1.0),
            "moment_y_loss": tf.constant(1.0),
            "twisting_moment_loss": tf.constant(1.0),
            "weighted_moment_x_loss": tf.constant(1.0),
            "weighted_moment_y_loss": tf.constant(1.0),
            "weighted_twisting_moment_loss": tf.constant(1.0),
            "moment_loss": tf.constant(1.0),
            "shear_x_loss": tf.constant(0.5),
            "shear_y_loss": tf.constant(1.5),
            "weighted_shear_x_loss": tf.constant(0.5),
            "weighted_shear_y_loss": tf.constant(1.5),
            "shear_loss": tf.constant(2.0),
            "equilibrium_loss": tf.constant(3.0),
            "physics_loss": tf.constant(999.0),
            "mxy_consistency_loss": tf.constant(0.25),
            "mxy_consistency_scale": tf.constant(0.5),
        }

    def fake_boundary_loss(**kwargs):
        del kwargs
        return {
            "deflection_bc_loss": tf.constant(0.25),
            "slope_bc_loss": tf.constant(0.75),
            "moment_bc_loss": tf.constant(1.25),
            "shear_bc_loss": tf.constant(1.75),
            "boundary_loss": tf.constant(4.0),
        }

    monkeypatch.setattr(total_module, "compute_physics_loss", fake_physics_loss)
    monkeypatch.setattr(total_module, "compute_boundary_loss", fake_boundary_loss)

    loss_dict = total_module.compute_total_loss(
        model=tf.keras.Sequential(),
        interior_xy=tf.zeros((2, 2), dtype=tf.float32),
        boundary_batch=BoundaryBatch(
            xy=tf.zeros((2, 2), dtype=tf.float32),
            normals=tf.ones((2, 2), dtype=tf.float32),
        ),
        boundary_type=BoundaryType.SIMPLE,
        load_fn=constant_load(0.0),
        parameters=PlateParameters(D=1.0, nu=0.3),
        weights=LossWeights(
            moment=2.0,
            shear=3.0,
            equilibrium=4.0,
            boundary=5.0,
            mxy_consistency=2.0,
        ),
    )

    assert tuple(loss_dict.keys()) == EXPECTED_TOTAL_KEYS
    tf.debugging.assert_near(loss_dict["physics_loss"], tf.constant(6.0))
    tf.debugging.assert_near(
        loss_dict["shear_loss"],
        loss_dict["shear_x_loss"] + loss_dict["shear_y_loss"],
    )
    tf.debugging.assert_near(
        loss_dict["moment_loss"],
        tf.constant(1.0),
    )
    tf.debugging.assert_near(loss_dict["weighted_moment_loss"], tf.constant(2.0))
    tf.debugging.assert_near(loss_dict["weighted_shear_loss"], tf.constant(6.0))
    tf.debugging.assert_near(loss_dict["weighted_equilibrium_loss"], tf.constant(12.0))
    tf.debugging.assert_near(loss_dict["weighted_boundary_loss"], tf.constant(20.0))
    tf.debugging.assert_near(
        loss_dict["weighted_mxy_consistency_loss"],
        tf.constant(0.5),
    )
    tf.debugging.assert_near(loss_dict["total_loss"], tf.constant(40.5))


def test_missing_boundary_components_are_scalar_zero_tensors() -> None:
    """Simple boundaries should normalize missing slope and shear terms to zero."""
    parameters = PlateParameters(D=1.0, nu=0.3)
    model = ZeroResidualModel(parameters)

    loss_dict = compute_total_loss(
        model=model,
        interior_xy=_interior_xy(),
        boundary_batch=_boundary_batch(),
        boundary_type=BoundaryType.SIMPLE,
        load_fn=constant_load(0.0),
        parameters=parameters,
    )

    assert loss_dict["slope_bc_loss"].shape == ()
    assert loss_dict["shear_bc_loss"].shape == ()
    tf.debugging.assert_near(loss_dict["slope_bc_loss"], tf.constant(0.0))
    tf.debugging.assert_near(loss_dict["shear_bc_loss"], tf.constant(0.0))


def test_total_loss_accepts_normalized_physics_loss_config() -> None:
    """Total loss should forward residual-normalization settings to physics loss."""
    parameters = PlateParameters(D=1.0, nu=0.3)
    model = QyBiasedZeroResidualModel(parameters, qy_bias=0.25)
    loss_dict = compute_total_loss(
        model=model,
        interior_xy=_interior_xy(),
        boundary_batch=_boundary_batch(),
        boundary_type=BoundaryType.SIMPLE,
        load_fn=constant_load(0.0),
        parameters=parameters,
        weights=LossWeights(boundary=0.0),
        physics_loss_config=PhysicsLossConfig(
            normalize_residuals=True,
            residual_scales=ResidualScales(shear_y=0.5),
        ),
    )

    tf.debugging.assert_near(loss_dict["shear_y_loss"], tf.constant(0.25))
    tf.debugging.assert_near(loss_dict["shear_loss"], tf.constant(0.25))
    tf.debugging.assert_near(loss_dict["total_loss"], tf.constant(0.25))


def test_all_total_loss_values_are_scalar_tensors() -> None:
    """Every returned loss component should be a scalar Tensor."""
    parameters = PlateParameters(D=1.0, nu=0.3)
    model = ZeroResidualModel(parameters)

    loss_dict = compute_total_loss(
        model=model,
        interior_xy=_interior_xy(),
        boundary_batch=_boundary_batch(),
        boundary_type=BoundaryType.SIMPLE,
        load_fn=constant_load(0.0),
        parameters=parameters,
    )

    assert tuple(loss_dict.keys()) == EXPECTED_TOTAL_KEYS
    for value in loss_dict.values():
        assert tf.is_tensor(value)
        assert value.shape == ()


def test_gradient_tape_computes_total_loss_gradients() -> None:
    """The total loss should remain differentiable with respect to model weights."""
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    interior_xy = _interior_xy()
    boundary_batch = _boundary_batch()
    parameters = PlateParameters(D=1.0, nu=0.3)
    model(interior_xy)

    with tf.GradientTape() as tape:
        loss_dict = compute_total_loss(
            model=model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=BoundaryType.FREE,
            load_fn=constant_load(0.0),
            parameters=parameters,
        )

    gradients = tape.gradient(loss_dict["total_loss"], model.trainable_variables)

    assert any(gradient is not None for gradient in gradients)


def test_total_loss_runs_inside_tf_function() -> None:
    """The total loss API should execute inside ``tf.function``."""
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    parameters = PlateParameters(D=1.0, nu=0.3)
    boundary_batch = _boundary_batch()

    @tf.function
    def loss_fn(interior_xy: tf.Tensor) -> tf.Tensor:
        loss_dict = compute_total_loss(
            model=model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=BoundaryType.FREE,
            load_fn=constant_load(0.0),
            parameters=parameters,
        )
        return loss_dict["total_loss"]

    loss_value = loss_fn(_interior_xy())

    assert loss_value.shape == ()


@pytest.mark.parametrize("hard_transform", [False, True])
def test_total_loss_supports_soft_and_hard_deflection_modes(hard_transform: bool) -> None:
    """Total loss should work whether deflection is soft or hard constrained."""
    kwargs = {}
    if hard_transform:
        kwargs = {
            "apply_deflection_transform": True,
            "deflection_transform_name": DEFLECTION_TRANSFORM_UNIT_SQUARE,
        }
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1, **kwargs)

    loss_dict = compute_total_loss(
        model=model,
        interior_xy=_interior_xy(),
        boundary_batch=_boundary_batch(),
        boundary_type=BoundaryType.SIMPLE,
        load_fn=constant_load(0.0),
        parameters=PlateParameters(D=1.0, nu=0.3),
    )

    assert loss_dict["total_loss"].shape == ()
