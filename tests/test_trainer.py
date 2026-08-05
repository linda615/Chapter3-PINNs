"""Tests for PINN training helpers."""

import tensorflow as tf

from losses import BoundaryType
from models import MultiSubNetPINN
from physics.plate_residuals import PlateParameters
from sampling import RectangularBoundarySampler, RectangularInteriorSampler, constant_load
from trainer import TrainingConfig, train_pinn, train_step


def test_train_step_updates_model_variables() -> None:
    """A single train step should produce losses and update at least one weight."""
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    parameters = PlateParameters(D=1.0, nu=0.3)
    interior_xy = RectangularInteriorSampler().sample(8, seed=1)
    boundary_xy, boundary_normals = RectangularBoundarySampler().sample_all_sides(2, seed=2)
    model(interior_xy)
    weights_before = [tf.identity(weight) for weight in model.trainable_variables]

    loss_dict = train_step(
        model=model,
        optimizer=optimizer,
        interior_xy=interior_xy,
        boundary_xy=boundary_xy,
        boundary_normals=boundary_normals,
        boundary_type=BoundaryType.FREE,
        load_fn=constant_load(0.0),
        parameters=parameters,
    )

    assert "total_loss" in loss_dict
    changed = [
        tf.reduce_any(tf.not_equal(before, after))
        for before, after in zip(weights_before, model.trainable_variables)
    ]
    assert any(bool(value.numpy()) for value in changed)


def test_train_pinn_returns_logged_history() -> None:
    """The high-level training loop should return configured loss snapshots."""
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    config = TrainingConfig(
        epochs=3,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=BoundaryType.FREE,
        log_every=2,
        seed=10,
    )

    history = train_pinn(
        model=model,
        optimizer=optimizer,
        interior_sampler=RectangularInteriorSampler(),
        boundary_sampler=RectangularBoundarySampler(),
        load_fn=constant_load(0.0),
        parameters=PlateParameters(D=1.0, nu=0.3),
        config=config,
    )

    assert [int(item["epoch"].numpy()) for item in history] == [1, 2, 3]
    for item in history:
        assert "total_loss" in item
        assert item["total_loss"].shape == ()
