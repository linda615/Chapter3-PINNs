"""Tests for the object-oriented PINN trainer."""

import tensorflow as tf

from boundary import BoundaryBatch
from losses import BoundaryType, LossWeightSchedule, LossWeightStage, LossWeights
from models import MultiSubNetPINN
from physics.plate_residuals import PlateParameters
from sampling import RectangularBoundarySampler, RectangularInteriorSampler, constant_load
from trainer import PINNTrainer, SharedTrunkPCGradConfig, TrainingConfig


def _make_trainer(epochs: int = 3) -> PINNTrainer:
    """Create a small trainer for tests."""
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    config = TrainingConfig(
        epochs=epochs,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=BoundaryType.FREE,
        log_every=2,
        seed=123,
    )
    return PINNTrainer(
        model=model,
        optimizer=optimizer,
        interior_sampler=RectangularInteriorSampler(),
        boundary_sampler=RectangularBoundarySampler(),
        load_fn=constant_load(0.0),
        parameters=PlateParameters(D=1.0, nu=0.3),
        config=config,
        loss_weights=LossWeights(),
    )


def test_pinn_trainer_compute_loss_returns_total_loss() -> None:
    """Trainer should compute the project total-loss dictionary."""
    trainer = _make_trainer()
    interior_xy = RectangularInteriorSampler().sample(8, seed=1)
    boundary_xy, boundary_normals = RectangularBoundarySampler().sample_all_sides(2, seed=2)

    loss_dict = trainer.compute_loss(
        interior_xy=interior_xy,
        boundary_batch=BoundaryBatch(xy=boundary_xy, normals=boundary_normals),
    )

    assert "total_loss" in loss_dict
    assert loss_dict["total_loss"].shape == ()


def test_pinn_trainer_train_step_updates_weights() -> None:
    """One trainer step should update at least one model variable."""
    trainer = _make_trainer()
    interior_xy, boundary_batch = trainer.sample_epoch(epoch=1)
    trainer.model(interior_xy)
    weights_before = [tf.identity(weight) for weight in trainer.model.trainable_variables]

    trainer.train_step(interior_xy, boundary_batch)

    changed = [
        tf.reduce_any(tf.not_equal(before, after))
        for before, after in zip(weights_before, trainer.model.trainable_variables)
    ]
    assert any(bool(value.numpy()) for value in changed)


def test_pinn_trainer_delays_pcgrad_until_configured_epoch() -> None:
    """Ordinary gradients precede the configured PCGrad start epoch."""
    trainer = _make_trainer()
    trainer.shared_pcgrad_config = SharedTrunkPCGradConfig(
        enabled=True,
        start_epoch=2,
    )
    interior_xy, boundary_batch = trainer.sample_epoch(epoch=1)
    trainer.model(interior_xy)

    before_start = trainer.train_step(
        interior_xy,
        boundary_batch,
        epoch=1,
    )
    after_start = trainer.train_step(
        interior_xy,
        boundary_batch,
        epoch=2,
    )

    tf.debugging.assert_equal(before_start["pcgrad_active"], 0.0)
    tf.debugging.assert_equal(before_start["pcgrad_projection_count"], 0.0)
    tf.debugging.assert_equal(after_start["pcgrad_active"], 1.0)
    assert "pcgrad_conflict_count" in after_start


def test_pinn_trainer_train_returns_logged_history() -> None:
    """Configured training should return snapshots at expected epochs."""
    trainer = _make_trainer(epochs=3)

    history = trainer.train()

    assert [int(item["epoch"].numpy()) for item in history] == [1, 2, 3]
    for item in history:
        assert "total_loss" in item
        assert item["total_loss"].shape == ()


def test_pinn_trainer_verbose_prints_logged_losses(capsys) -> None:
    """Verbose trainer should print loss summaries at logged epochs."""
    trainer = _make_trainer(epochs=2)
    trainer.verbose = True

    trainer.train()

    output = capsys.readouterr().out
    assert "epoch=1" in output
    assert "epoch=2" in output
    assert "total_loss=" in output
    assert "physics_loss=" in output
    assert "boundary_loss=" in output
    assert "equilibrium_weight=" in output
    assert "weighted_equilibrium_loss=" in output
    assert "mxy_consistency_weight=" in output
    assert "mxy_consistency_loss=" in output


def test_pinn_trainer_logs_scheduled_loss_weights() -> None:
    """Trainer should resolve staged loss weights by epoch."""
    trainer = _make_trainer(epochs=3)
    trainer.config = TrainingConfig(
        epochs=3,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=BoundaryType.FREE,
        log_every=1,
        seed=123,
    )
    trainer.loss_weights = LossWeightSchedule(
        stages=(
            LossWeightStage(end_epoch=1, weights=LossWeights(equilibrium=20.0)),
            LossWeightStage(end_epoch=3, weights=LossWeights(equilibrium=5.0)),
        )
    )

    history = trainer.train()

    assert [float(item["equilibrium_weight"].numpy()) for item in history] == [20.0, 5.0, 5.0]
    for item in history:
        assert "weighted_equilibrium_loss" in item
        assert "mxy_consistency_weight" in item
