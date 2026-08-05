"""Minimal convergence smoke test for the PINN training loop."""

import tensorflow as tf

from losses import BoundaryType
from models import MultiSubNetPINN
from physics.plate_residuals import PlateParameters
from sampling import RectangularBoundarySampler, RectangularInteriorSampler, constant_load
from trainer import PINNTrainer, TrainingConfig


def test_minimal_training_reduces_total_loss() -> None:
    """A tiny deterministic run should reduce total loss over a few steps."""
    tf.keras.utils.set_random_seed(2026)
    model = MultiSubNetPINN(grouping="six", hidden_width=8, hidden_depth=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    config = TrainingConfig(
        epochs=50,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=BoundaryType.FREE,
        log_every=1,
        seed=2026,
    )
    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        interior_sampler=RectangularInteriorSampler(),
        boundary_sampler=RectangularBoundarySampler(),
        load_fn=constant_load(0.0),
        parameters=PlateParameters(D=1.0, nu=0.3),
        config=config,
    )

    history = trainer.train()
    initial_loss = history[0]["total_loss"]
    final_loss = history[-1]["total_loss"]

    assert len(history) == 50
    for item in history:
        tf.debugging.assert_all_finite(item["total_loss"], "total_loss must stay finite")
    assert float(final_loss.numpy()) < float(initial_loss.numpy())
