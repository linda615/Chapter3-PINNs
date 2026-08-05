"""Training loop helpers for Kirchhoff thin plate PINNs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import tensorflow as tf

from boundary import BoundaryBatch
from losses import (
    BoundaryType,
    LossWeightSource,
    LossWeights,
    MxyConsistencyConfig,
    PhysicsLossConfig,
    compute_total_loss,
    resolve_loss_weights,
)
from physics.plate_residuals import PlateParameters
from sampling import RectangularBoundarySampler, RectangularInteriorSampler


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for residual-based PINN training."""

    epochs: int = 1000
    interior_points: int = 1024
    boundary_points_per_side: int = 256
    boundary_type: BoundaryType | str = BoundaryType.SIMPLE
    log_every: int = 100
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate training loop settings."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.interior_points <= 0:
            raise ValueError(f"interior_points must be positive, got {self.interior_points}.")
        if self.boundary_points_per_side <= 0:
            raise ValueError(
                "boundary_points_per_side must be positive, "
                f"got {self.boundary_points_per_side}."
            )
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}.")


def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    interior_xy: tf.Tensor,
    boundary_xy: tf.Tensor,
    boundary_normals: tf.Tensor,
    boundary_type: BoundaryType | str,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    loss_weights: LossWeights | None = None,
    foundation_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    physics_loss_config: PhysicsLossConfig | None = None,
    mxy_consistency_config: MxyConsistencyConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Run one optimization step and return the current loss dictionary."""
    with tf.GradientTape() as tape:
        loss_dict = compute_total_loss(
            model=model,
            interior_xy=interior_xy,
            boundary_batch=BoundaryBatch(xy=boundary_xy, normals=boundary_normals),
            boundary_type=boundary_type,
            load_fn=load_fn,
            parameters=parameters,
            foundation_fn=foundation_fn,
            weights=LossWeights() if loss_weights is None else loss_weights,
            physics_loss_config=physics_loss_config,
            mxy_consistency_config=mxy_consistency_config,
            training=True,
        )

    gradients = tape.gradient(loss_dict["total_loss"], model.trainable_variables)
    gradient_variable_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if not gradient_variable_pairs:
        raise ValueError("No gradients were produced for model trainable variables.")

    optimizer.apply_gradients(gradient_variable_pairs)
    return loss_dict


def train_pinn(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    interior_sampler: RectangularInteriorSampler,
    boundary_sampler: RectangularBoundarySampler,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    config: TrainingConfig,
    loss_weights: LossWeightSource | None = None,
    foundation_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    physics_loss_config: PhysicsLossConfig | None = None,
    mxy_consistency_config: MxyConsistencyConfig | None = None,
) -> list[dict[str, tf.Tensor]]:
    """Train a PINN with resampled interior and boundary collocation points.

    The returned history stores loss dictionaries at step 1, every
    ``config.log_every`` steps, and the final step.
    """
    history: list[dict[str, tf.Tensor]] = []

    for epoch in range(1, config.epochs + 1):
        interior_seed = _step_seed(config.seed, epoch, offset=0)
        boundary_seed = _step_seed(config.seed, epoch, offset=100000)
        interior_xy = interior_sampler.sample(config.interior_points, seed=interior_seed)
        boundary_xy, boundary_normals = boundary_sampler.sample_all_sides(
            config.boundary_points_per_side,
            seed=boundary_seed,
        )
        current_weights = resolve_loss_weights(loss_weights, epoch)

        loss_dict = train_step(
            model=model,
            optimizer=optimizer,
            interior_xy=interior_xy,
            boundary_xy=boundary_xy,
            boundary_normals=boundary_normals,
            boundary_type=config.boundary_type,
            load_fn=load_fn,
            parameters=parameters,
            loss_weights=current_weights,
            foundation_fn=foundation_fn,
            physics_loss_config=physics_loss_config,
            mxy_consistency_config=mxy_consistency_config,
        )
        loss_dict.update(_weight_snapshot(current_weights))

        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            history.append(_snapshot_losses(epoch, loss_dict))

    return history


def _step_seed(base_seed: int | None, epoch: int, offset: int) -> int | None:
    """Create deterministic per-step seeds when a base seed is provided."""
    if base_seed is None:
        return None
    return base_seed + offset + epoch


def _snapshot_losses(epoch: int, loss_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Copy loss tensors for history without converting them to NumPy."""
    snapshot = {"epoch": tf.constant(epoch, dtype=tf.int32)}
    snapshot.update({name: tf.identity(value) for name, value in loss_dict.items()})
    return snapshot


def _weight_snapshot(weights: LossWeights) -> dict[str, tf.Tensor]:
    """Return scalar tensors for logging the active loss weights."""
    return {
        "moment_weight": tf.constant(weights.moment, dtype=tf.float32),
        "shear_weight": tf.constant(weights.shear, dtype=tf.float32),
        "equilibrium_weight": tf.constant(weights.equilibrium, dtype=tf.float32),
        "boundary_weight": tf.constant(weights.boundary, dtype=tf.float32),
        "mxy_consistency_weight": tf.constant(
            weights.mxy_consistency,
            dtype=tf.float32,
        ),
    }
