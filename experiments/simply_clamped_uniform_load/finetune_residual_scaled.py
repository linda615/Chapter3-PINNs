"""Fine-tune the scaled mixed PINN with gradual residual normalization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import math
from pathlib import Path

import tensorflow as tf

from boundary import BoundaryLossConfig, BoundaryResidualScales
from configs import save_experiment_config
from losses import LossWeights, PhysicsLossConfig, ResidualScales

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
)
from .run import (
    BestValidationCheckpoint,
    load_initial_weights,
    save_history_csv,
    save_weights,
    set_reproducible_seed,
)
from .trainer import MixedBoundaryPINNTrainer


DEFAULT_INITIAL_CHECKPOINT = (
    Path("experiments/simply_clamped_uniform_load/")
    / "results_output_scaled_pinn/checkpoints/"
    / "best_validation_loss.weights.h5"
)
DEFAULT_RESULTS_DIR = (
    Path("experiments/simply_clamped_uniform_load/")
    / "results_residual_scaled_finetune"
)


@dataclass(frozen=True)
class ResidualScaledFineTuneConfig:
    """Controls for a gradual raw-to-normalized residual fine-tune."""

    experiment: ExperimentConfig = DEFAULT_CONFIG
    initial_checkpoint_path: Path = DEFAULT_INITIAL_CHECKPOINT
    results_dir: Path = DEFAULT_RESULTS_DIR
    epochs: int = 1500
    transition_epochs: int = 500
    initial_learning_rate: float = 1e-5
    final_learning_rate: float = 3e-6
    validation_interval: int = 100
    log_every: int = 100
    physics_scales: ResidualScales = ResidualScales(
        moment_x=0.01328375264867326,
        moment_y=0.022366591614178814,
        twisting_moment=0.006775997292221727,
        shear_x=0.08898337692695245,
        shear_y=0.2109545621763597,
        equilibrium=1.0,
    )
    boundary_scales: BoundaryResidualScales = BoundaryResidualScales(
        deflection=1.0,
        slope=0.0029383159708231688,
        moment=0.01328375264867326,
        shear=0.08898337692695245,
    )

    def __post_init__(self) -> None:
        """Validate fine-tuning controls before loading a checkpoint."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if not 0 < self.transition_epochs <= self.epochs:
            raise ValueError("transition_epochs must lie in [1, epochs].")
        if self.validation_interval <= 0 or self.log_every <= 0:
            raise ValueError("validation_interval and log_every must be positive.")
        rates = (self.initial_learning_rate, self.final_learning_rate)
        if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
            raise ValueError("Learning rates must be finite and positive.")


class GradualResidualScaleTrainer(MixedBoundaryPINNTrainer):
    """Apply logarithmically interpolated residual scales per epoch."""

    def __init__(
        self,
        *args,
        target_physics_scales: ResidualScales,
        target_boundary_scales: BoundaryResidualScales,
        transition_epochs: int,
        **kwargs,
    ) -> None:
        """Create a trainer with unit scales at epoch zero."""
        super().__init__(*args, **kwargs)
        self.target_physics_scales = target_physics_scales
        self.target_boundary_scales = target_boundary_scales
        self.transition_epochs = transition_epochs
        self._base_component_weights = self.physics_loss_config.component_weights

    def train_step(
        self,
        interior_xy: tf.Tensor,
        boundary_batch,
        loss_weights: LossWeights | None = None,
        epoch: int = 1,
    ) -> dict[str, tf.Tensor]:
        """Update effective scales, then perform one ordinary trainer step."""
        alpha = min(float(epoch) / float(self.transition_epochs), 1.0)
        physics_scales = _interpolate_dataclass_scales(
            self.target_physics_scales,
            alpha,
        )
        boundary_scales = _interpolate_dataclass_scales(
            self.target_boundary_scales,
            alpha,
        )
        self.physics_loss_config = PhysicsLossConfig(
            normalize_residuals=True,
            residual_scales=physics_scales,
            component_weights=self._base_component_weights,
        )
        self.boundary_loss_config = BoundaryLossConfig(
            normalize_residuals=True,
            residual_scales=boundary_scales,
        )
        losses = super().train_step(
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            loss_weights=loss_weights,
            epoch=epoch,
        )
        losses.update(_scale_snapshot(alpha, physics_scales, boundary_scales))
        return losses


def run_residual_scaled_finetuning(
    config: ResidualScaledFineTuneConfig = ResidualScaledFineTuneConfig(),
) -> list[dict[str, tf.Tensor]]:
    """Load the best raw-residual model and run gradual normalization."""
    save_experiment_config(
        config,
        run_entry=(
            "experiments.simply_clamped_uniform_load."
            "finetune_residual_scaled"
        ),
    )
    training = replace(
        config.experiment.training,
        epochs=config.epochs,
        log_every=config.log_every,
    )
    experiment = replace(
        config.experiment,
        results_dir=config.results_dir,
        initial_checkpoint_path=config.initial_checkpoint_path,
        training=training,
        validation_interval=config.validation_interval,
    )
    set_reproducible_seed(experiment.seed)
    model = create_model(experiment)
    load_initial_weights(model, config.initial_checkpoint_path)
    checkpoint = BestValidationCheckpoint(config=experiment)
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(config.transition_epochs - 1,),
        values=(config.initial_learning_rate, config.final_learning_rate),
    )
    trainer = GradualResidualScaleTrainer(
        model=model,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(experiment),
        parameters=experiment.plate,
        config=experiment.training,
        loss_weights=LossWeights(),
        physics_loss_config=PhysicsLossConfig(
            normalize_residuals=True,
        ),
        boundary_loss_config=BoundaryLossConfig(
            normalize_residuals=True,
        ),
        mxy_consistency_config=experiment.mxy_consistency_config,
        shared_pcgrad_config=replace(
            experiment.shared_pcgrad,
            enabled=False,
        ),
        verbose=True,
        callbacks=(checkpoint,),
        experiment_config=experiment,
        target_physics_scales=config.physics_scales,
        target_boundary_scales=config.boundary_scales,
        transition_epochs=config.transition_epochs,
    )
    history = trainer.train()
    save_history_csv(history, experiment.loss_history_path)
    save_history_csv(checkpoint.records, experiment.validation_history_path)
    save_weights(model, experiment.final_weights_path)
    print(
        "residual_scaled_finetune_complete "
        f"best_epoch={checkpoint.best_epoch} "
        f"best_validation_score={checkpoint.best_score:.6e} "
        f"best_weights={experiment.best_validation_loss_weights_path} "
        f"final_weights={experiment.final_weights_path}"
    )
    return history


def _interpolate_dataclass_scales(scale_config, alpha: float):
    """Interpolate from unit to target scales in logarithmic space."""
    values = {
        item.name: math.exp(alpha * math.log(getattr(scale_config, item.name)))
        for item in fields(scale_config)
    }
    return scale_config.__class__(**values)


def _scale_snapshot(
    alpha: float,
    physics: ResidualScales,
    boundary: BoundaryResidualScales,
) -> dict[str, tf.Tensor]:
    """Return active scales as scalar tensors for logs and CSV history."""
    snapshot = {
        "residual_scale_alpha": tf.constant(alpha, dtype=tf.float32),
        "boundary_scale_slope": tf.constant(boundary.slope, dtype=tf.float32),
        "boundary_scale_moment": tf.constant(boundary.moment, dtype=tf.float32),
    }
    for item in fields(physics):
        snapshot[f"residual_scale_{item.name}"] = tf.constant(
            getattr(physics, item.name),
            dtype=tf.float32,
        )
    return snapshot


def _parse_args() -> argparse.Namespace:
    """Parse fine-tuning paths and duration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=DEFAULT_INITIAL_CHECKPOINT,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--transition-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--final-learning-rate", type=float, default=3e-6)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ResidualScaledFineTuneConfig:
    """Build the immutable effective configuration."""
    return ResidualScaledFineTuneConfig(
        initial_checkpoint_path=args.initial_checkpoint,
        results_dir=args.results_dir,
        epochs=args.epochs,
        transition_epochs=args.transition_epochs,
        initial_learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        validation_interval=args.validation_interval,
        log_every=args.log_every,
    )


def main() -> None:
    """Run gradual residual-scale fine-tuning."""
    run_residual_scaled_finetuning(_config_from_args(_parse_args()))


if __name__ == "__main__":
    main()


__all__ = [
    "GradualResidualScaleTrainer",
    "ResidualScaledFineTuneConfig",
    "run_residual_scaled_finetuning",
]
