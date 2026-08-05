"""Train the mixed-boundary plate with adaptive gradient-balanced losses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path

import tensorflow as tf

from configs import save_experiment_config
from losses import LossWeights

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
)
from .run import (
    BestValidationCheckpoint,
    create_learning_rate_schedule,
    load_initial_weights,
    save_history_csv,
    save_weights,
    set_reproducible_seed,
)
from .trainer import MixedBoundaryPINNTrainer


_OBJECTIVE_NAMES = ("moment", "shear", "equilibrium", "boundary")
_LOSS_KEYS = (
    "moment_loss",
    "shear_loss",
    "equilibrium_loss",
    "boundary_loss",
)


@dataclass(frozen=True)
class AdaptiveLossConfig:
    """Controls for shared-gradient loss balancing."""

    warmup_epochs: int = 200
    update_every: int = 20
    ema_decay: float = 0.9
    min_weight: float = 0.1
    max_weight: float = 10.0
    epsilon: float = 1e-12
    
    def __post_init__(self) -> None:
        """Validate adaptive-weight hyperparameters."""
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        if self.update_every <= 0:
            raise ValueError("update_every must be positive.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        for name, value in (
            ("min_weight", self.min_weight),
            ("max_weight", self.max_weight),
            ("epsilon", self.epsilon),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight.")


@dataclass(frozen=True)
class AdaptiveRunConfig:
    """Experiment and output configuration for adaptive training."""

    experiment: ExperimentConfig = DEFAULT_CONFIG
    adaptive: AdaptiveLossConfig = AdaptiveLossConfig()
    
    results_dir: Path = Path(
        "experiments/simply_clamped_uniform_load/results_adaptive"
    )
    initial_checkpoint_path: Path | None = None


class AdaptiveMixedBoundaryTrainer(MixedBoundaryPINNTrainer):
    """Balance four objectives using shared-trunk gradient norms."""

    def __init__(self, *args, adaptive_config: AdaptiveLossConfig, **kwargs):
        """Create an adaptive trainer with unit initial weights."""
        super().__init__(*args, **kwargs)
        shared_trunk = getattr(self.model, "shared_trunk", None)
        if shared_trunk is None:
            raise ValueError(
                "Adaptive loss weighting requires model.shared_trunk."
            )
        self.adaptive_config = adaptive_config
        self.current_weights = LossWeights()
        self.last_gradient_norms = {
            name: tf.constant(0.0, dtype=tf.float32)
            for name in _OBJECTIVE_NAMES
        }
        self.last_weight_update = tf.constant(0, dtype=tf.int32)

    def train(self) -> list[dict[str, tf.Tensor]]:
        """Run Adam training and update weights on configured epochs."""
        history: list[dict[str, tf.Tensor]] = []
        for epoch in range(1, self.config.epochs + 1):
            interior_xy, boundary_groups = self.sample_epoch(epoch)
            if self._should_update_weights(epoch):
                self._update_weights(interior_xy, boundary_groups)
                self.last_weight_update = tf.constant(epoch, tf.int32)

            losses = self.train_step(
                interior_xy,
                boundary_groups,
                loss_weights=self.current_weights,
            )
            losses.update(self._adaptive_snapshot())
            if self._should_log(epoch):
                losses.update(
                    self.compute_log_diagnostics(
                        epoch=epoch,
                        interior_xy=interior_xy,
                        boundary_batch=boundary_groups,
                        loss_weights=self.current_weights,
                    )
                )
                record = {"epoch": tf.constant(epoch, tf.int32)}
                record.update(
                    {name: tf.identity(value) for name, value in losses.items()}
                )
                history.append(record)
                print(_format_adaptive_loss_line(epoch, losses))
            for callback in self.callbacks:
                callback(epoch, self.model, losses)
        return history

    def _should_update_weights(self, epoch: int) -> bool:
        """Return whether shared gradient norms should update weights."""
        return epoch > self.adaptive_config.warmup_epochs and (
            (epoch - self.adaptive_config.warmup_epochs)
            % self.adaptive_config.update_every
            == 0
        )

    def _update_weights(self, interior_xy, boundary_groups) -> None:
        """Update bounded EMA weights from unweighted shared gradients."""
        shared_variables = tuple(self.model.shared_trunk.trainable_variables)
        with tf.GradientTape(persistent=True) as tape:
            losses = self.compute_loss(
                interior_xy=interior_xy,
                boundary_batch=boundary_groups,
                loss_weights=LossWeights(),
                training=True,
            )
        norms = []
        for name, loss_key in zip(_OBJECTIVE_NAMES, _LOSS_KEYS):
            gradients = tape.gradient(losses[loss_key], shared_variables)
            norm = tf.linalg.global_norm(
                [gradient for gradient in gradients if gradient is not None]
            )
            tf.debugging.assert_all_finite(
                norm,
                f"Non-finite shared gradient norm for {name}.",
            )
            norms.append(norm)
        del tape

        dtype = norms[0].dtype
        epsilon = tf.cast(self.adaptive_config.epsilon, dtype)
        log_norms = [tf.math.log(norm + epsilon) for norm in norms]
        target_norm = tf.exp(tf.add_n(log_norms) / len(log_norms))
        candidates = tf.stack(
            [target_norm / (norm + epsilon) for norm in norms]
        )
        candidates = tf.clip_by_value(
            candidates,
            tf.cast(self.adaptive_config.min_weight, dtype),
            tf.cast(self.adaptive_config.max_weight, dtype),
        )
        candidates = candidates / tf.reduce_mean(candidates)
        previous = tf.constant(
            (
                self.current_weights.moment,
                self.current_weights.shear,
                self.current_weights.equilibrium,
                self.current_weights.boundary,
            ),
            dtype=dtype,
        )
        decay = tf.cast(self.adaptive_config.ema_decay, dtype)
        updated = decay * previous + (1.0 - decay) * candidates
        updated = updated / tf.reduce_mean(updated)
        updated = tf.clip_by_value(
            updated,
            tf.cast(self.adaptive_config.min_weight, dtype),
            tf.cast(self.adaptive_config.max_weight, dtype),
        )
        values = [float(value.numpy()) for value in tf.unstack(updated)]
        self.current_weights = LossWeights(
            moment=values[0],
            shear=values[1],
            equilibrium=values[2],
            boundary=values[3],
            mxy_consistency=0.0,
        )
        self.last_gradient_norms = {
            name: tf.identity(norm)
            for name, norm in zip(_OBJECTIVE_NAMES, norms)
        }

    def _adaptive_snapshot(self) -> dict[str, tf.Tensor]:
        """Return scalar diagnostics for CSV and console logging."""
        dtype = tf.float32
        snapshot = {
            "adaptive_moment_weight": tf.cast(
                self.current_weights.moment, dtype
            ),
            "adaptive_shear_weight": tf.cast(
                self.current_weights.shear, dtype
            ),
            "adaptive_equilibrium_weight": tf.cast(
                self.current_weights.equilibrium, dtype
            ),
            "adaptive_boundary_weight": tf.cast(
                self.current_weights.boundary, dtype
            ),
            "adaptive_last_update_epoch": tf.identity(
                self.last_weight_update
            ),
        }
        snapshot.update(
            {
                f"adaptive_grad_norm_{name}": tf.identity(norm)
                for name, norm in self.last_gradient_norms.items()
            }
        )
        return snapshot


def run_adaptive_experiment(
    run_config: AdaptiveRunConfig = AdaptiveRunConfig(),
) -> list[dict[str, tf.Tensor]]:
    """Train and save an independently reproducible adaptive experiment."""
    config = replace(
        run_config.experiment,
        results_dir=run_config.results_dir,
        initial_checkpoint_path=run_config.initial_checkpoint_path,
    )
    if not config.use_shared_trunk:
        raise ValueError("Set use_shared_trunk=True for adaptive training.")
    save_experiment_config(
        run_config,
        run_entry="experiments.simply_clamped_uniform_load.run_adaptive",
    )
    set_reproducible_seed(config.seed)
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    load_initial_weights(model, config.initial_checkpoint_path)
    checkpoint = BestValidationCheckpoint(config=config)
    trainer = AdaptiveMixedBoundaryTrainer(
        model=model,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=create_learning_rate_schedule(config)
        ),
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(config),
        parameters=config.plate,
        config=config.training,
        loss_weights=LossWeights(),
        physics_loss_config=config.physics_loss_config,
        mxy_consistency_config=None,
        shared_pcgrad_config=None,
        verbose=False,
        callbacks=(checkpoint,),
        experiment_config=config,
        adaptive_config=run_config.adaptive,
    )
    history = trainer.train()
    save_history_csv(history, config.loss_history_path)
    save_history_csv(checkpoint.records, config.validation_history_path)
    save_weights(model, config.final_weights_path)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return history


def _format_adaptive_loss_line(
    epoch: int,
    losses: dict[str, tf.Tensor],
) -> str:
    """Format losses, adaptive weights, and measured gradient norms."""
    keys = (
        "total_loss",
        "moment_loss",
        "shear_loss",
        "equilibrium_loss",
        "boundary_loss",
        "adaptive_moment_weight",
        "adaptive_shear_weight",
        "adaptive_equilibrium_weight",
        "adaptive_boundary_weight",
        "adaptive_grad_norm_moment",
        "adaptive_grad_norm_shear",
        "adaptive_grad_norm_equilibrium",
        "adaptive_grad_norm_boundary",
    )
    values = " ".join(
        f"{key}={float(losses[key].numpy()):.6e}" for key in keys
    )
    return f"adaptive epoch={epoch} {values}"


def _parse_args() -> argparse.Namespace:
    """Parse adaptive experiment command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=AdaptiveRunConfig().results_dir,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=200)
    parser.add_argument("--update-every", type=int, default=20)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--min-weight", type=float, default=0.1)
    parser.add_argument("--max-weight", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """Run adaptive training from the command line."""
    args = _parse_args()
    experiment = DEFAULT_CONFIG
    if args.epochs is not None:
        experiment = replace(
            experiment,
            training=replace(experiment.training, epochs=args.epochs),
        )
    run_config = AdaptiveRunConfig(
        experiment=experiment,
        adaptive=AdaptiveLossConfig(
            warmup_epochs=args.warmup_epochs,
            update_every=args.update_every,
            ema_decay=args.ema_decay,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        ),
        results_dir=args.results_dir,
        initial_checkpoint_path=args.initial_checkpoint,
    )
    history = run_adaptive_experiment(run_config)
    print(
        "simply_clamped_uniform_load_adaptive_complete "
        f"epochs={run_config.experiment.training.epochs} "
        f"results={run_config.results_dir} "
        f"final_total_loss="
        f"{float(history[-1]['total_loss'].numpy()):.6e}"
    )


if __name__ == "__main__":
    main()
