"""Train W-PINN on the uniform-load plate with mixed opposite edges."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
import time

import tensorflow as tf

from configs import save_experiment_config
from models import FIELD_NAMES

from .comparison_trainers import MixedBoundaryWPINNTrainer
from .config import ExperimentConfig
from .evaluate import make_test_grid
from .evaluate_w_pinn import evaluate_w_pinn_model, validation_score
from .problem import create_boundary_sampler, create_interior_sampler, create_load_fn
from .run import (
    create_learning_rate_schedule,
    load_initial_weights,
    save_history_csv,
    save_runtime_summary,
    save_weights,
    set_reproducible_seed,
)
from .w_pinn_config import W_PINN_CONFIG
from .w_pinn_problem import create_w_pinn_model


def run_experiment(config: ExperimentConfig = W_PINN_CONFIG) -> list[dict[str, tf.Tensor]]:
    """Train W-PINN and save independent comparison artifacts."""
    save_experiment_config(
        config,
        run_entry="experiments.simply_clamped_uniform_load.run_w_pinn",
    )
    set_reproducible_seed(config.seed)
    model = create_w_pinn_model(config)
    load_initial_weights(model, config.initial_checkpoint_path)
    checkpoint = BestWPINNValidationCheckpoint(config)
    trainer = MixedBoundaryWPINNTrainer(
        model=model,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=create_learning_rate_schedule(config)
        ),
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(config),
        parameters=config.plate,
        config=config.training,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        shared_pcgrad_config=config.shared_pcgrad,
        verbose=True,
        callbacks=(checkpoint,),
        experiment_config=config,
    )
    training_started = time.perf_counter()
    history = trainer.train()
    training_wall_time = time.perf_counter() - training_started
    save_history_csv(history, config.loss_history_path)
    save_history_csv(checkpoint.records, config.validation_history_path)
    save_weights(model, config.final_weights_path)
    save_runtime_summary(
        config=config,
        model_name="w_pinn",
        training_wall_time_seconds=training_wall_time,
        history=history,
        validation_records=checkpoint.records,
    )
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return history


@dataclass
class BestWPINNValidationCheckpoint:
    """Select W-PINN weights by mean six-field Levy-reference RelL2."""

    config: ExperimentConfig
    best_score: float = float("inf")
    best_epoch: int = 0
    records: list[dict[str, tf.Tensor]] = field(default_factory=list)

    def __call__(self, epoch: int, model: tf.keras.Model, loss_dict: dict[str, tf.Tensor]) -> None:
        del loss_dict
        if epoch % self.config.validation_interval and epoch != self.config.training.epochs:
            return
        validation_started = time.perf_counter()
        _, _, rel_l2 = evaluate_w_pinn_model(
            model,
            make_test_grid(self.config.validation_grid_size),
            self.config,
        )
        score = validation_score(rel_l2)
        record = {"epoch": tf.constant(epoch, tf.int32), "validation_score": tf.identity(score)}
        record["validation_seconds"] = tf.constant(
            time.perf_counter() - validation_started,
            dtype=tf.float64,
        )
        record.update({f"{name}_rel_l2": tf.identity(rel_l2[name]) for name in FIELD_NAMES})
        self.records.append(record)
        score_value = float(score.numpy())
        print(" ".join(
            [f"w_pinn_validation epoch={epoch}", f"validation_score={score_value:.6e}"]
            + [f"{name}_rel_l2={float(rel_l2[name].numpy()):.6e}" for name in FIELD_NAMES]
        ))
        if score_value < self.best_score:
            self.best_score = score_value
            self.best_epoch = epoch
            save_weights(model, self.config.best_validation_loss_weights_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=W_PINN_CONFIG.training.epochs)
    parser.add_argument("--interior-points", type=int, default=W_PINN_CONFIG.training.interior_points)
    parser.add_argument("--boundary-points-per-side", type=int, default=W_PINN_CONFIG.training.boundary_points_per_side)
    parser.add_argument("--log-every", type=int, default=W_PINN_CONFIG.training.log_every)
    parser.add_argument("--validation-interval", type=int, default=W_PINN_CONFIG.validation_interval)
    parser.add_argument("--validation-grid-size", type=int, default=W_PINN_CONFIG.validation_grid_size)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=W_PINN_CONFIG.results_dir)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    training = replace(
        W_PINN_CONFIG.training,
        epochs=args.epochs,
        interior_points=args.interior_points,
        boundary_points_per_side=args.boundary_points_per_side,
        log_every=args.log_every,
    )
    return replace(
        W_PINN_CONFIG,
        training=training,
        validation_interval=args.validation_interval,
        validation_grid_size=args.validation_grid_size,
        initial_checkpoint_path=args.initial_checkpoint,
        results_dir=args.results_dir,
    )


def main() -> None:
    config = _config_from_args(_parse_args())
    history = run_experiment(config)
    print(
        "simply_clamped_uniform_load_w_pinn "
        f"epochs={config.training.epochs} "
        f"final_total_loss={float(history[-1]['total_loss'].numpy()):.6e} "
        f"best_weights={config.best_validation_loss_weights_path}"
    )


if __name__ == "__main__":
    main()
