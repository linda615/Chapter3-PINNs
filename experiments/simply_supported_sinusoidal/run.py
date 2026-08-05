"""Run the minimal simply supported sinusoidal plate experiment."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs import save_experiment_config
from models import FIELD_NAMES
from trainer import LBFGSResult, PINNLBFGSTrainer, PINNTrainer, TrainingConfig

from .config import DEFAULT_CONFIG, ExperimentConfig, LearningRateScheduleConfig
from .evaluate import evaluate_model, make_test_grid
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
    get_boundary_type,
)


CONTINUE_EPOCHS = 2000


def run_experiment(config: ExperimentConfig = DEFAULT_CONFIG) -> list[dict[str, tf.Tensor]]:
    """Run a minimal training job and return logged loss history."""
    save_experiment_config(
        config,
        run_entry="experiments.simply_supported_sinusoidal.run",
    )
    set_reproducible_seed(config.seed)
    model = create_model(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=create_learning_rate_schedule(config))
    best_total_loss_checkpoint = BestTotalLossCheckpoint(config)
    best_checkpoint = BestValidationCheckpoint(config)
    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(),
        parameters=config.plate,
        config=config.training,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        verbose=True,
        callbacks=[best_total_loss_checkpoint, best_checkpoint],
    )
    history = trainer.train()
    save_history_csv(history, config.loss_history_path)
    save_history_csv(best_checkpoint.records, config.validation_history_path)
    save_weights(model, config.final_weights_path)
    if config.use_lbfgs:
        run_lbfgs_finetuning(
            config=config,
            model=model,
            initial_weights_path=config.best_adam_validation_loss_weights_path,
            initial_best_score=best_checkpoint.best_score,
        )
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return history


@dataclass
class BestTotalLossCheckpoint:
    """Save the checkpoint with the lowest training total loss."""

    config: ExperimentConfig
    best_loss: float = float("inf")
    best_epoch: int = 0

    def __call__(
        self,
        epoch: int,
        model: tf.keras.Model,
        loss_dict: dict[str, tf.Tensor],
    ) -> None:
        """Save weights whenever the current total loss improves."""
        total_loss = loss_dict.get("total_loss")
        if total_loss is None:
            raise KeyError("loss_dict must contain 'total_loss' for checkpoint monitoring.")

        loss_value = float(total_loss.numpy())
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_epoch = epoch
            save_weights(model, self.config.best_total_loss_weights_path)


@dataclass
class BestValidationCheckpoint:
    """Evaluate periodically and save the checkpoint with the best validation score."""

    config: ExperimentConfig
    best_score: float = float("inf")
    best_epoch: int = 0
    records: list[dict[str, tf.Tensor]] = field(default_factory=list)

    def __call__(
        self,
        epoch: int,
        model: tf.keras.Model,
        loss_dict: dict[str, tf.Tensor],
    ) -> None:
        """Evaluate every configured interval and save weights when score improves."""
        del loss_dict
        if epoch % self.config.validation_interval != 0 and epoch != self.config.training.epochs:
            return

        xy = make_test_grid(grid_size=self.config.validation_grid_size)
        _, _, rel_l2 = evaluate_model(model, xy, config=self.config)
        validation_score = tf.add_n([rel_l2[field] for field in FIELD_NAMES]) / float(len(FIELD_NAMES))
        record = {"epoch": tf.constant(epoch, dtype=tf.int32)}
        record.update({f"{field}_rel_l2": tf.identity(rel_l2[field]) for field in FIELD_NAMES})
        record["validation_score"] = tf.identity(validation_score)
        self.records.append(record)

        score = float(validation_score.numpy())
        print(_format_validation_line(epoch, rel_l2, score))
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            save_weights(model, self.config.best_adam_validation_loss_weights_path)
            save_weights(model, self.config.best_validation_loss_weights_path)
            save_weights(model, self.config.checkpoints_dir / "best.weights.h5")
            save_weights(model, self.config.checkpoints_dir / f"best_epoch_{epoch}.weights.h5")


@dataclass
class BestLBFGSValidationCheckpoint:
    """Save L-BFGS weights when validation score improves."""

    config: ExperimentConfig
    best_score: float = float("inf")
    global_best_score: float = float("inf")
    best_iteration: int = 0
    records: list[dict[str, tf.Tensor]] = field(default_factory=list)

    def __call__(
        self,
        iteration: int,
        model: tf.keras.Model,
        loss_dict: dict[str, tf.Tensor],
    ) -> None:
        """Evaluate periodically during L-BFGS fine tuning."""
        del loss_dict
        if (
            iteration % self.config.validation_interval != 0
            and iteration != self.config.lbfgs.max_iterations
        ):
            return
        self.evaluate_and_maybe_save(iteration, model)

    def evaluate_and_maybe_save(
        self,
        iteration: int,
        model: tf.keras.Model,
    ) -> None:
        """Evaluate L-BFGS validation score and save improved weights."""
        xy = make_test_grid(grid_size=self.config.validation_grid_size)
        _, _, rel_l2 = evaluate_model(model, xy, config=self.config)
        validation_score = tf.add_n([rel_l2[field] for field in FIELD_NAMES]) / float(len(FIELD_NAMES))
        record = {"iteration": tf.constant(iteration, dtype=tf.int32)}
        record.update({f"{field}_rel_l2": tf.identity(rel_l2[field]) for field in FIELD_NAMES})
        record["validation_score"] = tf.identity(validation_score)
        self.records.append(record)

        score = float(validation_score.numpy())
        print(_format_lbfgs_validation_line(iteration, rel_l2, score))
        if score < self.best_score:
            self.best_score = score
            self.best_iteration = iteration
            save_weights(model, self.config.best_lbfgs_validation_loss_weights_path)
            save_weights(model, self.config.checkpoints_dir / f"best_lbfgs_iter_{iteration}.weights.h5")
        if score < self.global_best_score:
            self.global_best_score = score
            save_weights(model, self.config.best_validation_loss_weights_path)
            save_weights(model, self.config.checkpoints_dir / "best.weights.h5")


def run_lbfgs_finetuning(
    config: ExperimentConfig = DEFAULT_CONFIG,
    model: tf.keras.Model | None = None,
    initial_weights_path: str | Path | None = None,
    initial_best_score: float = float("inf"),
) -> LBFGSResult:
    """Load Adam-best weights and fine tune them with fixed-batch L-BFGS."""
    initial_weights_path = Path(initial_weights_path or config.best_adam_validation_loss_weights_path)
    save_experiment_config(
        config,
        run_entry="experiments.simply_supported_sinusoidal.run.lbfgs",
        metadata={"initial_weights_path": initial_weights_path},
    )
    if not initial_weights_path.exists():
        raise FileNotFoundError(
            f"Cannot start L-BFGS because Adam-best weights were not found: {initial_weights_path}"
        )

    if model is None:
        set_reproducible_seed(config.seed)
        model = create_model(config)
        model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(initial_weights_path))

    lbfgs_checkpoint = BestLBFGSValidationCheckpoint(
        config=config,
        global_best_score=initial_best_score,
    )
    lbfgs_trainer = PINNLBFGSTrainer(
        model=model,
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(),
        parameters=config.plate,
        training_config=config.training,
        lbfgs_config=config.lbfgs,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        verbose=True,
        callbacks=[lbfgs_checkpoint],
    )
    result = lbfgs_trainer.train()
    lbfgs_checkpoint.evaluate_and_maybe_save(result.iterations, model)
    save_history_csv(result.history, config.lbfgs_history_path)
    save_history_csv(lbfgs_checkpoint.records, config.lbfgs_validation_history_path)
    save_weights(model, config.final_lbfgs_weights_path)
    print(
        "lbfgs_summary "
        f"success={result.success} "
        f"iterations={result.iterations} "
        f"function_evaluations={result.function_evaluations} "
        f"final_loss={result.final_loss:.6e} "
        f"message={result.message}"
    )
    return result


def continue_training(
    config: ExperimentConfig = DEFAULT_CONFIG,
    initial_weights_path: str | Path | None = None,
    extra_epochs: int = CONTINUE_EPOCHS,
) -> list[dict[str, tf.Tensor]]:
    """Continue training from existing weights using the minimum learning rate."""
    if extra_epochs <= 0:
        raise ValueError(f"extra_epochs must be positive, got {extra_epochs}.")

    initial_weights_path = Path(initial_weights_path or config.best_weights_path)
    save_experiment_config(
        config,
        run_entry="experiments.simply_supported_sinusoidal.run.continue",
        metadata={
            "initial_weights_path": initial_weights_path,
            "extra_epochs": extra_epochs,
        },
    )
    if not initial_weights_path.exists():
        raise FileNotFoundError(
            f"Cannot continue training because weights were not found: {initial_weights_path}"
        )

    set_reproducible_seed(config.seed)
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(initial_weights_path))

    minimum_learning_rate = config.learning_rate_schedule.final_learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=minimum_learning_rate)
    continue_config = TrainingConfig(
        epochs=extra_epochs,
        interior_points=config.training.interior_points,
        boundary_points_per_side=config.training.boundary_points_per_side,
        boundary_type=config.training.boundary_type,
        log_every=config.training.log_every,
        seed=config.training.seed,
    )
    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(),
        parameters=config.plate,
        config=continue_config,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        verbose=True,
    )

    history = trainer.train()
    target_epoch = _infer_epoch_from_weight_path(initial_weights_path, config.training.epochs)
    final_epoch = target_epoch + extra_epochs
    continued_history_path = config.history_dir / f"loss_continue_{target_epoch}_to_{final_epoch}.csv"
    continued_weights_path = config.checkpoints_dir / f"best_epoch_{final_epoch}.weights.h5"
    save_history_csv(history, continued_history_path)
    save_weights(model, continued_weights_path)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return history


def create_learning_rate_schedule(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Create the configured fixed-epoch four-stage Adam schedule."""
    schedule = config.learning_rate_schedule
    # Optimizer iterations are zero-based and there is one update per epoch.
    first_boundary = schedule.first_decay_epoch - 1
    second_boundary = schedule.second_decay_epoch - 1
    third_boundary = schedule.third_decay_epoch - 1
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(first_boundary, second_boundary, third_boundary),
        values=(
            schedule.initial_learning_rate,
            schedule.learning_rate_2,
            schedule.learning_rate_3,
            schedule.final_learning_rate,
        ),
    )


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds for repeatable experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Older TensorFlow builds or already-initialized runtimes may not allow this.
        pass


def save_history_csv(history: list[dict[str, tf.Tensor]], path: str | Path) -> None:
    """Save logged training losses to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("history must contain at least one loss snapshot.")

    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({key: _tensor_to_scalar(value) for key, value in item.items()})


def save_weights(model: tf.keras.Model, path: str | Path) -> None:
    """Save model weights to a Keras weights file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def _tensor_to_scalar(value: tf.Tensor):
    """Convert a scalar Tensor to a Python scalar for CSV output."""
    if not tf.is_tensor(value):
        return value
    array = value.numpy()
    if array.shape == ():
        return array.item()
    return array.tolist()


def _infer_epoch_from_weight_path(path: Path, fallback_epoch: int) -> int:
    """Infer epoch number from names like ``best_epoch_4000.weights.h5``."""
    stem = path.name.split(".weights.h5")[0]
    prefix = "best_epoch_"
    if stem.startswith(prefix):
        try:
            return int(stem[len(prefix):])
        except ValueError:
            return fallback_epoch
    return fallback_epoch


def _format_validation_line(
    epoch: int,
    rel_l2: dict[str, tf.Tensor],
    validation_score: float,
) -> str:
    """Format validation RelL2 metrics for console logging."""
    parts = [f"validation epoch={epoch}", f"validation_score={validation_score:.6e}"]
    for field in FIELD_NAMES:
        parts.append(f"{field}_rel_l2={float(rel_l2[field].numpy()):.6e}")
    return " ".join(parts)


def _format_lbfgs_validation_line(
    iteration: int,
    rel_l2: dict[str, tf.Tensor],
    validation_score: float,
) -> str:
    """Format L-BFGS validation RelL2 metrics for console logging."""
    parts = [f"lbfgs_validation iteration={iteration}", f"validation_score={validation_score:.6e}"]
    for field in FIELD_NAMES:
        parts.append(f"{field}_rel_l2={float(rel_l2[field].numpy()):.6e}")
    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    """Parse lightweight command-line training overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument(
        "--interior-points",
        type=int,
        default=DEFAULT_CONFIG.training.interior_points,
    )
    parser.add_argument(
        "--boundary-points-per-side",
        type=int,
        default=DEFAULT_CONFIG.training.boundary_points_per_side,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_CONFIG.training.log_every,
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=DEFAULT_CONFIG.validation_interval,
    )
    parser.add_argument(
        "--validation-grid-size",
        type=int,
        default=DEFAULT_CONFIG.validation_grid_size,
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Return a config carrying command-line training overrides."""
    training = replace(
        DEFAULT_CONFIG.training,
        epochs=args.epochs,
        interior_points=args.interior_points,
        boundary_points_per_side=args.boundary_points_per_side,
        log_every=args.log_every,
    )
    return replace(
        DEFAULT_CONFIG,
        training=training,
        validation_interval=args.validation_interval,
        validation_grid_size=args.validation_grid_size,
    )


def main() -> None:
    """Command-line entry point for training the experiment from scratch."""
    config = _config_from_args(_parse_args())
    history = run_experiment(config)
    first = history[0]["total_loss"]
    last = history[-1]["total_loss"]
    print(
        "simply_supported_sinusoidal "
        f"epochs={config.training.epochs} "
        f"boundary={get_boundary_type(config).value} "
        f"initial_total_loss={float(first.numpy()):.6e} "
        f"final_total_loss={float(last.numpy()):.6e} "
        f"normalize_physics_residuals={config.normalize_physics_residuals} "
        f"use_lbfgs={config.use_lbfgs} "
        f"adam_best_weights={config.best_adam_validation_loss_weights_path} "
        f"best_weights={config.best_weights_path} "
        f"final_weights={config.final_weights_path} "
        f"final_lbfgs_weights={config.final_lbfgs_weights_path} "
        f"history={config.loss_history_path} "
        f"validation={config.validation_history_path}"
    )


if __name__ == "__main__":
    main()
