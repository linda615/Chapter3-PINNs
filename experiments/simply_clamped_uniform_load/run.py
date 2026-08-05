"""Train the uniformly loaded plate with mixed opposite-edge supports."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs import save_experiment_config
from losses import resolve_loss_weights
from models import FIELD_NAMES

from .config import (
    DEFAULT_CONFIG,
    SUPPORTED_LOAD_MODES,
    ExperimentConfig,
)
from .evaluate import evaluate_model, make_test_grid
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
)
from .trainer import MixedBoundaryPINNTrainer


def run_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
    run_entry: str = "experiments.simply_clamped_uniform_load.run",
    model_name: str = "mixed_pinn",
) -> list[dict[str, tf.Tensor]]:
    """Train the mixed-edge plate and save all reproducibility artifacts."""
    save_experiment_config(
        config,
        run_entry=run_entry,
    )
    set_reproducible_seed(config.seed)
    model = create_model(config)
    print(
        f"head_output_activation={model.output_activation}"
    )
    print(
        "field_output_scales "
        + " ".join(
            f"{name}={value:.9e}"
            for name, value in model.get_field_output_scales().items()
        )
    )
    physics_scales = config.physics_loss_config.residual_scales
    print(
        "physics_residual_rms_scales "
        f"enabled={float(config.physics_loss_config.normalize_residuals):.0f} "
        f"moment_x={physics_scales.moment_x:.9e} "
        f"moment_y={physics_scales.moment_y:.9e} "
        f"twisting_moment={physics_scales.twisting_moment:.9e} "
        f"shear_x={physics_scales.shear_x:.9e} "
        f"shear_y={physics_scales.shear_y:.9e} "
        f"equilibrium={physics_scales.equilibrium:.9e}"
    )
    boundary_scales = config.boundary_loss_config.residual_scales
    print(
        "boundary_residual_rms_scales "
        f"enabled={float(config.boundary_loss_config.normalize_residuals):.0f} "
        f"deflection={boundary_scales.deflection:.9e} "
        f"slope={boundary_scales.slope:.9e} "
        f"moment={boundary_scales.moment:.9e} "
        f"shear={boundary_scales.shear:.9e}"
    )
    load_initial_weights(model, config.initial_checkpoint_path)
    checkpoint = BestValidationCheckpoint(config=config)
    trainer = MixedBoundaryPINNTrainer(
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
        mxy_consistency_config=config.mxy_consistency_config,
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
        model_name=model_name,
        training_wall_time_seconds=training_wall_time,
        history=history,
        validation_records=checkpoint.records,
    )
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return history


@dataclass
class BestValidationCheckpoint:
    """Select weights by mean six-field Levy-reference relative L2."""

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
        """Evaluate and save weights at configured epochs."""
        del loss_dict
        if (
            epoch % self.config.validation_interval != 0
            and epoch != self.config.training.epochs
        ):
            return

        validation_started = time.perf_counter()
        xy = make_test_grid(self.config.validation_grid_size)
        _, _, rel_l2 = evaluate_model(model, xy, self.config)
        score = tf.add_n(
            [rel_l2[name] for name in FIELD_NAMES]
        ) / float(len(FIELD_NAMES))
        record = {"epoch": tf.constant(epoch, dtype=tf.int32)}
        record.update(
            {
                f"{name}_rel_l2": tf.identity(rel_l2[name])
                for name in FIELD_NAMES
            }
        )
        record["validation_score"] = tf.identity(score)
        record["validation_seconds"] = tf.constant(
            time.perf_counter() - validation_started,
            dtype=tf.float64,
        )
        self.records.append(record)

        score_value = float(score.numpy())
        print(_format_validation_line(epoch, rel_l2, score_value))
        if score_value < self.best_score:
            self.best_score = score_value
            self.best_epoch = epoch
            save_weights(
                model,
                self.config.best_validation_loss_weights_path,
            )


def create_learning_rate_schedule(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Create the fixed-epoch four-stage Adam schedule."""
    schedule = config.learning_rate_schedule
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(
            schedule.first_decay_epoch - 1,
            schedule.second_decay_epoch - 1,
            schedule.third_decay_epoch - 1,
        ),
        values=(
            schedule.initial_learning_rate,
            schedule.learning_rate_2,
            schedule.learning_rate_3,
            schedule.final_learning_rate,
        ),
    )


def save_history_csv(
    history: list[dict[str, tf.Tensor]],
    path: Path,
) -> None:
    """Save scalar Tensor history records to CSV."""
    if not history:
        raise ValueError("history must contain at least one record.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    key: _tensor_to_scalar(record.get(key, ""))
                    for key in fieldnames
                }
            )


def save_weights(model: tf.keras.Model, path: Path) -> None:
    """Save model weights after creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def save_runtime_summary(
    config: ExperimentConfig,
    model_name: str,
    training_wall_time_seconds: float,
    history: list[dict[str, tf.Tensor]],
    validation_records: list[dict[str, tf.Tensor]],
) -> Path:
    """Save comparable optimization, validation, and wall-clock timings."""
    cumulative_train_step_seconds = 0.0
    if history and "cumulative_train_step_seconds" in history[-1]:
        cumulative_train_step_seconds = float(
            history[-1]["cumulative_train_step_seconds"].numpy()
        )
    validation_seconds = sum(
        float(record["validation_seconds"].numpy())
        for record in validation_records
        if "validation_seconds" in record
    )
    validation_count = len(validation_records)
    summary = {
        "model_name": model_name,
        "epochs": config.training.epochs,
        "interior_points": config.training.interior_points,
        "boundary_points_per_side": (
            config.training.boundary_points_per_side
        ),
        "validation_interval": config.validation_interval,
        "validation_grid_size": config.validation_grid_size,
        "training_wall_time_seconds": training_wall_time_seconds,
        "wall_seconds_per_epoch": (
            training_wall_time_seconds / config.training.epochs
        ),
        "cumulative_train_step_seconds": cumulative_train_step_seconds,
        "mean_train_step_seconds": (
            cumulative_train_step_seconds / config.training.epochs
        ),
        "cumulative_validation_seconds": validation_seconds,
        "mean_validation_seconds": (
            validation_seconds / validation_count
            if validation_count
            else 0.0
        ),
        "validation_count": validation_count,
    }
    path = config.results_dir / "runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(
        "runtime_summary "
        f"model={model_name} "
        f"wall_seconds={training_wall_time_seconds:.6f} "
        f"mean_train_step_seconds="
        f"{summary['mean_train_step_seconds']:.6f} "
        f"mean_validation_seconds="
        f"{summary['mean_validation_seconds']:.6f} "
        f"path={path}"
    )
    return path


def load_initial_weights(
    model: tf.keras.Model,
    checkpoint_path: Path | None,
) -> None:
    """Build the model and optionally load weights for continued training."""
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initial checkpoint does not exist: {checkpoint_path}"
        )
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(checkpoint_path))
    print(f"loaded_initial_checkpoint={checkpoint_path}")


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _tensor_to_scalar(value):
    """Convert a scalar Tensor into a CSV-compatible value."""
    if not tf.is_tensor(value):
        return value
    array = value.numpy()
    if array.shape == ():
        return array.item()
    return array.tolist()


def _format_validation_line(
    epoch: int,
    rel_l2: dict[str, tf.Tensor],
    score: float,
) -> str:
    """Format periodic Levy-reference validation metrics."""
    parts = [
        f"validation epoch={epoch}",
        f"validation_score={score:.6e}",
    ]
    parts.extend(
        f"{name}_rel_l2={float(rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    """Parse lightweight command-line overrides."""
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
    parser.add_argument(
        "--load-mode",
        choices=SUPPORTED_LOAD_MODES,
        default=DEFAULT_CONFIG.load_mode,
        help=(
            "Use 'uniform' for q=q0 or 'matched_levy' for the finite "
            "odd-sine load represented by the Levy reference."
        ),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Load model weights before starting a new optimizer run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_CONFIG.results_dir,
        help="Write checkpoints and histories to this directory.",
    )
    parser.add_argument(
        "--model-grouping",
        choices=("one", "two", "three", "six"),
        default=DEFAULT_CONFIG.model_grouping,
        help="Select the Mixed-PINN output grouping.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Return the effective command-line configuration."""
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
        load_mode=args.load_mode,
        initial_checkpoint_path=args.initial_checkpoint,
        results_dir=args.results_dir,
        model_grouping=args.model_grouping,
    )


def main() -> None:
    """Run mixed-edge training from the command line."""
    config = _config_from_args(_parse_args())
    history = run_experiment(config)
    simple_sides = ",".join(side.value for side in config.simple_sides)
    clamped_sides = ",".join(side.value for side in config.clamped_sides)
    print(
        "simply_clamped_uniform_load "
        f"epochs={config.training.epochs} "
        f"simple_sides={simple_sides} "
        f"clamped_sides={clamped_sides} "
        f"load_mode={config.load_mode} "
        f"initial_total_loss={float(history[0]['total_loss'].numpy()):.6e} "
        f"final_total_loss={float(history[-1]['total_loss'].numpy()):.6e} "
        f"best_weights={config.best_validation_loss_weights_path} "
        f"final_weights={config.final_weights_path} "
        f"history={config.loss_history_path}"
    )


if __name__ == "__main__":
    main()
