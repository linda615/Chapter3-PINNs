"""Run the nonuniform-support clamped rock-roof training smoke test."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from configs import save_experiment_config
from losses import compute_physics_loss, resolve_loss_weights
from models import FIELD_NAMES
from trainer import PINNTrainer

from .config import (
    DEFAULT_CONFIG,
    FOUNDATION_REFERENCE_PATH,
    FOUNDATION_RESULTS_DIR,
    NO_FOUNDATION_REFERENCE_PATH,
    NO_FOUNDATION_RESULTS_DIR,
    ExperimentConfig,
)
from .problem import (
    active_foundation_stiffness,
    create_boundary_sampler,
    create_foundation_fn,
    create_interior_sampler,
    create_load_fn,
    create_model,
    rock_roof_load,
)


def run_experiment(config: ExperimentConfig = DEFAULT_CONFIG) -> List[Dict[str, tf.Tensor]]:
    """Run Adam training and return training snapshots."""
    save_experiment_config(
        config,
        run_entry="experiments.clamped_rock_roof_nonuniform_support.run",
    )
    set_reproducible_seed(config.seed)
    model = create_model(config)
    print(f"head_output_activation={model.output_activation}")
    print(
        "field_output_scales "
        + " ".join(
            f"{name}={value:.9e}"
            for name, value in model.get_field_output_scales().items()
        )
    )
    physics_scales = config.physics_loss_config.residual_scales
    print(
        "physics_residual_scales "
        f"enabled={float(config.physics_loss_config.normalize_residuals):.0f} "
        f"moment_x={physics_scales.moment_x:.9e} "
        f"moment_y={physics_scales.moment_y:.9e} "
        f"twisting_moment={physics_scales.twisting_moment:.9e} "
        f"shear_x={physics_scales.shear_x:.9e} "
        f"shear_y={physics_scales.shear_y:.9e} "
        f"equilibrium={physics_scales.equilibrium:.9e}"
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=create_learning_rate_schedule(config))
    load_fn = create_load_fn(config)
    foundation_fn = create_foundation_fn(config)
    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=load_fn,
        foundation_fn=foundation_fn,
        parameters=config.plate,
        config=config.training,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        mxy_consistency_config=config.mxy_consistency_config,
        shared_pcgrad_config=config.shared_pcgrad,
        verbose=False,
    )

    monitor = FixedPhysicsMonitor(config=config, model=model)
    fdm_monitor = None
    if (
        config.fdm_evaluation_every > 0
        and config.training.epochs >= config.fdm_evaluation_every
    ):
        fdm_monitor = FDMReferenceMonitor(config=config, model=model)

    history = []
    for epoch in range(1, config.training.epochs + 1):
        interior_xy, boundary_batch = trainer.sample_epoch(epoch)
        current_weights = resolve_loss_weights(config.loss_weights, epoch)
        loss_dict = trainer.train_step(
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            loss_weights=current_weights,
            epoch=epoch,
        )
        monitor_snapshot = monitor.evaluate(epoch)
        loss_dict.update(monitor_snapshot)

        if _should_log(epoch, config):
            snapshot = _snapshot_losses(epoch, loss_dict)
            history.append(snapshot)
            print(_format_loss_line(epoch, snapshot))
            # Persist progress during long runs so an interruption does not
            # discard the complete in-memory training history.
            save_history_csv(history, config.loss_history_path)
            save_history_csv(monitor.records, config.monitor_history_path)

        if fdm_monitor is not None and fdm_monitor.should_evaluate(epoch):
            fdm_snapshot = fdm_monitor.evaluate(epoch)
            print(_format_fdm_validation_line(fdm_snapshot))
            save_history_csv(
                fdm_monitor.records,
                config.fdm_validation_history_path,
            )

    save_history_csv(history, config.loss_history_path)
    save_history_csv(monitor.records, config.monitor_history_path)
    if fdm_monitor is not None:
        save_history_csv(
            fdm_monitor.records,
            config.fdm_validation_history_path,
        )
    save_weights(model, config.final_weights_path)
    return history


@dataclass
class FixedPhysicsMonitor:
    """Monitor physics loss on one deterministic grid and save the best model."""

    config: ExperimentConfig
    model: tf.keras.Model
    best_loss: float = float("inf")
    best_epoch: int = 0

    def __post_init__(self) -> None:
        """Build the fixed monitor grid once."""
        self.xy = make_grid(self.config.monitor_grid_size)
        self.load_fn = create_load_fn(self.config)
        self.foundation_fn = create_foundation_fn(self.config)
        self.q_stats = tensor_stats(rock_roof_load(self.xy, config=self.config), prefix="q")
        if self.foundation_fn is None:
            k = tf.zeros((tf.shape(self.xy)[0], 1), dtype=self.xy.dtype)
        else:
            k = active_foundation_stiffness(self.xy, config=self.config)
        self.k_stats = tensor_stats(k, prefix="k")
        self.records = []

    def evaluate(self, epoch: int) -> Dict[str, tf.Tensor]:
        """Evaluate fixed-grid physics loss and update the best checkpoint."""
        losses = compute_physics_loss(
            model=self.model,
            xy=self.xy,
            load_fn=self.load_fn,
            parameters=self.config.plate,
            foundation_fn=self.foundation_fn,
            loss_config=self.config.physics_loss_config,
            mxy_consistency_config=self.config.mxy_consistency_config,
            training=False,
        )
        snapshot = {
            "foundation_enabled": tf.constant(int(self.config.foundation_enabled), dtype=tf.int32),
            "fixed_monitor_physics_loss": tf.identity(losses["physics_loss"]),
            "fixed_monitor_moment_loss": tf.identity(losses["moment_loss"]),
            "fixed_monitor_moment_x_loss": tf.identity(losses["moment_x_loss"]),
            "fixed_monitor_moment_y_loss": tf.identity(losses["moment_y_loss"]),
            "fixed_monitor_twisting_moment_loss": tf.identity(
                losses["twisting_moment_loss"]
            ),
            "fixed_monitor_shear_loss": tf.identity(losses["shear_loss"]),
            "fixed_monitor_equilibrium_loss": tf.identity(losses["equilibrium_loss"]),
            "fixed_monitor_mxy_consistency_loss": tf.identity(
                losses["mxy_consistency_loss"]
            ),
            "fixed_monitor_mxy_consistency_scale": tf.identity(
                losses["mxy_consistency_scale"]
            ),
        }
        snapshot.update(self.q_stats)
        snapshot.update(self.k_stats)
        record = {"epoch": tf.constant(epoch, dtype=tf.int32)}
        record.update(snapshot)
        self.records.append(record)

        monitor_loss = float(losses["physics_loss"].numpy())
        if monitor_loss < self.best_loss:
            self.best_loss = monitor_loss
            self.best_epoch = epoch
            save_weights(self.model, self.config.best_fixed_physics_loss_weights_path)
        return snapshot


@dataclass
class FDMReferenceMonitor:
    """Evaluate PINN fields against one preloaded FDM reference grid."""

    config: ExperimentConfig
    model: tf.keras.Model
    checkpoint_path: Path | None = None
    best_score: float = float("inf")
    best_epoch: int = 0

    def __post_init__(self) -> None:
        """Load and validate the FDM reference exactly once."""
        if self.config.fdm_evaluation_every <= 0:
            raise ValueError("fdm_evaluation_every must be positive.")
        if self.config.fdm_evaluation_batch_size <= 0:
            raise ValueError("fdm_evaluation_batch_size must be positive.")
        if self.checkpoint_path is None:
            self.checkpoint_path = self.config.best_fdm_validation_weights_path

        reference = _load_fdm_reference(self.config.reference_path)
        _validate_fdm_reference(reference, self.config)
        self.xy = tf.concat(
            (
                tf.convert_to_tensor(reference["x"], dtype=tf.float32),
                tf.convert_to_tensor(reference["y"], dtype=tf.float32),
            ),
            axis=1,
        )
        self.reference_fields = {
            field: tf.convert_to_tensor(reference[field], dtype=tf.float64)
            for field in FIELD_NAMES
        }
        self.reference_squared_norms = {
            field: tf.reduce_sum(tf.square(self.reference_fields[field]))
            for field in FIELD_NAMES
        }
        self.records = []

    def should_evaluate(self, epoch: int) -> bool:
        """Return whether this epoch is an FDM comparison epoch."""
        return epoch % self.config.fdm_evaluation_every == 0

    def evaluate(self, epoch: int) -> Dict[str, tf.Tensor]:
        """Compute six field-wise relative L2 errors in bounded-size batches."""
        squared_errors = {
            field: tf.constant(0.0, dtype=tf.float64)
            for field in FIELD_NAMES
        }
        point_count = int(self.xy.shape[0])
        batch_size = self.config.fdm_evaluation_batch_size

        for start in range(0, point_count, batch_size):
            stop = min(start + batch_size, point_count)
            predicted = self.model(self.xy[start:stop], training=False)
            for field in FIELD_NAMES:
                difference = (
                    tf.cast(predicted[field], tf.float64)
                    - self.reference_fields[field][start:stop]
                )
                squared_errors[field] = (
                    squared_errors[field]
                    + tf.reduce_sum(tf.square(difference))
                )

        rel_l2 = {
            field: tf.sqrt(
                squared_errors[field] / self.reference_squared_norms[field]
            )
            for field in FIELD_NAMES
        }
        validation_score = tf.add_n(
            [rel_l2[field] for field in FIELD_NAMES]
        ) / tf.cast(len(FIELD_NAMES), tf.float64)
        record = {"epoch": tf.constant(epoch, dtype=tf.int32)}
        record.update(
            {f"{field}_rel_l2": tf.identity(rel_l2[field]) for field in FIELD_NAMES}
        )
        record["fdm_validation_score"] = tf.identity(validation_score)
        self.records.append(record)

        score = float(validation_score.numpy())
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            save_weights(
                self.model,
                self.checkpoint_path,
            )
        return record


def make_grid(grid_size: int = 41, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Create a regular unit-square grid with shape ``(grid_size**2, 2)``."""
    values = tf.linspace(tf.cast(0.0, dtype), tf.cast(1.0, dtype), grid_size)
    x_grid, y_grid = tf.meshgrid(values, values, indexing="xy")
    return tf.stack([tf.reshape(x_grid, (-1,)), tf.reshape(y_grid, (-1,))], axis=1)


def tensor_stats(value: tf.Tensor, prefix: str) -> Dict[str, tf.Tensor]:
    """Return basic scalar statistics for a Tensor."""
    value = tf.convert_to_tensor(value)
    return {
        f"{prefix}_min": tf.reduce_min(value),
        f"{prefix}_max": tf.reduce_max(value),
        f"{prefix}_mean": tf.reduce_mean(value),
        f"{prefix}_rms": tf.sqrt(tf.reduce_mean(tf.square(value))),
    }


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
        pass


def save_history_csv(history: List[Dict[str, tf.Tensor]], path: str | Path) -> None:
    """Save Tensor scalar snapshots to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("history must contain at least one snapshot.")

    fieldnames = []
    for item in history:
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({key: _tensor_to_scalar(item.get(key, "")) for key in fieldnames})


def save_weights(model: tf.keras.Model, path: str | Path) -> None:
    """Save Keras weights to the requested path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def _snapshot_losses(epoch: int, loss_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Copy one training snapshot."""
    snapshot = {"epoch": tf.constant(epoch, dtype=tf.int32)}
    snapshot.update({name: tf.identity(value) for name, value in loss_dict.items()})
    return snapshot


def _should_log(epoch: int, config: ExperimentConfig) -> bool:
    """Return whether the epoch should be printed and stored."""
    return (
        epoch == 1
        or epoch == config.training.epochs
        or epoch % config.training.log_every == 0
    )


def _format_loss_line(epoch: int, loss_dict: Dict[str, tf.Tensor]) -> str:
    """Format a compact console loss line."""
    keys = (
        "foundation_enabled",
        "pcgrad_active",
        "pcgrad_conflict_count",
        "pcgrad_conflict_rate",
        "pcgrad_projection_count",
        "q_rms",
        "k_min",
        "k_max",
        "mxy_consistency_weight",
        "total_loss",
        "physics_loss",
        "moment_loss",
        "moment_x_loss",
        "moment_y_loss",
        "twisting_moment_loss",
        "mxy_consistency_loss",
        "mxy_consistency_scale",
        "weighted_mxy_consistency_loss",
        "shear_loss",
        "equilibrium_loss",
        "boundary_loss",
        "deflection_bc_loss",
        "slope_bc_loss",
        "moment_bc_loss",
        "shear_bc_loss",
        "fixed_monitor_physics_loss",
        "fixed_monitor_moment_loss",
        "fixed_monitor_moment_x_loss",
        "fixed_monitor_moment_y_loss",
        "fixed_monitor_twisting_moment_loss",
        "fixed_monitor_mxy_consistency_loss",
        "fixed_monitor_mxy_consistency_scale",
        "fixed_monitor_shear_loss",
        "fixed_monitor_equilibrium_loss",
    )
    parts = [f"epoch={epoch}"]
    for key in keys:
        if key in loss_dict:
            parts.append(f"{key}={float(tf.cast(loss_dict[key], tf.float32).numpy()):.6e}")
    return " ".join(parts)


def _format_fdm_validation_line(snapshot: Dict[str, tf.Tensor]) -> str:
    """Format periodic PINN-versus-FDM relative L2 errors."""
    epoch = int(snapshot["epoch"].numpy())
    score = float(snapshot["fdm_validation_score"].numpy())
    parts = [
        f"fdm_validation epoch={epoch}",
        f"fdm_validation_score={score:.6e}",
    ]
    for field in FIELD_NAMES:
        parts.append(
            f"{field}_rel_l2={float(snapshot[f'{field}_rel_l2'].numpy()):.6e}"
        )
    return " ".join(parts)


def _load_fdm_reference(path: str | Path) -> Dict[str, np.ndarray]:
    """Load required FDM arrays for periodic training evaluation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FDM reference file does not exist: {path}. Generate it before "
            "starting periodic FDM validation."
        )

    field_names = ("x", "y", "q", "k") + tuple(FIELD_NAMES)
    with np.load(str(path), allow_pickle=False) as data:
        required = field_names + ("D", "nu")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"FDM reference file {path} is missing fields: {missing}.")
        reference = {
            name: np.asarray(data[name], dtype=np.float64).reshape((-1, 1))
            for name in field_names
        }
        reference["D"] = np.asarray(data["D"], dtype=np.float64)
        reference["nu"] = np.asarray(data["nu"], dtype=np.float64)
    return reference


def _validate_fdm_reference(
    reference: Dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """Check that FDM data describes the same problem as the active training."""
    point_count = reference["x"].shape[0]
    for name in ("x", "y", "q", "k") + tuple(FIELD_NAMES):
        value = reference[name]
        if value.shape != (point_count, 1):
            raise ValueError(
                f"FDM reference field {name} must have shape "
                f"({point_count}, 1), got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"FDM reference field {name} contains non-finite values.")

    reference_D = float(np.asarray(reference["D"]).reshape(()))
    reference_nu = float(np.asarray(reference["nu"]).reshape(()))
    if not np.isclose(reference_D, config.plate.D, rtol=1e-12, atol=1e-14):
        raise ValueError(
            f"FDM reference D={reference_D} does not match training D={config.plate.D}."
        )
    if not np.isclose(reference_nu, config.plate.nu, rtol=1e-12, atol=1e-14):
        raise ValueError(
            f"FDM reference nu={reference_nu} does not match training nu={config.plate.nu}."
        )

    xy = tf.concat(
        (
            tf.convert_to_tensor(reference["x"], dtype=tf.float64),
            tf.convert_to_tensor(reference["y"], dtype=tf.float64),
        ),
        axis=1,
    )
    expected_q = rock_roof_load(xy, config=config).numpy()
    expected_k = active_foundation_stiffness(xy, config=config).numpy()
    _assert_matching_reference_field("q", reference["q"], expected_q)
    _assert_matching_reference_field("k", reference["k"], expected_k)

    for field in FIELD_NAMES:
        if np.linalg.norm(reference[field]) <= np.finfo(np.float64).eps:
            raise ValueError(
                f"FDM reference field {field} has zero L2 norm; relative L2 "
                "error is undefined."
            )


def _assert_matching_reference_field(
    name: str,
    reference: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Raise a clear error when reference and training problem fields differ."""
    if np.allclose(reference, expected, rtol=1e-7, atol=1e-10):
        return
    max_difference = float(np.max(np.abs(reference - expected)))
    raise ValueError(
        f"FDM reference {name} does not match the active training problem "
        f"(max absolute difference {max_difference:.6e}). Regenerate "
        "rock_roof_reference.npz with the same foundation configuration."
    )


def _tensor_to_scalar(value):
    """Convert scalar tensors to Python values for CSV output."""
    if isinstance(value, str):
        return value
    if not tf.is_tensor(value):
        return value
    array = value.numpy()
    if array.shape == ():
        return array.item()
    return array.tolist()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create a config with command-line overrides."""
    if args.foundation_enabled:
        results_dir = FOUNDATION_RESULTS_DIR
        reference_path = FOUNDATION_REFERENCE_PATH
    else:
        results_dir = NO_FOUNDATION_RESULTS_DIR
        reference_path = NO_FOUNDATION_REFERENCE_PATH
    training = replace(
        DEFAULT_CONFIG.training,
        epochs=args.epochs,
        interior_points=args.interior_points,
        boundary_points_per_side=args.boundary_points_per_side,
        log_every=args.log_every,
    )
    return replace(
        DEFAULT_CONFIG,
        foundation_enabled=args.foundation_enabled,
        fdm_evaluation_every=args.fdm_evaluation_every,
        fdm_evaluation_batch_size=args.fdm_evaluation_batch_size,
        model_grouping=args.model_grouping,
        results_dir=results_dir,
        reference_path=reference_path,
        training=training,
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--interior-points", type=int, default=DEFAULT_CONFIG.training.interior_points)
    parser.add_argument(
        "--boundary-points-per-side",
        type=int,
        default=DEFAULT_CONFIG.training.boundary_points_per_side,
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_CONFIG.training.log_every)
    foundation_group = parser.add_mutually_exclusive_group()
    foundation_group.add_argument(
        "--foundation-enabled",
        dest="foundation_enabled",
        action="store_true",
    )
    foundation_group.add_argument(
        "--no-foundation",
        dest="foundation_enabled",
        action="store_false",
    )
    parser.set_defaults(foundation_enabled=DEFAULT_CONFIG.foundation_enabled)
    parser.add_argument(
        "--fdm-evaluation-every",
        type=int,
        default=DEFAULT_CONFIG.fdm_evaluation_every,
    )
    parser.add_argument(
        "--fdm-evaluation-batch-size",
        type=int,
        default=DEFAULT_CONFIG.fdm_evaluation_batch_size,
    )
    parser.add_argument("--model-grouping", default=DEFAULT_CONFIG.model_grouping)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    config = _config_from_args(args)
    history = run_experiment(config)
    first = history[0]["total_loss"]
    last = history[-1]["total_loss"]
    print(
        "clamped_rock_roof_nonuniform_support "
        f"foundation_enabled={config.foundation_enabled} "
        f"initial_total_loss={float(first.numpy()):.6e} "
        f"final_total_loss={float(last.numpy()):.6e} "
        f"best_fixed_physics={config.best_fixed_physics_loss_weights_path} "
        f"best_fdm_validation={config.best_fdm_validation_weights_path} "
        f"final_weights={config.final_weights_path}"
    )


if __name__ == "__main__":
    main()
