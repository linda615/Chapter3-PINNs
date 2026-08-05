"""Audit whether the current network can represent the six Levy fields."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import tensorflow as tf

from configs import save_experiment_config
from models import FIELD_NAMES, FieldOutputScales
from sampling import BoundarySide

from .analytical import analytical_fields
from .config import DEFAULT_CONFIG, ExperimentConfig
from .evaluate import make_test_grid, relative_l2_error
from .problem import create_model
from .run import save_history_csv, save_weights, set_reproducible_seed


LOSS_MODE_NORMALIZED = "normalized"
LOSS_MODE_RAW = "raw"
SUPPORTED_LOSS_MODES = (LOSS_MODE_NORMALIZED, LOSS_MODE_RAW)
OUTPUT_SCALE_NONE = "none"
OUTPUT_SCALE_RMS = "rms"
SUPPORTED_OUTPUT_SCALE_MODES = (OUTPUT_SCALE_NONE, OUTPUT_SCALE_RMS)


@dataclass(frozen=True)
class CapacityAuditConfig:
    """Configuration for direct supervised fitting of the Levy fields."""

    experiment: ExperimentConfig = DEFAULT_CONFIG
    epochs: int = 2000
    training_points: int = 4096
    validation_grid_size: int = 101
    validation_interval: int = 100
    log_every: int = 100
    learning_rate: float = 1e-3
    learning_rate_2: float = 3e-4
    learning_rate_3: float = 1e-4
    final_learning_rate: float = 3e-5
    first_decay_epoch: int = 800
    second_decay_epoch: int = 1600
    third_decay_epoch: int = 2600
    loss_mode: str = LOSS_MODE_NORMALIZED
    output_scale_mode: str = OUTPUT_SCALE_RMS
    minimum_scale: float = 1e-8
    seed: int = 2051
    results_dir: Path = Path(
        "experiments/simply_clamped_uniform_load/results_capacity_audit"
    )

    def __post_init__(self) -> None:
        """Validate the standalone audit controls."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.training_points <= 0:
            raise ValueError("training_points must be positive.")
        if self.validation_grid_size <= 1:
            raise ValueError("validation_grid_size must be greater than 1.")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")
        learning_rates = (
            self.learning_rate,
            self.learning_rate_2,
            self.learning_rate_3,
            self.final_learning_rate,
        )
        if any(rate <= 0.0 for rate in learning_rates):
            raise ValueError("All learning rates must be positive.")
        if not (
            0
            < self.first_decay_epoch
            < self.second_decay_epoch
            < self.third_decay_epoch
        ):
            raise ValueError(
                "Learning-rate decay epochs must be strictly increasing."
            )
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        if self.loss_mode not in SUPPORTED_LOSS_MODES:
            raise ValueError(
                f"Unsupported loss_mode {self.loss_mode!r}; expected one of "
                f"{SUPPORTED_LOSS_MODES}."
            )
        if self.output_scale_mode not in SUPPORTED_OUTPUT_SCALE_MODES:
            raise ValueError(
                "Unsupported output_scale_mode "
                f"{self.output_scale_mode!r}; expected one of "
                f"{SUPPORTED_OUTPUT_SCALE_MODES}."
            )

    @property
    def checkpoints_dir(self) -> Path:
        """Directory containing capacity-audit checkpoints."""
        return self.results_dir / "checkpoints"

    @property
    def history_dir(self) -> Path:
        """Directory containing scalar audit histories."""
        return self.results_dir / "history"

    @property
    def best_weights_path(self) -> Path:
        """Checkpoint with the lowest mean validation RelL2."""
        return self.checkpoints_dir / "best_validation_loss.weights.h5"

    @property
    def final_weights_path(self) -> Path:
        """Checkpoint at the final supervised epoch."""
        return self.checkpoints_dir / (
            f"final_epoch_{self.epochs}.weights.h5"
        )

    @property
    def history_path(self) -> Path:
        """CSV containing train and validation fitting metrics."""
        return self.history_dir / "capacity_history.csv"

    @property
    def field_scales_path(self) -> Path:
        """CSV containing the exact-field RMS normalization scales."""
        return self.history_dir / "field_scales.csv"


@dataclass(frozen=True)
class CapacityAuditResult:
    """Results returned by the supervised network-capacity audit."""

    history: list[dict[str, tf.Tensor]]
    field_scales: dict[str, tf.Tensor]
    output_scales: dict[str, tf.Tensor]
    best_epoch: int
    best_validation_score: float


def run_capacity_audit(
    config: CapacityAuditConfig = CapacityAuditConfig(),
) -> CapacityAuditResult:
    """Fit the six analytical fields without physics or boundary losses."""
    # The audit applies candidate scales externally, so its model must stay unit-scaled.
    audit_model_config = replace(
        config.experiment,
        field_output_scales=FieldOutputScales(),
    )
    save_experiment_config(
        replace(config, experiment=audit_model_config),
        run_entry=(
            "experiments.simply_clamped_uniform_load."
            "audit_network_capacity"
        ),
    )
    set_reproducible_seed(config.seed)
    model = create_model(audit_model_config)
    xy_train = _sample_fixed_training_points(config)
    train_exact = _exact_fields(xy_train, config.experiment)
    field_scales = _compute_field_scales(train_exact, config)
    output_scales = _compute_output_scales(
        xy=xy_train,
        exact=train_exact,
        field_scales=field_scales,
        config=config,
    )
    loss_scales = _loss_scales(field_scales, config)
    xy_validation = make_test_grid(config.validation_grid_size)
    validation_exact = _exact_fields(xy_validation, config.experiment)
    learning_rate_schedule = _create_learning_rate_schedule(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)

    @tf.function
    def train_step() -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        with tf.GradientTape() as tape:
            predicted = _predict_physical_fields(
                model=model,
                xy=xy_train,
                output_scales=output_scales,
                output_scale_mode=config.output_scale_mode,
                training=True,
            )
            field_losses = _compute_field_losses(
                predicted,
                train_exact,
                loss_scales,
            )
            total_loss = tf.add_n(
                [field_losses[name] for name in FIELD_NAMES]
            ) / tf.cast(len(FIELD_NAMES), xy_train.dtype)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                model.trainable_variables,
            )
            if gradient is not None
        ]
        optimizer.apply_gradients(gradient_pairs)
        return total_loss, field_losses

    history: list[dict[str, tf.Tensor]] = []
    initial_record = _evaluate_epoch(
        epoch=0,
        model=model,
        xy_train=xy_train,
        train_exact=train_exact,
        xy_validation=xy_validation,
        validation_exact=validation_exact,
        loss_scales=loss_scales,
        output_scales=output_scales,
        output_scale_mode=config.output_scale_mode,
        learning_rate=_learning_rate_for_epoch(
            learning_rate_schedule,
            epoch=0,
        ),
    )
    history.append(initial_record)
    best_epoch = 0
    best_validation_score = float(
        initial_record["validation_score"].numpy()
    )
    save_weights(model, config.best_weights_path)
    _print_record(initial_record)

    for epoch in range(1, config.epochs + 1):
        train_step()
        should_evaluate = (
            epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
        if not should_evaluate:
            continue
        record = _evaluate_epoch(
            epoch=epoch,
            model=model,
            xy_train=xy_train,
            train_exact=train_exact,
            xy_validation=xy_validation,
            validation_exact=validation_exact,
            loss_scales=loss_scales,
            output_scales=output_scales,
            output_scale_mode=config.output_scale_mode,
            learning_rate=_learning_rate_for_epoch(
                learning_rate_schedule,
                epoch=epoch,
            ),
        )
        history.append(record)
        score = float(record["validation_score"].numpy())
        if score < best_validation_score:
            best_epoch = epoch
            best_validation_score = score
            save_weights(model, config.best_weights_path)
        if epoch % config.log_every == 0 or epoch == config.epochs:
            _print_record(record)

    save_weights(model, config.final_weights_path)
    save_history_csv(history, config.history_path)
    save_history_csv(
        [
            {
                f"{name}_rms_scale": field_scales[name]
                for name in FIELD_NAMES
            }
            | {
                f"{name}_output_scale": output_scales[name]
                for name in FIELD_NAMES
            }
        ],
        config.field_scales_path,
    )
    print(
        "network_capacity_audit_complete "
        f"best_epoch={best_epoch} "
        f"best_validation_score={best_validation_score:.6e} "
        f"best_weights={config.best_weights_path} "
        f"history={config.history_path}"
    )
    return CapacityAuditResult(
        history=history,
        field_scales=field_scales,
        output_scales=output_scales,
        best_epoch=best_epoch,
        best_validation_score=best_validation_score,
    )


def _sample_fixed_training_points(config: CapacityAuditConfig) -> tf.Tensor:
    """Return one reproducible set of points in the open unit square."""
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    return tf.random.stateless_uniform(
        shape=(config.training_points, 2),
        seed=(config.seed, config.seed + 1),
        minval=epsilon,
        maxval=tf.constant(1.0, dtype=tf.float32) - epsilon,
        dtype=tf.float32,
    )


def _create_learning_rate_schedule(
    config: CapacityAuditConfig,
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Create the fixed-epoch four-stage supervised learning rate."""
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(
            config.first_decay_epoch - 1,
            config.second_decay_epoch - 1,
            config.third_decay_epoch - 1,
        ),
        values=(
            config.learning_rate,
            config.learning_rate_2,
            config.learning_rate_3,
            config.final_learning_rate,
        ),
    )


def _learning_rate_for_epoch(
    schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
    epoch: int,
) -> tf.Tensor:
    """Return the rate applied during a one-based training epoch."""
    schedule_step = max(epoch - 1, 0)
    return tf.convert_to_tensor(schedule(schedule_step))


def _exact_fields(
    xy: tf.Tensor,
    config: ExperimentConfig,
) -> dict[str, tf.Tensor]:
    """Evaluate the finite Levy reference used by experiment validation."""
    return analytical_fields(
        xy=xy,
        parameters=config.plate,
        q0=config.q0,
        mode_count=config.analytical_modes,
    )


def _compute_field_scales(
    exact: dict[str, tf.Tensor],
    config: CapacityAuditConfig,
) -> dict[str, tf.Tensor]:
    """Return guarded RMS scales for equal relative field weighting."""
    minimum = tf.cast(config.minimum_scale, next(iter(exact.values())).dtype)
    return {
        name: tf.maximum(
            tf.sqrt(tf.reduce_mean(tf.square(exact[name]))),
            minimum,
        )
        for name in FIELD_NAMES
    }


def _loss_scales(
    field_scales: dict[str, tf.Tensor],
    config: CapacityAuditConfig,
) -> dict[str, tf.Tensor]:
    """Select normalized or raw supervised fitting scales."""
    if config.loss_mode == LOSS_MODE_NORMALIZED:
        return field_scales
    dtype = next(iter(field_scales.values())).dtype
    return {name: tf.ones((), dtype=dtype) for name in FIELD_NAMES}


def _compute_output_scales(
    xy: tf.Tensor,
    exact: dict[str, tf.Tensor],
    field_scales: dict[str, tf.Tensor],
    config: CapacityAuditConfig,
) -> dict[str, tf.Tensor]:
    """Return Head-output scales, accounting for the hard w envelope."""
    if config.experiment.output_activation == "tanh":
        output_scales = {
            name: 1.1 * tf.reduce_max(tf.abs(exact[name]))
            for name in FIELD_NAMES
        }
    else:
        output_scales = dict(field_scales)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    x_factor = x * (1.0 - x)
    y_factor = y * (1.0 - y)
    envelope = x_factor * y_factor
    if config.experiment.hard_clamped_slope:
        clamped_sides = {
            BoundarySide(side) for side in config.experiment.clamped_sides
        }
        if clamped_sides == {BoundarySide.BOTTOM, BoundarySide.TOP}:
            envelope = envelope * y_factor
        elif clamped_sides == {BoundarySide.LEFT, BoundarySide.RIGHT}:
            envelope = envelope * x_factor
        else:
            raise ValueError(
                "Hard clamped-slope scaling requires opposite clamped edges."
            )
    raw_w_target = tf.math.divide_no_nan(exact["w"], envelope)
    minimum = tf.cast(config.minimum_scale, exact["w"].dtype)
    if config.experiment.output_activation == "tanh":
        output_scales["w"] = tf.maximum(
            1.1 * tf.reduce_max(tf.abs(raw_w_target)),
            minimum,
        )
    else:
        output_scales["w"] = tf.maximum(
            tf.sqrt(tf.reduce_mean(tf.square(raw_w_target))),
            minimum,
        )
    return output_scales


def _compute_field_losses(
    predicted: dict[str, tf.Tensor],
    exact: dict[str, tf.Tensor],
    scales: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Return six scalar supervised MSE terms."""
    return {
        name: tf.reduce_mean(
            tf.square((predicted[name] - exact[name]) / scales[name])
        )
        for name in FIELD_NAMES
    }


def _predict_physical_fields(
    model: tf.keras.Model,
    xy: tf.Tensor,
    output_scales: dict[str, tf.Tensor],
    output_scale_mode: str,
    training: bool,
) -> dict[str, tf.Tensor]:
    """Map dimensionless network outputs back to physical field scales."""
    raw_fields = model(xy, training=training)
    if output_scale_mode == OUTPUT_SCALE_NONE:
        return raw_fields
    if output_scale_mode == OUTPUT_SCALE_RMS:
        return {
            name: raw_fields[name] * output_scales[name]
            for name in FIELD_NAMES
        }
    raise ValueError(
        f"Unsupported output_scale_mode {output_scale_mode!r}."
    )


def _evaluate_epoch(
    epoch: int,
    model: tf.keras.Model,
    xy_train: tf.Tensor,
    train_exact: dict[str, tf.Tensor],
    xy_validation: tf.Tensor,
    validation_exact: dict[str, tf.Tensor],
    loss_scales: dict[str, tf.Tensor],
    output_scales: dict[str, tf.Tensor],
    output_scale_mode: str,
    learning_rate: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Compute supervised losses and independent-grid relative errors."""
    train_predicted = _predict_physical_fields(
        model=model,
        xy=xy_train,
        output_scales=output_scales,
        output_scale_mode=output_scale_mode,
        training=False,
    )
    validation_predicted = _predict_physical_fields(
        model=model,
        xy=xy_validation,
        output_scales=output_scales,
        output_scale_mode=output_scale_mode,
        training=False,
    )
    field_losses = _compute_field_losses(
        train_predicted,
        train_exact,
        loss_scales,
    )
    total_loss = tf.add_n(
        [field_losses[name] for name in FIELD_NAMES]
    ) / tf.cast(len(FIELD_NAMES), xy_train.dtype)
    train_rel_l2 = {
        name: relative_l2_error(train_predicted[name], train_exact[name])
        for name in FIELD_NAMES
    }
    validation_rel_l2 = {
        name: relative_l2_error(
            validation_predicted[name],
            validation_exact[name],
        )
        for name in FIELD_NAMES
    }
    validation_score = tf.add_n(
        [validation_rel_l2[name] for name in FIELD_NAMES]
    ) / tf.cast(len(FIELD_NAMES), xy_train.dtype)
    record: dict[str, tf.Tensor] = {
        "epoch": tf.constant(epoch, dtype=tf.int32),
        "total_loss": total_loss,
        "validation_score": validation_score,
        "learning_rate": learning_rate,
    }
    for name in FIELD_NAMES:
        record[f"{name}_fit_loss"] = field_losses[name]
        record[f"train_{name}_rel_l2"] = train_rel_l2[name]
        record[f"validation_{name}_rel_l2"] = validation_rel_l2[name]
    return record


def _print_record(record: dict[str, tf.Tensor]) -> None:
    """Print a compact capacity-audit progress line."""
    parts = [
        "capacity_audit",
        f"epoch={int(record['epoch'].numpy())}",
        f"total_loss={float(record['total_loss'].numpy()):.6e}",
        f"learning_rate={float(record['learning_rate'].numpy()):.6e}",
        (
            "validation_score="
            f"{float(record['validation_score'].numpy()):.6e}"
        ),
    ]
    parts.extend(
        f"{name}_rel_l2="
        f"{float(record[f'validation_{name}_rel_l2'].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    print(" ".join(parts))


def _parse_args() -> argparse.Namespace:
    """Parse command-line controls for architecture comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--training-points", type=int, default=4096)
    parser.add_argument("--validation-grid-size", type=int, default=101)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate-2", type=float, default=3e-4)
    parser.add_argument("--learning-rate-3", type=float, default=1e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--first-decay-epoch", type=int, default=800)
    parser.add_argument("--second-decay-epoch", type=int, default=1600)
    parser.add_argument("--third-decay-epoch", type=int, default=2600)
    parser.add_argument(
        "--loss-mode",
        choices=SUPPORTED_LOSS_MODES,
        default=LOSS_MODE_NORMALIZED,
    )
    parser.add_argument(
        "--output-scale-mode",
        choices=SUPPORTED_OUTPUT_SCALE_MODES,
        default=OUTPUT_SCALE_RMS,
        help=(
            "Use RMS-scaled dimensionless Head outputs or preserve the "
            "legacy direct physical-output parameterization."
        ),
    )
    parser.add_argument(
        "--model-grouping",
        choices=("one", "two", "three", "six"),
        default=DEFAULT_CONFIG.model_grouping,
    )
    parser.add_argument(
        "--hidden-width",
        type=int,
        default=DEFAULT_CONFIG.hidden_width,
    )
    parser.add_argument(
        "--hidden-depth",
        type=int,
        default=DEFAULT_CONFIG.hidden_depth,
    )
    parser.add_argument(
        "--shared-width",
        type=int,
        default=DEFAULT_CONFIG.shared_width,
    )
    parser.add_argument(
        "--shared-depth",
        type=int,
        default=DEFAULT_CONFIG.shared_depth,
    )
    parser.add_argument("--disable-shared-trunk", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "experiments/simply_clamped_uniform_load/"
            "results_capacity_audit"
        ),
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> CapacityAuditConfig:
    """Construct the effective audit and model configuration."""
    experiment = replace(
        DEFAULT_CONFIG,
        model_grouping=args.model_grouping,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        use_shared_trunk=not args.disable_shared_trunk,
        shared_width=args.shared_width,
        shared_depth=args.shared_depth,
        results_dir=args.results_dir,
    )
    return CapacityAuditConfig(
        experiment=experiment,
        epochs=args.epochs,
        training_points=args.training_points,
        validation_grid_size=args.validation_grid_size,
        validation_interval=args.validation_interval,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        learning_rate_2=args.learning_rate_2,
        learning_rate_3=args.learning_rate_3,
        final_learning_rate=args.final_learning_rate,
        first_decay_epoch=args.first_decay_epoch,
        second_decay_epoch=args.second_decay_epoch,
        third_decay_epoch=args.third_decay_epoch,
        loss_mode=args.loss_mode,
        output_scale_mode=args.output_scale_mode,
        seed=args.seed,
        results_dir=args.results_dir,
    )


def main() -> None:
    """Run the supervised representation-capacity audit."""
    run_capacity_audit(_config_from_args(_parse_args()))


if __name__ == "__main__":
    main()


__all__ = [
    "CapacityAuditConfig",
    "CapacityAuditResult",
    "LOSS_MODE_NORMALIZED",
    "LOSS_MODE_RAW",
    "OUTPUT_SCALE_NONE",
    "OUTPUT_SCALE_RMS",
    "run_capacity_audit",
]
