"""Short-cycle block-coordinate fine-tuning for a six-branch PINN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable, Sequence

import tensorflow as tf

from configs import save_experiment_config
from losses import (
    LossWeights,
    MxyConsistencyConfig,
    compute_grouped_boundary_total_loss,
)
from models import FIELD_NAMES
from trainer import SharedTrunkPCGradConfig

from .config import DEFAULT_CONFIG, ExperimentConfig
from .evaluate import evaluate_model, make_test_grid
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
    sample_boundary_groups,
)
from .run import (
    BestValidationCheckpoint,
    load_initial_weights,
    save_history_csv,
    save_weights,
    set_reproducible_seed,
)


@dataclass(frozen=True)
class BlockwiseFineTuneConfig:
    """Controls for short accepted/rejected branch-training cycles."""

    initial_checkpoint_path: Path | None = None
    results_dir: Path = Path(
        "experiments/simply_clamped_uniform_load/"
        "results_blockwise_short_cycles"
    )
    cycles: int = 8
    moment_epochs_per_cycle: int = 20
    shear_epochs_per_cycle: int = 20
    deflection_epochs_per_cycle: int = 10
    joint_epochs: int = 1000
    block_learning_rate: float = 3e-5
    joint_learning_rate: float = 5e-6
    moment_shear_tether: float = 0.1
    shear_equilibrium_weight: float = 5.0
    block_min_delta: float = 0.0
    validation_interval: int = 100
    log_every: int = 50
    seed: int = DEFAULT_CONFIG.seed
    experiment: ExperimentConfig = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        """Validate short-cycle training controls."""
        for name, value in (
            ("cycles", self.cycles),
            ("moment_epochs_per_cycle", self.moment_epochs_per_cycle),
            ("shear_epochs_per_cycle", self.shear_epochs_per_cycle),
            ("deflection_epochs_per_cycle", self.deflection_epochs_per_cycle),
            ("joint_epochs", self.joint_epochs),
            ("validation_interval", self.validation_interval),
            ("log_every", self.log_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("block_learning_rate", self.block_learning_rate),
            ("joint_learning_rate", self.joint_learning_rate),
            ("moment_shear_tether", self.moment_shear_tether),
            ("shear_equilibrium_weight", self.shear_equilibrium_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.block_min_delta) or self.block_min_delta < 0.0:
            raise ValueError("block_min_delta must be finite and non-negative.")

    @property
    def block_epochs(self) -> int:
        """Return the number of head-only updates."""
        per_cycle = (
            self.moment_epochs_per_cycle
            + self.shear_epochs_per_cycle
            + self.deflection_epochs_per_cycle
        )
        return self.cycles * per_cycle

    @property
    def total_epochs(self) -> int:
        """Return block updates plus final joint updates."""
        return self.block_epochs + self.joint_epochs

    @property
    def checkpoints_dir(self) -> Path:
        return self.results_dir / "checkpoints"

    @property
    def history_dir(self) -> Path:
        return self.results_dir / "history"

    @property
    def loss_history_path(self) -> Path:
        return self.history_dir / "loss.csv"

    @property
    def validation_history_path(self) -> Path:
        return self.history_dir / "validation.csv"

    @property
    def block_history_path(self) -> Path:
        return self.history_dir / "block_validation.csv"

    @property
    def final_weights_path(self) -> Path:
        return self.checkpoints_dir / "final_blockwise.weights.h5"


def run_blockwise_finetuning(
    config: BlockwiseFineTuneConfig,
) -> list[dict[str, tf.Tensor]]:
    """Run short head-only cycles followed by joint fine-tuning."""
    if config.initial_checkpoint_path is None:
        raise ValueError(
            "Short-cycle blockwise training requires an initial six-branch "
            "checkpoint. Provide initial_checkpoint_path or "
            "--initial-checkpoint."
        )
    effective = _effective_experiment_config(config)
    save_experiment_config(
        config,
        run_entry=(
            "experiments.simply_clamped_uniform_load."
            "finetune_blockwise"
        ),
    )
    set_reproducible_seed(config.seed)
    model = create_model(effective)
    _validate_six_branch_model(model)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    load_initial_weights(model, config.initial_checkpoint_path)

    interior_sampler = create_interior_sampler()
    boundary_sampler = create_boundary_sampler()
    load_fn = create_load_fn(effective)
    validator = BestValidationCheckpoint(config=effective)
    history: list[dict[str, tf.Tensor]] = []
    block_history: list[dict[str, tf.Tensor]] = []
    global_epoch = 0

    moment_variables = _variables_from_layers(
        (model.net_Mx, model.net_My, model.net_Mxy)
    )
    shear_variables = _variables_from_layers((model.net_Qx, model.net_Qy))
    deflection_variables = tuple(model.net_w.trainable_variables)
    shared_variables = tuple(model.shared_trunk.trainable_variables)
    all_variables = tuple(model.trainable_variables)

    current_score, initial_rel_l2 = _validation_score(model, effective)
    _register_initial_validation(
        validator,
        model,
        current_score,
        initial_rel_l2,
    )
    print(
        "blockwise_initial_validation "
        f"validation_score={current_score:.6e}"
    )

    blocks = (
        (
            1,
            "moment",
            config.moment_epochs_per_cycle,
            moment_variables,
            _moment_stage_objective(config),
        ),
        (
            2,
            "shear",
            config.shear_epochs_per_cycle,
            shear_variables,
            _shear_stage_objective(config),
        ),
        (
            3,
            "deflection",
            config.deflection_epochs_per_cycle,
            deflection_variables,
            _deflection_stage_objective,
        ),
    )
    for cycle in range(1, config.cycles + 1):
        for stage_id, stage_name, stage_epochs, variables, objective_fn in blocks:
            before = _snapshot_variables(model.variables)
            score_before = current_score
            global_epoch = _run_stage(
                stage_id=stage_id,
                stage_name=stage_name,
                stage_epochs=stage_epochs,
                global_epoch=global_epoch,
                model=model,
                trainable_variables=variables,
                frozen_variables=_excluded_variables(
                    all_variables,
                    variables,
                ),
                objective_fn=objective_fn,
                learning_rate=config.block_learning_rate,
                config=config,
                effective=effective,
                interior_sampler=interior_sampler,
                boundary_sampler=boundary_sampler,
                load_fn=load_fn,
                validator=validator,
                history=history,
            )
            score_after, rel_l2 = _validation_score(model, effective)
            accepted = (
                score_after
                < score_before - config.block_min_delta
            )
            if accepted:
                current_score = score_after
                _register_block_validation(
                    validator,
                    model,
                    global_epoch,
                    score_after,
                    rel_l2,
                )
                save_weights(
                    model,
                    config.checkpoints_dir
                    / "latest_accepted_block.weights.h5",
                )
            else:
                _restore_variables(model.variables, before)
            block_history.append(
                {
                    "cycle": tf.constant(cycle, tf.int32),
                    "block_id": tf.constant(stage_id, tf.int32),
                    "epoch": tf.constant(global_epoch, tf.int32),
                    "score_before": tf.constant(score_before, tf.float32),
                    "score_after": tf.constant(score_after, tf.float32),
                    "accepted": tf.constant(float(accepted), tf.float32),
                }
            )
            print(
                f"block_validation cycle={cycle} block={stage_name} "
                f"score_before={score_before:.6e} "
                f"score_after={score_after:.6e} "
                f"accepted={int(accepted)}"
            )

    save_weights(
        model,
        config.checkpoints_dir / "after_short_cycles.weights.h5",
    )
    _run_stage(
        stage_id=4,
        stage_name="joint",
        stage_epochs=config.joint_epochs,
        global_epoch=global_epoch,
        model=model,
        trainable_variables=all_variables,
        frozen_variables=(),
        objective_fn=_joint_stage_objective,
        learning_rate=config.joint_learning_rate,
        config=config,
        effective=effective,
        interior_sampler=interior_sampler,
        boundary_sampler=boundary_sampler,
        load_fn=load_fn,
        validator=validator,
        history=history,
    )
    save_weights(
        model,
        config.checkpoints_dir / "after_joint.weights.h5",
    )
    save_weights(model, config.final_weights_path)
    save_history_csv(history, config.loss_history_path)
    save_history_csv(validator.records, config.validation_history_path)
    save_history_csv(block_history, config.block_history_path)
    return history


def _run_stage(
    *,
    stage_id: int,
    stage_name: str,
    stage_epochs: int,
    global_epoch: int,
    model: tf.keras.Model,
    trainable_variables: tuple[tf.Variable, ...],
    frozen_variables: tuple[tf.Variable, ...],
    objective_fn: Callable[[dict[str, tf.Tensor]], tf.Tensor],
    learning_rate: float,
    config: BlockwiseFineTuneConfig,
    effective: ExperimentConfig,
    interior_sampler,
    boundary_sampler,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    validator: BestValidationCheckpoint,
    history: list[dict[str, tf.Tensor]],
) -> int:
    """Run one short block and enforce its parameter ownership."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    frozen_snapshot = _snapshot_variables(frozen_variables)
    for local_epoch in range(1, stage_epochs + 1):
        global_epoch += 1
        interior_xy = interior_sampler.sample(
            effective.training.interior_points,
            seed=config.seed + global_epoch,
        )
        boundary_groups = sample_boundary_groups(
            sampler=boundary_sampler,
            points_per_side=effective.training.boundary_points_per_side,
            config=effective,
            seed=config.seed + 100000 + global_epoch,
        )
        with tf.GradientTape() as tape:
            losses = _compute_losses(
                model,
                interior_xy,
                boundary_groups,
                load_fn,
                effective,
                training=True,
            )
            objective = objective_fn(losses)
        _apply_gradients(
            tape,
            objective,
            trainable_variables,
            optimizer,
            stage_name,
        )
        if _should_observe(config, global_epoch):
            losses = _compute_losses(
                model,
                interior_xy,
                boundary_groups,
                load_fn,
                effective,
                training=False,
            )
        losses["stage_id"] = tf.constant(stage_id, tf.int32)
        losses["stage_local_epoch"] = tf.constant(local_epoch, tf.int32)
        losses["stage_objective"] = tf.identity(objective)
        losses["learning_rate"] = tf.cast(
            learning_rate,
            losses["total_loss"].dtype,
        )
        if _should_log(config, global_epoch):
            record = {"epoch": tf.constant(global_epoch, tf.int32)}
            record.update(
                {name: tf.identity(value) for name, value in losses.items()}
            )
            history.append(record)
            print(_format_progress_line(global_epoch, stage_name, losses))
        validator(global_epoch, model, losses)
    _assert_frozen_variables(stage_name, frozen_snapshot, frozen_variables)
    return global_epoch


def _moment_stage_objective(
    config: BlockwiseFineTuneConfig,
) -> Callable[[dict[str, tf.Tensor]], tf.Tensor]:
    """Build the moment-head objective with a derivative tether."""
    def objective(losses: dict[str, tf.Tensor]) -> tf.Tensor:
        dtype = losses["moment_loss"].dtype
        return (
            losses["weighted_moment_loss"]
            + tf.cast(config.moment_shear_tether, dtype)
            * losses["weighted_shear_loss"]
            + losses["moment_bc_loss"]
        )
    return objective


def _shear_stage_objective(
    config: BlockwiseFineTuneConfig,
) -> Callable[[dict[str, tf.Tensor]], tf.Tensor]:
    """Build the shear-head objective with equilibrium anchoring."""
    def objective(losses: dict[str, tf.Tensor]) -> tf.Tensor:
        dtype = losses["shear_loss"].dtype
        return (
            losses["weighted_shear_loss"]
            + tf.cast(config.shear_equilibrium_weight, dtype)
            * losses["weighted_equilibrium_loss"]
        )
    return objective


def _deflection_stage_objective(
    losses: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Couple the deflection head to moments and clamped-edge slopes."""
    return losses["weighted_moment_loss"] + losses["slope_bc_loss"]


def _joint_stage_objective(losses: dict[str, tf.Tensor]) -> tf.Tensor:
    return losses["total_loss"]


def _compute_losses(
    model: tf.keras.Model,
    interior_xy: tf.Tensor,
    boundary_groups: tuple,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    config: ExperimentConfig,
    training: bool,
) -> dict[str, tf.Tensor]:
    """Reuse the existing grouped-boundary total-loss implementation."""
    return compute_grouped_boundary_total_loss(
        model=model,
        interior_xy=interior_xy,
        boundary_groups=boundary_groups,
        load_fn=load_fn,
        parameters=config.plate,
        weights=LossWeights(),
        physics_loss_config=config.physics_loss_config,
        mxy_consistency_config=MxyConsistencyConfig(enabled=False),
        training=training,
    )


def _validation_score(
    model: tf.keras.Model,
    config: ExperimentConfig,
) -> tuple[float, dict[str, tf.Tensor]]:
    """Evaluate the mean six-field relative L2 score."""
    xy = make_test_grid(config.validation_grid_size)
    _, _, rel_l2 = evaluate_model(model, xy, config)
    score = tf.add_n([rel_l2[name] for name in FIELD_NAMES]) / float(
        len(FIELD_NAMES)
    )
    return float(score.numpy()), rel_l2


def _register_initial_validation(
    validator: BestValidationCheckpoint,
    model: tf.keras.Model,
    score: float,
    rel_l2: dict[str, tf.Tensor],
) -> None:
    """Make the input checkpoint a valid best-model candidate."""
    validator.best_score = score
    validator.best_epoch = 0
    _append_validation_record(validator, 0, score, rel_l2)
    save_weights(model, validator.config.best_validation_loss_weights_path)


def _register_block_validation(
    validator: BestValidationCheckpoint,
    model: tf.keras.Model,
    epoch: int,
    score: float,
    rel_l2: dict[str, tf.Tensor],
) -> None:
    """Register an accepted block if it establishes a global best."""
    _append_validation_record(validator, epoch, score, rel_l2)
    if score < validator.best_score:
        validator.best_score = score
        validator.best_epoch = epoch
        save_weights(model, validator.config.best_validation_loss_weights_path)


def _append_validation_record(
    validator: BestValidationCheckpoint,
    epoch: int,
    score: float,
    rel_l2: dict[str, tf.Tensor],
) -> None:
    record = {"epoch": tf.constant(epoch, tf.int32)}
    record.update(
        {
            f"{name}_rel_l2": tf.identity(rel_l2[name])
            for name in FIELD_NAMES
        }
    )
    record["validation_score"] = tf.constant(score, tf.float32)
    validator.records.append(record)


def _effective_experiment_config(
    config: BlockwiseFineTuneConfig,
) -> ExperimentConfig:
    training = replace(
        config.experiment.training,
        epochs=config.total_epochs,
        log_every=config.log_every,
        seed=config.seed,
    )
    return replace(
        config.experiment,
        model_grouping="six",
        use_shared_trunk=True,
        results_dir=config.results_dir,
        initial_checkpoint_path=config.initial_checkpoint_path,
        training=training,
        validation_interval=config.validation_interval,
        loss_weights=LossWeights(),
        shared_pcgrad=SharedTrunkPCGradConfig(enabled=False),
        mxy_consistency_config=MxyConsistencyConfig(enabled=False),
        seed=config.seed,
    )


def _validate_six_branch_model(model: tf.keras.Model) -> None:
    if model.get_grouping() != "six":
        raise ValueError("Blockwise training requires grouping='six'.")
    for attribute in (
        "shared_trunk",
        "net_w",
        "net_Mx",
        "net_My",
        "net_Mxy",
        "net_Qx",
        "net_Qy",
    ):
        if getattr(model, attribute, None) is None:
            raise ValueError(f"Blockwise training requires model.{attribute}.")


def _variables_from_layers(
    layers: Sequence[tf.keras.layers.Layer],
) -> tuple[tf.Variable, ...]:
    return tuple(
        variable
        for layer in layers
        for variable in layer.trainable_variables
    )


def _excluded_variables(
    all_variables: Sequence[tf.Variable],
    included_variables: Sequence[tf.Variable],
) -> tuple[tf.Variable, ...]:
    included = {variable.ref() for variable in included_variables}
    return tuple(
        variable
        for variable in all_variables
        if variable.ref() not in included
    )


def _snapshot_variables(
    variables: Sequence[tf.Variable],
) -> tuple[tf.Tensor, ...]:
    return tuple(tf.identity(variable) for variable in variables)


def _restore_variables(
    variables: Sequence[tf.Variable],
    values: Sequence[tf.Tensor],
) -> None:
    for variable, value in zip(variables, values):
        variable.assign(value)


def _assert_frozen_variables(
    stage_name: str,
    before: Sequence[tf.Tensor],
    after: Sequence[tf.Variable],
) -> None:
    for expected, actual in zip(before, after):
        tf.debugging.assert_equal(
            actual,
            expected,
            message=f"A frozen variable changed in the {stage_name} block.",
        )


def _apply_gradients(
    tape: tf.GradientTape,
    objective: tf.Tensor,
    variables: tuple[tf.Variable, ...],
    optimizer: tf.keras.optimizers.Optimizer,
    objective_name: str,
) -> None:
    gradients = tape.gradient(objective, variables)
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, variables)
        if gradient is not None
    ]
    if not pairs:
        raise ValueError(f"No gradients for the {objective_name} block.")
    optimizer.apply_gradients(pairs)


def _should_log(config: BlockwiseFineTuneConfig, epoch: int) -> bool:
    return epoch == 1 or epoch == config.total_epochs or (
        epoch % config.log_every == 0
    )


def _should_observe(config: BlockwiseFineTuneConfig, epoch: int) -> bool:
    return _should_log(config, epoch) or (
        epoch % config.validation_interval == 0
    )


def _format_progress_line(
    epoch: int,
    stage_name: str,
    losses: dict[str, tf.Tensor],
) -> str:
    keys = (
        "stage_objective",
        "total_loss",
        "moment_loss",
        "shear_loss",
        "equilibrium_loss",
        "boundary_loss",
        "learning_rate",
    )
    values = " ".join(
        f"{key}={float(losses[key].numpy()):.6e}" for key in keys
    )
    return f"blockwise epoch={epoch} stage={stage_name} {values}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BlockwiseFineTuneConfig().results_dir,
    )
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--moment-epochs-per-cycle", type=int, default=20)
    parser.add_argument("--shear-epochs-per-cycle", type=int, default=20)
    parser.add_argument("--deflection-epochs-per-cycle", type=int, default=10)
    parser.add_argument("--joint-epochs", type=int, default=1000)
    parser.add_argument("--block-learning-rate", type=float, default=3e-5)
    parser.add_argument("--joint-learning-rate", type=float, default=5e-6)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BlockwiseFineTuneConfig(
        initial_checkpoint_path=args.initial_checkpoint,
        results_dir=args.results_dir,
        cycles=args.cycles,
        moment_epochs_per_cycle=args.moment_epochs_per_cycle,
        shear_epochs_per_cycle=args.shear_epochs_per_cycle,
        deflection_epochs_per_cycle=args.deflection_epochs_per_cycle,
        joint_epochs=args.joint_epochs,
        block_learning_rate=args.block_learning_rate,
        joint_learning_rate=args.joint_learning_rate,
        validation_interval=args.validation_interval,
        log_every=args.log_every,
    )
    history = run_blockwise_finetuning(config)
    print(
        "simply_clamped_uniform_load_short_blockwise_complete "
        f"epochs={config.total_epochs} "
        f"best_weights="
        f"{config.checkpoints_dir / 'best_validation_loss.weights.h5'} "
        f"final_weights={config.final_weights_path} "
        f"block_history={config.block_history_path} "
        f"final_total_loss="
        f"{float(history[-1]['total_loss'].numpy()):.6e}"
    )


if __name__ == "__main__":
    main()
