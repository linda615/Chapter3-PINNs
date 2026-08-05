"""Resume the rock-roof experiment with Adam and L-BFGS fine tuning."""
#微调
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf

from configs import save_experiment_config
from models import FIELD_NAMES
from trainer import LBFGSResult, PINNLBFGSTrainer, PINNTrainer

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import (
    create_boundary_sampler,
    create_foundation_fn,
    create_interior_sampler,
    create_load_fn,
    create_model,
)
from .run import (
    FDMReferenceMonitor,
    save_history_csv,
    save_weights,
    set_reproducible_seed,
)


@dataclass
class AdamFineTuneResult:
    """Result of the resumed Adam stage."""

    model: tf.keras.Model
    history: List[Dict[str, tf.Tensor]]
    epochs_completed: int
    stopped_early: bool
    best_score: float
    best_epoch: int


def run_adam_finetuning(
    config: ExperimentConfig = DEFAULT_CONFIG,
    initial_weights_path: Optional[str | Path] = None,
) -> AdamFineTuneResult:
    """Resume Adam from the original best FDM-validation checkpoint.

    The baseline checkpoint is evaluated at epoch zero and copied to the
    fine-tuning checkpoint. Consequently, the saved fine-tuning model cannot
    be worse than the model used to start this stage.
    """
    fine_tuning = config.fine_tuning
    source_path = Path(
        initial_weights_path or config.best_fdm_validation_weights_path
    )
    save_experiment_config(
        config,
        run_entry=(
            "experiments.clamped_rock_roof_nonuniform_support."
            "finetune.adam"
        ),
        metadata={"initial_weights_path": source_path},
    )
    model = _load_model(config, source_path)
    training_config = replace(
        config.training,
        epochs=fine_tuning.adam_epochs,
        log_every=fine_tuning.adam_log_every,
        seed=_offset_seed(config.training.seed, 7000),
    )
    trainer = PINNTrainer(
        model=model,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=fine_tuning.adam_learning_rate
        ),
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(config),
        foundation_fn=create_foundation_fn(config),
        parameters=config.plate,
        config=training_config,
        loss_weights=fine_tuning.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        mxy_consistency_config=config.mxy_consistency_config,
        verbose=False,
    )
    monitor_config = replace(
        config,
        fdm_evaluation_every=fine_tuning.fdm_evaluation_every,
    )
    monitor = FDMReferenceMonitor(
        config=monitor_config,
        model=model,
        checkpoint_path=config.best_fdm_finetune_weights_path,
    )

    baseline = monitor.evaluate(0)
    print(_format_fdm_line("adam_fdm_validation", "epoch", baseline))
    early_stopping_best = float(baseline["fdm_validation_score"].numpy())
    stale_evaluations = 0
    stopped_early = False
    history: List[Dict[str, tf.Tensor]] = []
    epochs_completed = 0

    for epoch in range(1, fine_tuning.adam_epochs + 1):
        interior_xy, boundary_batch = trainer.sample_epoch(epoch)
        loss_dict = trainer.train_step(
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            loss_weights=fine_tuning.loss_weights,
        )
        epochs_completed = epoch

        if _should_log_adam(epoch, fine_tuning.adam_epochs, fine_tuning.adam_log_every):
            snapshot = _snapshot("epoch", epoch, loss_dict)
            history.append(snapshot)
            print(_format_loss_line("adam_finetune", "epoch", epoch, snapshot))

        should_evaluate = (
            epoch % fine_tuning.fdm_evaluation_every == 0
            or epoch == fine_tuning.adam_epochs
        )
        if not should_evaluate:
            continue

        fdm_snapshot = monitor.evaluate(epoch)
        print(_format_fdm_line("adam_fdm_validation", "epoch", fdm_snapshot))
        score = float(fdm_snapshot["fdm_validation_score"].numpy())
        if score < early_stopping_best - fine_tuning.early_stopping_min_delta:
            early_stopping_best = score
            stale_evaluations = 0
        else:
            stale_evaluations += 1

        if stale_evaluations >= fine_tuning.early_stopping_patience:
            stopped_early = True
            print(
                "adam_finetune_early_stop "
                f"epoch={epoch} "
                f"patience={fine_tuning.early_stopping_patience} "
                f"best_fdm_epoch={monitor.best_epoch} "
                f"best_fdm_validation_score={monitor.best_score:.6e}"
            )
            break

    save_weights(model, config.final_adam_finetune_weights_path)
    save_history_csv(history, config.adam_finetune_history_path)
    save_history_csv(monitor.records, config.adam_finetune_fdm_history_path)

    # Return the best validation state, not the final attempted Adam state.
    model.load_weights(str(config.best_fdm_finetune_weights_path))
    print(
        "adam_finetune_summary "
        f"source={source_path} "
        f"epochs_completed={epochs_completed} "
        f"stopped_early={stopped_early} "
        f"best_fdm_epoch={monitor.best_epoch} "
        f"best_fdm_validation_score={monitor.best_score:.6e} "
        f"best_weights={config.best_fdm_finetune_weights_path} "
        f"final_weights={config.final_adam_finetune_weights_path}"
    )
    return AdamFineTuneResult(
        model=model,
        history=history,
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
        best_score=monitor.best_score,
        best_epoch=monitor.best_epoch,
    )


def run_lbfgs_finetuning(
    config: ExperimentConfig = DEFAULT_CONFIG,
    initial_weights_path: Optional[str | Path] = None,
) -> LBFGSResult:
    """Fine tune the best resumed-Adam checkpoint with fixed-batch L-BFGS."""
    fine_tuning = config.fine_tuning
    source_path = Path(
        initial_weights_path or config.best_fdm_finetune_weights_path
    )
    save_experiment_config(
        config,
        run_entry=(
            "experiments.clamped_rock_roof_nonuniform_support."
            "finetune.lbfgs"
        ),
        metadata={"initial_weights_path": source_path},
    )
    model = _load_model(config, source_path)
    monitor_config = replace(
        config,
        fdm_evaluation_every=fine_tuning.lbfgs_fdm_evaluation_every,
    )
    monitor = FDMReferenceMonitor(
        config=monitor_config,
        model=model,
        checkpoint_path=config.best_fdm_lbfgs_weights_path,
    )

    baseline = monitor.evaluate(0)
    print(_format_fdm_line("lbfgs_fdm_validation", "iteration", baseline))

    def evaluate_callback(
        iteration: int,
        callback_model: tf.keras.Model,
        loss_dict: Dict[str, tf.Tensor],
    ) -> None:
        del callback_model, loss_dict
        if iteration % fine_tuning.lbfgs_fdm_evaluation_every != 0:
            return
        snapshot = monitor.evaluate(iteration)
        print(_format_fdm_line("lbfgs_fdm_validation", "iteration", snapshot))

    lbfgs_trainer = PINNLBFGSTrainer(
        model=model,
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(config),
        foundation_fn=create_foundation_fn(config),
        parameters=config.plate,
        training_config=config.training,
        lbfgs_config=fine_tuning.lbfgs,
        loss_weights=fine_tuning.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        mxy_consistency_config=config.mxy_consistency_config,
        verbose=True,
        callbacks=(evaluate_callback,),
    )
    result = lbfgs_trainer.train()

    if int(monitor.records[-1]["epoch"].numpy()) != result.iterations:
        final_fdm = monitor.evaluate(result.iterations)
        print(_format_fdm_line("lbfgs_fdm_validation", "iteration", final_fdm))

    save_weights(model, config.final_lbfgs_weights_path)
    save_history_csv(result.history, config.lbfgs_history_path)
    save_history_csv(
        _rename_index(monitor.records, source="epoch", target="iteration"),
        config.lbfgs_fdm_history_path,
    )

    # Leave the returned model state at the best FDM-validation iteration.
    model.load_weights(str(config.best_fdm_lbfgs_weights_path))
    print(
        "lbfgs_finetune_summary "
        f"source={source_path} "
        f"success={result.success} "
        f"iterations={result.iterations} "
        f"function_evaluations={result.function_evaluations} "
        f"final_loss={result.final_loss:.6e} "
        f"best_fdm_iteration={monitor.best_epoch} "
        f"best_fdm_validation_score={monitor.best_score:.6e} "
        f"best_weights={config.best_fdm_lbfgs_weights_path} "
        f"final_weights={config.final_lbfgs_weights_path} "
        f"message={result.message}"
    )
    return result


def run_finetuning(
    config: ExperimentConfig = DEFAULT_CONFIG,
    initial_weights_path: Optional[str | Path] = None,
) -> tuple[AdamFineTuneResult, LBFGSResult]:
    """Run resumed Adam and then L-BFGS from the best Adam checkpoint."""
    adam_result = run_adam_finetuning(
        config=config,
        initial_weights_path=initial_weights_path,
    )
    lbfgs_result = run_lbfgs_finetuning(
        config=config,
        initial_weights_path=config.best_fdm_finetune_weights_path,
    )
    return adam_result, lbfgs_result


def _load_model(config: ExperimentConfig, weights_path: Path) -> tf.keras.Model:
    """Build the configured architecture and load an existing checkpoint."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuning checkpoint does not exist: {weights_path}"
        )
    set_reproducible_seed(config.seed)
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(weights_path))
    return model


def _offset_seed(seed: Optional[int], offset: int) -> Optional[int]:
    """Return a deterministic but distinct seed for a resumed stage."""
    if seed is None:
        return None
    return seed + offset


def _should_log_adam(epoch: int, total_epochs: int, log_every: int) -> bool:
    """Return whether a resumed Adam loss should be printed and stored."""
    return epoch == 1 or epoch == total_epochs or epoch % log_every == 0


def _snapshot(
    index_name: str,
    index: int,
    values: Dict[str, tf.Tensor],
) -> Dict[str, tf.Tensor]:
    """Copy scalar tensors into one history record."""
    snapshot = {index_name: tf.constant(index, dtype=tf.int32)}
    snapshot.update({name: tf.identity(value) for name, value in values.items()})
    return snapshot


def _rename_index(
    records: List[Dict[str, tf.Tensor]],
    source: str,
    target: str,
) -> List[Dict[str, tf.Tensor]]:
    """Rename a scalar history index without modifying monitor records."""
    renamed = []
    for record in records:
        item = {target: tf.identity(record[source])}
        item.update(
            {
                name: tf.identity(value)
                for name, value in record.items()
                if name != source
            }
        )
        renamed.append(item)
    return renamed


def _format_loss_line(
    prefix: str,
    index_name: str,
    index: int,
    losses: Dict[str, tf.Tensor],
) -> str:
    """Format the active weights and principal losses."""
    keys = (
        "moment_weight",
        "shear_weight",
        "equilibrium_weight",
        "boundary_weight",
        "mxy_consistency_weight",
        "total_loss",
        "physics_loss",
        "weighted_moment_loss",
        "weighted_shear_loss",
        "weighted_equilibrium_loss",
        "weighted_boundary_loss",
        "weighted_mxy_consistency_loss",
        "moment_loss",
        "moment_x_loss",
        "moment_y_loss",
        "twisting_moment_loss",
        "mxy_consistency_loss",
        "mxy_consistency_scale",
        "shear_loss",
        "shear_x_loss",
        "shear_y_loss",
        "equilibrium_loss",
        "boundary_loss",
    )
    parts = [f"{prefix} {index_name}={index}"]
    for key in keys:
        if key in losses:
            parts.append(f"{key}={float(losses[key].numpy()):.6e}")
    return " ".join(parts)


def _format_fdm_line(
    prefix: str,
    index_name: str,
    snapshot: Dict[str, tf.Tensor],
) -> str:
    """Format six-field FDM validation output for either optimizer stage."""
    index = int(snapshot["epoch"].numpy())
    score = float(snapshot["fdm_validation_score"].numpy())
    parts = [
        f"{prefix} {index_name}={index}",
        f"fdm_validation_score={score:.6e}",
    ]
    for field in FIELD_NAMES:
        parts.append(
            f"{field}_rel_l2={float(snapshot[f'{field}_rel_l2'].numpy()):.6e}"
        )
    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    """Parse fine-tuning command-line settings."""
    fine_tuning = DEFAULT_CONFIG.fine_tuning
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("both", "adam", "lbfgs"),
        default="both",
        help="Run both stages, resumed Adam only, or L-BFGS only.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Override the default checkpoint used to start the selected stage.",
    )
    parser.add_argument("--adam-epochs", type=int, default=fine_tuning.adam_epochs)
    parser.add_argument(
        "--adam-learning-rate",
        type=float,
        default=fine_tuning.adam_learning_rate,
    )
    parser.add_argument(
        "--adam-log-every",
        type=int,
        default=fine_tuning.adam_log_every,
    )
    parser.add_argument(
        "--fdm-evaluation-every",
        type=int,
        default=fine_tuning.fdm_evaluation_every,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=fine_tuning.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=fine_tuning.early_stopping_min_delta,
    )
    parser.add_argument(
        "--lbfgs-iterations",
        type=int,
        default=fine_tuning.lbfgs.max_iterations,
    )
    parser.add_argument(
        "--lbfgs-log-every",
        type=int,
        default=fine_tuning.lbfgs.log_every,
    )
    parser.add_argument(
        "--lbfgs-fdm-evaluation-every",
        type=int,
        default=fine_tuning.lbfgs_fdm_evaluation_every,
    )
    parser.add_argument(
        "--fdm-evaluation-batch-size",
        type=int,
        default=DEFAULT_CONFIG.fdm_evaluation_batch_size,
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Apply command-line overrides to the immutable configuration."""
    lbfgs = replace(
        DEFAULT_CONFIG.fine_tuning.lbfgs,
        max_iterations=args.lbfgs_iterations,
        log_every=args.lbfgs_log_every,
    )
    fine_tuning = replace(
        DEFAULT_CONFIG.fine_tuning,
        adam_epochs=args.adam_epochs,
        adam_learning_rate=args.adam_learning_rate,
        adam_log_every=args.adam_log_every,
        fdm_evaluation_every=args.fdm_evaluation_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lbfgs=lbfgs,
        lbfgs_fdm_evaluation_every=args.lbfgs_fdm_evaluation_every,
    )
    return replace(
        DEFAULT_CONFIG,
        fine_tuning=fine_tuning,
        fdm_evaluation_batch_size=args.fdm_evaluation_batch_size,
    )


def main() -> None:
    """Run the selected fine-tuning stage from the command line."""
    args = _parse_args()
    config = _config_from_args(args)

    if args.stage == "adam":
        run_adam_finetuning(config, initial_weights_path=args.initial_checkpoint)
        return
    if args.stage == "lbfgs":
        run_lbfgs_finetuning(config, initial_weights_path=args.initial_checkpoint)
        return
    run_finetuning(config, initial_weights_path=args.initial_checkpoint)


if __name__ == "__main__":
    main()
