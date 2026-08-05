"""Train the strict all-shared low-order Mixed-PINN baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .all_shared_mixed_config import ALL_SHARED_MIXED_CONFIG
from .config import ExperimentConfig
from .run import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=ALL_SHARED_MIXED_CONFIG.training.epochs)
    parser.add_argument("--interior-points", type=int, default=ALL_SHARED_MIXED_CONFIG.training.interior_points)
    parser.add_argument("--boundary-points-per-side", type=int, default=ALL_SHARED_MIXED_CONFIG.training.boundary_points_per_side)
    parser.add_argument("--log-every", type=int, default=ALL_SHARED_MIXED_CONFIG.training.log_every)
    parser.add_argument("--validation-interval", type=int, default=ALL_SHARED_MIXED_CONFIG.validation_interval)
    parser.add_argument("--validation-grid-size", type=int, default=ALL_SHARED_MIXED_CONFIG.validation_grid_size)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=ALL_SHARED_MIXED_CONFIG.results_dir)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    training = replace(
        ALL_SHARED_MIXED_CONFIG.training,
        epochs=args.epochs,
        interior_points=args.interior_points,
        boundary_points_per_side=args.boundary_points_per_side,
        log_every=args.log_every,
    )
    return replace(
        ALL_SHARED_MIXED_CONFIG,
        training=training,
        validation_interval=args.validation_interval,
        validation_grid_size=args.validation_grid_size,
        initial_checkpoint_path=args.initial_checkpoint,
        results_dir=args.results_dir,
    )


def main() -> None:
    config = _config_from_args(_parse_args())
    history = run_experiment(
        config,
        run_entry=(
            "experiments.simply_clamped_uniform_load."
            "run_all_shared_mixed_pinn"
        ),
        model_name="all_shared_mixed_pinn",
    )
    print(
        "simply_clamped_uniform_load_all_shared_mixed_pinn "
        f"epochs={config.training.epochs} "
        f"final_total_loss={float(history[-1]['total_loss'].numpy()):.6e} "
        f"best_weights={config.best_validation_loss_weights_path}"
    )


if __name__ == "__main__":
    main()
