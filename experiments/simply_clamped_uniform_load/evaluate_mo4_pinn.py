"""Evaluate direct and w-derived fields from the mixed-edge MO4-PINN."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from models import FIELD_NAMES

from .analytical import analytical_fields
from .config import ExperimentConfig
from .evaluate import (
    make_test_grid,
    reconstruct_fields_from_w,
    relative_l2_error,
    save_pointwise_csv,
    save_rel_l2_csv,
)
from .mo4_config import MO4_PINN_CONFIG
from .problem import create_model


DEFAULT_CHECKPOINT = MO4_PINN_CONFIG.best_validation_loss_weights_path
DEFAULT_POINTWISE_CSV = MO4_PINN_CONFIG.results_dir / "evaluation_fields.csv"
DEFAULT_METRICS_CSV = MO4_PINN_CONFIG.results_dir / "evaluation_rel_l2.csv"


@dataclass(frozen=True)
class MO4Evaluation:
    direct: dict[str, tf.Tensor]
    from_w: dict[str, tf.Tensor]
    reference: dict[str, tf.Tensor]
    direct_rel_l2: dict[str, tf.Tensor]
    from_w_rel_l2: dict[str, tf.Tensor]
    consistency_rel_l2: dict[str, tf.Tensor]


def evaluate_mo4_model(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = MO4_PINN_CONFIG,
    batch_size: int = 1024,
) -> MO4Evaluation:
    """Evaluate direct fields and an independent reconstruction from w."""
    direct = model(xy, training=False)
    from_w = reconstruct_fields_from_w(model, xy, config, batch_size)
    reference = analytical_fields(
        xy=xy,
        parameters=config.plate,
        q0=config.q0,
        mode_count=config.analytical_modes,
    )
    direct_rel_l2 = {
        name: relative_l2_error(direct[name], reference[name])
        for name in FIELD_NAMES
    }
    from_w_rel_l2 = {
        name: relative_l2_error(from_w[name], reference[name])
        for name in FIELD_NAMES
    }
    consistency_rel_l2 = {
        name: relative_l2_error(direct[name], from_w[name])
        for name in FIELD_NAMES
    }
    return MO4Evaluation(
        direct,
        from_w,
        reference,
        direct_rel_l2,
        from_w_rel_l2,
        consistency_rel_l2,
    )


def validation_score(rel_l2: dict[str, tf.Tensor]) -> tf.Tensor:
    values = [rel_l2[name] for name in FIELD_NAMES]
    return tf.add_n(values) / tf.cast(len(values), values[0].dtype)


def run_evaluation(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    pointwise_csv: str | Path = DEFAULT_POINTWISE_CSV,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = MO4_PINN_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
) -> MO4Evaluation:
    """Load MO4 weights and save direct/from-w comparison artifacts."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(checkpoint_path))
    xy = make_test_grid(grid_size)
    evaluation = evaluate_mo4_model(model, xy, config, batch_size)
    save_pointwise_csv(Path(pointwise_csv), xy, evaluation.direct, evaluation.reference)
    _save_metrics(Path(metrics_csv), evaluation, checkpoint_path, grid_size)
    save_pointwise_csv(
        Path(pointwise_csv).with_name("evaluation_from_w_fields.csv"),
        xy,
        evaluation.from_w,
        evaluation.reference,
    )
    save_rel_l2_csv(
        Path(metrics_csv).with_name("evaluation_from_w_rel_l2.csv"),
        evaluation.from_w_rel_l2,
        checkpoint_path,
        grid_size,
    )
    if figures_dir is not None:
        from experiments.plotting import save_field_figures

        root = Path(figures_dir)
        save_field_figures(root / "direct", evaluation.direct, evaluation.reference, grid_size)
        save_field_figures(root / "from_w", evaluation.from_w, evaluation.reference, grid_size)
    return evaluation


def _save_metrics(path: Path, evaluation: MO4Evaluation, checkpoint: Path, grid_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = (
        ("direct", evaluation.direct_rel_l2),
        ("from_w", evaluation.from_w_rel_l2),
        ("consistency", evaluation.consistency_rel_l2),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "field", "rel_l2", "checkpoint", "grid_size"))
        for source, metrics in groups:
            for name in FIELD_NAMES:
                writer.writerow((source, name, float(metrics[name].numpy()), str(checkpoint), grid_size))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pointwise-csv", type=Path, default=DEFAULT_POINTWISE_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluation = run_evaluation(
        args.checkpoint,
        args.pointwise_csv,
        args.metrics_csv,
        args.grid_size,
        figures_dir=MO4_PINN_CONFIG.figures_dir if args.save_figures else None,
        batch_size=args.batch_size,
    )
    parts = [
        "simply_clamped_uniform_load_mo4_pinn_evaluation",
        f"validation_score={float(validation_score(evaluation.direct_rel_l2).numpy()):.6e}",
    ]
    parts.extend(f"{name}_rel_l2={float(evaluation.direct_rel_l2[name].numpy()):.6e}" for name in FIELD_NAMES)
    parts.extend(f"{name}_from_w_rel_l2={float(evaluation.from_w_rel_l2[name].numpy()):.6e}" for name in FIELD_NAMES)
    print(" ".join(parts))


if __name__ == "__main__":
    main()
