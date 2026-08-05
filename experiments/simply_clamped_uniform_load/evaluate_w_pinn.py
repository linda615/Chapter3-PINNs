"""Evaluate mixed-edge W-PINN using fields reconstructed from deflection."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from models import FIELD_NAMES
from physics import compute_w_pinn_fields

from .analytical import analytical_fields
from .config import ExperimentConfig
from .evaluate import (
    make_test_grid,
    relative_l2_error,
    save_pointwise_csv,
    save_rel_l2_csv,
)
from .w_pinn_config import W_PINN_CONFIG
from .w_pinn_problem import create_w_pinn_model


DEFAULT_CHECKPOINT = W_PINN_CONFIG.best_validation_loss_weights_path
DEFAULT_POINTWISE_CSV = W_PINN_CONFIG.results_dir / "evaluation_fields.csv"
DEFAULT_REL_L2_CSV = W_PINN_CONFIG.results_dir / "evaluation_rel_l2.csv"


def evaluate_w_pinn_model(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = W_PINN_CONFIG,
    batch_size: int = 1024,
) -> tuple[
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
]:
    """Return six w-derived fields, Levy fields, and field-wise RelL2."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    xy = tf.convert_to_tensor(xy)
    batches = {name: [] for name in FIELD_NAMES}
    point_count = int(xy.shape[0])
    for start in range(0, point_count, batch_size):
        fields = compute_w_pinn_fields(
            model=model,
            xy=xy[start : start + batch_size],
            parameters=config.plate,
            training=False,
        )
        for name in FIELD_NAMES:
            batches[name].append(fields[name])
    predicted = {
        name: tf.concat(batches[name], axis=0)
        for name in FIELD_NAMES
    }
    reference = analytical_fields(
        xy=xy,
        parameters=config.plate,
        q0=config.q0,
        mode_count=config.analytical_modes,
    )
    rel_l2 = {
        name: relative_l2_error(predicted[name], reference[name])
        for name in FIELD_NAMES
    }
    return predicted, reference, rel_l2


def validation_score(rel_l2: dict[str, tf.Tensor]) -> tf.Tensor:
    """Return the arithmetic mean of all six relative L2 errors."""
    values = [rel_l2[name] for name in FIELD_NAMES]
    return tf.add_n(values) / tf.cast(len(values), values[0].dtype)


def run_evaluation(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    pointwise_csv: str | Path = DEFAULT_POINTWISE_CSV,
    rel_l2_csv: str | Path = DEFAULT_REL_L2_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = W_PINN_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
) -> dict[str, tf.Tensor]:
    """Load W-PINN, reconstruct all fields, and save comparison outputs."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model = create_w_pinn_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(checkpoint_path))
    xy = make_test_grid(grid_size)
    predicted, reference, rel_l2 = evaluate_w_pinn_model(
        model,
        xy,
        config,
        batch_size,
    )
    save_pointwise_csv(Path(pointwise_csv), xy, predicted, reference)
    save_rel_l2_csv(
        Path(rel_l2_csv),
        rel_l2,
        checkpoint_path,
        grid_size,
    )
    if figures_dir is not None:
        from experiments.plotting import save_field_figures

        save_field_figures(
            Path(figures_dir),
            predicted,
            reference,
            grid_size,
            reference_title="Levy series reference",
        )
    return rel_l2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pointwise-csv", type=Path, default=DEFAULT_POINTWISE_CSV)
    parser.add_argument("--rel-l2-csv", type=Path, default=DEFAULT_REL_L2_CSV)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rel_l2 = run_evaluation(
        checkpoint_path=args.checkpoint,
        pointwise_csv=args.pointwise_csv,
        rel_l2_csv=args.rel_l2_csv,
        grid_size=args.grid_size,
        figures_dir=W_PINN_CONFIG.figures_dir if args.save_figures else None,
        batch_size=args.batch_size,
    )
    parts = [
        "simply_clamped_uniform_load_w_pinn_evaluation",
        f"validation_score={float(validation_score(rel_l2).numpy()):.6e}",
    ]
    parts.extend(
        f"{name}_rel_l2={float(rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    print(" ".join(parts))


if __name__ == "__main__":
    main()
