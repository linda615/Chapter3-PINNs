"""Evaluate the mixed-edge uniform-load plate against its Levy solution."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import tensorflow as tf

from models import FIELD_NAMES
from physics import compute_w_pinn_fields

from .analytical import analytical_fields
from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import create_model


DEFAULT_CHECKPOINT = DEFAULT_CONFIG.best_validation_loss_weights_path
DEFAULT_POINTWISE_CSV = DEFAULT_CONFIG.results_dir / "evaluation_fields.csv"
DEFAULT_REL_L2_CSV = DEFAULT_CONFIG.results_dir / "evaluation_rel_l2.csv"
DEFAULT_FROM_W_POINTWISE_CSV = (
    DEFAULT_CONFIG.results_dir / "evaluation_from_w_fields.csv"
)
DEFAULT_FROM_W_REL_L2_CSV = (
    DEFAULT_CONFIG.results_dir / "evaluation_from_w_rel_l2.csv"
)


@dataclass(frozen=True)
class MixedFieldEvaluation:
    """Direct, deflection-derived, and reference evaluation fields."""

    direct: dict[str, tf.Tensor]
    from_w: dict[str, tf.Tensor]
    reference: dict[str, tf.Tensor]
    direct_rel_l2: dict[str, tf.Tensor]
    from_w_rel_l2: dict[str, tf.Tensor]


def relative_l2_error(
    prediction: tf.Tensor,
    reference: tf.Tensor,
) -> tf.Tensor:
    """Return the relative L2 error with a guarded reference norm."""
    prediction = tf.convert_to_tensor(prediction)
    reference = tf.cast(reference, prediction.dtype)
    numerator = tf.sqrt(tf.reduce_sum(tf.square(prediction - reference)))
    denominator = tf.sqrt(tf.reduce_sum(tf.square(reference)))
    return numerator / tf.maximum(
        denominator,
        tf.cast(1e-12, prediction.dtype),
    )


def make_test_grid(
    grid_size: int = 101,
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    """Create a flattened regular grid on the unit square."""
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")
    values = tf.linspace(tf.cast(0.0, dtype), tf.cast(1.0, dtype), grid_size)
    x_grid, y_grid = tf.meshgrid(values, values, indexing="xy")
    return tf.stack(
        (
            tf.reshape(x_grid, (-1,)),
            tf.reshape(y_grid, (-1,)),
        ),
        axis=1,
    )


def evaluate_model(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
]:
    """Return physical predictions, Levy fields, and relative L2 errors.

    ``create_model`` restores the configured physical output scales inside
    the model call, so no additional post-processing scale is applied here.
    """
    predicted = model(xy, training=False)
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


def run_evaluation(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    pointwise_csv: Path = DEFAULT_POINTWISE_CSV,
    rel_l2_csv: Path = DEFAULT_REL_L2_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: Path | None = None,
    batch_size: int = 1024,
) -> dict[str, tf.Tensor]:
    """Run full evaluation while preserving the direct-error return API."""
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        pointwise_csv=pointwise_csv,
        rel_l2_csv=rel_l2_csv,
        grid_size=grid_size,
        config=config,
        figures_dir=figures_dir,
        batch_size=batch_size,
    )
    return evaluation.direct_rel_l2


def run_full_evaluation(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    pointwise_csv: Path = DEFAULT_POINTWISE_CSV,
    rel_l2_csv: Path = DEFAULT_REL_L2_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: Path | None = None,
    batch_size: int = 1024,
    from_w_pointwise_csv: Path | None = None,
    from_w_rel_l2_csv: Path | None = None,
) -> MixedFieldEvaluation:
    """Load a checkpoint and evaluate direct and reconstructed fields."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.load_weights(str(checkpoint_path))

    xy = make_test_grid(grid_size)
    predicted, reference, rel_l2 = evaluate_model(model, xy, config)
    from_w = reconstruct_fields_from_w(
        model=model,
        xy=xy,
        config=config,
        batch_size=batch_size,
    )
    from_w_rel_l2 = {
        name: relative_l2_error(from_w[name], reference[name])
        for name in FIELD_NAMES
    }
    if from_w_pointwise_csv is None:
        from_w_pointwise_csv = Path(pointwise_csv).with_name(
            "evaluation_from_w_fields.csv"
        )
    if from_w_rel_l2_csv is None:
        from_w_rel_l2_csv = Path(rel_l2_csv).with_name(
            "evaluation_from_w_rel_l2.csv"
        )
    save_pointwise_csv(pointwise_csv, xy, predicted, reference)
    save_rel_l2_csv(
        rel_l2_csv,
        rel_l2,
        checkpoint_path,
        grid_size,
    )
    save_pointwise_csv(from_w_pointwise_csv, xy, from_w, reference)
    save_rel_l2_csv(
        from_w_rel_l2_csv,
        from_w_rel_l2,
        checkpoint_path,
        grid_size,
    )
    if figures_dir is not None:
        from experiments.plotting import save_field_figures

        figures_dir = Path(figures_dir)
        save_field_figures(
            figures_dir=figures_dir,
            predicted=predicted,
            reference=reference,
            grid_size=grid_size,
            reference_title="Levy 级数解",
        )
        save_field_figures(
            figures_dir=figures_dir / "direct",
            predicted=predicted,
            reference=reference,
            grid_size=grid_size,
            reference_title="Levy 级数解",
        )
        save_field_figures(
            figures_dir=figures_dir / "from_w",
            predicted=from_w,
            reference=reference,
            grid_size=grid_size,
            reference_title="Levy 级数解",
        )
    return MixedFieldEvaluation(
        direct=predicted,
        from_w=from_w,
        reference=reference,
        direct_rel_l2=rel_l2,
        from_w_rel_l2=from_w_rel_l2,
    )


def reconstruct_fields_from_w(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
    batch_size: int = 1024,
) -> dict[str, tf.Tensor]:
    """Reconstruct all six fields from deflection in bounded batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    xy = tf.convert_to_tensor(xy)
    point_count = int(xy.shape[0])
    batches = {name: [] for name in FIELD_NAMES}
    for start in range(0, point_count, batch_size):
        fields = compute_w_pinn_fields(
            model=model,
            xy=xy[start : start + batch_size],
            parameters=config.plate,
            training=False,
        )
        for name in FIELD_NAMES:
            batches[name].append(fields[name])
    return {
        name: tf.concat(batches[name], axis=0)
        for name in FIELD_NAMES
    }


def save_pointwise_csv(
    path: Path,
    xy: tf.Tensor,
    predicted: dict[str, tf.Tensor],
    reference: dict[str, tf.Tensor],
) -> None:
    """Save coordinates and predicted/reference fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [xy[:, 0], xy[:, 1]]
    headers = ["x", "y"]
    for name in FIELD_NAMES:
        columns.extend((predicted[name][:, 0], reference[name][:, 0]))
        headers.extend((f"{name}_pred", f"{name}_exact"))
    values = tf.stack(columns, axis=1).numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(values.tolist())


def save_rel_l2_csv(
    path: Path,
    rel_l2: dict[str, tf.Tensor],
    checkpoint_path: Path,
    grid_size: int,
) -> None:
    """Save field-wise relative L2 metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "rel_l2", "checkpoint", "grid_size"))
        for name in FIELD_NAMES:
            writer.writerow(
                (
                    name,
                    float(rel_l2[name].numpy()),
                    str(checkpoint_path),
                    grid_size,
                )
            )


def _parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pointwise-csv", type=Path, default=None)
    parser.add_argument("--rel-l2-csv", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument(
        "--model-grouping",
        choices=("one", "two", "three", "six"),
        default=DEFAULT_CONFIG.model_grouping,
        help="Build the checkpoint-compatible Mixed-PINN grouping.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for evaluation CSV files and optional figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate a checkpoint from the command line."""
    args = _parse_args()
    results_dir = (
        DEFAULT_CONFIG.results_dir
        if args.results_dir is None
        else args.results_dir
    )
    config = replace(
        DEFAULT_CONFIG,
        model_grouping=args.model_grouping,
        results_dir=results_dir,
    )
    pointwise_csv = (
        results_dir / "evaluation_fields.csv"
        if args.pointwise_csv is None
        else args.pointwise_csv
    )
    rel_l2_csv = (
        results_dir / "evaluation_rel_l2.csv"
        if args.rel_l2_csv is None
        else args.rel_l2_csv
    )
    checkpoint_path = (
        config.best_validation_loss_weights_path
        if args.checkpoint is None
        else args.checkpoint
    )
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        pointwise_csv=pointwise_csv,
        rel_l2_csv=rel_l2_csv,
        grid_size=args.grid_size,
        config=config,
        figures_dir=config.figures_dir if args.save_figures else None,
        batch_size=args.batch_size,
    )
    print(
        "evaluation_field_output_scales "
        + " ".join(
            f"{name}={value:.9e}"
            for name, value in config.field_output_scales.as_dict().items()
        )
    )
    score = tf.add_n(
        [evaluation.direct_rel_l2[name] for name in FIELD_NAMES]
    ) / float(
        len(FIELD_NAMES)
    )
    parts = [
        "simply_clamped_uniform_load_evaluation",
        f"validation_score={float(score.numpy()):.6e}",
    ]
    parts.extend(
        f"{name}_rel_l2="
        f"{float(evaluation.direct_rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    parts.extend(
        f"{name}_from_w_rel_l2="
        f"{float(evaluation.from_w_rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    print(" ".join(parts))


if __name__ == "__main__":
    main()
