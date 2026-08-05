"""Evaluate a trained simply supported sinusoidal plate model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import tensorflow as tf

from experiments.plotting import save_field_figures
from experiments.simply_clamped_uniform_load.evaluate import (
    MixedFieldEvaluation,
    reconstruct_fields_from_w,
)
from models import FIELD_NAMES

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


def relative_l2_error(
    prediction: tf.Tensor,
    reference: tf.Tensor,
) -> tf.Tensor:
    """Return relative L2 error with a small denominator guard."""
    prediction = tf.convert_to_tensor(prediction)
    reference = tf.convert_to_tensor(reference)
    numerator = tf.sqrt(tf.reduce_mean(tf.square(prediction - reference)))
    denominator = tf.sqrt(tf.reduce_mean(tf.square(reference)))
    return numerator / tf.maximum(denominator, tf.cast(1e-12, prediction.dtype))


def make_test_grid(
    grid_size: int = 101,
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    """Create a square ``grid_size x grid_size`` test grid on ``[0, 1]^2``."""
    if grid_size <= 1:
        raise ValueError(f"grid_size must be greater than 1, got {grid_size}.")
    values = tf.linspace(tf.cast(0.0, dtype), tf.cast(1.0, dtype), grid_size)
    x_grid, y_grid = tf.meshgrid(values, values, indexing="xy")
    return tf.stack([tf.reshape(x_grid, (-1,)), tf.reshape(y_grid, (-1,))], axis=1)


def load_best_checkpoint(
    model: tf.keras.Model,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> str:
    """Load a best weights file or TensorFlow checkpoint into ``model``."""
    checkpoint_path = Path(checkpoint_path)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    if checkpoint_path.suffix == ".h5":
        weights_path = _resolve_h5_weights_path(checkpoint_path)
        model.load_weights(str(weights_path))
        return str(weights_path)

    checkpoint_prefix = _resolve_checkpoint_prefix(checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_prefix).expect_partial()
    return checkpoint_prefix


def evaluate_model(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Return predictions, analytical references, and six field RelL2 errors."""
    predicted = model(xy, training=False)
    reference = analytical_fields(xy, config.plate)
    rel_l2 = {
        field: relative_l2_error(predicted[field], reference[field])
        for field in FIELD_NAMES
    }
    return predicted, reference, rel_l2


def run_full_evaluation(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    pointwise_csv: str | Path = DEFAULT_POINTWISE_CSV,
    rel_l2_csv: str | Path = DEFAULT_REL_L2_CSV,
    from_w_pointwise_csv: str | Path = DEFAULT_FROM_W_POINTWISE_CSV,
    from_w_rel_l2_csv: str | Path = DEFAULT_FROM_W_REL_L2_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
) -> MixedFieldEvaluation:
    """Evaluate direct and deflection-reconstructed fields in one model load."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    model = create_model(config)
    resolved_checkpoint = load_best_checkpoint(model, checkpoint_path)
    xy = make_test_grid(grid_size=grid_size)
    predicted, reference, rel_l2 = evaluate_model(model, xy, config=config)
    reconstructed = reconstruct_fields_from_w(
        model=model,
        xy=xy,
        config=config,
        batch_size=batch_size,
    )
    from_w_rel_l2 = {
        field: relative_l2_error(reconstructed[field], reference[field])
        for field in FIELD_NAMES
    }

    save_pointwise_csv(pointwise_csv, xy, predicted, reference)
    save_rel_l2_csv(rel_l2_csv, rel_l2, resolved_checkpoint, grid_size)
    save_pointwise_csv(from_w_pointwise_csv, xy, reconstructed, reference)
    save_rel_l2_csv(
        from_w_rel_l2_csv,
        from_w_rel_l2,
        resolved_checkpoint,
        grid_size,
    )
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        # Keep root images for compatibility and mirror the uniform-load layout.
        save_field_figures(
            figures_dir,
            predicted,
            reference,
            grid_size=grid_size,
            reference_title="解析解",
        )
        save_field_figures(
            figures_dir / "direct",
            predicted,
            reference,
            grid_size=grid_size,
            reference_title="解析解",
        )
        save_field_figures(
            figures_dir / "from_w",
            reconstructed,
            reference,
            grid_size=grid_size,
            reference_title="解析解",
        )
    return MixedFieldEvaluation(
        direct=predicted,
        from_w=reconstructed,
        reference=reference,
        direct_rel_l2=rel_l2,
        from_w_rel_l2=from_w_rel_l2,
    )


def run_evaluation(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    pointwise_csv: str | Path = DEFAULT_POINTWISE_CSV,
    rel_l2_csv: str | Path = DEFAULT_REL_L2_CSV,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
) -> dict[str, tf.Tensor]:
    """Run full evaluation while preserving the legacy direct-error return."""
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        pointwise_csv=pointwise_csv,
        rel_l2_csv=rel_l2_csv,
        from_w_pointwise_csv=Path(pointwise_csv).with_name(
            "evaluation_from_w_fields.csv"
        ),
        from_w_rel_l2_csv=Path(rel_l2_csv).with_name(
            "evaluation_from_w_rel_l2.csv"
        ),
        grid_size=grid_size,
        config=config,
        figures_dir=figures_dir,
        batch_size=batch_size,
    )
    return evaluation.direct_rel_l2


def save_pointwise_csv(
    path: str | Path,
    xy: tf.Tensor,
    predicted: dict[str, tf.Tensor],
    reference: dict[str, tf.Tensor],
) -> None:
    """Save pointwise predicted and analytical fields to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["x", "y"]
    for field in FIELD_NAMES:
        header.extend([f"{field}_pred", f"{field}_ref"])

    rows = []
    xy_values = xy.numpy()
    predicted_values = {field: predicted[field].numpy() for field in FIELD_NAMES}
    reference_values = {field: reference[field].numpy() for field in FIELD_NAMES}
    for index in range(xy.shape[0]):
        row = [xy_values[index, 0], xy_values[index, 1]]
        for field in FIELD_NAMES:
            row.extend([predicted_values[field][index, 0], reference_values[field][index, 0]])
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_rel_l2_csv(
    path: str | Path,
    rel_l2: dict[str, tf.Tensor],
    checkpoint_path: str,
    grid_size: int,
) -> None:
    """Save six field relative L2 errors to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "rel_l2", "checkpoint", "grid_size"])
        for field in FIELD_NAMES:
            writer.writerow([field, float(rel_l2[field].numpy()), checkpoint_path, grid_size])


def _resolve_checkpoint_prefix(path: Path) -> str:
    """Resolve a checkpoint directory, prefix, or ``best`` checkpoint path."""
    if path.is_dir():
        latest = tf.train.latest_checkpoint(str(path))
        if latest is not None:
            return latest
        best_prefix = path / "best"
        if best_prefix.with_suffix(".index").exists():
            return str(best_prefix)
    if path.with_suffix(".index").exists():
        return str(path)
    candidates = sorted(path.parent.glob(f"{path.name}-*.index"))
    if candidates:
        return str(candidates[-1].with_suffix(""))
    raise FileNotFoundError(
        "Could not find checkpoint. Expected a checkpoint directory, a checkpoint "
        f"prefix, or files like {path}.index."
    )


def _resolve_h5_weights_path(path: Path) -> Path:
    """Resolve H5 weights, including legacy aliases for the best checkpoint."""
    if path.exists():
        return path

    if path.name == "best_weights.h5":
        for alias_name in ("best_validation_loss.weights.h5", "best.weights.h5"):
            alias = path.with_name(alias_name)
            if alias.exists():
                return alias
    elif path.name == "best.weights.h5":
        for alias_name in ("best_validation_loss.weights.h5", "best_weights.h5"):
            alias = path.with_name(alias_name)
            if alias.exists():
                return alias
    elif path.name == "best_validation_loss.weights.h5":
        for alias_name in ("best.weights.h5", "best_weights.h5"):
            alias = path.with_name(alias_name)
            if alias.exists():
                return alias

    if path.name in {"best_weights.h5", "best.weights.h5", "best_validation_loss.weights.h5"}:
        best_epoch_weights = _find_latest_best_epoch_weights(path.parent)
        if best_epoch_weights is not None:
            return best_epoch_weights

    available = _format_available_weights(path.parent)
    raise FileNotFoundError(
        f"Could not find weights file: {path}. Available weights: {available}"
    )


def _find_latest_best_epoch_weights(directory: Path) -> Path | None:
    """Return the highest numbered ``best_epoch_*.weights.h5`` file if present."""
    if not directory.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for candidate in directory.glob("best_epoch_*.weights.h5"):
        epoch_text = candidate.name.removeprefix("best_epoch_").removesuffix(".weights.h5")
        try:
            epoch = int(epoch_text)
        except ValueError:
            continue
        candidates.append((epoch, candidate))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _format_available_weights(directory: Path) -> str:
    """Format available H5 weights for clear file-not-found errors."""
    if not directory.exists():
        return "none; checkpoint directory does not exist"
    weights = sorted(path.name for path in directory.glob("*.h5"))
    if not weights:
        return "none"
    return ", ".join(weights)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--pointwise-csv", default=str(DEFAULT_POINTWISE_CSV))
    parser.add_argument("--rel-l2-csv", default=str(DEFAULT_REL_L2_CSV))
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for model evaluation."""
    args = _parse_args()
    figures_dir = DEFAULT_CONFIG.figures_dir if args.save_figures else None
    evaluation = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        pointwise_csv=args.pointwise_csv,
        rel_l2_csv=args.rel_l2_csv,
        from_w_pointwise_csv=Path(args.pointwise_csv).with_name(
            "evaluation_from_w_fields.csv"
        ),
        from_w_rel_l2_csv=Path(args.rel_l2_csv).with_name(
            "evaluation_from_w_rel_l2.csv"
        ),
        grid_size=args.grid_size,
        figures_dir=figures_dir,
        batch_size=args.batch_size,
    )
    direct_summary = " ".join(
        f"{field}_rel_l2="
        f"{float(evaluation.direct_rel_l2[field].numpy()):.6e}"
        for field in FIELD_NAMES
    )
    from_w_summary = " ".join(
        f"{field}_from_w_rel_l2="
        f"{float(evaluation.from_w_rel_l2[field].numpy()):.6e}"
        for field in FIELD_NAMES
    )
    print(
        "simply_supported_sinusoidal_evaluation "
        f"{direct_summary} {from_w_summary}"
    )


if __name__ == "__main__":
    main()
