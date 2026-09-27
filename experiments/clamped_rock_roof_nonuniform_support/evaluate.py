"""Evaluate the rock-roof PINN against the finite-difference reference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from experiments.simply_supported_sinusoidal.evaluate import load_best_checkpoint
from experiments.plotting import (
    relative_l2_error,
    save_field_figures,
)
from models import FIELD_NAMES
from physics import compute_w_pinn_fields

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import active_foundation_stiffness, create_model, rock_roof_load
from .run import make_grid


@dataclass(frozen=True)
class RockRoofEvaluation:
    """Direct, deflection-derived, and FDM reference evaluation fields."""

    direct: Dict[str, tf.Tensor]
    from_w: Dict[str, tf.Tensor]
    reference: Dict[str, tf.Tensor]
    direct_rel_l2: Dict[str, tf.Tensor]
    from_w_rel_l2: Dict[str, tf.Tensor]


def run_evaluation(
    checkpoint_path: str | Path = DEFAULT_CONFIG.best_fdm_validation_weights_path,
    output_csv: str | Path = DEFAULT_CONFIG.evaluation_csv_path,
    rel_l2_csv: str | Path = DEFAULT_CONFIG.relative_l2_csv_path,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
    from_w_output_csv: str | Path | None = None,
    from_w_rel_l2_csv: str | Path | None = None,
) -> Optional[Dict[str, tf.Tensor]]:
    """Run full evaluation while preserving the direct-error return API."""
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        rel_l2_csv=rel_l2_csv,
        grid_size=grid_size,
        config=config,
        figures_dir=figures_dir,
        batch_size=batch_size,
        from_w_output_csv=from_w_output_csv,
        from_w_rel_l2_csv=from_w_rel_l2_csv,
    )
    if evaluation is None:
        return None
    return evaluation.direct_rel_l2


def run_full_evaluation(
    checkpoint_path: str | Path = DEFAULT_CONFIG.best_fdm_validation_weights_path,
    output_csv: str | Path = DEFAULT_CONFIG.evaluation_csv_path,
    rel_l2_csv: str | Path = DEFAULT_CONFIG.relative_l2_csv_path,
    grid_size: int = 101,
    config: ExperimentConfig = DEFAULT_CONFIG,
    figures_dir: str | Path | None = None,
    batch_size: int = 1024,
    from_w_output_csv: str | Path | None = None,
    from_w_rel_l2_csv: str | Path | None = None,
) -> Optional[RockRoofEvaluation]:
    """Load one checkpoint and evaluate direct and reconstructed fields."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    model = create_model(config)
    resolved_checkpoint = load_best_checkpoint(model, checkpoint_path)
    xy = make_grid(grid_size)
    predicted = model(xy, training=False)
    q = rock_roof_load(xy, config=config)
    k = active_foundation_stiffness(xy, config=config)
    save_prediction_csv(output_csv, xy, predicted, q, k)

    reference = load_reference(config.reference_path)
    if reference is None:
        print(
            "reference_missing "
            f"path={config.reference_path} "
            "No FDM reference file is available; RelL2 metrics were skipped."
        )
        return None

    _validate_reference_parameters(reference, config)
    reference_xy = tf.concat(
        (
            tf.convert_to_tensor(reference["x"], dtype=xy.dtype),
            tf.convert_to_tensor(reference["y"], dtype=xy.dtype),
        ),
        axis=1,
    )
    reference_fields = {
        field: tf.convert_to_tensor(reference[field], dtype=xy.dtype)
        for field in FIELD_NAMES
    }
    direct = model(reference_xy, training=False)
    direct_rel_l2 = {
        field: relative_l2_error(direct[field], reference_fields[field])
        for field in FIELD_NAMES
    }

    from_w = reconstruct_fields_from_w(
        model=model,
        xy=reference_xy,
        config=config,
        batch_size=batch_size,
    )
    # Deflection is not reconstructed. Reusing the direct tensor avoids tiny
    # batch-size-dependent roundoff differences between two forward passes.
    from_w["w"] = direct["w"]
    from_w_rel_l2 = {
        field: relative_l2_error(from_w[field], reference_fields[field])
        for field in FIELD_NAMES
    }

    reference_grid_size = _reference_grid_size(reference)
    save_rel_l2_csv(
        rel_l2_csv,
        direct_rel_l2,
        resolved_checkpoint,
        reference_grid_size,
        config.reference_path,
    )
    if from_w_output_csv is None:
        from_w_output_csv = Path(output_csv).with_name(
            "evaluation_from_w_fields.csv"
        )
    if from_w_rel_l2_csv is None:
        from_w_rel_l2_csv = Path(rel_l2_csv).with_name(
            "evaluation_from_w_rel_l2.csv"
        )
    save_prediction_csv(
        from_w_output_csv,
        reference_xy,
        from_w,
        rock_roof_load(reference_xy, config=config),
        active_foundation_stiffness(reference_xy, config=config),
    )
    save_rel_l2_csv(
        from_w_rel_l2_csv,
        from_w_rel_l2,
        resolved_checkpoint,
        reference_grid_size,
        config.reference_path,
    )

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        save_field_figures(
            figures_dir / "direct",
            direct,
            reference_fields,
            grid_size=reference_grid_size,
            reference_title="有限差分参考解",
            prediction_title="PINN直接预测",
            error_title="绝对误差",
            panel_titles_below=True,
            include_field_title=False,
            dpi=300,
            panel_title_font_size=11.5,
        )
        save_field_figures(
            figures_dir / "from_w",
            from_w,
            reference_fields,
            grid_size=reference_grid_size,
            reference_title="有限差分参考解",
        )

    return RockRoofEvaluation(
        direct=direct,
        from_w=from_w,
        reference=reference_fields,
        direct_rel_l2=direct_rel_l2,
        from_w_rel_l2=from_w_rel_l2,
    )


def reconstruct_fields_from_w(
    model: tf.keras.Model,
    xy: tf.Tensor,
    config: ExperimentConfig = DEFAULT_CONFIG,
    batch_size: int = 1024,
) -> Dict[str, tf.Tensor]:
    """Reconstruct all six fields from deflection in bounded-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    xy = tf.convert_to_tensor(xy)
    point_count = int(xy.shape[0])
    batches = {field: [] for field in FIELD_NAMES}
    for start in range(0, point_count, batch_size):
        fields = compute_w_pinn_fields(
            model=model,
            xy=xy[start : start + batch_size],
            parameters=config.plate,
            training=False,
        )
        for field in FIELD_NAMES:
            batches[field].append(fields[field])
    return {
        field: tf.concat(batches[field], axis=0)
        for field in FIELD_NAMES
    }


def load_reference(path: str | Path) -> Optional[Dict[str, np.ndarray]]:
    """Load an optional finite-difference reference data set."""
    path = Path(path)
    if not path.exists():
        return None
    required_fields = (
        "x",
        "y",
        "w",
        "Mx",
        "My",
        "Mxy",
        "Qx",
        "Qy",
        "q",
        "k",
    )
    with np.load(str(path), allow_pickle=False) as data:
        required = required_fields + ("D", "nu")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                f"FDM reference file {path} is missing fields: {missing}."
            )
        reference = {
            name: np.asarray(data[name], dtype=np.float64).reshape((-1, 1))
            for name in required_fields
        }
        reference["D"] = np.asarray(data["D"], dtype=np.float64)
        reference["nu"] = np.asarray(data["nu"], dtype=np.float64)

    point_count = reference["x"].shape[0]
    for name in required_fields:
        if reference[name].shape != (point_count, 1):
            raise ValueError(
                f"FDM reference field {name} must have shape "
                f"({point_count}, 1), got {reference[name].shape}."
            )
        if not np.all(np.isfinite(reference[name])):
            raise ValueError(
                f"FDM reference field {name} contains non-finite values."
            )
    _reference_grid_size(reference)
    return reference


def save_prediction_csv(
    path: str | Path,
    xy: tf.Tensor,
    predicted: Dict[str, tf.Tensor],
    q: tf.Tensor,
    k: tf.Tensor,
) -> None:
    """Save pointwise PINN predictions and problem data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["x", "y", "q", "k"]
    header.extend([f"{field}_pred" for field in FIELD_NAMES])
    xy_values = xy.numpy()
    q_values = q.numpy()
    k_values = k.numpy()
    predicted_values = {
        field: predicted[field].numpy() for field in FIELD_NAMES
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(xy.shape[0]):
            row = [
                xy_values[index, 0],
                xy_values[index, 1],
                q_values[index, 0],
                k_values[index, 0],
            ]
            row.extend(
                predicted_values[field][index, 0] for field in FIELD_NAMES
            )
            writer.writerow(row)


def save_rel_l2_csv(
    path: str | Path,
    rel_l2: Dict[str, tf.Tensor],
    checkpoint_path: str,
    grid_size: int,
    reference_path: str | Path,
) -> None:
    """Save field-wise relative L2 errors against the FDM reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["field", "rel_l2", "checkpoint", "fdm_reference", "fdm_grid_size"]
        )
        for field in FIELD_NAMES:
            writer.writerow(
                [
                    field,
                    float(rel_l2[field].numpy()),
                    checkpoint_path,
                    str(reference_path),
                    grid_size,
                ]
            )


def _reference_grid_size(reference: Dict[str, np.ndarray]) -> int:
    """Infer and validate the square FDM grid size."""
    point_count = int(reference["x"].shape[0])
    grid_size = int(round(np.sqrt(point_count)))
    if grid_size * grid_size != point_count:
        raise ValueError(
            "FDM reference coordinates must describe a complete square grid, "
            f"got {point_count} points."
        )
    return grid_size


def _validate_reference_parameters(
    reference: Dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """Prevent comparison against a reference with different plate constants."""
    reference_D = float(np.asarray(reference["D"]).reshape(()))
    reference_nu = float(np.asarray(reference["nu"]).reshape(()))
    if not np.isclose(reference_D, config.plate.D, rtol=1e-12, atol=1e-14):
        raise ValueError(
            f"FDM reference D={reference_D} does not match PINN "
            f"D={config.plate.D}."
        )
    if not np.isclose(reference_nu, config.plate.nu, rtol=1e-12, atol=1e-14):
        raise ValueError(
            f"FDM reference nu={reference_nu} does not match PINN "
            f"nu={config.plate.nu}."
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CONFIG.best_fdm_validation_weights_path),
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_CONFIG.evaluation_csv_path),
    )
    parser.add_argument(
        "--rel-l2-csv",
        default=str(DEFAULT_CONFIG.relative_l2_csv_path),
    )
    parser.add_argument(
        "--from-w-output-csv",
        default=None,
        help="Optional pointwise CSV for fields reconstructed from deflection.",
    )
    parser.add_argument(
        "--from-w-rel-l2-csv",
        default=None,
        help="Optional RelL2 CSV for fields reconstructed from deflection.",
    )
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    figures_dir = DEFAULT_CONFIG.figures_dir if args.save_figures else None
    evaluation = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        rel_l2_csv=args.rel_l2_csv,
        from_w_output_csv=args.from_w_output_csv,
        from_w_rel_l2_csv=args.from_w_rel_l2_csv,
        grid_size=args.grid_size,
        figures_dir=figures_dir,
        batch_size=args.batch_size,
    )
    if evaluation is None:
        return

    direct_score = tf.add_n(
        [evaluation.direct_rel_l2[field] for field in FIELD_NAMES]
    ) / float(len(FIELD_NAMES))
    from_w_score = tf.add_n(
        [evaluation.from_w_rel_l2[field] for field in FIELD_NAMES]
    ) / float(len(FIELD_NAMES))
    parts = [
        "clamped_rock_roof_nonuniform_support_evaluation",
        f"validation_score={float(direct_score.numpy()):.6e}",
        f"from_w_validation_score={float(from_w_score.numpy()):.6e}",
    ]
    parts.extend(
        f"{field}_rel_l2="
        f"{float(evaluation.direct_rel_l2[field].numpy()):.6e}"
        for field in FIELD_NAMES
    )
    parts.extend(
        f"{field}_from_w_rel_l2="
        f"{float(evaluation.from_w_rel_l2[field].numpy()):.6e}"
        for field in FIELD_NAMES
    )
    print(" ".join(parts))


if __name__ == "__main__":
    main()
