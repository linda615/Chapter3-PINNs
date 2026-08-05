"""Reconstruct plate resultants from PINN deflection and compare with FDM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from experiments.simply_supported_sinusoidal.evaluate import load_best_checkpoint
from physics.autodiff import deflection_third_derivatives
from physics.plate_residuals import PlateParameters

from .config import DEFAULT_CONFIG, ExperimentConfig
from .evaluate import (
    _reference_grid_size,
    _validate_reference_parameters,
    load_reference,
)
from .problem import active_foundation_stiffness, create_model, rock_roof_load


AUXILIARY_FIELD_NAMES = ("Mx", "My", "Mxy", "Qx", "Qy")


def compute_auxiliary_fields_from_w(
    model: tf.keras.Model,
    xy: tf.Tensor,
    parameters: PlateParameters,
    training: bool = False,
) -> Dict[str, tf.Tensor]:
    """Reconstruct moments and shear forces from the predicted deflection.

    The signs are identical to ``physics.plate_residuals``:

    ``Mx = -D*(w_xx + nu*w_yy)``
    ``My = -D*(w_yy + nu*w_xx)``
    ``Mxy = -D*(1-nu)*w_xy``
    ``Qx = Mx_x + Mxy_y = -D*(w_xxx + w_xyy)``
    ``Qy = My_y + Mxy_x = -D*(w_yyy + w_xxy)``
    """
    derivatives = deflection_third_derivatives(
        model=model,
        xy=xy,
        training=training,
    )
    dtype = derivatives["w"].dtype
    D = tf.cast(parameters.D, dtype)
    nu = tf.cast(parameters.nu, dtype)

    return {
        "w": derivatives["w"],
        "Mx_from_w": -D
        * (derivatives["w_xx"] + nu * derivatives["w_yy"]),
        "My_from_w": -D
        * (derivatives["w_yy"] + nu * derivatives["w_xx"]),
        "Mxy_from_w": -D * (1.0 - nu) * derivatives["w_xy"],
        "Qx_from_w": -D
        * (derivatives["w_xxx"] + derivatives["w_xyy"]),
        "Qy_from_w": -D
        * (derivatives["w_yyy"] + derivatives["w_xxy"]),
    }


def run_auxiliary_evaluation(
    checkpoint_path: str | Path = DEFAULT_CONFIG.best_fdm_validation_weights_path,
    output_csv: str | Path = DEFAULT_CONFIG.auxiliary_fields_csv_path,
    metrics_csv: str | Path = DEFAULT_CONFIG.auxiliary_metrics_csv_path,
    batch_size: int = 2048,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[str, float]]:
    """Evaluate direct and deflection-reconstructed resultants on the FDM grid."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    reference = load_reference(config.reference_path)
    if reference is None:
        raise FileNotFoundError(
            f"FDM reference file does not exist: {config.reference_path}"
        )
    _validate_reference_parameters(reference, config)
    _validate_reference_problem(reference, config)

    model = create_model(config)
    resolved_checkpoint = load_best_checkpoint(model, checkpoint_path)
    xy = tf.concat(
        (
            tf.convert_to_tensor(reference["x"], dtype=tf.float32),
            tf.convert_to_tensor(reference["y"], dtype=tf.float32),
        ),
        axis=1,
    )

    direct_chunks = {field: [] for field in ("w",) + AUXILIARY_FIELD_NAMES}
    reconstructed_chunks = {
        f"{field}_from_w": [] for field in AUXILIARY_FIELD_NAMES
    }
    point_count = int(xy.shape[0])
    for start in range(0, point_count, batch_size):
        stop = min(start + batch_size, point_count)
        xy_batch = xy[start:stop]
        direct = model(xy_batch, training=False)
        reconstructed = compute_auxiliary_fields_from_w(
            model=model,
            xy=xy_batch,
            parameters=config.plate,
            training=False,
        )
        for field in direct_chunks:
            direct_chunks[field].append(
                np.asarray(direct[field].numpy(), dtype=np.float64)
            )
        for field in reconstructed_chunks:
            reconstructed_chunks[field].append(
                np.asarray(reconstructed[field].numpy(), dtype=np.float64)
            )

    direct_fields = {
        field: np.concatenate(chunks, axis=0)
        for field, chunks in direct_chunks.items()
    }
    reconstructed_fields = {
        field: np.concatenate(chunks, axis=0)
        for field, chunks in reconstructed_chunks.items()
    }
    metrics = _compute_metrics(
        reference=reference,
        direct_fields=direct_fields,
        reconstructed_fields=reconstructed_fields,
    )
    _save_pointwise_csv(
        path=output_csv,
        reference=reference,
        direct_fields=direct_fields,
        reconstructed_fields=reconstructed_fields,
    )
    _save_metrics_csv(
        path=metrics_csv,
        metrics=metrics,
        checkpoint_path=resolved_checkpoint,
        reference_path=config.reference_path,
        grid_size=_reference_grid_size(reference),
    )

    for field in AUXILIARY_FIELD_NAMES:
        values = metrics[field]
        print(
            "auxiliary_field_comparison "
            f"field={field} "
            f"direct_rel_l2={values['direct_rel_l2']:.6e} "
            f"from_w_rel_l2={values['from_w_rel_l2']:.6e} "
            f"direct_vs_from_w_rel_l2="
            f"{values['direct_vs_from_w_rel_l2']:.6e} "
            f"direct_rmse={values['direct_rmse']:.6e} "
            f"from_w_rmse={values['from_w_rmse']:.6e}"
        )
    print(
        "auxiliary_field_outputs "
        f"checkpoint={resolved_checkpoint} "
        f"pointwise_csv={output_csv} "
        f"metrics_csv={metrics_csv}"
    )
    return metrics


def _validate_reference_problem(
    reference: Dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """Check that reference load and foundation match the active problem."""
    xy = tf.concat(
        (
            tf.convert_to_tensor(reference["x"], dtype=tf.float64),
            tf.convert_to_tensor(reference["y"], dtype=tf.float64),
        ),
        axis=1,
    )
    expected_q = rock_roof_load(xy, config=config).numpy()
    expected_k = active_foundation_stiffness(xy, config=config).numpy()
    if not np.allclose(reference["q"], expected_q, rtol=1e-7, atol=1e-10):
        raise ValueError("FDM reference load does not match the active problem.")
    if not np.allclose(reference["k"], expected_k, rtol=1e-7, atol=1e-10):
        raise ValueError(
            "FDM reference foundation does not match the active problem."
        )


def _compute_metrics(
    reference: Dict[str, np.ndarray],
    direct_fields: Dict[str, np.ndarray],
    reconstructed_fields: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute field scales, RMSE, FDM RelL2, and mixed-field consistency."""
    metrics: Dict[str, Dict[str, float]] = {}
    for field in AUXILIARY_FIELD_NAMES:
        exact = reference[field]
        direct = direct_fields[field]
        from_w = reconstructed_fields[f"{field}_from_w"]
        metrics[field] = {
            "fdm_max_abs": float(np.max(np.abs(exact))),
            "direct_max_abs": float(np.max(np.abs(direct))),
            "from_w_max_abs": float(np.max(np.abs(from_w))),
            "direct_rmse": _rmse(direct, exact),
            "from_w_rmse": _rmse(from_w, exact),
            "direct_rel_l2": _relative_l2(direct, exact),
            "from_w_rel_l2": _relative_l2(from_w, exact),
            "direct_vs_from_w_rel_l2": _relative_l2(from_w, direct),
        }
    return metrics


def _save_pointwise_csv(
    path: str | Path,
    reference: Dict[str, np.ndarray],
    direct_fields: Dict[str, np.ndarray],
    reconstructed_fields: Dict[str, np.ndarray],
) -> None:
    """Save pointwise FDM, direct PINN, and reconstructed resultants."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["x", "y", "w_fdm", "w_pred"]
    for field in AUXILIARY_FIELD_NAMES:
        header.extend((f"{field}_fdm", f"{field}_direct", f"{field}_from_w"))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(reference["x"].shape[0]):
            row = [
                reference["x"][index, 0],
                reference["y"][index, 0],
                reference["w"][index, 0],
                direct_fields["w"][index, 0],
            ]
            for field in AUXILIARY_FIELD_NAMES:
                row.extend(
                    (
                        reference[field][index, 0],
                        direct_fields[field][index, 0],
                        reconstructed_fields[f"{field}_from_w"][index, 0],
                    )
                )
            writer.writerow(row)


def _save_metrics_csv(
    path: str | Path,
    metrics: Dict[str, Dict[str, float]],
    checkpoint_path: str,
    reference_path: str | Path,
    grid_size: int,
) -> None:
    """Save one summary row per reconstructed physical field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(next(iter(metrics.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("field",) + metric_names + ("checkpoint", "fdm_reference", "grid_size")
        )
        for field in AUXILIARY_FIELD_NAMES:
            writer.writerow(
                (field,)
                + tuple(metrics[field][name] for name in metric_names)
                + (checkpoint_path, str(reference_path), grid_size)
            )


def _rmse(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Return root-mean-square error."""
    return float(np.sqrt(np.mean(np.square(predicted - exact))))


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Return relative L2 error with a clear zero-norm check."""
    denominator = float(np.linalg.norm(exact))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Relative L2 denominator must be nonzero.")
    return float(np.linalg.norm(predicted - exact) / denominator)


def _parse_args() -> argparse.Namespace:
    """Parse auxiliary-field evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CONFIG.best_fdm_validation_weights_path),
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_CONFIG.auxiliary_fields_csv_path),
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_CONFIG.auxiliary_metrics_csv_path),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    """Run the auxiliary-field comparison command."""
    args = _parse_args()
    run_auxiliary_evaluation(
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        metrics_csv=args.metrics_csv,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
