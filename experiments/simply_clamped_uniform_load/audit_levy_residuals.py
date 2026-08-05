"""Audit Levy-series fields against the residuals used during training."""

from __future__ import annotations

import argparse

import tensorflow as tf

from physics import compute_plate_residuals

from .analytical import AnalyticalPlateModel, levy_series_load
from .config import DEFAULT_CONFIG
from .problem import create_interior_sampler, create_load_fn


def run_audit(
    point_count: int = 2048,
    mode_count: int = DEFAULT_CONFIG.analytical_modes,
    seed: int = 2051,
) -> None:
    """Print residual statistics for the training and truncated loads."""
    if point_count <= 0:
        raise ValueError("point_count must be positive.")
    if mode_count <= 0:
        raise ValueError("mode_count must be positive.")

    config = DEFAULT_CONFIG
    xy = create_interior_sampler(dtype=tf.float32).sample(
        point_count,
        seed=seed,
    )
    model = AnalyticalPlateModel(
        parameters=config.plate,
        q0=config.q0,
        mode_count=mode_count,
    )
    training_load_fn = create_load_fn(config)

    def truncated_load_fn(points: tf.Tensor) -> tf.Tensor:
        return levy_series_load(
            points,
            q0=config.q0,
            mode_count=mode_count,
        )

    training_residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=training_load_fn,
        parameters=config.plate,
    )
    truncated_residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=truncated_load_fn,
        parameters=config.plate,
    )

    print(
        "levy_residual_audit "
        f"points={point_count} modes={mode_count} seed={seed}"
    )
    _print_residuals("training_constant_q", training_residuals)
    _print_residuals("matched_truncated_qN", truncated_residuals)

    load_mismatch = training_load_fn(xy) - truncated_load_fn(xy)
    equilibrium_difference = (
        training_residuals["equilibrium"]
        - truncated_residuals["equilibrium"]
    )
    identity_error = equilibrium_difference - load_mismatch
    print(
        "equilibrium_identity "
        "r_eq_constant-r_eq_truncated=q-qN "
        f"rms_error={_rms(identity_error):.9e} "
        f"max_abs_error={_max_abs(identity_error):.9e}"
    )
    print(
        "load_truncation "
        f"rms={_rms(load_mismatch):.9e} "
        f"mean_abs={_mean_abs(load_mismatch):.9e} "
        f"max_abs={_max_abs(load_mismatch):.9e}"
    )


def _print_residuals(
    load_name: str,
    residuals: dict[str, tf.Tensor],
) -> None:
    """Print RMS, mean absolute, and maximum absolute residuals."""
    print(f"load={load_name}")
    for name, residual in residuals.items():
        print(
            f"  {name} "
            f"rms={_rms(residual):.9e} "
            f"mean_abs={_mean_abs(residual):.9e} "
            f"max_abs={_max_abs(residual):.9e}"
        )


def _rms(value: tf.Tensor) -> float:
    """Return the eager scalar RMS used by this command-line audit."""
    return float(tf.sqrt(tf.reduce_mean(tf.square(value))))


def _mean_abs(value: tf.Tensor) -> float:
    """Return the eager scalar mean absolute value."""
    return float(tf.reduce_mean(tf.abs(value)))


def _max_abs(value: tf.Tensor) -> float:
    """Return the eager scalar maximum absolute value."""
    return float(tf.reduce_max(tf.abs(value)))


def _parse_args() -> argparse.Namespace:
    """Parse command-line audit options."""
    parser = argparse.ArgumentParser(
        description=(
            "Substitute the truncated Levy fields into the plate residuals "
            "with both the training load and the matching truncated load."
        )
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument(
        "--modes",
        type=int,
        default=DEFAULT_CONFIG.analytical_modes,
    )
    parser.add_argument("--seed", type=int, default=2051)
    return parser.parse_args()


def main() -> None:
    """Run the command-line audit."""
    args = _parse_args()
    run_audit(
        point_count=args.points,
        mode_count=args.modes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
