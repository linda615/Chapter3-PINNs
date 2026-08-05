"""Generate the finite-difference reference solution for the rock-roof case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from ..config import DEFAULT_CONFIG, ExperimentConfig
from ..problem import foundation_stiffness, rock_roof_load
from .fdm_solver import FDMResult, solve_clamped_plate, two_region_foundation


def solve_reference(
    grid_size: int = 241,
    foundation_mode: Optional[str] = None,
    k1: float = 5.0,
    k2: float = 0.2,
    interface_x: float = 0.45,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> FDMResult:
    """Solve the configured problem with the independent FDM implementation."""
    resolved_foundation_mode = _resolve_foundation_mode(foundation_mode, config)
    load = _make_numpy_load(config)
    foundation = _make_numpy_foundation(
        mode=resolved_foundation_mode,
        config=config,
        k1=k1,
        k2=k2,
        interface_x=interface_x,
    )
    return solve_clamped_plate(
        grid_size=grid_size,
        D=config.plate.D,
        nu=config.plate.nu,
        load=load,
        foundation=foundation,
    )


def save_reference(
    result: FDMResult,
    output_path: str | Path,
    foundation_mode: str,
) -> Path:
    """Save a flattened FDM reference grid in the evaluation-file format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X, Y = np.meshgrid(result.x, result.y, indexing="xy")
    metadata = json.dumps(
        {
            "method": "13-point finite-difference reference",
            "boundary": "clamped: w=0 and dw/dn=0",
            "foundation_mode": foundation_mode,
            "grid_size": result.grid_size,
            "linear_system_relative_residual": result.relative_residual,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        output_path,
        x=X.reshape(-1),
        y=Y.reshape(-1),
        w=result.w.reshape(-1),
        Mx=result.Mx.reshape(-1),
        My=result.My.reshape(-1),
        Mxy=result.Mxy.reshape(-1),
        Qx=result.Qx.reshape(-1),
        Qy=result.Qy.reshape(-1),
        q=result.q.reshape(-1),
        k=result.k.reshape(-1),
        D=np.asarray(result.D, dtype=np.float64),
        nu=np.asarray(result.nu, dtype=np.float64),
        grid_size=np.asarray(result.grid_size, dtype=np.int64),
        linear_system_relative_residual=np.asarray(
            result.relative_residual,
            dtype=np.float64,
        ),
        metadata=np.asarray(metadata),
    )
    return output_path


def generate_reference(
    grid_size: int = 241,
    output_path: str | Path = DEFAULT_CONFIG.reference_path,
    foundation_mode: Optional[str] = None,
    k1: float = 5.0,
    k2: float = 0.2,
    interface_x: float = 0.45,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> FDMResult:
    """Solve and save the finite-difference reference solution."""
    resolved_foundation_mode = _resolve_foundation_mode(foundation_mode, config)
    result = solve_reference(
        grid_size=grid_size,
        foundation_mode=resolved_foundation_mode,
        k1=k1,
        k2=k2,
        interface_x=interface_x,
        config=config,
    )
    save_reference(
        result,
        output_path,
        foundation_mode=resolved_foundation_mode,
    )
    return result


def _resolve_foundation_mode(
    foundation_mode: Optional[str],
    config: ExperimentConfig,
) -> str:
    """Choose a reference foundation mode consistent with the active problem."""
    if foundation_mode is not None:
        return foundation_mode
    return "smooth" if config.foundation_enabled else "none"


def _make_numpy_load(
    config: ExperimentConfig,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Evaluate the project's canonical load function on an FDM grid."""

    def load(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return _evaluate_tensorflow_field(
            rock_roof_load,
            X,
            Y,
            config,
        )

    return load


def _make_numpy_foundation(
    mode: str,
    config: ExperimentConfig,
    k1: float,
    k2: float,
    interface_x: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create smooth, two-region, or zero foundation data."""
    normalized_mode = mode.strip().lower().replace("_", "-")
    if normalized_mode == "smooth":

        def smooth_foundation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return _evaluate_tensorflow_field(
                foundation_stiffness,
                X,
                Y,
                config,
            )

        return smooth_foundation

    if normalized_mode in ("two-region", "piecewise"):

        def piecewise_foundation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return two_region_foundation(
                x=X[0, :],
                y=Y[:, 0],
                k1=k1,
                k2=k2,
                interface_x=interface_x,
            )

        return piecewise_foundation

    if normalized_mode in ("none", "zero"):

        def zero_foundation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            del Y
            return np.zeros_like(X)

        return zero_foundation

    raise ValueError(
        "foundation_mode must be 'smooth', 'two-region', or 'none', "
        f"got {mode!r}."
    )


def _evaluate_tensorflow_field(
    field_fn: Callable[[tf.Tensor, ExperimentConfig], tf.Tensor],
    X: np.ndarray,
    Y: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Bridge an existing TensorFlow problem field to the NumPy FDM grid."""
    xy = tf.convert_to_tensor(
        np.column_stack((X.reshape(-1), Y.reshape(-1))),
        dtype=tf.float64,
    )
    values = field_fn(xy, config=config)
    return np.asarray(values.numpy(), dtype=np.float64).reshape(X.shape)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=241)
    parser.add_argument("--output", default=str(DEFAULT_CONFIG.reference_path))
    parser.add_argument(
        "--foundation-mode",
        choices=("smooth", "two-region", "none"),
        default=None,
        help="Defaults to 'smooth' when enabled in config, otherwise 'none'.",
    )
    parser.add_argument("--k1", type=float, default=5.0)
    parser.add_argument("--k2", type=float, default=0.2)
    parser.add_argument("--interface-x", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    """Generate and report the FDM reference solution."""
    args = _parse_args()
    foundation_mode = _resolve_foundation_mode(
        args.foundation_mode,
        DEFAULT_CONFIG,
    )
    result = generate_reference(
        grid_size=args.grid_size,
        output_path=args.output,
        foundation_mode=foundation_mode,
        k1=args.k1,
        k2=args.k2,
        interface_x=args.interface_x,
    )
    print(
        "fdm_reference_generated "
        f"grid_size={result.grid_size} "
        f"foundation_mode={foundation_mode} "
        f"unknowns={(result.grid_size - 2) ** 2} "
        f"relative_residual={result.relative_residual:.6e} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
