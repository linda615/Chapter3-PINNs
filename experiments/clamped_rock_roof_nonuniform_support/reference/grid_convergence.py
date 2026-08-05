"""Grid-convergence analysis for the finite-difference reference solution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import DEFAULT_CONFIG, ExperimentConfig
from .fdm_solver import FDMResult, ScalarField, solve_clamped_plate
from .generate_reference import solve_reference


FIELD_NAMES = ("w", "Mx", "My", "Mxy", "Qx", "Qy")


def run_grid_convergence(
    grid_sizes: Sequence[int] = (61, 121, 241),
    foundation_mode: Optional[str] = None,
    k1: float = 5.0,
    k2: float = 0.2,
    interface_x: float = 0.45,
    output_csv: Union[str, Path, None] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> List[Dict[str, float]]:
    """Run nested-grid self-convergence for the engineering problem."""
    _validate_nested_grids(grid_sizes)
    solutions = [
        solve_reference(
            grid_size=grid_size,
            foundation_mode=foundation_mode,
            k1=k1,
            k2=k2,
            interface_x=interface_x,
            config=config,
        )
        for grid_size in grid_sizes
    ]

    rows: List[Dict[str, float]] = []
    for index, result in enumerate(solutions):
        row: Dict[str, float] = {
            "grid_size": float(result.grid_size),
            "spacing": float(result.x[1] - result.x[0]),
            "linear_system_relative_residual": result.relative_residual,
        }
        if index < len(solutions) - 1:
            finer = solutions[index + 1]
            ratio = (finer.grid_size - 1) // (result.grid_size - 1)
            for field in FIELD_NAMES:
                coarse_values = result.fields[field]
                restricted_fine = finer.fields[field][::ratio, ::ratio]
                row[f"{field}_successive_rel_l2"] = _relative_l2(
                    coarse_values,
                    restricted_fine,
                )
        rows.append(row)

    if len(rows) >= 3:
        for index in range(1, len(rows) - 1):
            h_ratio = rows[index - 1]["spacing"] / rows[index]["spacing"]
            for field in FIELD_NAMES:
                coarse_error = rows[index - 1][f"{field}_successive_rel_l2"]
                fine_error = rows[index][f"{field}_successive_rel_l2"]
                rows[index][f"{field}_observed_order"] = float(
                    np.log(coarse_error / fine_error) / np.log(h_ratio)
                )

    if output_csv is not None:
        _save_rows(output_csv, rows)
    return rows


def run_manufactured_convergence(
    grid_sizes: Sequence[int] = (17, 33, 65),
    D: float = 1.0,
    nu: float = 0.25,
    foundation: ScalarField = 1.0,
) -> List[Dict[str, float]]:
    """Verify convergence with a clamped polynomial manufactured solution."""
    if len(grid_sizes) < 2:
        raise ValueError("At least two grid sizes are required.")
    rows: List[Dict[str, float]] = []

    for grid_size in grid_sizes:
        def load(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            exact_w = _manufactured_deflection(X, Y)
            k_values = _evaluate_foundation_for_manufactured(foundation, X, Y)
            return D * _manufactured_biharmonic(X, Y) + k_values * exact_w

        result = solve_clamped_plate(
            grid_size=grid_size,
            D=D,
            nu=nu,
            load=load,
            foundation=foundation,
        )
        X, Y = np.meshgrid(result.x, result.y, indexing="xy")
        exact_w = _manufactured_deflection(X, Y)
        error = _relative_l2(result.w, exact_w)
        row: Dict[str, float] = {
            "grid_size": float(grid_size),
            "spacing": float(result.x[1] - result.x[0]),
            "w_relative_l2": error,
            "linear_system_relative_residual": result.relative_residual,
        }
        if rows:
            previous = rows[-1]
            row["observed_order"] = float(
                np.log(previous["w_relative_l2"] / error)
                / np.log(previous["spacing"] / row["spacing"])
            )
        rows.append(row)
    return rows


def _manufactured_deflection(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return w=x^2(1-x)^2 y^2(1-y)^2."""
    return _quartic_factor(X) * _quartic_factor(Y)


def _manufactured_biharmonic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the exact biharmonic of the clamped polynomial."""
    factor_x = _quartic_factor(X)
    factor_y = _quartic_factor(Y)
    second_x = 2.0 - 12.0 * X + 12.0 * X**2
    second_y = 2.0 - 12.0 * Y + 12.0 * Y**2
    return 24.0 * factor_y + 2.0 * second_x * second_y + 24.0 * factor_x


def _quartic_factor(values: np.ndarray) -> np.ndarray:
    return values**2 * (1.0 - values) ** 2


def _evaluate_foundation_for_manufactured(
    foundation: ScalarField,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    values = foundation(X, Y) if callable(foundation) else foundation
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        return np.full_like(X, float(values))
    return np.broadcast_to(values, X.shape)


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(reference.reshape(-1))),
        np.finfo(np.float64).eps,
    )
    return float(np.linalg.norm((values - reference).reshape(-1)) / denominator)


def _validate_nested_grids(grid_sizes: Sequence[int]) -> None:
    if len(grid_sizes) < 2:
        raise ValueError("At least two grid sizes are required.")
    for coarse, fine in zip(grid_sizes[:-1], grid_sizes[1:]):
        if coarse < 5 or fine <= coarse:
            raise ValueError(f"Grid sizes must be increasing and >= 5, got {grid_sizes}.")
        if (fine - 1) % (coarse - 1) != 0:
            raise ValueError(
                "Grid sizes must be nested so fine-grid nodes contain all "
                f"coarse-grid nodes, got {coarse} and {fine}."
            )


def _save_rows(path: Union[str, Path], rows: List[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[61, 121, 241])
    parser.add_argument(
        "--foundation-mode",
        choices=("smooth", "two-region", "none"),
        default=None,
        help="Defaults to 'smooth' when enabled in config, otherwise 'none'.",
    )
    parser.add_argument("--k1", type=float, default=5.0)
    parser.add_argument("--k2", type=float, default=0.2)
    parser.add_argument("--interface-x", type=float, default=0.45)
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_CONFIG.history_dir / "fdm_grid_convergence.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested 61/121/241 FDM grid-convergence study."""
    args = _parse_args()
    rows = run_grid_convergence(
        grid_sizes=args.grid_sizes,
        foundation_mode=args.foundation_mode,
        k1=args.k1,
        k2=args.k2,
        interface_x=args.interface_x,
        output_csv=args.output_csv,
    )
    for row in rows:
        summary = " ".join(f"{key}={value:.6e}" for key, value in row.items())
        print(f"fdm_grid_convergence {summary}")
    print(f"fdm_grid_convergence_csv={args.output_csv}")


if __name__ == "__main__":
    main()
