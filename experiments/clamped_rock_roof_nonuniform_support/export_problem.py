"""Export rock-roof problem data for independent reference solvers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import active_foundation_stiffness, rock_roof_load
from .run import make_grid


def export_problem_data(
    config: ExperimentConfig = DEFAULT_CONFIG,
    grid_size: int = 101,
    output_csv: str | Path | None = None,
) -> Path:
    """Export ``x, y, q, k, D, nu`` and boundary type on a regular grid."""
    if output_csv is None:
        output_csv = config.exports_dir / f"rock_roof_problem_grid_{grid_size}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    xy = make_grid(grid_size)
    q = rock_roof_load(xy, config=config)
    k = active_foundation_stiffness(xy, config=config)
    xy_values = xy.numpy()
    q_values = q.numpy()
    k_values = k.numpy()

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "q", "k", "D", "nu", "boundary_type"])
        for index in range(xy.shape[0]):
            writer.writerow(
                [
                    xy_values[index, 0],
                    xy_values[index, 1],
                    q_values[index, 0],
                    k_values[index, 0],
                    config.plate.D,
                    config.plate.nu,
                    config.boundary_type.value,
                ]
            )
    print(f"rock_roof_problem_export path={output_csv}")
    return output_csv


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    export_problem_data(grid_size=args.grid_size, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
