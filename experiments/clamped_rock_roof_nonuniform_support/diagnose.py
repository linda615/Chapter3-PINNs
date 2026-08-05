"""Diagnose load and foundation fields for the rock-roof example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from .config import DEFAULT_CONFIG, ExperimentConfig
from .problem import active_foundation_stiffness, rock_roof_load
from .run import make_grid, tensor_stats


def run_diagnostics(
    config: ExperimentConfig = DEFAULT_CONFIG,
    grid_size: int = 101,
) -> Dict[str, tf.Tensor]:
    """Print statistics and save diagnostic contour figures."""
    xy = make_grid(grid_size)
    q = rock_roof_load(xy, config=config)
    k = active_foundation_stiffness(xy, config=config)
    stats = {}
    stats.update(tensor_stats(q, prefix="q"))
    stats.update(tensor_stats(k, prefix="k"))

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    _save_contour(
        path=config.figures_dir / "load_contour.png",
        xy=xy,
        value=q,
        grid_size=grid_size,
        title="Nonuniform Load q(x,y)",
        config=config,
    )
    _save_contour(
        path=config.figures_dir / "foundation_stiffness_contour.png",
        xy=xy,
        value=k,
        grid_size=grid_size,
        title=(
            "Winkler Foundation Stiffness k(x,y)"
            if config.foundation_enabled
            else "Foundation Disabled: k(x,y) = 0"
        ),
        config=config,
    )

    print(
        "rock_roof_diagnostics "
        + " ".join(f"{name}={float(value.numpy()):.6e}" for name, value in stats.items())
        + f" load_contour={config.figures_dir / 'load_contour.png'}"
        + f" foundation_contour={config.figures_dir / 'foundation_stiffness_contour.png'}"
    )
    return stats


def _save_contour(
    path: Path,
    xy: tf.Tensor,
    value: tf.Tensor,
    grid_size: int,
    title: str,
    config: ExperimentConfig,
) -> None:
    """Save one annotated contour plot."""
    x = xy[:, 0].numpy().reshape((grid_size, grid_size))
    y = xy[:, 1].numpy().reshape((grid_size, grid_size))
    z = value[:, 0].numpy().reshape((grid_size, grid_size))

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
    contour = ax.contourf(x, y, z, levels=32, cmap="viridis")
    fig.colorbar(contour, ax=ax, shrink=0.86)
    ax.plot(config.load_center_x, config.load_center_y, "ro", markersize=4)
    ax.text(config.load_center_x + 0.02, config.load_center_y, "local load center", color="red")
    if config.foundation_enabled:
        ax.axvline(
            config.foundation_transition_x,
            color="white",
            linestyle="--",
            linewidth=1.4,
        )
        ax.text(0.08, 0.92, "strong support", color="white", weight="bold")
        ax.text(0.62, 0.92, "weak support", color="white", weight="bold")
    else:
        ax.text(0.04, 0.92, "no foundation", color="white", weight="bold")
    ax.text(0.03, 0.03, "four clamped edges", color="white")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    run_diagnostics(grid_size=args.grid_size)


if __name__ == "__main__":
    main()
