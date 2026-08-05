"""MO4-PINN configuration for the mixed-edge plate experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, ExperimentConfig


# Only the equilibrium closure changes relative to Mixed PINN. Architecture,
# sampling, schedules, output scales, and validation settings remain identical.
MO4_PINN_CONFIG: ExperimentConfig = replace(
    DEFAULT_CONFIG,
    results_dir=Path(
        "experiments/simply_clamped_uniform_load/results_mo4_pinn"
    ),
)


__all__ = ["MO4_PINN_CONFIG"]
