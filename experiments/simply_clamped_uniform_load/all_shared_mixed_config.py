"""Strict all-shared low-order Mixed-PINN configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, ExperimentConfig


ALL_SHARED_MIXED_CONFIG: ExperimentConfig = replace(
    DEFAULT_CONFIG,
    model_grouping="one",
    hidden_width=64,
    hidden_depth=5,
    use_shared_trunk=False,
    shared_width=0,
    shared_depth=0,
    results_dir=Path(
        "experiments/simply_clamped_uniform_load/results_mixed_all_shared"
    ),
)


__all__ = ["ALL_SHARED_MIXED_CONFIG"]
