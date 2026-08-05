"""W-PINN baseline configuration for the mixed-edge plate experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from losses import LossWeights
from trainer import SharedTrunkPCGradConfig

from .config import DEFAULT_CONFIG, ExperimentConfig


W_PINN_CONFIG: ExperimentConfig = replace(
    DEFAULT_CONFIG,
    model_grouping="one",
    hidden_width=64,
    hidden_depth=5,
    use_shared_trunk=False,
    shared_width=0,
    shared_depth=0,
    gradient_diagnostics_enabled=False,
    shared_pcgrad=SharedTrunkPCGradConfig(enabled=False),
    results_dir=Path(
        "experiments/simply_clamped_uniform_load/results_w_pinn"
    ),
    loss_weights=LossWeights(
        moment=0.0,
        shear=0.0,
        equilibrium=1.0,
        boundary=1.0,
        mxy_consistency=0.0,
    ),
)


__all__ = ["W_PINN_CONFIG"]
