"""Model construction for the mixed-edge W-PINN baseline."""

from __future__ import annotations

from models import WPINN

from .config import ExperimentConfig
from .problem import _deflection_transform_name
from .w_pinn_config import W_PINN_CONFIG


def create_w_pinn_model(
    config: ExperimentConfig = W_PINN_CONFIG,
) -> WPINN:
    """Create W-PINN with the same hard w transform and scale as Mixed PINN."""
    return WPINN(
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
        activation=config.activation,
        apply_deflection_transform=True,
        deflection_transform_name=_deflection_transform_name(config),
        output_activation=config.output_activation,
        output_scale=config.field_output_scales.w,
    )


__all__ = ["create_w_pinn_model"]
