"""Loss functions for Kirchhoff thin plate PINNs."""

from boundary import (
    BoundaryType,
    compute_boundary_loss,
    compute_clamped_boundary,
    compute_free_boundary,
    compute_ss_boundary,
)
from .consistency import MxyConsistencyConfig, compute_mxy_consistency_loss
from .normalization import (
    PhysicsComponentWeights,
    PhysicsLossConfig,
    ResidualScales,
    unit_square_sinusoidal_residual_scales,
)
from .physics_loss import compute_physics_loss
from .mo4_loss import compute_mo4_physics_loss, compute_mo4_total_loss
from .total import compute_grouped_boundary_total_loss, compute_total_loss
from .weights import (
    LossWeightSchedule,
    LossWeightSource,
    LossWeightStage,
    LossWeights,
    resolve_loss_weights,
)
from .w_pinn_loss import (
    compute_w_pinn_boundary_loss,
    compute_w_pinn_total_loss,
)

__all__ = [
    "BoundaryType",
    "LossWeights",
    "LossWeightSchedule",
    "LossWeightSource",
    "LossWeightStage",
    "MxyConsistencyConfig",
    "PhysicsComponentWeights",
    "PhysicsLossConfig",
    "ResidualScales",
    "compute_boundary_loss",
    "compute_clamped_boundary",
    "compute_free_boundary",
    "compute_mxy_consistency_loss",
    "compute_physics_loss",
    "compute_ss_boundary",
    "compute_total_loss",
    "compute_grouped_boundary_total_loss",
    "compute_mo4_physics_loss",
    "compute_mo4_total_loss",
    "unit_square_sinusoidal_residual_scales",
    "resolve_loss_weights",
    "compute_w_pinn_boundary_loss",
    "compute_w_pinn_total_loss",
]
