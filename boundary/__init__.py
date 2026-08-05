"""Boundary condition framework for Kirchhoff thin plate PINNs."""

from .conditions import (
    BoundaryCondition,
    ClampedBoundary,
    FreeBoundary,
    SimplySupportedBoundary,
)
from .data import BoundaryBatch, BoundaryConditionBatch, BoundaryResiduals
from .loss import (
    boundary_residual_loss,
    compute_boundary_loss,
    compute_clamped_boundary,
    compute_free_boundary,
    compute_grouped_boundary_loss,
    compute_ss_boundary,
    get_boundary_condition,
)
from .operators import normal_moment, normal_shear, normal_slope
from .scaling import BoundaryLossConfig, BoundaryResidualScales
from .types import BoundaryType

__all__ = [
    "BoundaryBatch",
    "BoundaryConditionBatch",
    "BoundaryCondition",
    "BoundaryResiduals",
    "BoundaryLossConfig",
    "BoundaryResidualScales",
    "BoundaryType",
    "ClampedBoundary",
    "FreeBoundary",
    "SimplySupportedBoundary",
    "boundary_residual_loss",
    "compute_boundary_loss",
    "compute_clamped_boundary",
    "compute_free_boundary",
    "compute_grouped_boundary_loss",
    "compute_ss_boundary",
    "get_boundary_condition",
    "normal_moment",
    "normal_shear",
    "normal_slope",
]
