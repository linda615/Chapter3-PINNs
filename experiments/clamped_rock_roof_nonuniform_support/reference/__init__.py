"""Finite-difference reference tools for the nonuniform-support rock roof."""

from .fdm_solver import (
    FDMResult,
    assemble_clamped_plate_system,
    recover_plate_fields,
    solve_clamped_plate,
    two_region_foundation,
)

__all__ = [
    "FDMResult",
    "assemble_clamped_plate_system",
    "recover_plate_fields",
    "solve_clamped_plate",
    "two_region_foundation",
]
