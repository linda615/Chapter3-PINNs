"""Physics utilities for Kirchhoff thin plate PINNs."""

from .autodiff import (
    FIELD_NAMES,
    deflection_derivatives,
    deflection_third_derivatives,
    first_derivatives,
)
from .plate_residuals import PlateParameters, compute_plate_residuals
from .multi_output_fourth_order_residuals import (
    MultiOutputFourthOrderState,
    compute_multi_output_fourth_order_residuals,
    compute_multi_output_fourth_order_state,
)
from .w_pinn_fields import (
    compute_w_pinn_derivatives,
    compute_w_pinn_fields,
    fields_from_w_derivatives,
)
from .w_pinn_residuals import (
    compute_w_pinn_physics_loss,
    compute_w_pinn_residuals,
)

__all__ = [
    "FIELD_NAMES",
    "PlateParameters",
    "compute_plate_residuals",
    "deflection_derivatives",
    "deflection_third_derivatives",
    "first_derivatives",
    "compute_w_pinn_derivatives",
    "compute_w_pinn_fields",
    "compute_w_pinn_physics_loss",
    "compute_w_pinn_residuals",
    "fields_from_w_derivatives",
    "MultiOutputFourthOrderState",
    "compute_multi_output_fourth_order_residuals",
    "compute_multi_output_fourth_order_state",
]
