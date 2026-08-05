"""Kirchhoff thin plate residuals built on TensorFlow autodiff utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from .autodiff import deflection_derivatives, first_derivatives


@dataclass(frozen=True)
class PlateParameters:
    """Material parameters for an isotropic Kirchhoff thin plate."""

    D: float
    nu: float

    def __post_init__(self) -> None:
        """Validate bending stiffness and Poisson's ratio."""
        if self.D <= 0.0:
            raise ValueError(f"D must be positive, got {self.D}.")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"nu must satisfy -1 < nu < 0.5, got {self.nu}.")


@dataclass(frozen=True)
class PlateResidualState:
    """Fields, derivatives, and residuals produced by one autodiff pass."""

    fields: dict[str, tf.Tensor]
    field_derivatives: dict[str, dict[str, tf.Tensor]]
    deflection_derivatives: dict[str, tf.Tensor]
    residuals: dict[str, tf.Tensor]


def compute_plate_residuals(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute Kirchhoff thin plate bending residuals.

    The signs below follow the current internal-force and transverse-load
    convention used in this project. If the dissertation or reference paper
    defines moments, shear forces, or load directions differently, update the
    convention in this file only so the rest of the training code remains
    unchanged.
    """
    return compute_plate_residual_state(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    ).residuals


def compute_plate_residual_state(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> PlateResidualState:
    """Compute fields, derivatives, and residuals without duplicate autodiff."""
    xy = tf.convert_to_tensor(xy)
    fields, field_derivatives = first_derivatives(model, xy, training=training)
    w_derivatives = deflection_derivatives(model, xy, training=training)
    q = _validate_field_like("load_fn(xy)", load_fn(xy), xy)
    k = None
    if foundation_fn is not None:
        k = _validate_field_like(
            "foundation_fn(xy)",
            foundation_fn(xy),
            xy,
            require_nonnegative=True,
        )

    D = tf.cast(parameters.D, fields["w"].dtype)
    nu = tf.cast(parameters.nu, fields["w"].dtype)

    r_Mx = fields["Mx"] + D * (w_derivatives["w_xx"] + nu * w_derivatives["w_yy"])
    r_My = fields["My"] + D * (w_derivatives["w_yy"] + nu * w_derivatives["w_xx"])
    r_Mxy = fields["Mxy"] + D * (1.0 - nu) * w_derivatives["w_xy"]

    r_Qx = fields["Qx"] - (
        field_derivatives["Mx"]["x"] + field_derivatives["Mxy"]["y"]
    )
    r_Qy = fields["Qy"] - (
        field_derivatives["My"]["y"] + field_derivatives["Mxy"]["x"]
    )

    r_eq = field_derivatives["Qx"]["x"] + field_derivatives["Qy"]["y"] + q
    if k is not None:
        # Winkler foundation reaction follows the convention
        # r_eq = Qx_x + Qy_y + q - k*w. The signs for q and k*w are
        # centralized here because they depend on the load/displacement
        # convention used in the dissertation.
        r_eq = r_eq - k * fields["w"]

    residuals = {
        "moment_x": r_Mx,
        "moment_y": r_My,
        "twisting_moment": r_Mxy,
        "shear_x": r_Qx,
        "shear_y": r_Qy,
        "equilibrium": r_eq,
    }
    _validate_residual_shapes(residuals)
    return PlateResidualState(
        fields=fields,
        field_derivatives=field_derivatives,
        deflection_derivatives=w_derivatives,
        residuals=residuals,
    )


def _validate_field_like(
    name: str,
    value: tf.Tensor,
    xy: tf.Tensor,
    require_nonnegative: bool = False,
) -> tf.Tensor:
    """Validate a scalar field evaluated at the coordinate batch."""
    value = tf.convert_to_tensor(value)

    if not value.dtype.is_floating:
        raise ValueError(f"{name} must be a floating Tensor, got dtype {value.dtype}.")
    if value.dtype != xy.dtype:
        raise ValueError(f"{name} dtype must match xy dtype {xy.dtype}, got {value.dtype}.")

    if value.shape.rank is not None:
        if value.shape.rank != 2:
            raise ValueError(f"{name} must be a 2D tensor, got rank {value.shape.rank}.")
        if value.shape[-1] != 1:
            raise ValueError(f"{name} must have shape (batch_size, 1), got {value.shape}.")
    else:
        tf.debugging.assert_equal(tf.rank(value), 2, message=f"{name} must be 2D.")
        tf.debugging.assert_equal(
            tf.shape(value)[-1],
            1,
            message=f"{name} must have shape (batch_size, 1).",
        )

    tf.debugging.assert_equal(
        tf.shape(value)[0],
        tf.shape(xy)[0],
        message=f"{name} batch size must match xy batch size.",
    )
    tf.debugging.assert_all_finite(value, message=f"{name} must contain only finite values.")
    if require_nonnegative:
        tf.debugging.assert_greater_equal(
            value,
            tf.zeros_like(value),
            message=f"{name} must be non-negative.",
        )
    return value


def _validate_residual_shapes(residuals: dict[str, tf.Tensor]) -> None:
    """Check that every residual has one scalar value per coordinate."""
    for name, value in residuals.items():
        if value.shape.rank is not None:
            if value.shape.rank != 2:
                raise ValueError(f"{name} residual must be 2D, got rank {value.shape.rank}.")
            if value.shape[-1] != 1:
                raise ValueError(f"{name} residual must have shape (batch_size, 1), got {value.shape}.")
        else:
            tf.debugging.assert_equal(tf.rank(value), 2, message=f"{name} residual must be 2D.")
            tf.debugging.assert_equal(
                tf.shape(value)[-1],
                1,
                message=f"{name} residual must have shape (batch_size, 1).",
            )
