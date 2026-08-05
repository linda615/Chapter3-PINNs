"""Fourth-order Kirchhoff residuals for a deflection-only W-PINN."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf

from .plate_residuals import PlateParameters
from .w_pinn_fields import compute_w_pinn_derivatives


def compute_w_pinn_residuals(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute ``D*biharmonic(w) + k*w - q`` on interior points."""
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as fourth_tape:
        fourth_tape.watch(xy)
        derivatives = compute_w_pinn_derivatives(
            model=model,
            xy=xy,
            training=training,
        )
        w_xxx = derivatives["w_xxx"]
        w_xxy = derivatives["w_xxy"]
        w_xyy = derivatives["w_xyy"]
        w_yyy = derivatives["w_yyy"]

    grad_w_xxx = _require_gradient(
        fourth_tape.gradient(w_xxx, xy),
        "w_xxx",
    )
    grad_w_xxy = _require_gradient(
        fourth_tape.gradient(w_xxy, xy),
        "w_xxy",
    )
    grad_w_xyy = _require_gradient(
        fourth_tape.gradient(w_xyy, xy),
        "w_xyy",
    )
    grad_w_yyy = _require_gradient(
        fourth_tape.gradient(w_yyy, xy),
        "w_yyy",
    )
    del fourth_tape

    w_xxxx = grad_w_xxx[:, 0:1]
    w_yyyy = grad_w_yyy[:, 1:2]
    w_xxyy = 0.5 * (
        grad_w_xxy[:, 1:2]
        + grad_w_xyy[:, 0:1]
    )
    biharmonic_w = w_xxxx + 2.0 * w_xxyy + w_yyyy

    q = _validate_scalar_field("load_fn(xy)", load_fn(xy), xy)
    D = tf.cast(parameters.D, xy.dtype)
    residual = D * biharmonic_w - q

    if foundation_fn is not None:
        k = _validate_scalar_field(
            "foundation_fn(xy)",
            foundation_fn(xy),
            xy,
            require_nonnegative=True,
        )
        residual = residual + k * derivatives["w"]

    _validate_column("equilibrium", residual)
    return {
        "equilibrium": residual,
        "w_xxxx": w_xxxx,
        "w_xxyy": w_xxyy,
        "w_yyyy": w_yyyy,
        "biharmonic_w": biharmonic_w,
    }


def compute_w_pinn_physics_loss(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Return the scalar fourth-order equilibrium and physics losses."""
    residuals = compute_w_pinn_residuals(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    )
    equilibrium_loss = tf.reduce_mean(tf.square(residuals["equilibrium"]))
    return {
        "equilibrium_loss": equilibrium_loss,
        "physics_loss": equilibrium_loss,
    }


def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate floating coordinate input."""
    xy = tf.convert_to_tensor(xy)
    if not xy.dtype.is_floating:
        raise TypeError(f"xy must be a floating Tensor, got {xy.dtype}.")
    if xy.shape.rank is not None:
        if xy.shape.rank != 2 or xy.shape[-1] != 2:
            raise ValueError(
                f"xy must have shape (batch_size, 2), got {xy.shape}."
            )
        return xy
    tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be 2D.")
    tf.debugging.assert_equal(
        tf.shape(xy)[-1],
        2,
        message="xy must have shape (batch_size, 2).",
    )
    return xy


def _validate_scalar_field(
    name: str,
    value: tf.Tensor,
    xy: tf.Tensor,
    require_nonnegative: bool = False,
) -> tf.Tensor:
    """Validate a scalar field evaluated at every coordinate."""
    value = tf.convert_to_tensor(value)
    if not value.dtype.is_floating:
        raise TypeError(f"{name} must be floating-point, got {value.dtype}.")
    if value.dtype != xy.dtype:
        raise ValueError(
            f"{name} dtype must match xy dtype {xy.dtype}, got {value.dtype}."
        )
    _validate_column(name, value)
    tf.debugging.assert_equal(
        tf.shape(value)[0],
        tf.shape(xy)[0],
        message=f"{name} batch size must match xy batch size.",
    )
    tf.debugging.assert_all_finite(
        value,
        message=f"{name} must contain finite values.",
    )
    if require_nonnegative:
        tf.debugging.assert_greater_equal(
            value,
            tf.zeros_like(value),
            message=f"{name} must be non-negative.",
        )
    return value


def _validate_column(name: str, value: tf.Tensor) -> None:
    """Validate a column tensor."""
    if value.shape.rank is not None:
        if value.shape.rank != 2 or value.shape[-1] != 1:
            raise ValueError(
                f"{name} must have shape (batch_size, 1), got {value.shape}."
            )
        return
    tf.debugging.assert_equal(tf.rank(value), 2, message=f"{name} must be 2D.")
    tf.debugging.assert_equal(
        tf.shape(value)[-1],
        1,
        message=f"{name} must have shape (batch_size, 1).",
    )


def _require_gradient(
    gradient: tf.Tensor | None,
    name: str,
) -> tf.Tensor:
    """Return a derivative or raise a clear disconnected-graph error."""
    if gradient is None:
        raise ValueError(
            f"Gradient for {name!r} is None. Check model dependence on xy."
        )
    return gradient
