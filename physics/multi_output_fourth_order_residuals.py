"""Fourth-order equilibrium residuals for a six-output Kirchhoff PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from models import FIELD_NAMES

from .plate_residuals import PlateParameters


@dataclass(frozen=True)
class MultiOutputFourthOrderState:
    """Fields, deflection derivatives, and MO4 residual tensors."""

    fields: dict[str, tf.Tensor]
    deflection_derivatives: dict[str, tf.Tensor]
    residuals: dict[str, tf.Tensor]


def compute_multi_output_fourth_order_residuals(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Return constitutive, shear, and fourth-order equilibrium residuals."""
    return compute_multi_output_fourth_order_state(
        model=model,
        xy=xy,
        load_fn=load_fn,
        parameters=parameters,
        foundation_fn=foundation_fn,
        training=training,
    ).residuals


def compute_multi_output_fourth_order_state(
    model: tf.keras.Model,
    xy: tf.Tensor,
    load_fn: Callable[[tf.Tensor], tf.Tensor],
    parameters: PlateParameters,
    foundation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    training: bool = False,
) -> MultiOutputFourthOrderState:
    """Compute all MO4 quantities from one model evaluation.

    MO4 retains the mixed model's moment and shear consistency equations, but
    closes equilibrium with ``D*biharmonic(w) + k*w - q = 0`` instead of
    ``div(Q) + q - k*w = 0``.
    """
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as fourth_tape:
        fourth_tape.watch(xy)
        with tf.GradientTape(persistent=True) as third_tape:
            third_tape.watch(xy)
            with tf.GradientTape(persistent=True) as second_tape:
                second_tape.watch(xy)
                with tf.GradientTape(persistent=True) as first_tape:
                    first_tape.watch(xy)
                    fields = _validate_fields(
                        model(xy, training=training),
                        xy,
                    )
                    w = fields["w"]

                grad_w = _require_gradient(
                    first_tape.gradient(w, xy),
                    "w",
                )
                grad_mx = _require_gradient(
                    first_tape.gradient(fields["Mx"], xy),
                    "Mx",
                )
                grad_my = _require_gradient(
                    first_tape.gradient(fields["My"], xy),
                    "My",
                )
                grad_mxy = _require_gradient(
                    first_tape.gradient(fields["Mxy"], xy),
                    "Mxy",
                )
                w_x = grad_w[:, 0:1]
                w_y = grad_w[:, 1:2]

            grad_w_x = _require_gradient(
                second_tape.gradient(w_x, xy),
                "w_x",
            )
            grad_w_y = _require_gradient(
                second_tape.gradient(w_y, xy),
                "w_y",
            )
            w_xx = grad_w_x[:, 0:1]
            w_yy = grad_w_y[:, 1:2]
            w_xy = 0.5 * (
                grad_w_x[:, 1:2]
                + grad_w_y[:, 0:1]
            )

        grad_w_xx = _require_gradient(
            third_tape.gradient(w_xx, xy),
            "w_xx",
        )
        grad_w_yy = _require_gradient(
            third_tape.gradient(w_yy, xy),
            "w_yy",
        )
        w_xxx = grad_w_xx[:, 0:1]
        w_xxy = grad_w_xx[:, 1:2]
        w_xyy = grad_w_yy[:, 0:1]
        w_yyy = grad_w_yy[:, 1:2]

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

    del first_tape
    del second_tape
    del third_tape
    del fourth_tape

    w_xxxx = grad_w_xxx[:, 0:1]
    w_xxyy = 0.5 * (
        grad_w_xxy[:, 1:2]
        + grad_w_xyy[:, 0:1]
    )
    w_yyyy = grad_w_yyy[:, 1:2]
    biharmonic_w = w_xxxx + 2.0 * w_xxyy + w_yyyy

    dtype = w.dtype
    D = tf.cast(parameters.D, dtype)
    nu = tf.cast(parameters.nu, dtype)
    q = _validate_scalar_field("load_fn(xy)", load_fn(xy), xy)

    r_mx = fields["Mx"] + D * (w_xx + nu * w_yy)
    r_my = fields["My"] + D * (w_yy + nu * w_xx)
    r_mxy = fields["Mxy"] + D * (1.0 - nu) * w_xy
    r_qx = fields["Qx"] - (
        grad_mx[:, 0:1]
        + grad_mxy[:, 1:2]
    )
    r_qy = fields["Qy"] - (
        grad_my[:, 1:2]
        + grad_mxy[:, 0:1]
    )
    r_fourth = D * biharmonic_w - q

    if foundation_fn is not None:
        k = _validate_scalar_field(
            "foundation_fn(xy)",
            foundation_fn(xy),
            xy,
            require_nonnegative=True,
        )
        r_fourth = r_fourth + k * w

    derivatives = {
        "w": w,
        "w_x": w_x,
        "w_y": w_y,
        "w_xx": w_xx,
        "w_yy": w_yy,
        "w_xy": w_xy,
        "w_xxx": w_xxx,
        "w_xxy": w_xxy,
        "w_xyy": w_xyy,
        "w_yyy": w_yyy,
        "w_xxxx": w_xxxx,
        "w_xxyy": w_xxyy,
        "w_yyyy": w_yyyy,
        "biharmonic_w": biharmonic_w,
    }
    residuals = {
        "moment_x": r_mx,
        "moment_y": r_my,
        "twisting_moment": r_mxy,
        "shear_x": r_qx,
        "shear_y": r_qy,
        "equilibrium": r_fourth,
    }
    _validate_columns(residuals)
    return MultiOutputFourthOrderState(
        fields=fields,
        deflection_derivatives=derivatives,
        residuals=residuals,
    )


def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate floating coordinates with shape ``(batch_size, 2)``."""
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


def _validate_fields(
    fields: dict[str, tf.Tensor],
    xy: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Validate the fixed six-field model output contract."""
    if not isinstance(fields, dict):
        raise TypeError("MO4 model output must be a field dictionary.")
    missing = [name for name in FIELD_NAMES if name not in fields]
    if missing:
        raise KeyError(f"MO4 model output is missing fields: {missing}.")

    validated: dict[str, tf.Tensor] = {}
    for name in FIELD_NAMES:
        value = tf.convert_to_tensor(fields[name])
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
        validated[name] = value
    return validated


def _validate_scalar_field(
    name: str,
    value: tf.Tensor,
    xy: tf.Tensor,
    require_nonnegative: bool = False,
) -> tf.Tensor:
    """Validate a scalar load or foundation field."""
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


def _validate_columns(values: dict[str, tf.Tensor]) -> None:
    """Validate all residual column tensors."""
    for name, value in values.items():
        _validate_column(name, value)


def _validate_column(name: str, value: tf.Tensor) -> None:
    """Validate one ``(batch_size, 1)`` tensor."""
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
