"""Derivative-based physical field reconstruction for a deflection-only PINN."""

from __future__ import annotations

import tensorflow as tf

from models import FIELD_NAMES

from .plate_residuals import PlateParameters


def compute_w_pinn_derivatives(
    model: tf.keras.Model,
    xy: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Return deflection derivatives through third order.

    The model only needs to provide ``{"w": tensor}``. Models returning
    additional fields are also accepted so analytical mixed-field models can
    be reused for verification.
    """
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as third_tape:
        third_tape.watch(xy)
        with tf.GradientTape(persistent=True) as second_tape:
            second_tape.watch(xy)
            with tf.GradientTape() as first_tape:
                first_tape.watch(xy)
                w = _extract_w(model(xy, training=training), xy)

            grad_w = _require_gradient(first_tape.gradient(w, xy), "w")
            w_x = grad_w[:, 0:1]
            w_y = grad_w[:, 1:2]

        grad_w_x = _require_gradient(second_tape.gradient(w_x, xy), "w_x")
        grad_w_y = _require_gradient(second_tape.gradient(w_y, xy), "w_y")
        w_xx = grad_w_x[:, 0:1]
        w_yy = grad_w_y[:, 1:2]
        w_xy = 0.5 * (grad_w_x[:, 1:2] + grad_w_y[:, 0:1])

    grad_w_xx = _require_gradient(third_tape.gradient(w_xx, xy), "w_xx")
    grad_w_yy = _require_gradient(third_tape.gradient(w_yy, xy), "w_yy")

    del second_tape
    del third_tape

    return {
        "w": w,
        "w_x": w_x,
        "w_y": w_y,
        "w_xx": w_xx,
        "w_yy": w_yy,
        "w_xy": w_xy,
        "w_xxx": grad_w_xx[:, 0:1],
        "w_xxy": grad_w_xx[:, 1:2],
        "w_xyy": grad_w_yy[:, 0:1],
        "w_yyy": grad_w_yy[:, 1:2],
    }


def compute_w_pinn_fields(
    model: tf.keras.Model,
    xy: tf.Tensor,
    parameters: PlateParameters,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Reconstruct all six Kirchhoff fields from the predicted deflection."""
    derivatives = compute_w_pinn_derivatives(
        model=model,
        xy=xy,
        training=training,
    )
    return fields_from_w_derivatives(derivatives, parameters)


def fields_from_w_derivatives(
    derivatives: dict[str, tf.Tensor],
    parameters: PlateParameters,
) -> dict[str, tf.Tensor]:
    """Construct Kirchhoff moments and shears from known derivatives of ``w``."""
    w = derivatives["w"]
    dtype = w.dtype
    D = tf.cast(parameters.D, dtype)
    nu = tf.cast(parameters.nu, dtype)

    mx = -D * (derivatives["w_xx"] + nu * derivatives["w_yy"])
    my = -D * (derivatives["w_yy"] + nu * derivatives["w_xx"])
    mxy = -D * (1.0 - nu) * derivatives["w_xy"]

    # Expanding Mx_x + Mxy_y and My_y + Mxy_x cancels nu and avoids
    # differentiating already assembled moment tensors a second time.
    qx = -D * (derivatives["w_xxx"] + derivatives["w_xyy"])
    qy = -D * (derivatives["w_xxy"] + derivatives["w_yyy"])

    fields = dict(zip(FIELD_NAMES, (w, mx, my, mxy, qx, qy)))
    _validate_column_fields(fields, w)
    return fields


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


def _extract_w(
    outputs: dict[str, tf.Tensor],
    xy: tf.Tensor,
) -> tf.Tensor:
    """Validate and return the model deflection output."""
    if not isinstance(outputs, dict) or "w" not in outputs:
        raise KeyError("W-PINN model output must contain a 'w' tensor.")
    w = tf.convert_to_tensor(outputs["w"])
    if w.dtype != xy.dtype:
        raise ValueError(f"w dtype must match xy dtype {xy.dtype}, got {w.dtype}.")
    if w.shape.rank is not None:
        if w.shape.rank != 2 or w.shape[-1] != 1:
            raise ValueError(
                f"w must have shape (batch_size, 1), got {w.shape}."
            )
    else:
        tf.debugging.assert_equal(tf.rank(w), 2, message="w must be 2D.")
        tf.debugging.assert_equal(
            tf.shape(w)[-1],
            1,
            message="w must have shape (batch_size, 1).",
        )
    tf.debugging.assert_equal(
        tf.shape(w)[0],
        tf.shape(xy)[0],
        message="w batch size must match xy batch size.",
    )
    return w


def _validate_column_fields(
    fields: dict[str, tf.Tensor],
    reference: tf.Tensor,
) -> None:
    """Validate reconstructed fields without leaving TensorFlow execution."""
    for name in FIELD_NAMES:
        value = tf.convert_to_tensor(fields[name])
        if value.dtype != reference.dtype:
            raise ValueError(
                f"{name} dtype must match w dtype {reference.dtype}, got {value.dtype}."
            )
        if value.shape.rank is not None:
            if value.shape.rank != 2 or value.shape[-1] != 1:
                raise ValueError(
                    f"{name} must have shape (batch_size, 1), got {value.shape}."
                )
        else:
            tf.debugging.assert_equal(
                tf.rank(value),
                2,
                message=f"{name} must be 2D.",
            )
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
