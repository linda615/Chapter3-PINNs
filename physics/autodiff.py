"""TensorFlow automatic differentiation helpers for PINN fields."""

from __future__ import annotations

import tensorflow as tf


FIELD_NAMES = ("w", "Mx", "My", "Mxy", "Qx", "Qy")


def first_derivatives(
    model: tf.keras.Model,
    xy: tf.Tensor,
    training: bool = False,
) -> tuple[dict[str, tf.Tensor], dict[str, dict[str, tf.Tensor]]]:
    """Compute model fields and first spatial derivatives.

    Args:
        model: A TensorFlow model returning a dictionary with the fixed field
            names ``("w", "Mx", "My", "Mxy", "Qx", "Qy")``.
        xy: Floating-point coordinate tensor with shape ``(batch_size, 2)``.
        training: Whether to call the model in training mode.

    Returns:
        A pair ``(fields, derivatives)``. ``fields[field]`` has shape
        ``(batch_size, 1)``, and ``derivatives[field]["x"]`` /
        ``derivatives[field]["y"]`` contain first derivatives with the same
        shape.
    """
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        fields = _validate_fields(model(xy, training=training))

    derivatives: dict[str, dict[str, tf.Tensor]] = {}
    for field_name in FIELD_NAMES:
        grad = tape.gradient(fields[field_name], xy)
        grad = _require_gradient(grad, field_name)
        derivatives[field_name] = {
            "x": grad[:, 0:1],
            "y": grad[:, 1:2],
        }

    del tape
    return fields, derivatives


def deflection_derivatives(
    model: tf.keras.Model,
    xy: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute deflection and first/second derivatives of ``w``.

    Returns a dictionary containing ``w``, ``w_x``, ``w_y``, ``w_xx``,
    ``w_yy``, and ``w_xy``. Every tensor has shape ``(batch_size, 1)``.
    """
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as second_tape:
        second_tape.watch(xy)
        with tf.GradientTape(persistent=True) as first_tape:
            first_tape.watch(xy)
            fields = _validate_fields(model(xy, training=training))
            w = fields["w"]

        grad_w = first_tape.gradient(w, xy)
        grad_w = _require_gradient(grad_w, "w")
        w_x = grad_w[:, 0:1]
        w_y = grad_w[:, 1:2]

    grad_w_x = second_tape.gradient(w_x, xy)
    grad_w_y = second_tape.gradient(w_y, xy)
    grad_w_x = _require_gradient(grad_w_x, "w_x")
    grad_w_y = _require_gradient(grad_w_y, "w_y")

    del first_tape
    del second_tape

    return {
        "w": w,
        "w_x": w_x,
        "w_y": w_y,
        "w_xx": grad_w_x[:, 0:1],
        "w_yy": grad_w_y[:, 1:2],
        "w_xy": grad_w_x[:, 1:2],
    }


def deflection_third_derivatives(
    model: tf.keras.Model,
    xy: tf.Tensor,
    training: bool = False,
) -> dict[str, tf.Tensor]:
    """Compute deflection derivatives through third order.

    The returned third derivatives are the four independent terms required
    to reconstruct Kirchhoff shear forces from the deflection field:
    ``w_xxx``, ``w_xxy``, ``w_xyy``, and ``w_yyy``.
    """
    xy = _validate_xy(xy)

    with tf.GradientTape(persistent=True) as third_tape:
        third_tape.watch(xy)
        with tf.GradientTape(persistent=True) as second_tape:
            second_tape.watch(xy)
            with tf.GradientTape(persistent=True) as first_tape:
                first_tape.watch(xy)
                fields = _validate_fields(model(xy, training=training))
                w = fields["w"]

            grad_w = _require_gradient(first_tape.gradient(w, xy), "w")
            w_x = grad_w[:, 0:1]
            w_y = grad_w[:, 1:2]

        grad_w_x = _require_gradient(second_tape.gradient(w_x, xy), "w_x")
        grad_w_y = _require_gradient(second_tape.gradient(w_y, xy), "w_y")
        w_xx = grad_w_x[:, 0:1]
        w_yy = grad_w_y[:, 1:2]
        # Average the two equivalent mixed partials to reduce roundoff noise.
        w_xy = 0.5 * (grad_w_x[:, 1:2] + grad_w_y[:, 0:1])

    grad_w_xx = _require_gradient(third_tape.gradient(w_xx, xy), "w_xx")
    grad_w_yy = _require_gradient(third_tape.gradient(w_yy, xy), "w_yy")

    del first_tape
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


def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
    """Validate coordinates without leaving TensorFlow graph execution."""
    xy = tf.convert_to_tensor(xy)

    if not xy.dtype.is_floating:
        raise TypeError(f"xy must be a floating-point Tensor, got dtype {xy.dtype}.")

    if xy.shape.rank is not None:
        if xy.shape.rank != 2:
            raise ValueError(f"xy must be a 2D tensor, got rank {xy.shape.rank}.")
        if xy.shape[-1] != 2:
            raise ValueError(f"xy must have shape (batch_size, 2), got {xy.shape}.")
        return xy

    tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be a 2D tensor.")
    tf.debugging.assert_equal(
        tf.shape(xy)[-1],
        2,
        message="xy must have shape (batch_size, 2).",
    )
    return xy


def _validate_fields(fields: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Validate model output field names and tensor shapes."""
    missing = [field for field in FIELD_NAMES if field not in fields]
    if missing:
        raise KeyError(f"Model output is missing fields: {missing}.")

    validated = {}
    for field_name in FIELD_NAMES:
        value = tf.convert_to_tensor(fields[field_name])
        if value.shape.rank is not None:
            if value.shape.rank != 2:
                raise ValueError(f"{field_name} must be a 2D tensor, got rank {value.shape.rank}.")
            if value.shape[-1] != 1:
                raise ValueError(f"{field_name} must have shape (batch_size, 1), got {value.shape}.")
        else:
            tf.debugging.assert_equal(tf.rank(value), 2, message=f"{field_name} must be 2D.")
            tf.debugging.assert_equal(
                tf.shape(value)[-1],
                1,
                message=f"{field_name} must have shape (batch_size, 1).",
            )
        validated[field_name] = value

    return validated


def _require_gradient(gradient: tf.Tensor | None, name: str) -> tf.Tensor:
    """Return a gradient tensor or raise a clear error if it is disconnected."""
    if gradient is None:
        raise ValueError(f"Gradient for {name!r} is None. Check model dependence on xy.")
    return gradient
