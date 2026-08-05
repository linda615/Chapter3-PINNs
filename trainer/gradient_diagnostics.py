"""Gradient diagnostics for competing PINN loss groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import tensorflow as tf

from .pcgrad import resolve_pcgrad_shared_variables


WEIGHTED_LOSS_KEYS = {
    "moment": "weighted_moment_loss",
    "shear": "weighted_shear_loss",
    "equilibrium": "weighted_equilibrium_loss",
    "boundary": "weighted_boundary_loss",
}


def compute_shared_gradient_diagnostics(
    model: tf.keras.Model,
    loss_fn: Callable[[], dict[str, tf.Tensor]],
) -> dict[str, tf.Tensor]:
    """Return weighted-loss gradient norms and cosines on the shared trunk.

    ``loss_fn`` is evaluated once under a persistent tape. The four gradients
    therefore use exactly the same model state and collocation points.
    """
    with tf.GradientTape(persistent=True) as tape:
        loss_dict = loss_fn()

    variables = resolve_pcgrad_shared_variables(model)

    missing = [
        key for key in WEIGHTED_LOSS_KEYS.values() if key not in loss_dict
    ]
    if missing:
        del tape
        raise KeyError(
            "Loss dictionary is missing gradient diagnostic terms: "
            + ", ".join(missing)
        )

    gradients = {
        name: tape.gradient(loss_dict[key], variables)
        for name, key in WEIGHTED_LOSS_KEYS.items()
    }
    del tape

    dtype = tf.convert_to_tensor(
        loss_dict["weighted_moment_loss"]
    ).dtype
    norms = {
        name: _global_norm(values, dtype)
        for name, values in gradients.items()
    }
    diagnostics = {
        f"shared_grad_norm_{name}": norm
        for name, norm in norms.items()
    }

    names = tuple(WEIGHTED_LOSS_KEYS)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            dot = _gradient_dot(
                gradients[left_name],
                gradients[right_name],
                dtype,
            )
            cosine = tf.math.divide_no_nan(
                dot,
                norms[left_name] * norms[right_name],
            )
            diagnostics[
                f"shared_grad_cos_{left_name}_{right_name}"
            ] = cosine
    return diagnostics


def _global_norm(
    gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return the global norm of all available gradient tensors."""
    tensors = [
        tf.cast(tf.convert_to_tensor(gradient), dtype)
        for gradient in gradients
        if gradient is not None
    ]
    if not tensors:
        return tf.zeros((), dtype=dtype)
    return tf.linalg.global_norm(tensors)


def _gradient_dot(
    left: Sequence[tf.Tensor | tf.IndexedSlices | None],
    right: Sequence[tf.Tensor | tf.IndexedSlices | None],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return the dot product between two aligned gradient collections."""
    terms = []
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        left_tensor = tf.cast(tf.convert_to_tensor(left_value), dtype)
        right_tensor = tf.cast(tf.convert_to_tensor(right_value), dtype)
        terms.append(tf.reduce_sum(left_tensor * right_tensor))
    if not terms:
        return tf.zeros((), dtype=dtype)
    return tf.add_n(terms)


__all__ = [
    "WEIGHTED_LOSS_KEYS",
    "compute_shared_gradient_diagnostics",
]
