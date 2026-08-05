"""Gradient surgery utilities for shared PINN parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import tensorflow as tf


PCGRAD_TASK_LOSS_KEYS = {
    "moment": "weighted_moment_loss",
    "shear": "weighted_shear_loss",
    "equilibrium": "weighted_equilibrium_loss",
    "boundary": "weighted_boundary_loss",
}
PCGRAD_TASK_NAMES = tuple(PCGRAD_TASK_LOSS_KEYS)


def resolve_pcgrad_shared_variables(
    model: tf.keras.Model,
) -> tuple[tf.Variable, ...]:
    """Return parameters shared by every physical objective.

    Multi-branch models expose an explicit ``shared_trunk``. In the strict
    all-shared ``grouping='one'`` configuration there is no separate trunk,
    so every model parameter is shared and must participate in PCGrad.
    """
    shared_trunk = getattr(model, "shared_trunk", None)
    if shared_trunk is not None:
        variables = tuple(shared_trunk.trainable_variables)
        if variables:
            return variables

    get_grouping = getattr(model, "get_grouping", None)
    if callable(get_grouping) and get_grouping() == "one":
        variables = tuple(model.trainable_variables)
        if variables:
            return variables

    raise ValueError(
        "PCGrad requires an explicit shared trunk or a fully shared "
        "grouping='one' model."
    )


@dataclass(frozen=True)
class SharedTrunkPCGradConfig:
    """Configuration for dynamic four-objective shared-trunk PCGrad."""

    enabled: bool = False
    start_epoch: int = 1
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        """Validate the numerical projection guard."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if self.start_epoch <= 0:
            raise ValueError("start_epoch must be positive.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")


@dataclass(frozen=True)
class BoundaryAnchoredPCGradConfig(SharedTrunkPCGradConfig):
    """Configuration for moment-to-boundary shared-gradient projection."""


def apply_shared_trunk_pcgrad(
    tape: tf.GradientTape,
    loss_dict: dict[str, tf.Tensor],
    all_variables: Sequence[tf.Variable],
    shared_variables: Sequence[tf.Variable],
    total_gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
    config: SharedTrunkPCGradConfig,
) -> tuple[
    list[tf.Tensor | tf.IndexedSlices | None],
    dict[str, tf.Tensor],
]:
    """Apply dynamic PCGrad to four weighted shared-trunk objectives.

    Conflict decisions are recomputed from the current step's original task
    gradients. Only shared-variable entries are corrected; branch-specific
    entries remain exactly equal to the ordinary total-loss gradients.
    """
    if not config.enabled:
        return list(total_gradients), {}
    if not shared_variables:
        raise ValueError(
            "Shared-trunk PCGrad requires shared trainable variables."
        )
    missing = [
        loss_key
        for loss_key in PCGRAD_TASK_LOSS_KEYS.values()
        if loss_key not in loss_dict
    ]
    if missing:
        raise KeyError(
            "Shared-trunk PCGrad requires loss entries: "
            + ", ".join(missing)
        )

    dtype = tf.convert_to_tensor(
        loss_dict["weighted_moment_loss"]
    ).dtype
    original = {
        task_name: _dense_gradients(
            tape.gradient(loss_dict[loss_key], shared_variables),
            shared_variables,
            dtype,
        )
        for task_name, loss_key in PCGRAD_TASK_LOSS_KEYS.items()
    }
    projected = {
        task_name: list(gradients)
        for task_name, gradients in original.items()
    }
    original_norms = {
        task_name: tf.linalg.global_norm(gradients)
        for task_name, gradients in original.items()
    }
    norm_squared = {
        task_name: _norm_squared(gradients, dtype)
        for task_name, gradients in original.items()
    }

    diagnostics: dict[str, tf.Tensor] = {}
    conflict_flags = []
    for left_index, left_name in enumerate(PCGRAD_TASK_NAMES):
        for right_name in PCGRAD_TASK_NAMES[left_index + 1:]:
            dot = _dot(
                original[left_name],
                original[right_name],
                dtype,
            )
            conflict = tf.logical_and(
                dot < tf.zeros((), dtype=dtype),
                tf.logical_and(
                    norm_squared[left_name] > tf.cast(
                        config.epsilon,
                        dtype,
                    ),
                    norm_squared[right_name] > tf.cast(
                        config.epsilon,
                        dtype,
                    ),
                ),
            )
            conflict_flags.append(tf.cast(conflict, dtype))
            pair_name = f"{left_name}_{right_name}"
            diagnostics[f"pcgrad_conflict_{pair_name}"] = tf.cast(
                conflict,
                dtype,
            )
            diagnostics[f"pcgrad_cos_{pair_name}_before"] = _cosine(
                original[left_name],
                original[right_name],
                dtype,
            )

    projection_flags = []
    task_count = len(PCGRAD_TASK_NAMES)
    for left_index, left_name in enumerate(PCGRAD_TASK_NAMES):
        reference_order = (
            PCGRAD_TASK_NAMES[left_index + 1:]
            + PCGRAD_TASK_NAMES[:left_index]
        )
        for right_name in reference_order:
            original_dot = _dot(
                original[left_name],
                original[right_name],
                dtype,
            )
            current_dot = _dot(
                projected[left_name],
                original[right_name],
                dtype,
            )
            apply_projection = tf.logical_and(
                original_dot < tf.zeros((), dtype=dtype),
                tf.logical_and(
                    current_dot < tf.zeros((), dtype=dtype),
                    norm_squared[right_name] > tf.cast(
                        config.epsilon,
                        dtype,
                    ),
                ),
            )
            coefficient = tf.where(
                apply_projection,
                tf.math.divide_no_nan(
                    current_dot,
                    norm_squared[right_name],
                ),
                tf.zeros((), dtype=dtype),
            )
            projected[left_name] = [
                left_gradient - coefficient * right_gradient
                for left_gradient, right_gradient in zip(
                    projected[left_name],
                    original[right_name],
                )
            ]
            projection_flags.append(tf.cast(apply_projection, dtype))

    corrections = [
        tf.zeros_like(variable, dtype=dtype)
        for variable in shared_variables
    ]
    for task_name in PCGRAD_TASK_NAMES:
        corrections = [
            correction + projected_gradient - original_gradient
            for correction, projected_gradient, original_gradient in zip(
                corrections,
                projected[task_name],
                original[task_name],
            )
        ]

    variable_indices = {
        variable.ref(): index
        for index, variable in enumerate(all_variables)
    }
    projected_total = list(total_gradients)
    for variable, correction in zip(shared_variables, corrections):
        index = variable_indices[variable.ref()]
        total_gradient = projected_total[index]
        if total_gradient is None:
            projected_total[index] = correction
        else:
            projected_total[index] = (
                tf.convert_to_tensor(total_gradient) + correction
            )

    for task_name in PCGRAD_TASK_NAMES:
        diagnostics[f"pcgrad_norm_{task_name}_before"] = (
            original_norms[task_name]
        )
        diagnostics[f"pcgrad_norm_{task_name}_after"] = (
            tf.linalg.global_norm(projected[task_name])
        )
    for left_index, left_name in enumerate(PCGRAD_TASK_NAMES):
        for right_name in PCGRAD_TASK_NAMES[left_index + 1:]:
            pair_name = f"{left_name}_{right_name}"
            diagnostics[f"pcgrad_cos_{pair_name}_after"] = _cosine(
                projected[left_name],
                projected[right_name],
                dtype,
            )

    conflict_count = (
        tf.add_n(conflict_flags)
        if conflict_flags
        else tf.zeros((), dtype=dtype)
    )
    projection_count = (
        tf.add_n(projection_flags)
        if projection_flags
        else tf.zeros((), dtype=dtype)
    )
    pair_count = task_count * (task_count - 1) / 2
    diagnostics["pcgrad_conflict_count"] = conflict_count
    diagnostics["pcgrad_conflict_rate"] = (
        conflict_count / tf.cast(pair_count, dtype)
    )
    diagnostics["pcgrad_projection_count"] = projection_count
    return projected_total, diagnostics


def apply_boundary_anchored_pcgrad(
    tape: tf.GradientTape,
    loss_dict: dict[str, tf.Tensor],
    all_variables: Sequence[tf.Variable],
    shared_variables: Sequence[tf.Variable],
    total_gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
    config: BoundaryAnchoredPCGradConfig,
) -> tuple[
    list[tf.Tensor | tf.IndexedSlices | None],
    dict[str, tf.Tensor],
]:
    """Project a conflicting moment gradient while preserving the anchor.

    Only shared-trunk entries in ``total_gradients`` are changed. Gradients
    for every branch-specific variable remain exactly as produced by the
    weighted total loss.
    """
    if not config.enabled:
        return list(total_gradients), {}
    if not shared_variables:
        raise ValueError(
            "Boundary-anchored PCGrad requires shared trainable variables."
        )

    moment_gradients = tape.gradient(
        loss_dict["weighted_moment_loss"],
        shared_variables,
    )
    boundary_gradients = tape.gradient(
        loss_dict["weighted_boundary_loss"],
        shared_variables,
    )
    dtype = tf.convert_to_tensor(
        loss_dict["weighted_moment_loss"]
    ).dtype
    moment_dense = _dense_gradients(
        moment_gradients,
        shared_variables,
        dtype,
    )
    boundary_dense = _dense_gradients(
        boundary_gradients,
        shared_variables,
        dtype,
    )

    moment_norm = tf.linalg.global_norm(moment_dense)
    boundary_norm = tf.linalg.global_norm(boundary_dense)
    dot = _dot(moment_dense, boundary_dense, dtype)
    boundary_norm_squared = tf.add_n(
        [tf.reduce_sum(tf.square(value)) for value in boundary_dense]
    )
    conflict = tf.logical_and(
        dot < tf.zeros((), dtype=dtype),
        boundary_norm_squared > tf.cast(config.epsilon, dtype),
    )
    projection_coefficient = tf.where(
        conflict,
        tf.math.divide_no_nan(dot, boundary_norm_squared),
        tf.zeros((), dtype=dtype),
    )
    projected_moment = [
        moment - projection_coefficient * boundary
        for moment, boundary in zip(moment_dense, boundary_dense)
    ]
    projected_moment_norm = tf.linalg.global_norm(projected_moment)

    variable_indices = {
        variable.ref(): index
        for index, variable in enumerate(all_variables)
    }
    projected_total = list(total_gradients)
    for variable, original_moment, new_moment in zip(
        shared_variables,
        moment_dense,
        projected_moment,
    ):
        index = variable_indices[variable.ref()]
        correction = new_moment - original_moment
        total_gradient = projected_total[index]
        if total_gradient is None:
            projected_total[index] = correction
        else:
            projected_total[index] = (
                tf.convert_to_tensor(total_gradient) + correction
            )

    cosine_before = tf.math.divide_no_nan(
        dot,
        moment_norm * boundary_norm,
    )
    cosine_after = tf.math.divide_no_nan(
        _dot(projected_moment, boundary_dense, dtype),
        projected_moment_norm * boundary_norm,
    )
    diagnostics = {
        "pcgrad_moment_boundary_applied": tf.cast(conflict, dtype),
        "pcgrad_moment_boundary_dot": dot,
        "pcgrad_moment_boundary_cos_before": cosine_before,
        "pcgrad_moment_boundary_cos_after": cosine_after,
        "pcgrad_moment_norm_before": moment_norm,
        "pcgrad_moment_norm_after": projected_moment_norm,
        "pcgrad_boundary_norm": boundary_norm,
    }
    return projected_total, diagnostics


def _dense_gradients(
    gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
    variables: Sequence[tf.Variable],
    dtype: tf.dtypes.DType,
) -> list[tf.Tensor]:
    """Return dense aligned gradients, replacing disconnected entries by zero."""
    dense = []
    for gradient, variable in zip(gradients, variables):
        if gradient is None:
            dense.append(tf.zeros_like(variable, dtype=dtype))
        else:
            dense.append(tf.cast(tf.convert_to_tensor(gradient), dtype))
    return dense


def _dot(
    left: Sequence[tf.Tensor],
    right: Sequence[tf.Tensor],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return the dot product of aligned dense gradient collections."""
    terms = [
        tf.reduce_sum(left_value * right_value)
        for left_value, right_value in zip(left, right)
    ]
    if not terms:
        return tf.zeros((), dtype=dtype)
    return tf.add_n(terms)


def _norm_squared(
    gradients: Sequence[tf.Tensor],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return the squared global norm of aligned gradients."""
    terms = [
        tf.reduce_sum(tf.square(gradient))
        for gradient in gradients
    ]
    if not terms:
        return tf.zeros((), dtype=dtype)
    return tf.add_n(terms)


def _cosine(
    left: Sequence[tf.Tensor],
    right: Sequence[tf.Tensor],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return a safe cosine similarity for aligned gradients."""
    return tf.math.divide_no_nan(
        _dot(left, right, dtype),
        tf.linalg.global_norm(left) * tf.linalg.global_norm(right),
    )


__all__ = [
    "BoundaryAnchoredPCGradConfig",
    "PCGRAD_TASK_LOSS_KEYS",
    "PCGRAD_TASK_NAMES",
    "SharedTrunkPCGradConfig",
    "apply_boundary_anchored_pcgrad",
    "apply_shared_trunk_pcgrad",
]
