"""One-way consistency losses for mixed Kirchhoff plate fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import tensorflow as tf

from physics.plate_residuals import PlateParameters, PlateResidualState


@dataclass(frozen=True)
class MxyConsistencyConfig:
    """Configuration for normalized direct-Mxy versus deflection consistency."""

    enabled: bool = False
    minimum_scale: float = 1e-3
    stop_gradient_target: bool = True

    def __post_init__(self) -> None:
        """Validate consistency-loss controls."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError(
                "minimum_scale must be finite and positive, "
                f"got {self.minimum_scale}."
            )
        if not isinstance(self.stop_gradient_target, bool):
            raise TypeError("stop_gradient_target must be a bool.")


def compute_mxy_consistency_loss(
    state: PlateResidualState,
    parameters: PlateParameters,
    config: Optional[MxyConsistencyConfig] = None,
) -> dict[str, tf.Tensor]:
    """Compare direct ``Mxy`` with the value reconstructed from ``w_xy``.

    The deflection-derived target is detached by default, so this auxiliary
    objective calibrates the direct mixed-variable output instead of allowing
    both sides of the constitutive relation to move toward one another.
    """
    resolved = MxyConsistencyConfig() if config is None else config
    mxy_pred = tf.convert_to_tensor(state.fields["Mxy"])
    zero = tf.zeros((), dtype=mxy_pred.dtype)
    if not resolved.enabled:
        return {
            "mxy_consistency_loss": zero,
            "mxy_consistency_scale": zero,
        }

    D = tf.cast(parameters.D, mxy_pred.dtype)
    nu = tf.cast(parameters.nu, mxy_pred.dtype)
    mxy_from_w = (
        -D
        * (tf.cast(1.0, mxy_pred.dtype) - nu)
        * state.deflection_derivatives["w_xy"]
    )
    target = (
        tf.stop_gradient(mxy_from_w)
        if resolved.stop_gradient_target
        else mxy_from_w
    )
    target_rms = tf.sqrt(tf.reduce_mean(tf.square(target)))
    minimum_scale = tf.cast(resolved.minimum_scale, mxy_pred.dtype)
    scale = tf.stop_gradient(tf.maximum(target_rms, minimum_scale))
    normalized_difference = (mxy_pred - target) / scale

    return {
        "mxy_consistency_loss": tf.reduce_mean(
            tf.square(normalized_difference)
        ),
        "mxy_consistency_scale": scale,
    }
