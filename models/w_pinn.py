"""Single-output deflection PINN for Kirchhoff thin plates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import tensorflow as tf

from .multi_subnet_pinn import (
    DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
    DEFLECTION_TRANSFORM_CUSTOM,
    DEFLECTION_TRANSFORM_NONE,
    DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    OUTPUT_ACTIVATION_LINEAR,
    OUTPUT_ACTIVATION_NAMES,
    OUTPUT_ACTIVATION_TANH,
    MultiSubNetPINN,
)
from .subnet import IndependentSubNet


W_PINN_DEFLECTION_TRANSFORM_NAMES = (
    DEFLECTION_TRANSFORM_NONE,
    DEFLECTION_TRANSFORM_CUSTOM,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
    DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
)


class WPINN(tf.keras.Model):
    """A fully connected PINN that predicts only the plate deflection ``w``."""

    def __init__(
        self,
        hidden_width: int = 64,
        hidden_depth: int = 5,
        activation: str | tf.keras.layers.Layer = "tanh",
        hidden_units: Sequence[int] | None = None,
        apply_deflection_transform: bool = False,
        deflection_transform_name: str = DEFLECTION_TRANSFORM_NONE,
        deflection_transform: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
        output_activation: str = OUTPUT_ACTIVATION_LINEAR,
        output_scale: float = 1.0,
        name: str = "w_pinn",
    ) -> None:
        """Create a single-output deflection network."""
        if deflection_transform_name not in W_PINN_DEFLECTION_TRANSFORM_NAMES:
            valid = ", ".join(
                f'"{item}"' for item in W_PINN_DEFLECTION_TRANSFORM_NAMES
            )
            raise ValueError(
                "Unsupported deflection_transform_name "
                f"{deflection_transform_name!r}. Expected one of: {valid}."
            )
        if (
            deflection_transform is not None
            and deflection_transform_name == DEFLECTION_TRANSFORM_NONE
        ):
            deflection_transform_name = DEFLECTION_TRANSFORM_CUSTOM
        if (
            deflection_transform is not None
            and deflection_transform_name != DEFLECTION_TRANSFORM_CUSTOM
        ):
            raise ValueError(
                "Use deflection_transform_name='custom' when passing a custom "
                "deflection_transform callable."
            )
        if apply_deflection_transform and deflection_transform is None:
            deflection_transform = self._get_named_deflection_transform(
                deflection_transform_name
            )
        if apply_deflection_transform and deflection_transform is None:
            raise ValueError(
                "A built-in or custom deflection transform is required when "
                "apply_deflection_transform=True."
            )
        if output_activation not in OUTPUT_ACTIVATION_NAMES:
            raise ValueError(
                f"Unsupported output_activation {output_activation!r}; "
                f"expected one of {OUTPUT_ACTIVATION_NAMES}."
            )
        if (
            not isinstance(output_scale, (int, float))
            or not math.isfinite(output_scale)
            or output_scale <= 0.0
        ):
            raise ValueError("output_scale must be a positive finite number.")

        super().__init__(name=name)
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.hidden_units = None if hidden_units is None else tuple(hidden_units)
        self.apply_deflection_transform = bool(apply_deflection_transform)
        self.deflection_transform_name = deflection_transform_name
        self.deflection_transform = deflection_transform
        self.output_activation = output_activation
        self.output_scale = float(output_scale)
        self.net_w = IndependentSubNet(
            output_dim=1,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            activation=activation,
            hidden_units=hidden_units,
            input_dim=2,
            name="net_w",
        )

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        """Return the transformed deflection as ``{"w": (batch, 1)}``."""
        xy = self._validate_xy(xy)
        raw_w = self.net_w(xy, training=training)
        w = tf.tanh(raw_w) if self.output_activation == OUTPUT_ACTIVATION_TANH else raw_w
        if self.apply_deflection_transform:
            if self.deflection_transform is None:
                raise ValueError("deflection_transform is not configured.")
            w = tf.convert_to_tensor(self.deflection_transform(xy, w))
            self._validate_w(w)
        return {"w": w * tf.cast(self.output_scale, w.dtype)}

    def call_raw(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        """Return the untransformed deflection for diagnostics."""
        xy = self._validate_xy(xy)
        return {"w": self.net_w(xy, training=training)}

    def predict_fields(self, xy: tf.Tensor) -> dict[str, tf.Tensor]:
        """Predict deflection in inference mode."""
        return self(xy, training=False)

    def uses_deflection_transform(self) -> bool:
        """Return whether a hard deflection transform is active."""
        return self.apply_deflection_transform

    def get_trainable_parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return int(
            sum(tf.size(variable).numpy() for variable in self.trainable_variables)
        )

    def get_config(self) -> dict:
        """Return a serializable model configuration."""
        if (
            self.apply_deflection_transform
            and self.deflection_transform_name == DEFLECTION_TRANSFORM_CUSTOM
        ):
            raise ValueError(
                "Models using a custom deflection_transform callable cannot be "
                "fully restored from get_config()."
            )
        return {
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "apply_deflection_transform": self.apply_deflection_transform,
            "deflection_transform_name": self.deflection_transform_name,
            "output_activation": self.output_activation,
            "output_scale": self.output_scale,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict) -> "WPINN":
        """Restore a model from ``get_config`` output."""
        return cls(**config)

    @staticmethod
    def _get_named_deflection_transform(
        name: str,
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None:
        """Resolve a built-in hard output transform."""
        if name in (DEFLECTION_TRANSFORM_NONE, DEFLECTION_TRANSFORM_CUSTOM):
            return None
        if name in (
            DEFLECTION_TRANSFORM_UNIT_SQUARE,
            DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE,
            DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
            DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
        ):
            return MultiSubNetPINN._get_named_deflection_transform(name)
        raise ValueError(f"Unsupported deflection_transform_name {name!r}.")

    @staticmethod
    def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
        """Validate floating coordinates with shape ``(batch_size, 2)``."""
        xy = tf.convert_to_tensor(xy)
        if not xy.dtype.is_floating:
            raise TypeError(f"xy must be floating-point, got {xy.dtype}.")
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

    @staticmethod
    def _validate_w(w: tf.Tensor) -> None:
        """Validate a transformed scalar deflection output."""
        if w.shape.rank is not None:
            if w.shape.rank != 2 or w.shape[-1] != 1:
                raise ValueError(
                    "deflection_transform output must have shape "
                    f"(batch_size, 1), got {w.shape}."
                )
            return
        tf.debugging.assert_equal(tf.rank(w), 2, message="w must be 2D.")
        tf.debugging.assert_equal(
            tf.shape(w)[-1],
            1,
            message="w must have shape (batch_size, 1).",
        )
