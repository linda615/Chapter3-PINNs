"""Independent fully connected subnetworks for PINN field prediction."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


class IndependentSubNet(tf.keras.Model):
    """A standalone MLP that maps spatial coordinates ``(x, y)`` to fields.

    Each instance owns its own hidden layers and trainable variables. No layers
    are shared between different ``IndependentSubNet`` objects.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_width: int = 64,
        hidden_depth: int = 4,
        activation: str | tf.keras.layers.Layer = "tanh",
        hidden_units: Sequence[int] | None = None,
        input_dim: int | None = 2,
        name: str | None = None,
    ) -> None:
        """Create an independent fully connected subnetwork."""
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if input_dim is not None and input_dim <= 0:
            raise ValueError(f"input_dim must be positive or None, got {input_dim}.")
        if hidden_units is None:
            if hidden_width <= 0:
                raise ValueError(f"hidden_width must be positive, got {hidden_width}.")
            if hidden_depth < 0:
                raise ValueError(f"hidden_depth must be non-negative, got {hidden_depth}.")
            hidden_units = tuple(hidden_width for _ in range(hidden_depth))
        if any(unit <= 0 for unit in hidden_units):
            raise ValueError("All hidden layer widths must be positive.")

        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.hidden_units = tuple(hidden_units)
        self.input_dim = input_dim

        self.hidden_layers = [
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name=f"hidden_{index}",
            )
            for index, units in enumerate(self.hidden_units)
        ]
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name="output",
        )

    def call(self, xy: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Evaluate the subnetwork on a rank-2 input tensor."""
        inputs = self._validate_inputs(xy)
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z, training=training)
        return self.output_layer(z, training=training)

    def _validate_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """Validate that the input is a rank-2 tensor with the configured width."""
        inputs = tf.convert_to_tensor(inputs)

        if inputs.shape.rank is not None:
            if inputs.shape.rank != 2:
                raise ValueError(f"inputs must be a 2D tensor, got rank {inputs.shape.rank}.")
            if self.input_dim is not None and inputs.shape[-1] != self.input_dim:
                raise ValueError(
                    "inputs must have shape "
                    f"(batch_size, {self.input_dim}), got {inputs.shape}."
                )
            return inputs

        tf.debugging.assert_equal(tf.rank(inputs), 2, message="inputs must be a 2D tensor.")
        if self.input_dim is not None:
            tf.debugging.assert_equal(
                tf.shape(inputs)[-1],
                self.input_dim,
                message=f"inputs must have shape (batch_size, {self.input_dim}).",
            )
        return inputs

    def get_config(self) -> dict:
        """Return a serializable configuration dictionary."""
        return {
            "output_dim": self.output_dim,
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "input_dim": self.input_dim,
            "name": self.name,
        }
