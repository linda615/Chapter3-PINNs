"""Shared plotting helpers for experiment evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import FIELD_NAMES


def relative_l2_error(prediction: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    """Return the relative L2 error with a guarded reference norm."""
    prediction = tf.convert_to_tensor(prediction)
    reference = tf.cast(reference, prediction.dtype)
    numerator = tf.sqrt(tf.reduce_sum(tf.square(prediction - reference)))
    denominator = tf.sqrt(tf.reduce_sum(tf.square(reference)))
    return numerator / tf.maximum(denominator, tf.cast(1e-12, prediction.dtype))


def save_field_figures(
    figures_dir: str | Path,
    predicted: Mapping[str, tf.Tensor],
    reference: Mapping[str, tf.Tensor],
    grid_size: int,
    reference_title: str = "Reference",
) -> None:
    """Save prediction, reference, and absolute-error maps for six fields."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name in FIELD_NAMES:
        prediction_values = _grid_values(predicted[name], grid_size)
        reference_values = _grid_values(reference[name], grid_size)
        error_values = np.abs(prediction_values - reference_values)

        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        panels = (
            (prediction_values, "PINN prediction"),
            (reference_values, reference_title),
            (error_values, "Absolute error"),
        )
        for axis, (values, title) in zip(axes, panels):
            contour = axis.imshow(
                values,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal",
                cmap="viridis",
            )
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(contour, ax=axis, shrink=0.82)
        figure.suptitle(name)
        figure.savefig(figures_dir / f"{name}.png", dpi=200)
        plt.close(figure)


def _grid_values(values: tf.Tensor, grid_size: int) -> np.ndarray:
    """Convert a flattened tensor to a checked square NumPy grid."""
    array = np.asarray(tf.convert_to_tensor(values).numpy()).reshape(-1)
    expected = grid_size * grid_size
    if array.size != expected:
        raise ValueError(f"Expected {expected} values for a {grid_size}x{grid_size} grid, got {array.size}.")
    return array.reshape(grid_size, grid_size)
