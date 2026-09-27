"""Shared plotting helpers for experiment evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    prediction_title: str = "PINN prediction",
    error_title: str = "Absolute error",
    panel_titles_below: bool = False,
    include_field_title: bool = True,
    dpi: int = 200,
    panel_labels: Sequence[str] = ("a", "b", "c"),
    panel_title_font_size: float = 10.5,
) -> None:
    """Save prediction, reference, and absolute-error maps for six fields."""
    if len(panel_labels) != 3:
        raise ValueError("panel_labels must contain exactly three labels.")
    _configure_publication_font()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name in FIELD_NAMES:
        prediction_values = _grid_values(predicted[name], grid_size)
        reference_values = _grid_values(reference[name], grid_size)
        error_values = np.abs(prediction_values - reference_values)

        figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.65))
        field_min = min(float(np.min(prediction_values)), float(np.min(reference_values)))
        field_max = max(float(np.max(prediction_values)), float(np.max(reference_values)))
        if np.isclose(field_min, field_max):
            padding = max(abs(field_min) * 1e-6, 1e-12)
            field_min -= padding
            field_max += padding
        field_exponent = _colorbar_exponent(field_min, field_max)
        error_exponent = _colorbar_exponent(0.0, float(np.max(error_values)))
        panels = (
            (prediction_values, prediction_title),
            (reference_values, reference_title),
            (error_values, error_title),
        )
        for index, (axis, (values, title)) in enumerate(zip(axes, panels)):
            exponent = field_exponent if index < 2 else error_exponent
            scale = 10.0 ** exponent
            displayed_values = values / scale
            image_kwargs = {}
            if index < 2:
                image_kwargs = {
                    "vmin": field_min / scale,
                    "vmax": field_max / scale,
                }
            contour = axis.imshow(
                displayed_values,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal",
                cmap="viridis" if index < 2 else "magma",
                **image_kwargs,
            )
            caption = f"({panel_labels[index]}) {title}"
            if panel_titles_below:
                axis.text(
                    0.5,
                    -0.28,
                    caption,
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=panel_title_font_size,
                    clip_on=False,
                )
            else:
                axis.set_title(caption, fontsize=panel_title_font_size)
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$y$")
            colorbar = figure.colorbar(contour, ax=axis, shrink=0.82, pad=0.04)
            if exponent != 0:
                colorbar.ax.set_title(
                    rf"$\times 10^{{{exponent}}}$",
                    fontsize=9.0,
                    pad=5.0,
                )
        if include_field_title:
            figure.suptitle(name)
        figure.subplots_adjust(
            left=0.055,
            right=0.985,
            top=0.91 if include_field_title else 0.97,
            bottom=0.24 if panel_titles_below else 0.14,
            wspace=0.34,
        )
        figure.savefig(
            figures_dir / f"{name}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def _configure_publication_font() -> None:
    """Use an installed Chinese font and thesis-sized plot text."""
    override = os.environ.get("PINN_CHINESE_FONT")
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        Path(override) if override else None,
        windows_fonts / "simsun.ttc",
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for font_path in candidates:
        if font_path is None or not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font_name, "DejaVu Sans"],
                "font.size": 10.5,
                "axes.titlesize": 10.5,
                "axes.labelsize": 10.5,
                "xtick.labelsize": 9.0,
                "ytick.labelsize": 9.0,
                "axes.unicode_minus": False,
                "mathtext.fontset": "stix",
            }
        )
        return


def _colorbar_exponent(minimum: float, maximum: float) -> int:
    """Return a stable base-10 exponent for one colorbar."""
    magnitude = max(abs(minimum), abs(maximum))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return 0
    exponent = int(np.floor(np.log10(magnitude)))
    return exponent if exponent != 0 else 0


def _grid_values(values: tf.Tensor, grid_size: int) -> np.ndarray:
    """Convert a flattened tensor to a checked square NumPy grid."""
    array = np.asarray(tf.convert_to_tensor(values).numpy()).reshape(-1)
    expected = grid_size * grid_size
    if array.size != expected:
        raise ValueError(f"Expected {expected} values for a {grid_size}x{grid_size} grid, got {array.size}.")
    return array.reshape(grid_size, grid_size)
