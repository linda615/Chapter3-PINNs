"""Boundary differential and resultant operators."""

from __future__ import annotations

import tensorflow as tf


def normal_moment(fields: dict[str, tf.Tensor], normals: tf.Tensor) -> tf.Tensor:
    """Return normal bending moment ``M_n``."""
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    return nx**2 * fields["Mx"] + ny**2 * fields["My"] + 2.0 * nx * ny * fields["Mxy"]


def normal_shear(fields: dict[str, tf.Tensor], normals: tf.Tensor) -> tf.Tensor:
    """Return normal shear force ``Q_n``."""
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    return nx * fields["Qx"] + ny * fields["Qy"]


def normal_slope(derivatives: dict[str, tf.Tensor], normals: tf.Tensor) -> tf.Tensor:
    """Return normal deflection slope ``dw/dn``."""
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    return nx * derivatives["w_x"] + ny * derivatives["w_y"]
