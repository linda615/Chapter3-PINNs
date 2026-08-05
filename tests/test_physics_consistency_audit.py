"""Physics consistency audit for the simply supported sinusoidal example."""

from pathlib import Path

import pytest
import tensorflow as tf

from boundary import BoundaryBatch, BoundaryType, compute_boundary_loss
from experiments.simply_supported_sinusoidal.analytical import (
    AnalyticalPlateModel,
    analytical_fields,
)
from experiments.simply_supported_sinusoidal.config import DEFAULT_CONFIG
from experiments.simply_supported_sinusoidal.evaluate import (
    evaluate_model,
    load_best_checkpoint,
    make_test_grid,
    relative_l2_error,
)
from experiments.simply_supported_sinusoidal.problem import (
    create_boundary_sampler,
    create_load_fn,
    create_model,
)
from models import FIELD_NAMES
from physics.plate_residuals import compute_plate_residuals


def _audit_grid() -> tf.Tensor:
    """Use an interior grid to avoid evaluating derivatives exactly at corners."""
    values = tf.linspace(tf.constant(0.1, dtype=tf.float32), tf.constant(0.9, dtype=tf.float32), 9)
    x_grid, y_grid = tf.meshgrid(values, values, indexing="ij")
    return tf.stack([tf.reshape(x_grid, (-1,)), tf.reshape(y_grid, (-1,))], axis=1)


def _load_available_best_checkpoint(model: tf.keras.Model) -> None:
    """Load the trained checkpoint or skip when it belongs to an old architecture."""
    try:
        load_best_checkpoint(model, DEFAULT_CONFIG.best_weights_path)
    except ValueError as exc:
        pytest.skip(f"Trained best weights are incompatible with the current model: {exc}")


def test_a_analytical_model_residual_groups_are_near_zero() -> None:
    """A. Analytical model should satisfy moment, shear, and equilibrium residuals."""
    model = AnalyticalPlateModel(DEFAULT_CONFIG.plate)
    residuals = compute_plate_residuals(
        model=model,
        xy=_audit_grid(),
        load_fn=create_load_fn(),
        parameters=DEFAULT_CONFIG.plate,
    )

    groups = {
        "moment": ("moment_x", "moment_y", "twisting_moment"),
        "shear": ("shear_x", "shear_y"),
        "equilibrium": ("equilibrium",),
    }
    for group_name, keys in groups.items():
        max_abs = max(float(tf.reduce_max(tf.abs(residuals[key])).numpy()) for key in keys)
        print(f"audit_residual_group {group_name} max_abs={max_abs:.6e}")

    for key, residual in residuals.items():
        print(f"audit_residual {key} max_abs={float(tf.reduce_max(tf.abs(residual)).numpy()):.6e}")
        tf.debugging.assert_near(residual, tf.zeros_like(residual), atol=1e-5)


def test_b_analytical_model_satisfies_all_simple_boundaries() -> None:
    """B. Analytical model should satisfy four-edge simply supported conditions."""
    model = AnalyticalPlateModel(DEFAULT_CONFIG.plate)
    boundary_xy, normals = create_boundary_sampler().sample_all_sides(11, seed=7)

    losses = compute_boundary_loss(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        boundary_type=BoundaryType.SIMPLE,
    )

    for key, value in losses.items():
        print(f"audit_boundary {key}={float(value.numpy()):.6e}")
    tf.debugging.assert_near(losses["deflection_bc_loss"], tf.constant(0.0), atol=1e-10)
    tf.debugging.assert_near(losses["moment_bc_loss"], tf.constant(0.0), atol=1e-10)


@pytest.mark.skipif(
    not DEFAULT_CONFIG.best_weights_path.exists(),
    reason="Trained best weights are not available for prediction audit.",
)
def test_c_trained_model_field_error_report() -> None:
    """C. Print exact/predicted scales, RMSE, and RelL2 for each field."""
    model = create_model(DEFAULT_CONFIG)
    _load_available_best_checkpoint(model)
    xy = make_test_grid(grid_size=101)
    predicted, reference, rel_l2 = evaluate_model(model, xy, config=DEFAULT_CONFIG)

    for field in FIELD_NAMES:
        error = predicted[field] - reference[field]
        exact_max_abs = tf.reduce_max(tf.abs(reference[field]))
        pred_max_abs = tf.reduce_max(tf.abs(predicted[field]))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(error)))
        print(
            f"audit_field {field} "
            f"exact_max_abs={float(exact_max_abs.numpy()):.6e} "
            f"pred_max_abs={float(pred_max_abs.numpy()):.6e} "
            f"RMSE={float(rmse.numpy()):.6e} "
            f"RelL2={float(rel_l2[field].numpy()):.6e}"
        )
        tf.debugging.assert_all_finite(rel_l2[field], f"{field} RelL2 must be finite")


@pytest.mark.skipif(
    not DEFAULT_CONFIG.best_weights_path.exists(),
    reason="Trained best weights are not available for center-point audit.",
)
def test_d_trained_model_center_point_report() -> None:
    """D. Print predicted and exact w, Mx, My at the plate center."""
    model = create_model(DEFAULT_CONFIG)
    _load_available_best_checkpoint(model)
    xy = tf.constant([[0.5, 0.5]], dtype=tf.float32)
    predicted = model(xy, training=False)
    reference = analytical_fields(xy, DEFAULT_CONFIG.plate)

    for field in ("w", "Mx", "My"):
        print(
            f"audit_center {field} "
            f"pred={float(predicted[field].numpy()[0, 0]):.6e} "
            f"exact={float(reference[field].numpy()[0, 0]):.6e}"
        )
