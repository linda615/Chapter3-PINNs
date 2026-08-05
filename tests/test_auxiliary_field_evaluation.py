"""Tests for resultants reconstructed from the PINN deflection field."""

from __future__ import annotations

from dataclasses import replace

import tensorflow as tf

from experiments.clamped_rock_roof_nonuniform_support.config import DEFAULT_CONFIG
from experiments.clamped_rock_roof_nonuniform_support.evaluate_auxiliary_fields import (
    AUXILIARY_FIELD_NAMES,
    compute_auxiliary_fields_from_w,
    run_auxiliary_evaluation,
)
from experiments.clamped_rock_roof_nonuniform_support.problem import create_model
from experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference import (
    save_reference,
    solve_reference,
)
from physics.plate_residuals import PlateParameters


class PolynomialDeflectionModel(tf.keras.Model):
    """Analytic model with w=x^3+x*y^2 and unused zero mixed fields."""

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        del training
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        w = x * x * x + x * y * y
        zero = tf.zeros_like(w)
        return {
            "w": w,
            "Mx": zero,
            "My": zero,
            "Mxy": zero,
            "Qx": zero,
            "Qy": zero,
        }


def test_auxiliary_fields_from_w_match_closed_form() -> None:
    """Reconstructed resultants must follow the project sign convention."""
    xy = tf.constant(
        [[0.0, 0.0], [0.25, 0.75], [1.0, 0.5]],
        dtype=tf.float32,
    )
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    parameters = PlateParameters(D=2.0, nu=0.3)

    fields = compute_auxiliary_fields_from_w(
        PolynomialDeflectionModel(),
        xy,
        parameters,
    )

    tf.debugging.assert_near(fields["Mx_from_w"], -13.2 * x)
    tf.debugging.assert_near(fields["My_from_w"], -7.6 * x)
    tf.debugging.assert_near(fields["Mxy_from_w"], -2.8 * y)
    tf.debugging.assert_near(
        fields["Qx_from_w"],
        tf.fill((3, 1), -16.0),
    )
    tf.debugging.assert_near(
        fields["Qy_from_w"],
        tf.zeros((3, 1), dtype=tf.float32),
    )


def test_auxiliary_evaluation_saves_pointwise_and_metric_csv(tmp_path) -> None:
    """The complete helper should load weights and compare with a small FDM grid."""
    reference_path = tmp_path / "reference.npz"
    reference = solve_reference(grid_size=9, foundation_mode="none")
    save_reference(reference, reference_path, foundation_mode="none")
    config = replace(
        DEFAULT_CONFIG,
        foundation_enabled=False,
        hidden_width=4,
        hidden_depth=1,
        reference_path=reference_path,
        results_dir=tmp_path / "results",
    )
    checkpoint_path = config.best_fdm_validation_weights_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    model.save_weights(str(checkpoint_path))

    metrics = run_auxiliary_evaluation(
        checkpoint_path=checkpoint_path,
        output_csv=config.auxiliary_fields_csv_path,
        metrics_csv=config.auxiliary_metrics_csv_path,
        batch_size=17,
        config=config,
    )

    assert tuple(metrics) == AUXILIARY_FIELD_NAMES
    assert config.auxiliary_fields_csv_path.exists()
    assert config.auxiliary_metrics_csv_path.exists()
    for field in AUXILIARY_FIELD_NAMES:
        assert metrics[field]["direct_rel_l2"] >= 0.0
        assert metrics[field]["from_w_rel_l2"] >= 0.0
        assert metrics[field]["direct_vs_from_w_rel_l2"] >= 0.0
