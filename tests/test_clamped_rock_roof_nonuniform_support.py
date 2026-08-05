"""Tests for the nonuniform-support clamped rock-roof example."""

from __future__ import annotations

from dataclasses import replace
import sys

import pytest
import tensorflow as tf

from boundary import BoundaryBatch, compute_clamped_boundary
from losses import resolve_loss_weights
from experiments.clamped_rock_roof_nonuniform_support.config import DEFAULT_CONFIG
from experiments.clamped_rock_roof_nonuniform_support.evaluate import (
    run_full_evaluation,
)
from experiments.clamped_rock_roof_nonuniform_support.problem import (
    active_foundation_stiffness,
    create_boundary_sampler,
    create_foundation_fn,
    create_model,
    foundation_stiffness,
    rock_roof_load,
)
from models import FIELD_NAMES
from physics.autodiff import deflection_derivatives
from trainer import TrainingConfig
from experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference import (
    save_reference,
    solve_reference,
)
from experiments.clamped_rock_roof_nonuniform_support.run import (
    FDMReferenceMonitor,
    _parse_args,
    create_learning_rate_schedule,
    run_experiment,
)


def test_load_and_foundation_shapes_and_values() -> None:
    """q and k must have shape (N, 1), finite values, positive q, and nonnegative k."""
    xy = tf.constant([[0.0, 0.0], [0.68, 0.42], [1.0, 1.0]], dtype=tf.float32)
    q = rock_roof_load(xy)
    k = foundation_stiffness(xy)

    assert q.shape == (3, 1)
    assert k.shape == (3, 1)
    assert q.dtype == xy.dtype
    assert k.dtype == xy.dtype
    tf.debugging.assert_all_finite(q, "q must be finite")
    tf.debugging.assert_all_finite(k, "k must be finite")
    tf.debugging.assert_greater(q, tf.zeros_like(q))
    tf.debugging.assert_greater_equal(k, tf.zeros_like(k))


def test_foundation_can_be_disabled_by_config() -> None:
    """The foundation function is optional and controlled by config."""
    disabled = replace(DEFAULT_CONFIG, foundation_enabled=False)
    enabled = replace(DEFAULT_CONFIG, foundation_enabled=True)

    assert create_foundation_fn(disabled) is None
    assert create_foundation_fn(enabled) is not None
    xy = tf.constant([[0.25, 0.5], [0.75, 0.5]], dtype=tf.float32)
    tf.debugging.assert_equal(
        active_foundation_stiffness(xy, config=disabled),
        tf.zeros((2, 1), dtype=tf.float32),
    )


def test_loss_weight_schedule_matches_uniform_load_stages() -> None:
    """The configured progressive loss stages must resolve by epoch."""
    expected = {
        1: (1.0, 0.2, 0.05, 5.0),
        3000: (1.0, 0.2, 0.05, 5.0),
        3001: (1.0, 2.0, 1.0, 2.0),
        8000: (1.0, 2.0, 1.0, 2.0),
        8001: (1.0, 2.0, 5.0, 1.0),
        20000: (1.0, 2.0, 5.0, 1.0),
    }
    for epoch, expected_values in expected.items():
        weights = resolve_loss_weights(DEFAULT_CONFIG.loss_weights, epoch)
        assert (
            weights.moment,
            weights.shear,
            weights.equilibrium,
            weights.boundary,
        ) == expected_values
        assert weights.mxy_consistency == 0.0
    assert not DEFAULT_CONFIG.mxy_consistency_config.enabled
    assert DEFAULT_CONFIG.physics_loss_config.normalize_residuals
    assert DEFAULT_CONFIG.boundary_loss_config.normalize_residuals
    assert DEFAULT_CONFIG.shared_pcgrad.enabled
    assert DEFAULT_CONFIG.shared_pcgrad.start_epoch == 5001


def test_learning_rate_schedule_uses_fixed_epoch_boundaries() -> None:
    """Learning-rate transitions must not scale with the requested epoch count."""
    schedule = create_learning_rate_schedule(DEFAULT_CONFIG)
    settings = DEFAULT_CONFIG.learning_rate_schedule

    tf.debugging.assert_near(
        schedule(tf.constant(settings.first_decay_epoch - 1)),
        tf.constant(settings.initial_learning_rate, dtype=tf.float32),
    )
    tf.debugging.assert_near(
        schedule(tf.constant(settings.first_decay_epoch)),
        tf.constant(settings.learning_rate_2, dtype=tf.float32),
    )
    tf.debugging.assert_near(
        schedule(tf.constant(settings.second_decay_epoch)),
        tf.constant(settings.learning_rate_3, dtype=tf.float32),
    )
    tf.debugging.assert_near(
        schedule(tf.constant(settings.third_decay_epoch)),
        tf.constant(settings.final_learning_rate, dtype=tf.float32),
    )

    short_config = replace(
        DEFAULT_CONFIG,
        training=replace(DEFAULT_CONFIG.training, epochs=10),
    )
    short_schedule = create_learning_rate_schedule(short_config)
    tf.debugging.assert_near(
        short_schedule(tf.constant(settings.first_decay_epoch)),
        tf.constant(settings.learning_rate_2, dtype=tf.float32),
    )


def test_hard_transform_enforces_w_and_normal_slope_on_all_edges() -> None:
    """The full-clamped transform exactly enforces both essential conditions."""
    model = create_model(DEFAULT_CONFIG)
    boundary_xy, normals = create_boundary_sampler().sample_all_sides(8, seed=123)
    _ = model(boundary_xy, training=False)

    derivatives = deflection_derivatives(model, boundary_xy, training=False)
    boundary_loss = compute_clamped_boundary(
        model,
        BoundaryBatch(xy=boundary_xy, normals=normals),
        training=True,
    )

    tf.debugging.assert_less(tf.reduce_max(tf.abs(derivatives["w"])), 1e-7)
    tf.debugging.assert_near(
        boundary_loss["slope_bc_loss"],
        tf.zeros_like(boundary_loss["slope_bc_loss"]),
        atol=1e-12,
    )
    tf.debugging.assert_near(
        boundary_loss["deflection_bc_loss"],
        tf.zeros_like(boundary_loss["deflection_bc_loss"]),
        atol=1e-12,
    )

    raw = model.call_raw(boundary_xy, training=False)
    normalized = model.predict_normalized_fields(boundary_xy)
    final = model(boundary_xy, training=False)
    for field in FIELD_NAMES:
        if field == "w":
            continue
        tf.debugging.assert_near(normalized[field], tf.tanh(raw[field]))
        tf.debugging.assert_near(
            final[field],
            normalized[field]
            * getattr(DEFAULT_CONFIG.field_output_scales, field),
        )
        tf.debugging.assert_less_equal(normalized[field], 1.0)
        tf.debugging.assert_greater_equal(normalized[field], -1.0)


def test_smoke_training_one_epoch_returns_finite_loss(tmp_path) -> None:
    """A tiny version of the engineering example can complete one train step."""
    training = TrainingConfig(
        epochs=1,
        interior_points=16,
        boundary_points_per_side=4,
        boundary_type=DEFAULT_CONFIG.boundary_type,
        log_every=1,
        seed=DEFAULT_CONFIG.seed,
    )
    config = replace(
        DEFAULT_CONFIG,
        foundation_enabled=False,
        training=training,
        monitor_grid_size=5,
        results_dir=tmp_path / "results",
    )

    history = run_experiment(config)

    assert len(history) == 1
    tf.debugging.assert_all_finite(history[-1]["total_loss"], "total_loss must be finite")
    tf.debugging.assert_all_finite(
        history[-1]["fixed_monitor_physics_loss"],
        "fixed monitor loss must be finite",
    )
    tf.debugging.assert_near(history[-1]["slope_bc_loss"], 0.0, atol=1e-12)
    tf.debugging.assert_equal(history[-1]["pcgrad_active"], 0.0)


def test_fdm_monitor_computes_six_relative_l2_errors(tmp_path) -> None:
    """Periodic validation must compare all PINN fields on the FDM grid."""
    reference_path = tmp_path / "rock_roof_reference.npz"
    result = solve_reference(grid_size=9, foundation_mode="none")
    save_reference(result, reference_path, foundation_mode="none")
    config = replace(
        DEFAULT_CONFIG,
        foundation_enabled=False,
        reference_path=reference_path,
        results_dir=tmp_path / "results",
        fdm_evaluation_every=200,
        fdm_evaluation_batch_size=17,
    )
    monitor = FDMReferenceMonitor(config=config, model=create_model(config))

    assert not monitor.should_evaluate(199)
    assert monitor.should_evaluate(200)
    snapshot = monitor.evaluate(200)

    assert snapshot["epoch"].shape == ()
    assert snapshot["fdm_validation_score"].shape == ()
    assert monitor.best_epoch == 200
    assert monitor.best_score == float(snapshot["fdm_validation_score"].numpy())
    assert config.best_fdm_validation_weights_path.exists()
    monitor.best_score = 0.0
    monitor.evaluate(400)
    assert monitor.best_epoch == 200
    assert monitor.best_score == 0.0
    tf.debugging.assert_all_finite(
        snapshot["fdm_validation_score"],
        "FDM validation score must be finite",
    )
    for field in FIELD_NAMES:
        value = snapshot[f"{field}_rel_l2"]
        assert value.shape == ()
        tf.debugging.assert_all_finite(value, f"{field} RelL2 must be finite")


def test_full_evaluation_reports_direct_and_from_w_errors(tmp_path) -> None:
    """Evaluation must export direct and deflection-derived FDM metrics."""
    reference_path = tmp_path / "rock_roof_reference.npz"
    result = solve_reference(grid_size=9, foundation_mode="smooth")
    save_reference(result, reference_path, foundation_mode="smooth")
    config = replace(
        DEFAULT_CONFIG,
        reference_path=reference_path,
        results_dir=tmp_path / "results",
    )
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    checkpoint = tmp_path / "checkpoint.weights.h5"
    model.save_weights(str(checkpoint))

    direct_csv = tmp_path / "evaluation_fields.csv"
    direct_rel_l2_csv = tmp_path / "evaluation_rel_l2.csv"
    from_w_csv = tmp_path / "evaluation_from_w_fields.csv"
    from_w_rel_l2_csv = tmp_path / "evaluation_from_w_rel_l2.csv"
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint,
        output_csv=direct_csv,
        rel_l2_csv=direct_rel_l2_csv,
        from_w_output_csv=from_w_csv,
        from_w_rel_l2_csv=from_w_rel_l2_csv,
        grid_size=5,
        config=config,
        batch_size=17,
    )

    assert evaluation is not None
    assert direct_csv.exists()
    assert direct_rel_l2_csv.exists()
    assert from_w_csv.exists()
    assert from_w_rel_l2_csv.exists()
    for field in FIELD_NAMES:
        assert evaluation.direct_rel_l2[field].shape == ()
        assert evaluation.from_w_rel_l2[field].shape == ()
        tf.debugging.assert_all_finite(
            evaluation.direct_rel_l2[field],
            f"direct {field} RelL2 must be finite",
        )
        tf.debugging.assert_all_finite(
            evaluation.from_w_rel_l2[field],
            f"from-w {field} RelL2 must be finite",
        )
    tf.debugging.assert_equal(evaluation.direct["w"], evaluation.from_w["w"])


def test_fdm_monitor_rejects_foundation_mismatch(tmp_path) -> None:
    """A smooth-foundation reference cannot validate no-foundation training."""
    reference_path = tmp_path / "rock_roof_reference.npz"
    result = solve_reference(grid_size=9, foundation_mode="smooth")
    save_reference(result, reference_path, foundation_mode="smooth")
    config = replace(
        DEFAULT_CONFIG,
        foundation_enabled=False,
        reference_path=reference_path,
    )

    with pytest.raises(ValueError, match="reference k does not match"):
        FDMReferenceMonitor(config=config, model=create_model(config))


def test_command_line_foundation_default_inherits_config(monkeypatch) -> None:
    """Omitting the flag must not silently disable the configured foundation."""
    monkeypatch.setattr(sys, "argv", ["run"])
    args = _parse_args()

    assert args.foundation_enabled is DEFAULT_CONFIG.foundation_enabled
