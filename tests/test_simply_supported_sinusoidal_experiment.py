"""Tests for the simply supported sinusoidal plate experiment."""

import math

import pytest
import tensorflow as tf

from boundary import BoundaryBatch, BoundaryType, compute_boundary_loss
from losses import resolve_loss_weights
from experiments.simply_supported_sinusoidal.analytical import AnalyticalPlateModel
from experiments.simply_supported_sinusoidal.config import DEFAULT_CONFIG, ExperimentConfig
from experiments.simply_supported_sinusoidal.problem import (
    create_boundary_sampler,
    create_load_fn,
    create_model,
)
from experiments.simply_supported_sinusoidal.evaluate import (
    FIELD_NAMES,
    make_test_grid,
    run_evaluation,
)
from experiments.simply_supported_sinusoidal.run import (
    continue_training,
    create_learning_rate_schedule,
    run_lbfgs_finetuning,
    run_experiment,
)
from physics.plate_residuals import compute_plate_residuals
from trainer import LBFGSConfig, TrainingConfig


def test_analytical_solution_has_zero_physics_residuals() -> None:
    """Analytical fields should satisfy the current mixed-variable residuals."""
    model = AnalyticalPlateModel(DEFAULT_CONFIG.plate)
    xy = tf.constant(
        [
            [0.25, 0.25],
            [0.50, 0.50],
            [0.75, 0.25],
            [0.25, 0.75],
        ],
        dtype=tf.float32,
    )

    residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=create_load_fn(),
        parameters=DEFAULT_CONFIG.plate,
    )

    for residual in residuals.values():
        tf.debugging.assert_near(residual, tf.zeros_like(residual), atol=1e-5)


def test_analytical_solution_satisfies_simply_supported_edges() -> None:
    """Analytical solution should satisfy w = 0 and M_n = 0 on all edges."""
    model = AnalyticalPlateModel(DEFAULT_CONFIG.plate)
    boundary_xy, normals = create_boundary_sampler().sample_all_sides(4, seed=1)

    losses = compute_boundary_loss(
        model=model,
        batch=BoundaryBatch(xy=boundary_xy, normals=normals),
        boundary_type=BoundaryType.SIMPLE,
    )

    tf.debugging.assert_near(losses["deflection_bc_loss"], tf.constant(0.0), atol=1e-10)
    tf.debugging.assert_near(losses["moment_bc_loss"], tf.constant(0.0), atol=1e-10)


def test_hard_output_transform_sets_edge_deflection_to_zero() -> None:
    """The configured PINN hard transform should force w = 0 on all edges."""
    model = create_model(DEFAULT_CONFIG)
    boundary_xy, _ = create_boundary_sampler().sample_all_sides(4, seed=2)

    fields = model(boundary_xy, training=False)

    tf.debugging.assert_near(fields["w"], tf.zeros_like(fields["w"]), atol=1e-7)


def test_experiment_runs_one_training_step_with_finite_loss(tmp_path) -> None:
    """The experiment should run through the full training stack."""
    config = ExperimentConfig(
        results_dir=tmp_path,
        use_lbfgs=False,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        training=TrainingConfig(
            epochs=1,
            interior_points=8,
            boundary_points_per_side=2,
            boundary_type=BoundaryType.SIMPLE,
            log_every=1,
            seed=2026,
        )
    )

    history = run_experiment(config)

    assert len(history) == 1
    tf.debugging.assert_all_finite(history[0]["total_loss"], "total_loss must be finite")
    assert (tmp_path / "checkpoints" / "best_total_loss.weights.h5").exists()
    assert (tmp_path / "checkpoints" / "best_adam_validation_loss.weights.h5").exists()
    assert (tmp_path / "checkpoints" / "best_validation_loss.weights.h5").exists()
    assert (tmp_path / "checkpoints" / "best.weights.h5").exists()
    assert (tmp_path / "checkpoints" / "best_epoch_1.weights.h5").exists()
    assert (tmp_path / "checkpoints" / "final_epoch_1.weights.h5").exists()
    assert (tmp_path / "history" / "loss.csv").exists()
    assert (tmp_path / "history" / "validation.csv").exists()
    assert (tmp_path / "figures").is_dir()


def test_experiment_one_step_is_reproducible_with_same_seed(tmp_path) -> None:
    """A minimal experiment run should be repeatable when the same seeds are used."""
    training = TrainingConfig(
        epochs=1,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=BoundaryType.SIMPLE,
        log_every=1,
        seed=2026,
    )
    first_config = ExperimentConfig(
        results_dir=tmp_path / "first",
        training=training,
        seed=2026,
        use_lbfgs=False,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
    )
    second_config = ExperimentConfig(
        results_dir=tmp_path / "second",
        training=training,
        seed=2026,
        use_lbfgs=False,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
    )

    first_history = run_experiment(first_config)
    second_history = run_experiment(second_config)

    tf.debugging.assert_near(
        first_history[0]["total_loss"],
        second_history[0]["total_loss"],
        atol=1e-7,
    )


def test_continue_training_uses_existing_weights_and_minimum_lr(tmp_path) -> None:
    """Continue training should load current weights and save the next epoch file."""
    config = ExperimentConfig(
        results_dir=tmp_path,
        use_lbfgs=False,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        training=TrainingConfig(
            epochs=4,
            interior_points=8,
            boundary_points_per_side=2,
            boundary_type=BoundaryType.SIMPLE,
            log_every=1,
            seed=2026,
        ),
    )
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    source_weights = tmp_path / "checkpoints" / "best_epoch_4.weights.h5"
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(source_weights))

    history = continue_training(
        config=config,
        initial_weights_path=source_weights,
        extra_epochs=1,
    )

    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "best_epoch_5.weights.h5").exists()
    assert (tmp_path / "history" / "loss_continue_4_to_5.csv").exists()
    tf.debugging.assert_all_finite(history[0]["total_loss"], "continued loss must be finite")


def test_piecewise_learning_rate_schedule_uses_fixed_epochs() -> None:
    """Learning-rate transitions must not scale with total training epochs."""
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

    short_config = ExperimentConfig(
        training=TrainingConfig(
            epochs=10,
            interior_points=8,
            boundary_points_per_side=2,
            boundary_type=BoundaryType.SIMPLE,
            log_every=1,
            seed=2026,
        ),
        use_lbfgs=False,
    )
    short_schedule = create_learning_rate_schedule(short_config)
    tf.debugging.assert_near(
        short_schedule(tf.constant(settings.first_decay_epoch)),
        tf.constant(settings.learning_rate_2, dtype=tf.float32),
    )


def test_default_physics_loss_uses_analytical_rms_scales() -> None:
    """The default sinusoidal run should normalize all residuals by exact RMS scales."""
    config = DEFAULT_CONFIG.physics_loss_config
    scales = config.residual_scales

    assert config.normalize_residuals is True
    assert scales.moment_x == pytest.approx((1.0 + 0.3) / (8.0 * math.pi**2))
    assert scales.moment_y == pytest.approx((1.0 + 0.3) / (8.0 * math.pi**2))
    assert scales.twisting_moment == pytest.approx((1.0 - 0.3) / (8.0 * math.pi**2))
    assert scales.shear_x == pytest.approx(1.0 / (4.0 * math.pi))
    assert scales.shear_y == pytest.approx(1.0 / (4.0 * math.pi))
    assert scales.equilibrium == pytest.approx(0.5)


def test_default_boundary_loss_uses_characteristic_rms_scales() -> None:
    """The default run should normalize simple-edge residuals consistently."""
    config = DEFAULT_CONFIG.boundary_loss_config
    scales = config.residual_scales

    assert config.normalize_residuals is True
    assert scales.deflection == pytest.approx(1.0 / (8.0 * math.pi**4))
    assert scales.slope == pytest.approx(1.0 / (8.0 * math.pi**3))
    assert scales.moment == pytest.approx((1.0 + 0.3) / (8.0 * math.pi**2))
    assert scales.shear == pytest.approx(1.0 / (4.0 * math.pi))


def test_loss_weight_schedule_matches_reproduction_settings() -> None:
    """The sinusoidal reproduction run should keep unit loss weights."""
    expected = {
        1: (1.0, 1.0, 1.0, 1.0),
        1500: (1.0, 1.0, 1.0, 1.0),
        1501: (1.0, 1.0, 1.0, 1.0),
        3600: (1.0, 1.0, 1.0, 1.0),
        3601: (1.0, 1.0, 1.0, 1.0),
        6000: (1.0, 1.0, 1.0, 1.0),
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


def test_lbfgs_finetuning_loads_adam_best_and_saves_outputs(tmp_path) -> None:
    """L-BFGS fine tuning should start from Adam-best weights and save its outputs."""
    config = ExperimentConfig(
        results_dir=tmp_path,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        lbfgs=LBFGSConfig(max_iterations=1, log_every=1, seed=2026),
        training=TrainingConfig(
            epochs=1,
            interior_points=4,
            boundary_points_per_side=1,
            boundary_type=BoundaryType.SIMPLE,
            log_every=1,
            seed=2026,
        ),
    )
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    config.best_adam_validation_loss_weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(config.best_adam_validation_loss_weights_path))

    result = run_lbfgs_finetuning(
        config=config,
        initial_weights_path=config.best_adam_validation_loss_weights_path,
        initial_best_score=0.0,
    )

    assert result.iterations >= 0
    tf.debugging.assert_all_finite(tf.constant(result.final_loss), "L-BFGS final loss must be finite")
    assert config.final_lbfgs_weights_path.exists()
    assert config.lbfgs_history_path.exists()
    assert config.lbfgs_validation_history_path.exists()
    assert config.best_lbfgs_validation_loss_weights_path.exists()


def test_evaluate_loads_checkpoint_and_saves_csv(tmp_path) -> None:
    """Evaluation should load a best checkpoint, evaluate fields, and save CSV."""
    model = create_model(DEFAULT_CONFIG)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    weights_path = tmp_path / "checkpoints" / "best_epoch_1000.weights.h5"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))

    pointwise_csv = tmp_path / "fields.csv"
    rel_l2_csv = tmp_path / "rel_l2.csv"
    rel_l2 = run_evaluation(
        checkpoint_path=weights_path,
        pointwise_csv=pointwise_csv,
        rel_l2_csv=rel_l2_csv,
        grid_size=5,
        figures_dir=tmp_path / "figures",
        batch_size=7,
    )

    assert pointwise_csv.exists()
    assert rel_l2_csv.exists()
    assert set(rel_l2.keys()) == set(FIELD_NAMES)
    assert len(pointwise_csv.read_text(encoding="utf-8").splitlines()) == 26
    assert len(rel_l2_csv.read_text(encoding="utf-8").splitlines()) == 7
    assert (tmp_path / "figures" / "w.png").exists()
    assert (tmp_path / "figures" / "direct" / "w.png").exists()
    assert (tmp_path / "figures" / "from_w" / "w.png").exists()


def test_evaluate_accepts_legacy_best_weights_alias(tmp_path) -> None:
    """Evaluation should accept ``best_weights.h5`` as an alias for validation-best weights."""
    model = create_model(DEFAULT_CONFIG)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    saved_weights = tmp_path / "checkpoints" / "best_validation_loss.weights.h5"
    saved_weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(saved_weights))

    rel_l2 = run_evaluation(
        checkpoint_path=tmp_path / "checkpoints" / "best_weights.h5",
        pointwise_csv=tmp_path / "fields_alias.csv",
        rel_l2_csv=tmp_path / "rel_l2_alias.csv",
        grid_size=5,
    )

    assert set(rel_l2.keys()) == set(FIELD_NAMES)
    assert (tmp_path / "fields_alias.csv").exists()
    assert (tmp_path / "rel_l2_alias.csv").exists()


def test_make_test_grid_defaults_to_101_by_101_points() -> None:
    """The evaluation grid default should be 101 x 101 points."""
    xy = make_test_grid()

    assert xy.shape == (101 * 101, 2)
