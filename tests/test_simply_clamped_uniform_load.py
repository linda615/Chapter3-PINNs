"""Tests for the opposite simple/clamped edge uniform-load experiment."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import tensorflow as tf

from boundary import (
    BoundaryType,
    compute_boundary_loss,
    compute_grouped_boundary_loss,
)
from experiments.simply_clamped_uniform_load.audit_network_capacity import (
    CapacityAuditConfig,
    run_capacity_audit,
)
from experiments.simply_clamped_uniform_load.analytical import (
    AnalyticalPlateModel,
    analytical_fields,
    levy_series_load,
)
from experiments.simply_clamped_uniform_load.config import (
    DEFAULT_CONFIG,
    LOAD_MODE_MATCHED_LEVY,
    LOAD_MODE_UNIFORM,
    ExperimentConfig,
)
from experiments.simply_clamped_uniform_load.evaluate import (
    run_full_evaluation,
)
from experiments.simply_clamped_uniform_load.finetune_blockwise import (
    BlockwiseFineTuneConfig,
    run_blockwise_finetuning,
)
from experiments.simply_clamped_uniform_load.problem import (
    create_boundary_sampler,
    create_load_fn,
    create_model,
    sample_boundary_groups,
)
from experiments.simply_clamped_uniform_load.run import (
    create_learning_rate_schedule,
    load_initial_weights,
    run_experiment,
)
from experiments.simply_clamped_uniform_load.run_adaptive import (
    AdaptiveLossConfig,
    AdaptiveRunConfig,
    run_adaptive_experiment,
)
from models import FIELD_NAMES, FieldOutputScales
from physics import compute_plate_residuals
from sampling import BoundarySide


def test_boundary_groups_assign_opposite_edges() -> None:
    """Vertical edges are simple and horizontal edges are clamped."""
    groups = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=7,
        config=DEFAULT_CONFIG,
        seed=31,
    )
    simple_group, clamped_group = groups

    assert simple_group.boundary_type == BoundaryType.SIMPLE
    assert clamped_group.boundary_type == BoundaryType.CLAMPED
    assert simple_group.batch.xy.shape == (14, 2)
    assert clamped_group.batch.xy.shape == (14, 2)

    simple_x = simple_group.batch.xy[:, 0]
    clamped_y = clamped_group.batch.xy[:, 1]
    tf.debugging.assert_equal(
        tf.logical_or(simple_x == 0.0, simple_x == 1.0),
        tf.ones_like(simple_x, dtype=tf.bool),
    )
    tf.debugging.assert_equal(
        tf.logical_or(clamped_y == 0.0, clamped_y == 1.0),
        tf.ones_like(clamped_y, dtype=tf.bool),
    )


def test_load_mode_selects_uniform_or_matched_levy_load() -> None:
    """The closure experiment must use the exact finite-series load."""
    xy = tf.constant(
        ((0.03, 0.2), (0.27, 0.5), (0.71, 0.8)),
        dtype=tf.float32,
    )
    uniform_config = replace(
        DEFAULT_CONFIG,
        load_mode=LOAD_MODE_UNIFORM,
        q0=1.7,
        analytical_modes=6,
    )
    matched_config = replace(
        uniform_config,
        load_mode=LOAD_MODE_MATCHED_LEVY,
    )

    tf.debugging.assert_near(
        create_load_fn(uniform_config)(xy),
        tf.fill((3, 1), tf.constant(1.7, dtype=tf.float32)),
    )
    tf.debugging.assert_near(
        create_load_fn(matched_config)(xy),
        levy_series_load(xy, q0=1.7, mode_count=6),
    )


def test_invalid_load_mode_is_rejected() -> None:
    """Configuration errors should fail before training starts."""
    with pytest.raises(ValueError, match="Unsupported load_mode"):
        ExperimentConfig(load_mode="not-a-load")


def test_experiment_model_uses_configured_physical_output_scales() -> None:
    """The formal PINN model restores physical units inside its call path."""
    model = create_model(DEFAULT_CONFIG)

    assert isinstance(DEFAULT_CONFIG.field_output_scales, FieldOutputScales)
    assert model.get_field_output_scales() == (
        DEFAULT_CONFIG.field_output_scales.as_dict()
    )
    assert model.output_activation == "tanh"

    normalized = model.predict_normalized_fields(
        tf.constant(((0.2, 0.3), (0.8, 0.7)), dtype=tf.float32)
    )
    for value in normalized.values():
        tf.debugging.assert_less_equal(value, 1.0)
        tf.debugging.assert_greater_equal(value, -1.0)


def test_experiment_uses_analytical_rms_residual_normalization() -> None:
    """Physics and soft boundary residuals use declared analytical RMS scales."""
    physics = DEFAULT_CONFIG.physics_loss_config
    boundary = DEFAULT_CONFIG.boundary_loss_config

    assert physics.normalize_residuals is True
    assert boundary.normalize_residuals is True
    assert physics.residual_scales.moment_x == pytest.approx(
        0.01328375181234
    )
    assert physics.residual_scales.moment_y == pytest.approx(
        0.02236658893387
    )
    assert physics.residual_scales.twisting_moment == pytest.approx(
        0.00677599537661
    )
    assert physics.residual_scales.shear_x == pytest.approx(
        0.08898336779952
    )
    assert physics.residual_scales.shear_y == pytest.approx(
        0.2109545493085
    )
    assert physics.residual_scales.equilibrium == pytest.approx(1.0)
    assert boundary.residual_scales.moment == pytest.approx(
        physics.residual_scales.moment_x
    )


def test_experiment_uses_fixed_epoch_four_stage_learning_rate() -> None:
    """The physical run follows the schedule validated by the capacity audit."""
    schedule = create_learning_rate_schedule(DEFAULT_CONFIG)
    settings = DEFAULT_CONFIG.learning_rate_schedule

    checks = (
        (0, settings.initial_learning_rate),
        (settings.first_decay_epoch - 1, settings.initial_learning_rate),
        (settings.first_decay_epoch, settings.learning_rate_2),
        (settings.second_decay_epoch, settings.learning_rate_3),
        (settings.third_decay_epoch, settings.final_learning_rate),
    )
    for step, expected in checks:
        assert float(schedule(tf.constant(step)).numpy()) == pytest.approx(expected)


def test_network_capacity_audit_saves_finite_supervised_metrics(
    tmp_path: Path,
) -> None:
    """A small audit should fit all six fields and save its artifacts."""
    results_dir = tmp_path / "capacity"
    experiment = replace(
        DEFAULT_CONFIG,
        model_grouping="six",
        hidden_width=4,
        hidden_depth=1,
        use_shared_trunk=True,
        shared_width=4,
        shared_depth=1,
        analytical_modes=4,
        results_dir=results_dir,
    )
    config = CapacityAuditConfig(
        experiment=experiment,
        epochs=4,
        training_points=16,
        validation_grid_size=5,
        validation_interval=1,
        log_every=1,
        learning_rate=1e-2,
        learning_rate_2=5e-3,
        learning_rate_3=1e-3,
        final_learning_rate=1e-4,
        first_decay_epoch=1,
        second_decay_epoch=2,
        third_decay_epoch=3,
        seed=113,
        results_dir=results_dir,
    )

    result = run_capacity_audit(config)

    assert len(result.history) == 5
    assert 0 <= result.best_epoch <= config.epochs
    assert result.best_validation_score >= 0.0
    actual_learning_rates = [
        float(record["learning_rate"].numpy())
        for record in result.history
    ]
    assert actual_learning_rates == pytest.approx(
        (1e-2, 1e-2, 5e-3, 1e-3, 1e-4)
    )
    for scale in result.field_scales.values():
        assert scale.shape == ()
        tf.debugging.assert_all_finite(scale, "field scale must be finite")
        tf.debugging.assert_positive(scale)
    for name, scale in result.output_scales.items():
        assert scale.shape == ()
        tf.debugging.assert_all_finite(scale, "output scale must be finite")
        tf.debugging.assert_positive(scale)
        if name != "w":
            if experiment.output_activation == "tanh":
                tf.debugging.assert_greater_equal(
                    scale,
                    result.field_scales[name],
                )
            else:
                tf.debugging.assert_near(scale, result.field_scales[name])
    tf.debugging.assert_greater(
        result.output_scales["w"],
        result.field_scales["w"],
    )
    for record in result.history:
        for name, value in record.items():
            assert value.shape == ()
            if name == "epoch":
                assert int(value.numpy()) >= 0
            else:
                tf.debugging.assert_all_finite(
                    value,
                    "audit metric must be finite",
                )
    assert config.best_weights_path.is_file()
    assert config.final_weights_path.is_file()
    assert config.history_path.is_file()
    assert config.field_scales_path.is_file()
    assert (config.results_dir / "config.json").is_file()


def test_levy_fields_satisfy_existing_plate_residuals() -> None:
    """The Levy fields obey the project's mixed-variable sign convention."""
    mode_count = 8
    config = replace(
        DEFAULT_CONFIG,
        analytical_modes=mode_count,
        load_mode=LOAD_MODE_MATCHED_LEVY,
    )
    xy = tf.constant(
        (
            (0.13, 0.21),
            (0.37, 0.44),
            (0.72, 0.63),
            (0.89, 0.82),
        ),
        dtype=tf.float32,
    )
    model = AnalyticalPlateModel(
        parameters=config.plate,
        q0=config.q0,
        mode_count=mode_count,
    )

    residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=create_load_fn(config),
        parameters=config.plate,
    )
    for residual in residuals.values():
        tf.debugging.assert_near(
            residual,
            tf.zeros_like(residual),
            atol=1e-5,
        )


def test_levy_fields_expose_truncated_load_mismatch() -> None:
    """The finite Levy reference does not exactly represent constant load."""
    mode_count = 8
    xy = tf.constant(
        (
            (0.013, 0.21),
            (0.19, 0.44),
            (0.57, 0.63),
            (0.91, 0.82),
        ),
        dtype=tf.float32,
    )
    model = AnalyticalPlateModel(
        parameters=DEFAULT_CONFIG.plate,
        q0=DEFAULT_CONFIG.q0,
        mode_count=mode_count,
    )
    training_load_fn = create_load_fn(DEFAULT_CONFIG)

    def truncated_load_fn(points: tf.Tensor) -> tf.Tensor:
        return levy_series_load(
            points,
            q0=DEFAULT_CONFIG.q0,
            mode_count=mode_count,
        )

    training_residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=training_load_fn,
        parameters=DEFAULT_CONFIG.plate,
    )
    truncated_residuals = compute_plate_residuals(
        model=model,
        xy=xy,
        load_fn=truncated_load_fn,
        parameters=DEFAULT_CONFIG.plate,
    )

    for name in (
        "moment_x",
        "moment_y",
        "twisting_moment",
        "shear_x",
        "shear_y",
    ):
        tf.debugging.assert_near(
            training_residuals[name],
            tf.zeros_like(training_residuals[name]),
            atol=1e-5,
        )

    expected_mismatch = training_load_fn(xy) - truncated_load_fn(xy)
    tf.debugging.assert_near(
        training_residuals["equilibrium"]
        - truncated_residuals["equilibrium"],
        expected_mismatch,
        atol=1e-6,
    )
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(training_residuals["equilibrium"])),
        tf.constant(1e-2, dtype=xy.dtype),
    )


def test_levy_solution_satisfies_both_boundary_types() -> None:
    """Grouped boundary loss dispatches simple and clamped formulas."""
    model = AnalyticalPlateModel(
        parameters=DEFAULT_CONFIG.plate,
        q0=DEFAULT_CONFIG.q0,
        mode_count=8,
    )
    groups = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=11,
        config=DEFAULT_CONFIG,
        seed=47,
    )

    losses = compute_grouped_boundary_loss(model, groups)

    for name in (
        "deflection_bc_loss",
        "slope_bc_loss",
        "moment_bc_loss",
        "boundary_loss",
    ):
        tf.debugging.assert_near(
            losses[name],
            tf.zeros_like(losses[name]),
            atol=1e-12,
        )
    tf.debugging.assert_equal(
        losses["shear_bc_loss"],
        tf.zeros_like(losses["shear_bc_loss"]),
    )


def test_hard_transform_enforces_deflection_and_simple_edge_moment() -> None:
    """Hard transforms fix w, clamped slope, and selected simple-edge moment."""
    tf.keras.utils.set_random_seed(59)
    model = create_model(
        replace(
            DEFAULT_CONFIG,
            model_grouping="one",
            hidden_width=8,
            hidden_depth=1,
            use_shared_trunk=False,
            hard_simple_moment=True,
        )
    )
    boundary_xy, _ = create_boundary_sampler().sample_all_sides(
        points_per_side=8,
        seed=53,
    )
    fields = model(boundary_xy, training=False)
    tf.debugging.assert_near(
        fields["w"],
        tf.zeros_like(fields["w"]),
        atol=1e-7,
    )
    simple_group, clamped_group = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=8,
        config=DEFAULT_CONFIG,
        seed=61,
    )
    simple_fields = model(simple_group.batch.xy, training=False)
    tf.debugging.assert_near(
        simple_fields["Mx"],
        tf.zeros_like(simple_fields["Mx"]),
        atol=1e-7,
    )
    simple_losses = compute_boundary_loss(
        model=model,
        batch=simple_group.batch,
        boundary_type=BoundaryType.SIMPLE,
    )
    tf.debugging.assert_near(
        simple_losses["moment_bc_loss"],
        tf.zeros_like(simple_losses["moment_bc_loss"]),
        atol=1e-12,
    )
    clamped_losses = compute_boundary_loss(
        model=model,
        batch=clamped_group.batch,
        boundary_type=BoundaryType.CLAMPED,
    )
    tf.debugging.assert_near(
        clamped_losses["slope_bc_loss"],
        tf.zeros_like(clamped_losses["slope_bc_loss"]),
        atol=1e-12,
    )


def test_hard_clamped_slope_transform_can_be_disabled() -> None:
    """The experiment retains a soft-slope ablation switch."""
    tf.keras.utils.set_random_seed(71)
    config = replace(
        DEFAULT_CONFIG,
        model_grouping="one",
        hidden_width=8,
        hidden_depth=1,
        use_shared_trunk=False,
        hard_clamped_slope=False,
    )
    model = create_model(config)
    _, clamped_group = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=8,
        config=config,
        seed=73,
    )

    losses = compute_boundary_loss(
        model=model,
        batch=clamped_group.batch,
        boundary_type=BoundaryType.CLAMPED,
    )

    assert float(losses["slope_bc_loss"].numpy()) > 0.0


def test_hard_moment_transform_can_be_disabled() -> None:
    """The experiment keeps an explicit switch for soft-moment ablations."""
    model = create_model(
        replace(
            DEFAULT_CONFIG,
            model_grouping="one",
            hidden_width=8,
            hidden_depth=1,
            use_shared_trunk=False,
            hard_simple_moment=False,
        )
    )

    assert model.uses_moment_transform() is False


def test_soft_moment_boundary_is_the_experiment_default() -> None:
    """The empirically more stable soft moment condition is the default."""
    model = create_model(
        replace(
            DEFAULT_CONFIG,
            model_grouping="one",
            hidden_width=8,
            hidden_depth=1,
            use_shared_trunk=False,
        )
    )

    assert DEFAULT_CONFIG.hard_simple_moment is False
    assert model.uses_moment_transform() is False


def test_hard_moment_transform_follows_horizontal_simple_edges() -> None:
    """Swapping edge types should hard-enforce My on horizontal edges."""
    config = replace(
        DEFAULT_CONFIG,
        simple_sides=(BoundarySide.BOTTOM, BoundarySide.TOP),
        clamped_sides=(BoundarySide.LEFT, BoundarySide.RIGHT),
        model_grouping="one",
        hidden_width=8,
        hidden_depth=1,
        use_shared_trunk=False,
        hard_simple_moment=True,
    )
    model = create_model(config)
    simple_group, _ = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=8,
        config=config,
        seed=67,
    )
    raw = model.predict_raw_fields(simple_group.batch.xy)
    normalized = model.predict_normalized_fields(simple_group.batch.xy)
    final = model.predict_fields(simple_group.batch.xy)

    tf.debugging.assert_near(
        final["My"],
        tf.zeros_like(final["My"]),
        atol=1e-7,
    )
    tf.debugging.assert_near(
        final["Mx"],
        normalized["Mx"] * config.field_output_scales.Mx,
    )
    tf.debugging.assert_near(normalized["Mx"], tf.tanh(raw["Mx"]))

    _, clamped_group = sample_boundary_groups(
        sampler=create_boundary_sampler(),
        points_per_side=8,
        config=config,
        seed=69,
    )
    clamped_losses = compute_boundary_loss(
        model=model,
        batch=clamped_group.batch,
        boundary_type=BoundaryType.CLAMPED,
    )
    tf.debugging.assert_near(
        clamped_losses["slope_bc_loss"],
        tf.zeros_like(clamped_losses["slope_bc_loss"]),
        atol=1e-12,
    )


def test_analytical_fields_are_complete_column_tensors() -> None:
    """The reference exposes the same six fields as the PINN."""
    xy = tf.constant(
        ((0.2, 0.3), (0.5, 0.5), (0.8, 0.7)),
        dtype=tf.float32,
    )
    fields = analytical_fields(
        xy,
        parameters=DEFAULT_CONFIG.plate,
        q0=DEFAULT_CONFIG.q0,
        mode_count=6,
    )

    assert tuple(fields) == ("w", "Mx", "My", "Mxy", "Qx", "Qy")
    for value in fields.values():
        assert value.shape == (3, 1)


def test_high_mode_levy_fields_are_finite() -> None:
    """The scaled Levy formulation remains finite at the N=40 default."""
    xy = tf.constant(
        (
            (0.0, 0.0),
            (0.01, 0.25),
            (0.5, 0.5),
            (0.99, 0.75),
            (1.0, 1.0),
        ),
        dtype=tf.float32,
    )
    fields = analytical_fields(
        xy,
        parameters=DEFAULT_CONFIG.plate,
        q0=DEFAULT_CONFIG.q0,
        mode_count=40,
    )

    for name, value in fields.items():
        tf.debugging.assert_all_finite(
            value,
            f"{name} must remain finite for 40 Levy modes.",
        )


def test_one_epoch_training_is_finite_and_saves_config(
    tmp_path: Path,
) -> None:
    """The complete grouped-boundary training path runs once."""
    training = replace(
        DEFAULT_CONFIG.training,
        epochs=1,
        interior_points=8,
        boundary_points_per_side=3,
        log_every=1,
    )
    config = replace(
        DEFAULT_CONFIG,
        model_grouping="three",
        hidden_width=4,
        hidden_depth=1,
        use_shared_trunk=True,
        shared_width=4,
        shared_depth=1,
        analytical_modes=4,
        validation_interval=1,
        validation_grid_size=7,
        gradient_diagnostics_interior_points=4,
        gradient_diagnostics_boundary_points_per_side=2,
        shared_pcgrad=replace(DEFAULT_CONFIG.shared_pcgrad, enabled=True),
        results_dir=tmp_path / "results",
        training=training,
    )

    history = run_experiment(config)

    assert len(history) == 1
    assert bool(tf.math.is_finite(history[0]["total_loss"]).numpy())
    gradient_keys = (
        "shared_grad_norm_moment",
        "shared_grad_norm_shear",
        "shared_grad_norm_equilibrium",
        "shared_grad_norm_boundary",
        "shared_grad_cos_moment_shear",
        "shared_grad_cos_moment_equilibrium",
        "shared_grad_cos_moment_boundary",
        "shared_grad_cos_shear_equilibrium",
        "shared_grad_cos_shear_boundary",
        "shared_grad_cos_equilibrium_boundary",
    )
    for key in gradient_keys:
        assert history[0][key].shape == ()
        tf.debugging.assert_all_finite(
            history[0][key],
            f"{key} must be finite.",
        )
    pcgrad_keys = (
        "pcgrad_conflict_count",
        "pcgrad_conflict_rate",
        "pcgrad_projection_count",
        "pcgrad_conflict_moment_shear",
        "pcgrad_conflict_moment_equilibrium",
        "pcgrad_conflict_moment_boundary",
        "pcgrad_conflict_shear_equilibrium",
        "pcgrad_conflict_shear_boundary",
        "pcgrad_conflict_equilibrium_boundary",
        "pcgrad_cos_moment_shear_before",
        "pcgrad_cos_moment_shear_after",
        "pcgrad_cos_moment_equilibrium_before",
        "pcgrad_cos_moment_equilibrium_after",
        "pcgrad_cos_moment_boundary_before",
        "pcgrad_cos_moment_boundary_after",
        "pcgrad_cos_shear_equilibrium_before",
        "pcgrad_cos_shear_equilibrium_after",
        "pcgrad_cos_shear_boundary_before",
        "pcgrad_cos_shear_boundary_after",
        "pcgrad_cos_equilibrium_boundary_before",
        "pcgrad_cos_equilibrium_boundary_after",
    )
    for key in pcgrad_keys:
        assert history[0][key].shape == ()
        tf.debugging.assert_all_finite(
            history[0][key],
            f"{key} must be finite.",
        )
    tf.debugging.assert_equal(history[0]["pcgrad_active"], 0.0)
    assert 0.0 <= float(
        history[0]["pcgrad_conflict_rate"].numpy()
    ) <= 1.0
    config_path = config.results_dir / "config.json"
    assert config_path.is_file()
    with config_path.open("r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    assert saved_config["simple_sides"] == [
        BoundarySide.LEFT.value,
        BoundarySide.RIGHT.value,
    ]
    assert saved_config["clamped_sides"] == [
        BoundarySide.BOTTOM.value,
        BoundarySide.TOP.value,
    ]
    assert saved_config["load_mode"] == config.load_mode
    assert saved_config["field_output_scales"] == (
        config.field_output_scales.as_dict()
    )
    assert saved_config["output_activation"] == "tanh"
    assert saved_config["hard_clamped_slope"] is True
    component_weights = config.physics_loss_config.component_weights
    assert saved_config["physics_loss_config"]["component_weights"] == {
        "moment_x": component_weights.moment_x,
        "moment_y": component_weights.moment_y,
        "twisting_moment": component_weights.twisting_moment,
        "shear_x": component_weights.shear_x,
        "shear_y": component_weights.shear_y,
    }


def test_initial_checkpoint_restores_model_weights(
    tmp_path: Path,
) -> None:
    """The experiment can start a new optimizer run from saved weights."""
    config = replace(
        DEFAULT_CONFIG,
        model_grouping="three",
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
    )
    xy = tf.constant(
        ((0.2, 0.3), (0.7, 0.8)),
        dtype=tf.float32,
    )
    tf.keras.utils.set_random_seed(701)
    source = create_model(config)
    expected = source(xy, training=False)
    checkpoint_path = tmp_path / "initial.weights.h5"
    source.save_weights(str(checkpoint_path))

    tf.keras.utils.set_random_seed(907)
    restored = create_model(config)
    load_initial_weights(restored, checkpoint_path)
    actual = restored(xy, training=False)

    for name in FIELD_NAMES:
        tf.debugging.assert_near(actual[name], expected[name])


def test_blockwise_finetuning_keeps_shared_trunk_frozen(
    tmp_path: Path,
) -> None:
    """Six-branch blockwise training runs all three ownership stages."""
    experiment = replace(
        DEFAULT_CONFIG,
        model_grouping="six",
        use_shared_trunk=True,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        analytical_modes=4,
        validation_grid_size=5,
        training=replace(
            DEFAULT_CONFIG.training,
            epochs=6,
            interior_points=8,
            boundary_points_per_side=3,
            log_every=1,
        ),
    )
    tf.keras.utils.set_random_seed(1009)
    source = create_model(experiment)
    source(tf.zeros((1, 2), dtype=tf.float32), training=False)
    initial_shared = [
        tf.identity(variable)
        for variable in source.shared_trunk.trainable_variables
    ]
    initial_moment = [
        tf.identity(variable)
        for layer in (source.net_Mx, source.net_My, source.net_Mxy)
        for variable in layer.trainable_variables
    ]
    initial_shear = [
        tf.identity(variable)
        for layer in (source.net_Qx, source.net_Qy)
        for variable in layer.trainable_variables
    ]
    initial_w = [
        tf.identity(variable)
        for variable in source.net_w.trainable_variables
    ]
    initial_checkpoint = tmp_path / "source.weights.h5"
    source.save_weights(str(initial_checkpoint))

    config = BlockwiseFineTuneConfig(
        initial_checkpoint_path=initial_checkpoint,
        results_dir=tmp_path / "blockwise",
        cycles=1,
        moment_epochs_per_cycle=1,
        shear_epochs_per_cycle=1,
        deflection_epochs_per_cycle=1,
        joint_epochs=1,
        validation_interval=1,
        log_every=1,
        experiment=experiment,
    )
    history = run_blockwise_finetuning(config)

    restored = create_model(experiment)
    restored(tf.zeros((1, 2), dtype=tf.float32), training=False)
    restored.load_weights(str(config.final_weights_path))

    assert len(history) == config.total_epochs
    assert any(
        bool(tf.reduce_any(tf.not_equal(before, after)).numpy())
        for before, after in zip(
            initial_shared,
            restored.shared_trunk.trainable_variables,
        )
    )
    assert any(
        bool(tf.reduce_any(tf.not_equal(before, after)).numpy())
        for before, after in zip(
            initial_w,
            restored.net_w.trainable_variables,
        )
    )
    assert any(
        bool(tf.reduce_any(tf.not_equal(before, after)).numpy())
        for before, after in zip(
          initial_moment,
          [
              variable
              for layer in (
                  restored.net_Mx,
                  restored.net_My,
                  restored.net_Mxy,
              )
              for variable in layer.trainable_variables
          ],
          )
      )
    assert any(
        bool(tf.reduce_any(tf.not_equal(before, after)).numpy())
        for before, after in zip(
            initial_shear,
            [
                variable
                for layer in (restored.net_Qx, restored.net_Qy)
                for variable in layer.trainable_variables
            ],
        )
    )
    for checkpoint_name in (
        "after_short_cycles.weights.h5",
        "after_joint.weights.h5",
        "best_validation_loss.weights.h5",
        "final_blockwise.weights.h5",
    ):
        assert (config.checkpoints_dir / checkpoint_name).is_file()
    assert config.loss_history_path.is_file()
    assert config.validation_history_path.is_file()
    assert config.block_history_path.is_file()
    assert (config.results_dir / "config.json").is_file()


def test_blockwise_finetuning_can_start_without_checkpoint(
    tmp_path: Path,
) -> None:
    """Short-cycle calibration requires a trained six-branch checkpoint."""
    experiment = replace(
        DEFAULT_CONFIG,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        analytical_modes=4,
        validation_grid_size=5,
        training=replace(
            DEFAULT_CONFIG.training,
            interior_points=8,
            boundary_points_per_side=3,
            log_every=1,
        ),
    )
    config = BlockwiseFineTuneConfig(
        initial_checkpoint_path=None,
        results_dir=tmp_path / "random_blockwise",
        cycles=1,
        moment_epochs_per_cycle=1,
        shear_epochs_per_cycle=1,
        deflection_epochs_per_cycle=1,
        joint_epochs=1,
        validation_interval=1,
        log_every=1,
        experiment=experiment,
    )

    with pytest.raises(ValueError, match="initial six-branch checkpoint"):
        run_blockwise_finetuning(config)


def test_adaptive_training_updates_finite_positive_weights(
    tmp_path: Path,
) -> None:
    """Shared-gradient balancing must update and persist bounded weights."""
    experiment = replace(
        DEFAULT_CONFIG,
        model_grouping="three",
        use_shared_trunk=True,
        hidden_width=4,
        hidden_depth=1,
        shared_width=4,
        shared_depth=1,
        analytical_modes=4,
        validation_grid_size=5,
        gradient_diagnostics_enabled=False,
        validation_interval=3,
        training=replace(
            DEFAULT_CONFIG.training,
            epochs=3,
            interior_points=8,
            boundary_points_per_side=3,
            log_every=1,
        ),
    )
    results_dir = tmp_path / "adaptive"
    history = run_adaptive_experiment(
        AdaptiveRunConfig(
            experiment=experiment,
            adaptive=AdaptiveLossConfig(
                warmup_epochs=0,
                update_every=1,
                ema_decay=0.0,
                min_weight=0.1,
                max_weight=10.0,
            ),
            results_dir=results_dir,
        )
    )

    assert len(history) == 3
    final_weights = tf.stack(
        [
            history[-1]["adaptive_moment_weight"],
            history[-1]["adaptive_shear_weight"],
            history[-1]["adaptive_equilibrium_weight"],
            history[-1]["adaptive_boundary_weight"],
        ]
    )
    tf.debugging.assert_all_finite(final_weights, "weights must be finite")
    tf.debugging.assert_greater(final_weights, tf.zeros_like(final_weights))
    assert bool(tf.reduce_any(tf.abs(final_weights - 1.0) > 1e-5).numpy())
    assert (
        results_dir / "checkpoints" / "final_epoch_3.weights.h5"
    ).is_file()
    assert (results_dir / "history" / "loss.csv").is_file()
    assert (results_dir / "history" / "validation.csv").is_file()


def test_full_evaluation_reports_direct_and_from_w_errors(
    tmp_path: Path,
) -> None:
    """Evaluation should save and return both direct and from-w fields."""
    config = replace(
        DEFAULT_CONFIG,
        hidden_width=8,
        hidden_depth=1,
        shared_width=8,
        shared_depth=1,
        analytical_modes=4,
        results_dir=tmp_path / "results",
    )
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    checkpoint_path = config.checkpoints_dir / "test.weights.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path))

    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        pointwise_csv=config.results_dir / "evaluation_fields.csv",
        rel_l2_csv=config.results_dir / "evaluation_rel_l2.csv",
        from_w_pointwise_csv=(
            config.results_dir / "evaluation_from_w_fields.csv"
        ),
        from_w_rel_l2_csv=(
            config.results_dir / "evaluation_from_w_rel_l2.csv"
        ),
        grid_size=5,
        config=config,
        batch_size=8,
    )

    assert (config.results_dir / "evaluation_fields.csv").exists()
    assert (config.results_dir / "evaluation_rel_l2.csv").exists()
    assert (config.results_dir / "evaluation_from_w_fields.csv").exists()
    assert (config.results_dir / "evaluation_from_w_rel_l2.csv").exists()
    for name in FIELD_NAMES:
        assert evaluation.direct[name].shape == (25, 1)
        assert evaluation.from_w[name].shape == (25, 1)
        assert evaluation.direct_rel_l2[name].shape == ()
        assert evaluation.from_w_rel_l2[name].shape == ()
        tf.debugging.assert_all_finite(
            evaluation.from_w_rel_l2[name],
            f"{name} from-w RelL2 must be finite.",
        )
    tf.debugging.assert_near(
        evaluation.direct["w"],
        evaluation.from_w["w"],
    )
