"""Tests for W-PINN, MO4-PINN, and Mixed-PINN comparison pipelines."""

from dataclasses import replace

import tensorflow as tf

from boundary import BoundaryType
from experiments.simply_clamped_uniform_load.evaluate_mo4_pinn import (
    evaluate_mo4_model,
)
from experiments.simply_clamped_uniform_load.all_shared_mixed_config import (
    ALL_SHARED_MIXED_CONFIG,
)
from experiments.simply_clamped_uniform_load.evaluate_w_pinn import (
    evaluate_w_pinn_model,
)
from experiments.simply_clamped_uniform_load.mo4_config import MO4_PINN_CONFIG
from experiments.simply_clamped_uniform_load.problem import (
    create_boundary_sampler,
    create_model,
    sample_boundary_groups,
)
from experiments.simply_clamped_uniform_load.run_mo4_pinn import (
    run_experiment as run_mo4_experiment,
)
from experiments.simply_clamped_uniform_load.run_w_pinn import (
    run_experiment as run_w_experiment,
)
from experiments.simply_clamped_uniform_load.run import (
    run_experiment as run_mixed_experiment,
)
from experiments.simply_clamped_uniform_load.w_pinn_config import W_PINN_CONFIG
from experiments.simply_clamped_uniform_load.w_pinn_problem import (
    create_w_pinn_model,
)
from losses import compute_w_pinn_boundary_loss
from models import FIELD_NAMES
from physics import compute_w_pinn_derivatives
from trainer import SharedTrunkPCGradConfig


def test_w_pinn_uses_mixed_edge_hard_transform_and_physical_scale() -> None:
    """W-PINN should enforce w everywhere and slope only on clamped edges."""
    config = replace(W_PINN_CONFIG, hidden_width=8, hidden_depth=1)
    model = create_w_pinn_model(config)
    simple, clamped = sample_boundary_groups(
        create_boundary_sampler(),
        points_per_side=5,
        config=config,
        seed=401,
    )
    all_xy = tf.concat((simple.batch.xy, clamped.batch.xy), axis=0)
    tf.debugging.assert_near(
        model(all_xy)["w"],
        tf.zeros((20, 1), dtype=tf.float32),
        atol=1e-7,
    )
    derivatives = compute_w_pinn_derivatives(model, clamped.batch.xy)
    normal_slope = (
        clamped.batch.normals[:, 0:1] * derivatives["w_x"]
        + clamped.batch.normals[:, 1:2] * derivatives["w_y"]
    )
    tf.debugging.assert_near(
        normal_slope,
        tf.zeros_like(normal_slope),
        atol=1e-6,
    )
    simple_loss = compute_w_pinn_boundary_loss(
        model,
        simple.batch,
        BoundaryType.SIMPLE,
        config.plate,
    )
    assert simple_loss["moment_bc_loss"].shape == ()
    tf.debugging.assert_all_finite(simple_loss["moment_bc_loss"], "moment BC")
    assert model.output_scale == config.field_output_scales.w


def test_all_shared_mixed_is_one_64_by_5_network() -> None:
    """The strict baseline must not contain a trunk-plus-head decomposition."""
    model = create_model(ALL_SHARED_MIXED_CONFIG)
    outputs = model(tf.zeros((2, 2), dtype=tf.float32))

    assert model.get_grouping() == "one"
    assert model.shared_trunk is None
    assert len(model.subnets) == 1
    assert model.subnets[0] is model.net_all
    assert len(model.net_all.hidden_layers) == 5
    assert all(layer.units == 64 for layer in model.net_all.hidden_layers)
    assert model.net_all.output_layer.units == 6
    assert tuple(outputs) == FIELD_NAMES


def test_all_shared_mixed_pcgrad_covers_complete_network(tmp_path) -> None:
    """PCGrad should run without an explicit trunk in grouping-one mode."""
    config = replace(
        ALL_SHARED_MIXED_CONFIG,
        hidden_width=4,
        hidden_depth=1,
        shared_pcgrad=SharedTrunkPCGradConfig(enabled=True, start_epoch=1),
        gradient_diagnostics_enabled=False,
        analytical_modes=3,
        validation_interval=1,
        validation_grid_size=5,
        results_dir=tmp_path / "all_shared",
        training=replace(
            ALL_SHARED_MIXED_CONFIG.training,
            epochs=1,
            interior_points=8,
            boundary_points_per_side=2,
            log_every=1,
        ),
    )

    history = run_mixed_experiment(
        config,
        run_entry="tests.all_shared_mixed",
    )

    tf.debugging.assert_equal(history[0]["pcgrad_active"], 1.0)
    tf.debugging.assert_all_finite(
        history[0]["pcgrad_conflict_rate"],
        "PCGrad diagnostics must be finite.",
    )
    assert (config.results_dir / "runtime.json").is_file()
    assert history[0]["mean_train_step_seconds"].shape == ()


def test_variant_evaluators_return_six_column_fields() -> None:
    """Both comparison evaluators should expose the common six-field schema."""
    xy = tf.constant(
        ((0.2, 0.3), (0.5, 0.5), (0.8, 0.7)),
        dtype=tf.float32,
    )
    w_config = replace(W_PINN_CONFIG, hidden_width=8, hidden_depth=1, analytical_modes=4)
    predicted, reference, rel_l2 = evaluate_w_pinn_model(
        create_w_pinn_model(w_config),
        xy,
        w_config,
        batch_size=2,
    )
    mo4_config = replace(
        MO4_PINN_CONFIG,
        model_grouping="one",
        hidden_width=8,
        hidden_depth=1,
        use_shared_trunk=False,
        analytical_modes=4,
        shared_pcgrad=SharedTrunkPCGradConfig(enabled=False),
    )
    mo4 = evaluate_mo4_model(create_model(mo4_config), xy, mo4_config, batch_size=2)
    for name in FIELD_NAMES:
        assert predicted[name].shape == (3, 1)
        assert reference[name].shape == (3, 1)
        assert rel_l2[name].shape == ()
        assert mo4.direct[name].shape == (3, 1)
        assert mo4.from_w[name].shape == (3, 1)


def test_w_and_mo4_one_epoch_runs_save_separate_artifacts(tmp_path) -> None:
    """Tiny runs should train, validate, and keep checkpoints isolated."""
    common_training = replace(
        W_PINN_CONFIG.training,
        epochs=1,
        interior_points=4,
        boundary_points_per_side=2,
        log_every=1,
    )
    w_config = replace(
        W_PINN_CONFIG,
        hidden_width=8,
        hidden_depth=1,
        training=common_training,
        analytical_modes=3,
        validation_interval=1,
        validation_grid_size=5,
        results_dir=tmp_path / "w_pinn",
    )
    mo4_config = replace(
        MO4_PINN_CONFIG,
        model_grouping="one",
        hidden_width=8,
        hidden_depth=1,
        use_shared_trunk=False,
        shared_pcgrad=SharedTrunkPCGradConfig(enabled=False),
        gradient_diagnostics_enabled=False,
        training=replace(common_training, seed=MO4_PINN_CONFIG.training.seed),
        analytical_modes=3,
        validation_interval=1,
        validation_grid_size=5,
        results_dir=tmp_path / "mo4_pinn",
    )

    w_history = run_w_experiment(w_config)
    mo4_history = run_mo4_experiment(mo4_config)

    for history, config in ((w_history, w_config), (mo4_history, mo4_config)):
        assert len(history) == 1
        tf.debugging.assert_all_finite(history[0]["total_loss"], "total loss")
        assert config.best_validation_loss_weights_path.is_file()
        assert config.final_weights_path.is_file()
        assert config.loss_history_path.is_file()
        assert config.validation_history_path.is_file()
        assert (config.results_dir / "runtime.json").is_file()
        assert history[0]["mean_train_step_seconds"].shape == ()
