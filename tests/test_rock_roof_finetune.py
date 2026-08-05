"""Tests for resumed Adam and L-BFGS rock-roof fine tuning."""

from __future__ import annotations

from dataclasses import replace

import tensorflow as tf

from experiments.clamped_rock_roof_nonuniform_support.config import (
    DEFAULT_CONFIG,
    FineTuningConfig,
)
from experiments.clamped_rock_roof_nonuniform_support.finetune import (
    run_adam_finetuning,
    run_lbfgs_finetuning,
)
from experiments.clamped_rock_roof_nonuniform_support.problem import create_model
from experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference import (
    save_reference,
    solve_reference,
)
from losses import LossWeights
from trainer import LBFGSConfig, TrainingConfig


def test_adam_and_lbfgs_finetuning_use_separate_fdm_checkpoints(tmp_path) -> None:
    """Both fine-tuning stages must run from saved FDM-selected weights."""
    reference_path = tmp_path / "reference" / "rock_roof_reference.npz"
    reference = solve_reference(grid_size=9, foundation_mode="none")
    save_reference(reference, reference_path, foundation_mode="none")

    training = TrainingConfig(
        epochs=1,
        interior_points=8,
        boundary_points_per_side=2,
        boundary_type=DEFAULT_CONFIG.boundary_type,
        log_every=1,
        seed=151,
    )
    fine_tuning = FineTuningConfig(
        adam_epochs=1,
        adam_learning_rate=1e-5,
        adam_log_every=1,
        fdm_evaluation_every=1,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        loss_weights=LossWeights(
            moment=10.0,
            shear=5.0,
            equilibrium=1.0,
            boundary=1.0,
        ),
        lbfgs=LBFGSConfig(
            max_iterations=1,
            max_line_search_iterations=5,
            function_tolerance=1e-8,
            gradient_tolerance=1e-6,
            log_every=1,
            seed=152,
        ),
        lbfgs_fdm_evaluation_every=1,
    )
    config = replace(
        DEFAULT_CONFIG,
        foundation_enabled=False,
        hidden_width=4,
        hidden_depth=1,
        training=training,
        fine_tuning=fine_tuning,
        reference_path=reference_path,
        results_dir=tmp_path / "results",
        fdm_evaluation_batch_size=17,
    )

    initial_model = create_model(config)
    initial_model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    config.best_fdm_validation_weights_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    initial_model.save_weights(str(config.best_fdm_validation_weights_path))

    adam_result = run_adam_finetuning(config)

    assert adam_result.epochs_completed == 1
    assert len(adam_result.history) == 1
    assert config.best_fdm_validation_weights_path.exists()
    assert config.best_fdm_finetune_weights_path.exists()
    assert config.final_adam_finetune_weights_path.exists()
    assert config.adam_finetune_history_path.exists()
    assert config.adam_finetune_fdm_history_path.exists()
    assert tf.math.is_finite(adam_result.history[-1]["total_loss"])

    lbfgs_result = run_lbfgs_finetuning(config)

    assert lbfgs_result.iterations <= fine_tuning.lbfgs.max_iterations
    assert config.best_fdm_lbfgs_weights_path.exists()
    assert config.final_lbfgs_weights_path.exists()
    assert config.lbfgs_history_path.exists()
    assert config.lbfgs_fdm_history_path.exists()
    assert tf.math.is_finite(lbfgs_result.history[-1]["total_loss"])
