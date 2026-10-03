"""Tests for SCSC output/residual scale sensitivity configurations."""

import json
from pathlib import Path

import pytest

from experiments.simply_clamped_uniform_load.config import DEFAULT_CONFIG
from experiments.simply_clamped_uniform_load.scale_ablation_config import (
    SCALE_VARIANTS,
    VARIANT_ALPHA_DOUBLE,
    VARIANT_ALPHA_HALF,
    VARIANT_ALPHA_QUADRUPLE,
    VARIANT_UNIT_ALPHA_BETA,
    make_scale_ablation_config,
    scale_audit,
)
from experiments.simply_clamped_uniform_load.summarize_scale_ablation import (
    BASELINE_NAME,
    collect_summary_rows,
)


METRICS = ("w", "Mx", "My", "Mxy", "Qx", "Qy", "six_field_mean")


def test_unit_alpha_beta_uses_only_dimensional_scales() -> None:
    config = make_scale_ablation_config(VARIANT_UNIT_ALPHA_BETA, seed=2051)
    audit = scale_audit(config, VARIANT_UNIT_ALPHA_BETA)

    assert set(audit["alpha"].values()) == {1.0}
    assert set(audit["beta_physics"].values()) == {1.0}
    assert set(audit["beta_boundary"].values()) == {1.0}
    assert set(config.field_output_scales.as_dict().values()) == {1.0}


@pytest.mark.parametrize(
    ("variant", "factor"),
    (
        (VARIANT_ALPHA_HALF, 0.5),
        (VARIANT_ALPHA_DOUBLE, 2.0),
        (VARIANT_ALPHA_QUADRUPLE, 4.0),
    ),
)
def test_alpha_variants_scale_only_field_outputs(
    variant: str,
    factor: float,
) -> None:
    config = make_scale_ablation_config(variant, seed=2052)
    baseline = DEFAULT_CONFIG

    for name, baseline_value in baseline.field_output_scales.as_dict().items():
        assert config.field_output_scales.as_dict()[name] == pytest.approx(
            factor * baseline_value
        )
    assert (
        config.physics_loss_config.residual_scales
        == baseline.physics_loss_config.residual_scales
    )
    assert (
        config.boundary_loss_config.residual_scales
        == baseline.boundary_loss_config.residual_scales
    )


@pytest.mark.parametrize("variant", SCALE_VARIANTS)
def test_scale_variants_preserve_training_and_model_settings(variant: str) -> None:
    config = make_scale_ablation_config(variant, seed=2053)

    assert config.model_grouping == DEFAULT_CONFIG.model_grouping == "six"
    assert config.hidden_width == DEFAULT_CONFIG.hidden_width
    assert config.hidden_depth == DEFAULT_CONFIG.hidden_depth
    assert config.shared_width == DEFAULT_CONFIG.shared_width
    assert config.shared_depth == DEFAULT_CONFIG.shared_depth
    assert config.loss_weights == DEFAULT_CONFIG.loss_weights
    assert config.learning_rate_schedule == DEFAULT_CONFIG.learning_rate_schedule
    assert config.training.epochs == DEFAULT_CONFIG.training.epochs == 20000
    assert config.training.interior_points == DEFAULT_CONFIG.training.interior_points
    assert (
        config.training.boundary_points_per_side
        == DEFAULT_CONFIG.training.boundary_points_per_side
    )


def test_unknown_scale_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported scale variant"):
        make_scale_ablation_config("unknown", seed=2051)


def test_scale_summary_includes_baseline_and_scale_variants(
    tmp_path: Path,
) -> None:
    scale_root = tmp_path / "scale"
    for variant_index, variant in enumerate(SCALE_VARIANTS, start=1):
        for seed_index, seed in enumerate((2051, 2052, 2053), start=1):
            metrics = {
                name: float(variant_index + seed_index) for name in METRICS
            }
            result_path = (
                scale_root / "runs" / f"{variant}_seed{seed}" / "result.json"
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"best": metrics, "final": metrics}),
                encoding="utf-8",
            )

    baseline_summary = tmp_path / "baseline.csv"
    baseline_lines = [
        "model,checkpoint,field,mean_percent,sample_sd_percent"
    ]
    for checkpoint in ("best", "final"):
        for metric in METRICS:
            baseline_lines.append(
                f"Mixed-6B,{checkpoint},{metric},3.0,0.2"
            )
    baseline_summary.write_text(
        "\n".join(baseline_lines) + "\n",
        encoding="utf-8",
    )

    rows = collect_summary_rows(
        scale_root=scale_root,
        baseline_root=tmp_path / "missing_raw_baseline",
        baseline_summary=baseline_summary,
    )

    assert len(rows) == (1 + len(SCALE_VARIANTS)) * 2 * len(METRICS)
    baseline_rows = [row for row in rows if row["variant"] == BASELINE_NAME]
    assert len(baseline_rows) == 2 * len(METRICS)
    assert {row["runs"] for row in baseline_rows} == {3}
