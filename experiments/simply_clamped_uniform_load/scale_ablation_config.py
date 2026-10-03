"""Scale-sensitivity configurations for the SCSC Mixed-6B experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

from boundary import BoundaryLossConfig, BoundaryResidualScales
from losses import PhysicsLossConfig, ResidualScales
from models import FieldOutputScales

from .config import DEFAULT_CONFIG, ExperimentConfig


VARIANT_UNIT_ALPHA_BETA = "unit_alpha_beta"
VARIANT_ALPHA_HALF = "alpha_half"
VARIANT_ALPHA_DOUBLE = "alpha_double"
VARIANT_ALPHA_QUADRUPLE = "alpha_quadruple"
SCALE_VARIANTS: Tuple[str, ...] = (
    VARIANT_UNIT_ALPHA_BETA,
    VARIANT_ALPHA_HALF,
    VARIANT_ALPHA_DOUBLE,
    VARIANT_ALPHA_QUADRUPLE,
)

DEFAULT_SCALE_ABLATION_ROOT = Path(
    "experiments/simply_clamped_uniform_load/results_scale_ablation"
)


def make_scale_ablation_config(
    variant: str,
    seed: int,
    root: Path = DEFAULT_SCALE_ABLATION_ROOT,
    pilot: bool = False,
) -> ExperimentConfig:
    """Return one SCSC scale ablation while preserving all other settings."""
    if variant not in SCALE_VARIANTS:
        raise ValueError(
            f"Unsupported scale variant {variant!r}; expected one of "
            f"{SCALE_VARIANTS}."
        )

    base = DEFAULT_CONFIG
    if variant == VARIANT_UNIT_ALPHA_BETA:
        field_scales = _dimensional_field_scales(base)
        physics_config = replace(
            base.physics_loss_config,
            residual_scales=_dimensional_physics_scales(base),
        )
        boundary_config = replace(
            base.boundary_loss_config,
            residual_scales=_dimensional_boundary_scales(base),
        )
    else:
        factors = {
            VARIANT_ALPHA_HALF: 0.5,
            VARIANT_ALPHA_DOUBLE: 2.0,
            VARIANT_ALPHA_QUADRUPLE: 4.0,
        }
        factor = factors[variant]
        field_scales = _multiply_field_scales(
            base.field_output_scales,
            factor,
        )
        physics_config = base.physics_loss_config
        boundary_config = base.boundary_loss_config

    run_group = "pilot" if pilot else "runs"
    return replace(
        base,
        seed=seed,
        training=replace(base.training, seed=seed),
        gradient_diagnostics_enabled=False,
        initial_checkpoint_path=None,
        results_dir=Path(root) / run_group / f"{variant}_seed{seed}",
        field_output_scales=field_scales,
        physics_loss_config=physics_config,
        boundary_loss_config=boundary_config,
    )


def scale_audit(config: ExperimentConfig, variant: str) -> Dict[str, object]:
    """Return explicit alpha and beta coefficients for one unit-square run."""
    dimensional_fields = _dimensional_field_scales(config)
    dimensional_physics = _dimensional_physics_scales(config)
    dimensional_boundary = _dimensional_boundary_scales(config)
    return {
        "variant": variant,
        "definition": _variant_definition(variant),
        "alpha": _ratios(
            config.field_output_scales.as_dict(),
            dimensional_fields.as_dict(),
        ),
        "beta_physics": _ratios(
            _physics_scale_dict(config.physics_loss_config),
            _physics_scale_dict(
                PhysicsLossConfig(
                    normalize_residuals=True,
                    residual_scales=dimensional_physics,
                )
            ),
        ),
        "beta_boundary": _ratios(
            _boundary_scale_dict(config.boundary_loss_config),
            _boundary_scale_dict(
                BoundaryLossConfig(
                    normalize_residuals=True,
                    residual_scales=dimensional_boundary,
                )
            ),
        ),
        "field_output_scales": config.field_output_scales.as_dict(),
        "physics_residual_scales": _physics_scale_dict(
            config.physics_loss_config
        ),
        "boundary_residual_scales": _boundary_scale_dict(
            config.boundary_loss_config
        ),
    }


def _dimensional_field_scales(config: ExperimentConfig) -> FieldOutputScales:
    load = abs(config.q0)
    length = 1.0
    bending = config.plate.D
    return FieldOutputScales(
        w=load * length**4 / bending,
        Mx=load * length**2,
        My=load * length**2,
        Mxy=load * length**2,
        Qx=load * length,
        Qy=load * length,
    )


def _dimensional_physics_scales(config: ExperimentConfig) -> ResidualScales:
    load = abs(config.q0)
    length = 1.0
    return ResidualScales(
        moment_x=load * length**2,
        moment_y=load * length**2,
        twisting_moment=load * length**2,
        shear_x=load * length,
        shear_y=load * length,
        equilibrium=load,
    )


def _dimensional_boundary_scales(
    config: ExperimentConfig,
) -> BoundaryResidualScales:
    load = abs(config.q0)
    length = 1.0
    bending = config.plate.D
    return BoundaryResidualScales(
        deflection=load * length**4 / bending,
        slope=load * length**3 / bending,
        moment=load * length**2,
        shear=load * length,
    )


def _multiply_field_scales(
    scales: FieldOutputScales,
    factor: float,
) -> FieldOutputScales:
    return FieldOutputScales(
        **{name: factor * value for name, value in scales.as_dict().items()}
    )


def _physics_scale_dict(config: PhysicsLossConfig) -> Dict[str, float]:
    scales = config.residual_scales
    return {
        "moment_x": scales.moment_x,
        "moment_y": scales.moment_y,
        "twisting_moment": scales.twisting_moment,
        "shear_x": scales.shear_x,
        "shear_y": scales.shear_y,
        "equilibrium": scales.equilibrium,
    }


def _boundary_scale_dict(config: BoundaryLossConfig) -> Dict[str, float]:
    scales = config.residual_scales
    return {
        "deflection": scales.deflection,
        "slope": scales.slope,
        "moment": scales.moment,
        "shear": scales.shear,
    }


def _ratios(
    values: Dict[str, float],
    dimensional_values: Dict[str, float],
) -> Dict[str, float]:
    return {
        name: values[name] / dimensional_values[name]
        for name in values
    }


def _variant_definition(variant: str) -> str:
    definitions = {
        VARIANT_UNIT_ALPHA_BETA: (
            "All alpha and beta coefficients equal one; dimensional scales "
            "only."
        ),
        VARIANT_ALPHA_HALF: (
            "All current field-output alpha coefficients multiplied by 0.5; "
            "current beta coefficients unchanged."
        ),
        VARIANT_ALPHA_DOUBLE: (
            "All current field-output alpha coefficients multiplied by 2; "
            "current beta coefficients unchanged."
        ),
        VARIANT_ALPHA_QUADRUPLE: (
            "All current field-output alpha coefficients multiplied by 4; "
            "current beta coefficients unchanged."
        ),
    }
    return definitions[variant]


__all__ = [
    "DEFAULT_SCALE_ABLATION_ROOT",
    "SCALE_VARIANTS",
    "VARIANT_ALPHA_DOUBLE",
    "VARIANT_ALPHA_HALF",
    "VARIANT_ALPHA_QUADRUPLE",
    "VARIANT_UNIT_ALPHA_BETA",
    "make_scale_ablation_config",
    "scale_audit",
]
