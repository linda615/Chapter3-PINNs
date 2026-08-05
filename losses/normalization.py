"""Residual normalization settings for physics losses."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from physics.plate_residuals import PlateParameters


@dataclass(frozen=True)
class ResidualScales:
    """Positive scales used to nondimensionalize Kirchhoff residuals."""

    moment_x: float = 1.0
    moment_y: float = 1.0
    twisting_moment: float = 1.0
    shear_x: float = 1.0
    shear_y: float = 1.0
    equilibrium: float = 1.0

    def __post_init__(self) -> None:
        """Validate all residual scales."""
        for name, value in (
            ("moment_x", self.moment_x),
            ("moment_y", self.moment_y),
            ("twisting_moment", self.twisting_moment),
            ("shear_x", self.shear_x),
            ("shear_y", self.shear_y),
            ("equilibrium", self.equilibrium),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} scale must be finite and positive, got {value}.")


@dataclass(frozen=True)
class PhysicsComponentWeights:
    """Non-negative weights for moment and shear residual components."""

    moment_x: float = 1.0
    moment_y: float = 1.0
    twisting_moment: float = 1.0
    shear_x: float = 1.0
    shear_y: float = 1.0

    def __post_init__(self) -> None:
        """Validate all component weights."""
        for name, value in (
            ("moment_x", self.moment_x),
            ("moment_y", self.moment_y),
            ("twisting_moment", self.twisting_moment),
            ("shear_x", self.shear_x),
            ("shear_y", self.shear_y),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} component weight must be finite and non-negative, "
                    f"got {value}."
                )


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Options controlling how raw plate residuals are converted to losses."""

    normalize_residuals: bool = False
    residual_scales: ResidualScales = field(default_factory=ResidualScales)
    component_weights: PhysicsComponentWeights = field(
        default_factory=PhysicsComponentWeights
    )


def unit_square_sinusoidal_residual_scales(
    parameters: PlateParameters,
    load_amplitude: float = 1.0,
) -> ResidualScales:
    """Return analytical residual scales for q=A sin(pi x) sin(pi y) on [0, 1]^2."""
    if not math.isfinite(load_amplitude) or load_amplitude <= 0.0:
        raise ValueError(f"load_amplitude must be finite and positive, got {load_amplitude}.")

    pi = math.pi
    moment_scale = load_amplitude * (1.0 + parameters.nu) / (4.0 * pi**2)
    twisting_moment_scale = load_amplitude * (1.0 - parameters.nu) / (4.0 * pi**2)
    shear_scale = load_amplitude / (2.0 * pi)
    equilibrium_scale = load_amplitude
    return ResidualScales(
        moment_x=moment_scale,
        moment_y=moment_scale,
        twisting_moment=twisting_moment_scale,
        shear_x=shear_scale,
        shear_y=shear_scale,
        equilibrium=equilibrium_scale,
    )
