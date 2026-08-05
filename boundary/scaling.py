"""Scale configuration for boundary-condition residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class BoundaryResidualScales:
    """Positive characteristic scales for boundary residual components."""

    deflection: float = 1.0
    slope: float = 1.0
    moment: float = 1.0
    shear: float = 1.0

    def __post_init__(self) -> None:
        """Require finite positive component scales."""
        for name, value in (
            ("deflection", self.deflection),
            ("slope", self.slope),
            ("moment", self.moment),
            ("shear", self.shear),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} boundary scale must be finite and positive, got {value}."
                )


@dataclass(frozen=True)
class BoundaryLossConfig:
    """Options controlling boundary-residual normalization."""

    normalize_residuals: bool = False
    residual_scales: BoundaryResidualScales = field(
        default_factory=BoundaryResidualScales
    )

