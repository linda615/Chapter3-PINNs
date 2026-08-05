"""Loss weighting configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Union


@dataclass(frozen=True)
class LossWeights:
    """Non-negative finite weights for total loss aggregation."""

    moment: float = 1.0
    shear: float = 1.0
    equilibrium: float = 1.0
    boundary: float = 1.0
    mxy_consistency: float = 0.0

    def __post_init__(self) -> None:
        """Validate that all weights are finite and non-negative."""
        for name, value in (
            ("moment", self.moment),
            ("shear", self.shear),
            ("equilibrium", self.equilibrium),
            ("boundary", self.boundary),
            ("mxy_consistency", self.mxy_consistency),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative, got {value}.")


@dataclass(frozen=True)
class LossWeightStage:
    """Loss weights used until and including ``end_epoch``."""

    end_epoch: int
    weights: LossWeights

    def __post_init__(self) -> None:
        """Validate one staged-loss segment."""
        if self.end_epoch <= 0:
            raise ValueError(f"end_epoch must be positive, got {self.end_epoch}.")
        if not isinstance(self.weights, LossWeights):
            raise TypeError("weights must be a LossWeights instance.")


@dataclass(frozen=True)
class LossWeightSchedule:
    """Piecewise-constant loss weights selected by epoch."""

    stages: tuple[LossWeightStage, ...]

    def __post_init__(self) -> None:
        """Validate that stages are non-empty and strictly ordered."""
        if not self.stages:
            raise ValueError("LossWeightSchedule requires at least one stage.")

        previous_end_epoch = 0
        for stage in self.stages:
            if not isinstance(stage, LossWeightStage):
                raise TypeError("All stages must be LossWeightStage instances.")
            if stage.end_epoch <= previous_end_epoch:
                raise ValueError("LossWeightSchedule stages must have increasing end_epoch values.")
            previous_end_epoch = stage.end_epoch

    def for_epoch(self, epoch: int) -> LossWeights:
        """Return the weights active at a 1-based training epoch."""
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}.")
        for stage in self.stages:
            if epoch <= stage.end_epoch:
                return stage.weights
        return self.stages[-1].weights


LossWeightSource = Union[LossWeights, LossWeightSchedule, Callable[[int], LossWeights]]


def resolve_loss_weights(
    source: LossWeightSource | None,
    epoch: int,
) -> LossWeights:
    """Resolve static, scheduled, callable, or missing loss weights."""
    if source is None:
        return LossWeights()
    if isinstance(source, LossWeights):
        return source
    if isinstance(source, LossWeightSchedule):
        return source.for_epoch(epoch)
    if callable(source):
        weights = source(epoch)
        if not isinstance(weights, LossWeights):
            raise TypeError("Callable loss weight source must return a LossWeights instance.")
        return weights
    raise TypeError(
        "loss_weights must be LossWeights, LossWeightSchedule, callable, or None."
    )
