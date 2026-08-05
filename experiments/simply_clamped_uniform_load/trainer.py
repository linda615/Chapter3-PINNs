"""Trainer adapter for edge-specific plate boundary conditions."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from boundary import BoundaryConditionBatch, BoundaryLossConfig
from losses import (
    LossWeights,
    compute_grouped_boundary_total_loss,
    resolve_loss_weights,
)
from trainer import PINNTrainer, compute_shared_gradient_diagnostics

from .config import ExperimentConfig
from .problem import sample_boundary_groups


class MixedBoundaryPINNTrainer(PINNTrainer):
    """Reuse the project trainer with grouped simple/clamped edge batches."""

    def __init__(
        self,
        *args: Any,
        experiment_config: ExperimentConfig,
        **kwargs: Any,
    ) -> None:
        """Create the base trainer and retain the edge assignment."""
        self.boundary_loss_config: BoundaryLossConfig | None = kwargs.pop(
            "boundary_loss_config",
            experiment_config.boundary_loss_config,
        )
        super().__init__(*args, **kwargs)
        self.experiment_config = experiment_config
        self._gradient_diagnostic_interior_xy = None
        self._gradient_diagnostic_boundary_groups = None
        if (
            experiment_config.gradient_diagnostics_enabled
            and (
                getattr(self.model, "shared_trunk", None) is not None
                or (
                    callable(getattr(self.model, "get_grouping", None))
                    and self.model.get_grouping() == "one"
                )
            )
        ):
            self._gradient_diagnostic_interior_xy = (
                self.interior_sampler.sample(
                    experiment_config.gradient_diagnostics_interior_points,
                    seed=experiment_config.gradient_diagnostics_seed,
                )
            )
            self._gradient_diagnostic_boundary_groups = (
                sample_boundary_groups(
                    sampler=self.boundary_sampler,
                    points_per_side=(
                        experiment_config
                        .gradient_diagnostics_boundary_points_per_side
                    ),
                    config=experiment_config,
                    seed=experiment_config.gradient_diagnostics_seed + 100000,
                )
            )

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: tuple[BoundaryConditionBatch, ...],
        loss_weights: LossWeights | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute physics once and dispatch each edge group independently."""
        weights = (
            resolve_loss_weights(self.loss_weights, 1)
            if loss_weights is None
            else loss_weights
        )
        return compute_grouped_boundary_total_loss(
            model=self.model,
            interior_xy=interior_xy,
            boundary_groups=boundary_batch,
            load_fn=self.load_fn,
            parameters=self.parameters,
            foundation_fn=self.foundation_fn,
            weights=weights,
            physics_loss_config=self.physics_loss_config,
            mxy_consistency_config=self.mxy_consistency_config,
            boundary_loss_config=self.boundary_loss_config,
            training=training,
        )

    def compute_log_diagnostics(
        self,
        epoch: int,
        interior_xy: tf.Tensor,
        boundary_batch: object,
        loss_weights: LossWeights,
    ) -> dict[str, tf.Tensor]:
        """Measure weighted loss competition on fixed shared-trunk points."""
        del epoch, interior_xy, boundary_batch
        if (
            self._gradient_diagnostic_interior_xy is None
            or self._gradient_diagnostic_boundary_groups is None
        ):
            return {}

        return compute_shared_gradient_diagnostics(
            model=self.model,
            loss_fn=lambda: self.compute_loss(
                interior_xy=self._gradient_diagnostic_interior_xy,
                boundary_batch=self._gradient_diagnostic_boundary_groups,
                loss_weights=loss_weights,
                training=False,
            ),
        )

    def sample_epoch(
        self,
        epoch: int,
    ) -> tuple[tf.Tensor, tuple[BoundaryConditionBatch, ...]]:
        """Sample interior points and the two edge-condition groups."""
        interior_seed = _step_seed(self.config.seed, epoch, offset=0)
        boundary_seed = _step_seed(
            self.config.seed,
            epoch,
            offset=100000,
        )
        interior_xy = self.interior_sampler.sample(
            self.config.interior_points,
            seed=interior_seed,
        )
        boundary_groups = sample_boundary_groups(
            sampler=self.boundary_sampler,
            points_per_side=self.config.boundary_points_per_side,
            config=self.experiment_config,
            seed=boundary_seed,
        )
        return interior_xy, boundary_groups

def _step_seed(
    base_seed: int | None,
    epoch: int,
    offset: int,
) -> int | None:
    """Create deterministic per-epoch seeds."""
    if base_seed is None:
        return None
    return base_seed + offset + epoch


__all__ = ["MixedBoundaryPINNTrainer"]
