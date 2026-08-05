"""Training adapter for the multi-output fourth-order PINN."""

from __future__ import annotations

import tensorflow as tf

from boundary import BoundaryBatch
from losses import LossWeights, compute_mo4_total_loss, resolve_loss_weights

from .trainer import PINNTrainer


class MO4PINNTrainer(PINNTrainer):
    """Reuse standard training infrastructure with the MO4 loss closure."""

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: BoundaryBatch,
        loss_weights: LossWeights | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute the multi-output fourth-order total loss."""
        weights = (
            resolve_loss_weights(self.loss_weights, 1)
            if loss_weights is None
            else loss_weights
        )
        return compute_mo4_total_loss(
            model=self.model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=self.config.boundary_type,
            load_fn=self.load_fn,
            parameters=self.parameters,
            weights=weights,
            foundation_fn=self.foundation_fn,
            training=training,
        )
