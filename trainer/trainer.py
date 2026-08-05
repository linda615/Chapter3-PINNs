"""Object-oriented trainer for Kirchhoff thin plate PINNs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time

import tensorflow as tf

from boundary import BoundaryBatch, BoundaryLossConfig
from losses import (
    LossWeightSource,
    LossWeights,
    MxyConsistencyConfig,
    PhysicsLossConfig,
    compute_total_loss,
    resolve_loss_weights,
)
from physics.plate_residuals import PlateParameters
from sampling import RectangularBoundarySampler, RectangularInteriorSampler

from .pinn_trainer import TrainingConfig
from .pcgrad import (
    BoundaryAnchoredPCGradConfig,
    SharedTrunkPCGradConfig,
    apply_shared_trunk_pcgrad,
    resolve_pcgrad_shared_variables,
)


@dataclass
class TrainerHistory:
    """Container for logged training loss snapshots."""

    records: list[dict[str, tf.Tensor]] = field(default_factory=list)

    def append(self, epoch: int, loss_dict: dict[str, tf.Tensor]) -> None:
        """Append one loss snapshot without converting tensors to NumPy."""
        snapshot = {"epoch": tf.constant(epoch, dtype=tf.int32)}
        snapshot.update({name: tf.identity(value) for name, value in loss_dict.items()})
        self.records.append(snapshot)


class PINNTrainer:
    """High-level trainer that wires samplers, losses, and optimizer together."""

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        interior_sampler: RectangularInteriorSampler,
        boundary_sampler: RectangularBoundarySampler,
        load_fn: Callable[[tf.Tensor], tf.Tensor],
        parameters: PlateParameters,
        config: TrainingConfig,
        loss_weights: LossWeightSource | None = None,
        foundation_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        physics_loss_config: PhysicsLossConfig | None = None,
        boundary_loss_config: BoundaryLossConfig | None = None,
        mxy_consistency_config: MxyConsistencyConfig | None = None,
        boundary_pcgrad_config: BoundaryAnchoredPCGradConfig | None = None,
        shared_pcgrad_config: SharedTrunkPCGradConfig | None = None,
        verbose: bool = False,
        callbacks: Sequence[Callable[[int, tf.keras.Model, dict[str, tf.Tensor]], None]] | None = None,
    ) -> None:
        """Create a trainer for residual-based PINN optimization."""
        self.model = model
        self.optimizer = optimizer
        self.interior_sampler = interior_sampler
        self.boundary_sampler = boundary_sampler
        self.load_fn = load_fn
        self.foundation_fn = foundation_fn
        self.parameters = parameters
        self.config = config
        self.loss_weights = loss_weights
        self.physics_loss_config = physics_loss_config
        self.boundary_loss_config = boundary_loss_config
        self.mxy_consistency_config = mxy_consistency_config
        if (
            boundary_pcgrad_config is not None
            and shared_pcgrad_config is not None
        ):
            raise ValueError(
                "Provide only shared_pcgrad_config; "
                "boundary_pcgrad_config is the legacy compatibility option."
            )
        self.shared_pcgrad_config = (
            shared_pcgrad_config
            if shared_pcgrad_config is not None
            else (
                SharedTrunkPCGradConfig()
                if boundary_pcgrad_config is None
                else SharedTrunkPCGradConfig(
                    enabled=boundary_pcgrad_config.enabled,
                    start_epoch=boundary_pcgrad_config.start_epoch,
                    epsilon=boundary_pcgrad_config.epsilon,
                )
            )
        )
        self.boundary_pcgrad_config = self.shared_pcgrad_config
        self.verbose = verbose
        self.callbacks = tuple(callbacks or ())

    def train(self) -> list[dict[str, tf.Tensor]]:
        """Run the configured training loop and return logged loss snapshots."""
        history = TrainerHistory()
        cumulative_train_step_seconds = 0.0
        interval_train_step_seconds = 0.0
        interval_epoch_count = 0

        for epoch in range(1, self.config.epochs + 1):
            step_started = time.perf_counter()
            interior_xy, boundary_batch = self.sample_epoch(epoch)
            current_weights = resolve_loss_weights(self.loss_weights, epoch)
            loss_dict = self.train_step(
                interior_xy,
                boundary_batch,
                loss_weights=current_weights,
                epoch=epoch,
            )
            step_seconds = time.perf_counter() - step_started
            cumulative_train_step_seconds += step_seconds
            interval_train_step_seconds += step_seconds
            interval_epoch_count += 1

            if self._should_log(epoch):
                loss_dict.update(
                    {
                        "mean_train_step_seconds": tf.constant(
                            interval_train_step_seconds
                            / interval_epoch_count,
                            dtype=tf.float64,
                        ),
                        "cumulative_train_step_seconds": tf.constant(
                            cumulative_train_step_seconds,
                            dtype=tf.float64,
                        ),
                    }
                )
                loss_dict.update(
                    self.compute_log_diagnostics(
                        epoch=epoch,
                        interior_xy=interior_xy,
                        boundary_batch=boundary_batch,
                        loss_weights=current_weights,
                    )
                )
                history.append(epoch, loss_dict)
                if self.verbose:
                    print(_format_loss_line(epoch, loss_dict))
                interval_train_step_seconds = 0.0
                interval_epoch_count = 0
            for callback in self.callbacks:
                callback(epoch, self.model, loss_dict)

        return history.records

    def train_step(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: BoundaryBatch,
        loss_weights: LossWeights | None = None,
        epoch: int = 1,
    ) -> dict[str, tf.Tensor]:
        """Run one gradient update from already sampled collocation points."""
        use_pcgrad = (
            self.shared_pcgrad_config.enabled
            and epoch >= self.shared_pcgrad_config.start_epoch
        )
        with tf.GradientTape(persistent=use_pcgrad) as tape:
            loss_dict = self.compute_loss(
                interior_xy=interior_xy,
                boundary_batch=boundary_batch,
                loss_weights=loss_weights,
                training=True,
            )

        variables = tuple(self.model.trainable_variables)
        gradients = tape.gradient(loss_dict["total_loss"], variables)
        if use_pcgrad:
            shared_variables = resolve_pcgrad_shared_variables(self.model)
            gradients, pcgrad_diagnostics = (
                apply_shared_trunk_pcgrad(
                    tape=tape,
                    loss_dict=loss_dict,
                    all_variables=variables,
                    shared_variables=shared_variables,
                    total_gradients=gradients,
                    config=self.shared_pcgrad_config,
                )
            )
            loss_dict.update(pcgrad_diagnostics)
            del tape
        if self.shared_pcgrad_config.enabled:
            loss_dict.update(
                {"pcgrad_active": tf.constant(float(use_pcgrad))}
            )
            if not use_pcgrad:
                loss_dict.update(_inactive_pcgrad_diagnostics())
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if not gradient_variable_pairs:
            raise ValueError("No gradients were produced for model trainable variables.")

        self.optimizer.apply_gradients(gradient_variable_pairs)
        loss_dict.update(_weight_snapshot(LossWeights() if loss_weights is None else loss_weights))
        return loss_dict

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: BoundaryBatch,
        loss_weights: LossWeights | None = None,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute the weighted total loss using the project loss API."""
        weights = resolve_loss_weights(self.loss_weights, 1) if loss_weights is None else loss_weights
        return compute_total_loss(
            model=self.model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=self.config.boundary_type,
            load_fn=self.load_fn,
            parameters=self.parameters,
            foundation_fn=self.foundation_fn,
            weights=weights,
            physics_loss_config=self.physics_loss_config,
            boundary_loss_config=self.boundary_loss_config,
            mxy_consistency_config=self.mxy_consistency_config,
            training=training,
        )

    def compute_log_diagnostics(
        self,
        epoch: int,
        interior_xy: tf.Tensor,
        boundary_batch: object,
        loss_weights: LossWeights,
    ) -> dict[str, tf.Tensor]:
        """Return optional diagnostics evaluated only at logged epochs."""
        del epoch, interior_xy, boundary_batch, loss_weights
        return {}

    def sample_epoch(self, epoch: int) -> tuple[tf.Tensor, BoundaryBatch]:
        """Sample interior and boundary collocation points for one epoch."""
        interior_seed = _step_seed(self.config.seed, epoch, offset=0)
        boundary_seed = _step_seed(self.config.seed, epoch, offset=100000)
        interior_xy = self.interior_sampler.sample(
            self.config.interior_points,
            seed=interior_seed,
        )
        boundary_xy, boundary_normals = self.boundary_sampler.sample_all_sides(
            self.config.boundary_points_per_side,
            seed=boundary_seed,
        )
        return interior_xy, BoundaryBatch(xy=boundary_xy, normals=boundary_normals)

    def _should_log(self, epoch: int) -> bool:
        """Return whether an epoch should be added to history."""
        return (
            epoch == 1
            or epoch == self.config.epochs
            or epoch % self.config.log_every == 0
        )


def _step_seed(base_seed: int | None, epoch: int, offset: int) -> int | None:
    """Create deterministic per-step seeds when a base seed is provided."""
    if base_seed is None:
        return None
    return base_seed + offset + epoch


def _format_loss_line(epoch: int, loss_dict: dict[str, tf.Tensor]) -> str:
    """Format a compact training loss line for console logging."""
    keys = (
        "moment_weight",
        "shear_weight",
        "equilibrium_weight",
        "boundary_weight",
        "mxy_consistency_weight",
        "moment_x_weight",
        "moment_y_weight",
        "twisting_moment_weight",
        "shear_x_weight",
        "shear_y_weight",
        "total_loss",
        "physics_loss",
        "weighted_moment_loss",
        "weighted_shear_loss",
        "weighted_equilibrium_loss",
        "weighted_boundary_loss",
        "weighted_mxy_consistency_loss",
        "moment_loss",
        "moment_x_loss",
        "moment_y_loss",
        "twisting_moment_loss",
        "weighted_moment_x_loss",
        "weighted_moment_y_loss",
        "weighted_twisting_moment_loss",
        "mxy_consistency_loss",
        "mxy_consistency_scale",
        "shear_loss",
        "shear_x_loss",
        "shear_y_loss",
        "weighted_shear_x_loss",
        "weighted_shear_y_loss",
        "fourth_order_loss",
        "equilibrium_loss",
        "boundary_loss",
        "deflection_bc_loss",
        "slope_bc_loss",
        "moment_bc_loss",
        "shear_bc_loss",
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
        "pcgrad_conflict_count",
        "pcgrad_active",
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
        "pcgrad_norm_moment_before",
        "pcgrad_norm_moment_after",
        "pcgrad_norm_shear_before",
        "pcgrad_norm_shear_after",
        "pcgrad_norm_equilibrium_before",
        "pcgrad_norm_equilibrium_after",
        "pcgrad_norm_boundary_before",
        "pcgrad_norm_boundary_after",
        "residual_scale_alpha",
        "residual_scale_moment_x",
        "residual_scale_moment_y",
        "residual_scale_twisting_moment",
        "residual_scale_shear_x",
        "residual_scale_shear_y",
        "residual_scale_equilibrium",
        "boundary_scale_slope",
        "boundary_scale_moment",
        "mean_train_step_seconds",
        "cumulative_train_step_seconds",
    )
    parts = [f"epoch={epoch}"]
    for key in keys:
        if key in loss_dict:
            parts.append(f"{key}={float(loss_dict[key].numpy()):.6e}")
    return " ".join(parts)


def _weight_snapshot(weights: LossWeights) -> dict[str, tf.Tensor]:
    """Return scalar tensors for logging the active loss weights."""
    return {
        "moment_weight": tf.constant(weights.moment, dtype=tf.float32),
        "shear_weight": tf.constant(weights.shear, dtype=tf.float32),
        "equilibrium_weight": tf.constant(weights.equilibrium, dtype=tf.float32),
        "boundary_weight": tf.constant(weights.boundary, dtype=tf.float32),
        "mxy_consistency_weight": tf.constant(
            weights.mxy_consistency,
            dtype=tf.float32,
        ),
    }


def _inactive_pcgrad_diagnostics() -> dict[str, tf.Tensor]:
    """Return a stable all-zero diagnostic schema before PCGrad starts."""
    zero = tf.constant(0.0, dtype=tf.float32)
    task_names = ("moment", "shear", "equilibrium", "boundary")
    diagnostics = {
        "pcgrad_conflict_count": zero,
        "pcgrad_conflict_rate": zero,
        "pcgrad_projection_count": zero,
    }
    for task_name in task_names:
        diagnostics[f"pcgrad_norm_{task_name}_before"] = zero
        diagnostics[f"pcgrad_norm_{task_name}_after"] = zero
    for left_index, left_name in enumerate(task_names):
        for right_name in task_names[left_index + 1:]:
            pair_name = f"{left_name}_{right_name}"
            diagnostics[f"pcgrad_conflict_{pair_name}"] = zero
            diagnostics[f"pcgrad_cos_{pair_name}_before"] = zero
            diagnostics[f"pcgrad_cos_{pair_name}_after"] = zero
    return diagnostics
