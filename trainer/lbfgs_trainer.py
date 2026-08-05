"""SciPy L-BFGS fine-tuning utilities for Kirchhoff thin plate PINNs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
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


@dataclass(frozen=True)
class LBFGSConfig:
    """Configuration for the fixed-batch L-BFGS fine-tuning stage."""

    max_iterations: int = 500
    max_line_search_iterations: int = 50
    function_tolerance: float = 1e-12
    gradient_tolerance: float = 1e-8
    log_every: int = 20
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate L-BFGS settings."""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}.")
        if self.max_line_search_iterations <= 0:
            raise ValueError(
                "max_line_search_iterations must be positive, "
                f"got {self.max_line_search_iterations}."
            )
        if self.function_tolerance <= 0.0:
            raise ValueError(f"function_tolerance must be positive, got {self.function_tolerance}.")
        if self.gradient_tolerance <= 0.0:
            raise ValueError(f"gradient_tolerance must be positive, got {self.gradient_tolerance}.")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}.")


@dataclass
class LBFGSResult:
    """L-BFGS optimization result and logged loss snapshots."""

    history: list[dict[str, tf.Tensor]] = field(default_factory=list)
    success: bool = False
    message: str = ""
    iterations: int = 0
    function_evaluations: int = 0
    final_loss: float = float("nan")


class PINNLBFGSTrainer:
    """Fixed-collocation L-BFGS fine tuner for an already initialized PINN."""

    def __init__(
        self,
        model: tf.keras.Model,
        interior_sampler: RectangularInteriorSampler,
        boundary_sampler: RectangularBoundarySampler,
        load_fn: Callable[[tf.Tensor], tf.Tensor],
        parameters: PlateParameters,
        training_config: TrainingConfig,
        lbfgs_config: LBFGSConfig,
        loss_weights: LossWeightSource | None = None,
        foundation_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        physics_loss_config: PhysicsLossConfig | None = None,
        boundary_loss_config: BoundaryLossConfig | None = None,
        mxy_consistency_config: MxyConsistencyConfig | None = None,
        verbose: bool = False,
        callbacks: Sequence[Callable[[int, tf.keras.Model, dict[str, tf.Tensor]], None]] | None = None,
    ) -> None:
        """Create a fixed-batch L-BFGS trainer."""
        self.model = model
        self.interior_sampler = interior_sampler
        self.boundary_sampler = boundary_sampler
        self.load_fn = load_fn
        self.parameters = parameters
        self.training_config = training_config
        self.lbfgs_config = lbfgs_config
        self.loss_weights = loss_weights
        self.foundation_fn = foundation_fn
        self.physics_loss_config = physics_loss_config
        self.boundary_loss_config = boundary_loss_config
        self.mxy_consistency_config = mxy_consistency_config
        self.verbose = verbose
        self.callbacks = tuple(callbacks or ())
        self._iteration = 0

    def train(self) -> LBFGSResult:
        """Run L-BFGS on one fixed set of collocation points."""
        from scipy.optimize import minimize

        interior_xy, boundary_batch = self.sample_fixed_batch()
        self.model(interior_xy[:1], training=False)
        initial_weights = self._pack_trainable_variables()
        history: list[dict[str, tf.Tensor]] = []

        def objective(flat_weights: np.ndarray) -> tuple[float, np.ndarray]:
            return self._loss_and_gradient(flat_weights, interior_xy, boundary_batch)

        def callback(flat_weights: np.ndarray) -> None:
            self._iteration += 1
            self._assign_flat_weights(flat_weights)
            loss_dict = self.compute_loss(interior_xy, boundary_batch, training=False)
            for user_callback in self.callbacks:
                user_callback(self._iteration, self.model, loss_dict)
            if self._should_log(self._iteration):
                snapshot = _snapshot_losses(self._iteration, loss_dict)
                history.append(snapshot)
                if self.verbose:
                    print(_format_loss_line("lbfgs_iter", self._iteration, loss_dict))

        result = minimize(
            fun=objective,
            x0=initial_weights,
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": self.lbfgs_config.max_iterations,
                "maxls": self.lbfgs_config.max_line_search_iterations,
                "ftol": self.lbfgs_config.function_tolerance,
                "gtol": self.lbfgs_config.gradient_tolerance,
            },
        )
        self._assign_flat_weights(result.x)

        final_loss_dict = self.compute_loss(interior_xy, boundary_batch, training=False)
        if not history or int(history[-1]["iteration"].numpy()) != int(result.nit):
            history.append(_snapshot_losses(int(result.nit), final_loss_dict))

        return LBFGSResult(
            history=history,
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            final_loss=float(result.fun),
        )

    def sample_fixed_batch(self) -> tuple[tf.Tensor, BoundaryBatch]:
        """Sample the fixed collocation set used by L-BFGS."""
        seed = self.lbfgs_config.seed
        if seed is None:
            seed = self.training_config.seed
        interior_seed = _step_seed(seed, offset=0)
        boundary_seed = _step_seed(seed, offset=100000)
        interior_xy = self.interior_sampler.sample(
            self.training_config.interior_points,
            seed=interior_seed,
        )
        boundary_xy, boundary_normals = self.boundary_sampler.sample_all_sides(
            self.training_config.boundary_points_per_side,
            seed=boundary_seed,
        )
        return interior_xy, BoundaryBatch(xy=boundary_xy, normals=boundary_normals)

    def compute_loss(
        self,
        interior_xy: tf.Tensor,
        boundary_batch: BoundaryBatch,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Compute total loss on the fixed L-BFGS collocation set."""
        weights = resolve_loss_weights(self.loss_weights, self.training_config.epochs)
        loss_dict = compute_total_loss(
            model=self.model,
            interior_xy=interior_xy,
            boundary_batch=boundary_batch,
            boundary_type=self.training_config.boundary_type,
            load_fn=self.load_fn,
            parameters=self.parameters,
            foundation_fn=self.foundation_fn,
            weights=weights,
            physics_loss_config=self.physics_loss_config,
            boundary_loss_config=self.boundary_loss_config,
            mxy_consistency_config=self.mxy_consistency_config,
            training=training,
        )
        loss_dict.update(_weight_snapshot(weights))
        return loss_dict

    def _loss_and_gradient(
        self,
        flat_weights: np.ndarray,
        interior_xy: tf.Tensor,
        boundary_batch: BoundaryBatch,
    ) -> tuple[float, np.ndarray]:
        """Assign weights, then return scalar loss and flattened gradient."""
        self._assign_flat_weights(flat_weights)
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(interior_xy, boundary_batch, training=True)
            total_loss = loss_dict["total_loss"]

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        flat_gradient = self._pack_gradients(gradients)
        return float(total_loss.numpy()), flat_gradient

    def _pack_trainable_variables(self) -> np.ndarray:
        """Flatten model trainable variables into one NumPy vector."""
        values = [tf.reshape(variable, (-1,)).numpy() for variable in self.model.trainable_variables]
        if not values:
            raise ValueError("Model has no trainable variables for L-BFGS.")
        return np.concatenate(values).astype(np.float64)

    def _pack_gradients(self, gradients: Sequence[tf.Tensor | None]) -> np.ndarray:
        """Flatten gradients, replacing disconnected gradients with zeros."""
        values = []
        for gradient, variable in zip(gradients, self.model.trainable_variables):
            if gradient is None:
                gradient = tf.zeros_like(variable)
            values.append(tf.reshape(gradient, (-1,)).numpy())
        return np.concatenate(values).astype(np.float64)

    def _assign_flat_weights(self, flat_weights: np.ndarray) -> None:
        """Assign one flattened NumPy vector back into model variables."""
        offset = 0
        for variable in self.model.trainable_variables:
            size = int(tf.size(variable).numpy())
            values = flat_weights[offset : offset + size]
            reshaped = np.reshape(values, variable.shape)
            variable.assign(tf.cast(reshaped, variable.dtype))
            offset += size
        if offset != flat_weights.size:
            raise ValueError(
                f"Unexpected L-BFGS weight vector size {flat_weights.size}; consumed {offset}."
            )

    def _should_log(self, iteration: int) -> bool:
        """Return whether an L-BFGS iteration should be recorded."""
        return (
            iteration == 1
            or iteration == self.lbfgs_config.max_iterations
            or iteration % self.lbfgs_config.log_every == 0
        )


def _step_seed(base_seed: int | None, offset: int) -> int | None:
    """Create deterministic fixed-batch seeds."""
    if base_seed is None:
        return None
    return base_seed + offset


def _snapshot_losses(iteration: int, loss_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Copy one L-BFGS loss dictionary for CSV history."""
    snapshot = {"iteration": tf.constant(iteration, dtype=tf.int32)}
    snapshot.update({name: tf.identity(value) for name, value in loss_dict.items()})
    return snapshot


def _format_loss_line(prefix: str, iteration: int, loss_dict: dict[str, tf.Tensor]) -> str:
    """Format a compact L-BFGS loss line."""
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
        "equilibrium_loss",
        "boundary_loss",
    )
    parts = [f"{prefix}={iteration}"]
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
