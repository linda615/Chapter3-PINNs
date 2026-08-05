"""Configuration for the simply supported sinusoidal plate example."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from boundary import BoundaryLossConfig, BoundaryResidualScales
from losses import (
    BoundaryType,
    LossWeightSchedule,
    LossWeightSource,
    LossWeightStage,
    LossWeights,
    PhysicsLossConfig,
    ResidualScales,
    unit_square_sinusoidal_residual_scales,
)
from physics.plate_residuals import PlateParameters
from trainer import LBFGSConfig, TrainingConfig


SINUSOIDAL_ANALYTICAL_RMS_SCALES = ResidualScales(
    moment_x=(1.0 + 0.3) / (8.0 * math.pi**2),
    moment_y=(1.0 + 0.3) / (8.0 * math.pi**2),
    twisting_moment=(1.0 - 0.3) / (8.0 * math.pi**2),
    shear_x=1.0 / (4.0 * math.pi),
    shear_y=1.0 / (4.0 * math.pi),
    equilibrium=0.5,
)

# The exact simply supported boundary residuals vanish, so their RMS cannot
# provide a nonzero denominator. Use the corresponding exact field RMS values
# as characteristic boundary scales instead.
SINUSOIDAL_BOUNDARY_RMS_SCALES = BoundaryResidualScales(
    deflection=1.0 / (8.0 * math.pi**4),
    slope=1.0 / (8.0 * math.pi**3),
    moment=SINUSOIDAL_ANALYTICAL_RMS_SCALES.moment_x,
    shear=SINUSOIDAL_ANALYTICAL_RMS_SCALES.shear_x,
)


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    """Piecewise-constant Adam settings with fixed epoch transitions."""

    initial_learning_rate: float = 1e-3
    learning_rate_2: float = 5e-4
    learning_rate_3: float = 1e-4
    final_learning_rate: float = 3e-5
    first_decay_epoch: int = 3000
    second_decay_epoch: int = 7000
    third_decay_epoch: int = 12000

    def __post_init__(self) -> None:
        """Validate learning rates and fixed epoch transitions."""
        rates = (
            self.initial_learning_rate,
            self.learning_rate_2,
            self.learning_rate_3,
            self.final_learning_rate,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in rates):
            raise ValueError("All learning rates must be finite and positive.")
        if not (
            0
            < self.first_decay_epoch
            < self.second_decay_epoch
            < self.third_decay_epoch
        ):
            raise ValueError(
                "Expected 0 < first_decay_epoch < second_decay_epoch "
                "< third_decay_epoch."
            )


SINUSOIDAL_LOSS_WEIGHT_SCHEDULE = LossWeightSchedule(
    stages=(
        LossWeightStage(
            end_epoch=3000,
            weights=LossWeights(
                moment=1.0,
                shear=0.2,
                equilibrium=0.05,
                boundary=5.0,
                mxy_consistency=0.0,
            ),
        ),
        LossWeightStage(
            end_epoch=8000,
            weights=LossWeights(
                moment=1.0,
                shear=2.0,
                equilibrium=1.0,
                boundary=2.0,
                mxy_consistency=0.0,
            ),
        ),
        LossWeightStage(
            end_epoch=15000,
            weights=LossWeights(
                moment=1.0,
                shear=2.0,
                equilibrium=5.0,
                boundary=1.0,
                mxy_consistency=0.0,
            ),
        ),
    )
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal runnable experiment settings."""

    plate: PlateParameters = PlateParameters(D=1.0, nu=0.3)
    boundary_type: BoundaryType = BoundaryType.SIMPLE
    model_grouping: str = "six"
    hidden_width: int = 32
    hidden_depth: int = 3
    activation: str = "tanh"
    use_shared_trunk: bool = True
    shared_width: int = 64
    shared_depth: int = 2
    learning_rate: float = 1e-3
    learning_rate_schedule: LearningRateScheduleConfig = LearningRateScheduleConfig()
    validation_interval: int = 200
    validation_grid_size: int = 101
    normalize_physics_residuals: bool = True
    physics_residual_scales: ResidualScales | None = (
        SINUSOIDAL_ANALYTICAL_RMS_SCALES
    )
    boundary_loss_config: BoundaryLossConfig = BoundaryLossConfig(
        normalize_residuals=True,
        residual_scales=SINUSOIDAL_BOUNDARY_RMS_SCALES,
    )
    use_lbfgs: bool = False
    lbfgs: LBFGSConfig = LBFGSConfig(max_iterations=500, log_every=20, seed=2026)
    seed: int = 2026
    results_dir: Path = Path(
        "experiments/simply_supported_sinusoidal/"
        "results_six_physics_boundary_rms_normalized"
    )
    training: TrainingConfig = TrainingConfig(
        epochs=20000,
        interior_points=2048,
        boundary_points_per_side=20,
        boundary_type=BoundaryType.SIMPLE,
        log_every=100,
        seed=2026,
    )
    loss_weights: LossWeightSource = SINUSOIDAL_LOSS_WEIGHT_SCHEDULE

    @property
    def physics_loss_config(self) -> PhysicsLossConfig:
        """Return the selected raw or normalized physics-loss configuration."""
        scales = self.physics_residual_scales
        if scales is None:
            scales = unit_square_sinusoidal_residual_scales(self.plate)
        return PhysicsLossConfig(
            normalize_residuals=self.normalize_physics_residuals,
            residual_scales=scales,
        )

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for experiment checkpoints."""
        return self.results_dir / "checkpoints"

    @property
    def history_dir(self) -> Path:
        """Directory for loss history files."""
        return self.results_dir / "history"

    @property
    def figures_dir(self) -> Path:
        """Directory for generated figures."""
        return self.results_dir / "figures"

    @property
    def best_weights_path(self) -> Path:
        """Backward-compatible alias for the best validation checkpoint path."""
        return self.best_validation_loss_weights_path

    @property
    def best_total_loss_weights_path(self) -> Path:
        """Checkpoint path for the lowest training total loss."""
        return self.checkpoints_dir / "best_total_loss.weights.h5"

    @property
    def best_validation_loss_weights_path(self) -> Path:
        """Checkpoint path for the lowest validation score."""
        return self.checkpoints_dir / "best_validation_loss.weights.h5"

    @property
    def best_adam_validation_loss_weights_path(self) -> Path:
        """Checkpoint path for the Adam-stage lowest validation score."""
        return self.checkpoints_dir / "best_adam_validation_loss.weights.h5"

    @property
    def best_lbfgs_validation_loss_weights_path(self) -> Path:
        """Checkpoint path for the lowest validation score during L-BFGS."""
        return self.checkpoints_dir / "best_lbfgs_validation_loss.weights.h5"

    @property
    def final_lbfgs_weights_path(self) -> Path:
        """Final L-BFGS weight file for this experiment."""
        return self.checkpoints_dir / f"final_lbfgs_iter_{self.lbfgs.max_iterations}.weights.h5"

    @property
    def final_weights_path(self) -> Path:
        """Default final-epoch weight file for this experiment."""
        return self.checkpoints_dir / f"final_epoch_{self.training.epochs}.weights.h5"

    @property
    def validation_history_path(self) -> Path:
        """Default validation history CSV path."""
        return self.history_dir / "validation.csv"

    @property
    def loss_history_path(self) -> Path:
        """Default loss history CSV path."""
        return self.history_dir / "loss.csv"

    @property
    def lbfgs_history_path(self) -> Path:
        """Default L-BFGS history CSV path."""
        return self.history_dir / "loss_lbfgs.csv"

    @property
    def lbfgs_validation_history_path(self) -> Path:
        """Default L-BFGS validation history CSV path."""
        return self.history_dir / "validation_lbfgs.csv"


DEFAULT_CONFIG = ExperimentConfig()
