"""Configuration for the nonuniform-support clamped rock-roof example."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from losses import (
    BoundaryType,
    LossWeightSchedule,
    LossWeightSource,
    LossWeightStage,
    LossWeights,
    MxyConsistencyConfig,
    PhysicsComponentWeights,
    PhysicsLossConfig,
    ResidualScales,
)
from boundary import BoundaryLossConfig, BoundaryResidualScales
from models import FieldOutputScales
from physics.plate_residuals import PlateParameters
from trainer import LBFGSConfig, SharedTrunkPCGradConfig, TrainingConfig


EXPERIMENT_DIR = Path("experiments/clamped_rock_roof_nonuniform_support")
FOUNDATION_RESULTS_DIR = EXPERIMENT_DIR / "results_bounded_output_rms_loss"
NO_FOUNDATION_RESULTS_DIR = FOUNDATION_RESULTS_DIR / "no_foundation"
FOUNDATION_REFERENCE_PATH = EXPERIMENT_DIR / "reference" / "rock_roof_reference.npz"
NO_FOUNDATION_REFERENCE_PATH = (
    EXPERIMENT_DIR / "reference" / "rock_roof_reference_no_foundation.npz"
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


ROCK_ROOF_LOSS_WEIGHT_SCHEDULE = LossWeightSchedule(
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
class FineTuningConfig:
    """Adam and L-BFGS settings used after the initial PINN training."""

    adam_epochs: int = 1500
    adam_learning_rate: float = 3e-5
    adam_log_every: int = 100
    fdm_evaluation_every: int = 200
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    loss_weights: LossWeights = LossWeights(
        moment=1.0,
        shear=1.0,
        equilibrium=1.0,
        boundary=1.0,
        mxy_consistency=0.0,
    )
    lbfgs: LBFGSConfig = LBFGSConfig(
        max_iterations=500,
        log_every=20,
        seed=9031,
    )
    lbfgs_fdm_evaluation_every: int = 20

    def __post_init__(self) -> None:
        """Validate fine-tuning controls."""
        if self.adam_epochs <= 0:
            raise ValueError("adam_epochs must be positive.")
        if self.adam_learning_rate <= 0.0:
            raise ValueError("adam_learning_rate must be positive.")
        if self.adam_log_every <= 0:
            raise ValueError("adam_log_every must be positive.")
        if self.fdm_evaluation_every <= 0:
            raise ValueError("fdm_evaluation_every must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if self.lbfgs_fdm_evaluation_every <= 0:
            raise ValueError("lbfgs_fdm_evaluation_every must be positive.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Runnable settings for the engineering-equivalent rock-roof example."""

    plate: PlateParameters = PlateParameters(D=1.0, nu=0.25)
    boundary_type: BoundaryType = BoundaryType.CLAMPED
    foundation_enabled: bool = True
    foundation_min: float = 20.0
    foundation_max: float = 200.0
    foundation_transition_x: float = 0.45
    foundation_transition_width: float = 0.08
    load_background: float = 1.0
    load_gradient_x: float = 0.3
    load_local_amplitude: float = 0.8
    load_center_x: float = 0.68
    load_center_y: float = 0.42
    load_width_x: float = 0.18
    load_width_y: float = 0.15
    model_grouping: str = "six"
    hidden_width: int = 32
    hidden_depth: int = 3
    activation: str = "tanh"
    output_activation: str = "tanh"
    use_shared_trunk: bool = True
    shared_width: int = 64
    shared_depth: int = 2
    field_output_scales: FieldOutputScales = FieldOutputScales(
        # Four times the physics-derived characteristic amplitudes. The w
        # scale also compensates the full-clamped envelope maximum (1/256).
        w=0.62969522304338,
        Mx=0.110065241956486,
        My=0.110065241956486,
        Mxy=0.110065241956486,
        Qx=0.736258077137578,
        Qy=0.736258077137578,
    )
    hard_clamped_slope: bool = True
    learning_rate_schedule: LearningRateScheduleConfig = LearningRateScheduleConfig()
    fine_tuning: FineTuningConfig = FineTuningConfig()
    loss_weights: LossWeightSource = ROCK_ROOF_LOSS_WEIGHT_SCHEDULE
    mxy_consistency_config: MxyConsistencyConfig = MxyConsistencyConfig(
        enabled=False,
        minimum_scale=1e-3,
        stop_gradient_target=True,
    )
    physics_loss_config: PhysicsLossConfig = field(
        default_factory=lambda: PhysicsLossConfig(
            normalize_residuals=True,
            residual_scales=ResidualScales(
                moment_x=0.0275163104891215,
                moment_y=0.0275163104891215,
                twisting_moment=0.0275163104891215,
                shear_x=0.184064519284395,
                shear_y=0.184064519284395,
                equilibrium=1.2312605381012,
            ),
            component_weights=PhysicsComponentWeights(),
        )
    )
    boundary_loss_config: BoundaryLossConfig = BoundaryLossConfig(
        normalize_residuals=True,
        residual_scales=BoundaryResidualScales(
            deflection=0.000614936741253301,
            slope=0.00411348882379597,
            moment=0.0275163104891215,
            shear=0.184064519284395,
        ),
    )
    shared_pcgrad: SharedTrunkPCGradConfig = SharedTrunkPCGradConfig(
        enabled=True,
        start_epoch=5001,
    )
    seed: int = 2031
    monitor_grid_size: int = 41
    fdm_evaluation_every: int = 200
    fdm_evaluation_batch_size: int = 8192
    results_dir: Path = FOUNDATION_RESULTS_DIR
    reference_path: Path = FOUNDATION_REFERENCE_PATH
    training: TrainingConfig = TrainingConfig(
        epochs=20000,
        interior_points=2048,
        boundary_points_per_side=20,
        boundary_type=BoundaryType.CLAMPED,
        log_every=100,
        seed=2031,
    )

    def __post_init__(self) -> None:
        """Validate bounded-output and hard-boundary controls."""
        if self.output_activation not in ("linear", "tanh"):
            raise ValueError(
                "output_activation must be either 'linear' or 'tanh'."
            )
        if not isinstance(self.hard_clamped_slope, bool):
            raise ValueError("hard_clamped_slope must be a boolean.")

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for experiment checkpoints."""
        return self.results_dir / "checkpoints"

    @property
    def history_dir(self) -> Path:
        """Directory for training history files."""
        return self.results_dir / "history"

    @property
    def figures_dir(self) -> Path:
        """Directory for diagnostic figures."""
        return self.results_dir / "figures"

    @property
    def exports_dir(self) -> Path:
        """Directory for exported problem-definition data."""
        return self.results_dir / "exports"

    @property
    def best_fixed_physics_loss_weights_path(self) -> Path:
        """Checkpoint selected by fixed-monitor-grid physics loss."""
        return self.checkpoints_dir / "best_fixed_physics_loss.weights.h5"

    @property
    def best_fdm_validation_weights_path(self) -> Path:
        """Checkpoint selected by the periodic six-field FDM validation score."""
        return self.checkpoints_dir / "best_fdm_validation.weights.h5"

    @property
    def best_fdm_finetune_weights_path(self) -> Path:
        """Best FDM checkpoint produced by the resumed Adam stage."""
        return self.checkpoints_dir / "best_fdm_finetune.weights.h5"

    @property
    def final_adam_finetune_weights_path(self) -> Path:
        """Final, possibly early-stopped, resumed Adam weights."""
        return self.checkpoints_dir / "final_adam_finetune.weights.h5"

    @property
    def best_fdm_lbfgs_weights_path(self) -> Path:
        """Best FDM checkpoint produced during L-BFGS."""
        return self.checkpoints_dir / "best_fdm_lbfgs.weights.h5"

    @property
    def final_lbfgs_weights_path(self) -> Path:
        """Final weights returned by the L-BFGS optimizer."""
        return self.checkpoints_dir / (
            f"final_lbfgs_iter_{self.fine_tuning.lbfgs.max_iterations}.weights.h5"
        )

    @property
    def final_weights_path(self) -> Path:
        """Final Adam weight file for this experiment."""
        return self.checkpoints_dir / f"final_epoch_{self.training.epochs}.weights.h5"

    @property
    def loss_history_path(self) -> Path:
        """Training loss CSV path."""
        return self.history_dir / "loss.csv"

    @property
    def monitor_history_path(self) -> Path:
        """Fixed-monitor-grid physics history CSV path."""
        return self.history_dir / "fixed_monitor.csv"

    @property
    def fdm_validation_history_path(self) -> Path:
        """Periodic PINN-versus-FDM relative-L2 history CSV path."""
        return self.history_dir / "fdm_validation.csv"

    @property
    def adam_finetune_history_path(self) -> Path:
        """Resumed Adam loss history CSV path."""
        return self.history_dir / "loss_adam_finetune.csv"

    @property
    def adam_finetune_fdm_history_path(self) -> Path:
        """Resumed Adam FDM validation history CSV path."""
        return self.history_dir / "fdm_validation_adam_finetune.csv"

    @property
    def lbfgs_history_path(self) -> Path:
        """L-BFGS loss history CSV path."""
        return self.history_dir / "loss_lbfgs_finetune.csv"

    @property
    def lbfgs_fdm_history_path(self) -> Path:
        """L-BFGS FDM validation history CSV path."""
        return self.history_dir / "fdm_validation_lbfgs.csv"

    @property
    def evaluation_csv_path(self) -> Path:
        """Default evaluation prediction CSV path."""
        return self.results_dir / "evaluation_fields.csv"

    @property
    def relative_l2_csv_path(self) -> Path:
        """Default reference-comparison CSV path."""
        return self.results_dir / "evaluation_rel_l2.csv"

    @property
    def auxiliary_fields_csv_path(self) -> Path:
        """Pointwise direct, reconstructed, and FDM field comparison."""
        return self.results_dir / "auxiliary_fields_from_w.csv"

    @property
    def auxiliary_metrics_csv_path(self) -> Path:
        """Summary metrics for fields reconstructed from deflection."""
        return self.results_dir / "auxiliary_fields_from_w_metrics.csv"


DEFAULT_CONFIG = ExperimentConfig()
