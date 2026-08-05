"""Configuration for the mixed simply supported/clamped plate experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from losses import (
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
from physics.plate_residuals import PlateParameters
from models import FieldOutputScales
from sampling import BoundarySide
from trainer import SharedTrunkPCGradConfig


LOAD_MODE_UNIFORM = "uniform"
LOAD_MODE_MATCHED_LEVY = "matched_levy"
SUPPORTED_LOAD_MODES = (
    LOAD_MODE_UNIFORM,
    LOAD_MODE_MATCHED_LEVY,
)


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    """Four-stage piecewise-constant Adam settings."""

    initial_learning_rate: float = 1e-3
    learning_rate_2: float = 5e-4       # 稍温和一点的衰减
    learning_rate_3: float = 1e-4
    final_learning_rate: float = 3e-5

    # ---> 把触发降速的 Epoch 大幅拉长！
    first_decay_epoch: int  = 3000      # 3000 步前给足 1e-3 的动能冲透梯度屏障
    second_decay_epoch: int = 7000
    third_decay_epoch: int  = 12000

    def __post_init__(self) -> None:
        """Validate learning rates and transition epochs."""
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
                "Learning-rate decay epochs must be strictly increasing."
            )


@dataclass(frozen=True)
class MixedBoundaryTrainingConfig:
    """Training controls for an experiment with edge-specific conditions."""

    epochs: int = 20000
    interior_points: int = 2048
    boundary_points_per_side: int = 80
    log_every: int = 100
    seed: int | None = 2051

    def __post_init__(self) -> None:
        """Validate training controls."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.interior_points <= 0:
            raise ValueError("interior_points must be positive.")
        if self.boundary_points_per_side <= 0:
            raise ValueError("boundary_points_per_side must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


MIXED_EDGE_LOSS_WEIGHT_SCHEDULE = LossWeightSchedule(
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
    """Settings for a unit plate with two simple and two clamped edges."""

    plate: PlateParameters = PlateParameters(D=1.0, nu=0.3)
    q0: float = 1.0
    load_mode: str = LOAD_MODE_UNIFORM
    simple_sides: tuple[BoundarySide, ...] = (
        BoundarySide.LEFT,
        BoundarySide.RIGHT,
    )
    clamped_sides: tuple[BoundarySide, ...] = (
        BoundarySide.BOTTOM,
        BoundarySide.TOP,
    )
    model_grouping: str = "six"
    hidden_width: int = 32
    hidden_depth: int = 3
    activation: str = "tanh"
    output_activation: str = "tanh"
    use_shared_trunk: bool = True
    shared_width: int = 64
    shared_depth: int = 2
    field_output_scales: FieldOutputScales = FieldOutputScales(
        # Each scale is 1.1 times the analytical peak. For w, the peak is
        # divided by the maximum hard-envelope value (1/64).
        w=0.134966506938208,
        Mx=0.026826018075626,
        My=0.076821036344195,
        Mxy=0.01589473767002,
        Qx=0.35638489827633,
        Qy=0.56807355315425,
    )
    hard_clamped_slope: bool = True
    hard_simple_moment: bool = False
    learning_rate_schedule: LearningRateScheduleConfig = (
        LearningRateScheduleConfig()
    )
    analytical_modes: int = 40
    validation_interval: int = 200
    validation_grid_size: int = 101
    gradient_diagnostics_enabled: bool = True
    gradient_diagnostics_interior_points: int = 256
    gradient_diagnostics_boundary_points_per_side: int = 20
    gradient_diagnostics_seed: int = 12051
    shared_pcgrad: SharedTrunkPCGradConfig = (
        SharedTrunkPCGradConfig(enabled=True, start_epoch=5001)
    )
    seed: int = 2051
    results_dir: Path = Path(
        "experiments/simply_clamped_uniform_load/"
        "results_bounded_output_rms_loss_six"
    )
    initial_checkpoint_path: Path | None = None
    training: MixedBoundaryTrainingConfig = MixedBoundaryTrainingConfig()
    loss_weights: LossWeightSource = MIXED_EDGE_LOSS_WEIGHT_SCHEDULE
    physics_loss_config: PhysicsLossConfig = PhysicsLossConfig(
        normalize_residuals=True,
        residual_scales=ResidualScales(
            moment_x=0.01328375181234,
            moment_y=0.02236658893387,
            twisting_moment=0.00677599537661,
            shear_x=0.08898336779952,
            shear_y=0.2109545493085,
            equilibrium=1.0,
        ),
        component_weights=PhysicsComponentWeights(
            moment_x=1.0,
            moment_y=1.0,
            twisting_moment=1.0,
            shear_x=1.0,
            shear_y=1.0,
        ),
    )
    boundary_loss_config: BoundaryLossConfig = BoundaryLossConfig(
        normalize_residuals=True,
        residual_scales=BoundaryResidualScales(
            deflection=0.000889362046453,
            slope=0.000889362046453,
            moment=0.01328375181234,
            shear=0.08898336779952,
        ),
    )
    mxy_consistency_config: MxyConsistencyConfig = MxyConsistencyConfig(
        enabled=False,
        minimum_scale=1e-3,
        stop_gradient_target=True,
    )

    def __post_init__(self) -> None:
        """Validate physical parameters and the four-edge partition."""
        if not math.isfinite(self.q0):
            raise ValueError(f"q0 must be finite, got {self.q0}.")
        if self.load_mode not in SUPPORTED_LOAD_MODES:
            raise ValueError(
                f"Unsupported load_mode {self.load_mode!r}; expected one of "
                f"{SUPPORTED_LOAD_MODES}."
            )
        if self.analytical_modes <= 0:
            raise ValueError("analytical_modes must be positive.")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")
        if self.validation_grid_size <= 1:
            raise ValueError("validation_grid_size must be greater than 1.")
        if self.output_activation not in ("linear", "tanh"):
            raise ValueError(
                "output_activation must be either 'linear' or 'tanh'."
            )
        if not isinstance(self.gradient_diagnostics_enabled, bool):
            raise ValueError("gradient_diagnostics_enabled must be a boolean.")
        if self.gradient_diagnostics_interior_points <= 0:
            raise ValueError(
                "gradient_diagnostics_interior_points must be positive."
            )
        if self.gradient_diagnostics_boundary_points_per_side <= 0:
            raise ValueError(
                "gradient_diagnostics_boundary_points_per_side must be "
                "positive."
            )
        if not isinstance(self.hard_simple_moment, bool):
            raise ValueError("hard_simple_moment must be a boolean.")
        if not isinstance(self.hard_clamped_slope, bool):
            raise ValueError("hard_clamped_slope must be a boolean.")

        simple = tuple(BoundarySide(side) for side in self.simple_sides)
        clamped = tuple(BoundarySide(side) for side in self.clamped_sides)
        if len(simple) != 2 or len(clamped) != 2:
            raise ValueError(
                "Exactly two simple sides and two clamped sides are required."
            )
        if set(simple).intersection(clamped):
            raise ValueError("Simple and clamped side sets must not overlap.")
        if set(simple).union(clamped) != set(BoundarySide):
            raise ValueError(
                "Simple and clamped sides must cover all rectangle edges."
            )
        if set(simple) not in (
            {BoundarySide.LEFT, BoundarySide.RIGHT},
            {BoundarySide.BOTTOM, BoundarySide.TOP},
        ):
            raise ValueError("The simply supported sides must be opposite.")
        if set(clamped) not in (
            {BoundarySide.LEFT, BoundarySide.RIGHT},
            {BoundarySide.BOTTOM, BoundarySide.TOP},
        ):
            raise ValueError("The clamped sides must be opposite.")

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model checkpoints."""
        return self.results_dir / "checkpoints"

    @property
    def boundary_pcgrad(self) -> SharedTrunkPCGradConfig:
        """Return shared PCGrad under the legacy configuration name."""
        return self.shared_pcgrad

    @property
    def history_dir(self) -> Path:
        """Directory for training histories."""
        return self.results_dir / "history"

    @property
    def figures_dir(self) -> Path:
        """Directory for evaluation figures."""
        return self.results_dir / "figures"

    @property
    def best_validation_loss_weights_path(self) -> Path:
        """Checkpoint selected by analytical six-field validation."""
        return self.checkpoints_dir / "best_validation_loss.weights.h5"

    @property
    def final_weights_path(self) -> Path:
        """Final Adam checkpoint path."""
        return self.checkpoints_dir / (
            f"final_epoch_{self.training.epochs}.weights.h5"
        )

    @property
    def loss_history_path(self) -> Path:
        """Training loss CSV path."""
        return self.history_dir / "loss.csv"

    @property
    def validation_history_path(self) -> Path:
        """Analytical validation history CSV path."""
        return self.history_dir / "validation.csv"


DEFAULT_CONFIG = ExperimentConfig()
