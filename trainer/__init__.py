"""Training utilities for Kirchhoff thin plate PINNs."""

from .gradient_diagnostics import compute_shared_gradient_diagnostics
from .lbfgs_trainer import LBFGSConfig, LBFGSResult, PINNLBFGSTrainer
from .pcgrad import (
    BoundaryAnchoredPCGradConfig,
    SharedTrunkPCGradConfig,
)
from .pinn_trainer import TrainingConfig, train_pinn, train_step
from .trainer import PINNTrainer, TrainerHistory
from .w_pinn_trainer import WPINNTrainer
from .mo4_trainer import MO4PINNTrainer

__all__ = [
    "LBFGSConfig",
    "LBFGSResult",
    "BoundaryAnchoredPCGradConfig",
    "SharedTrunkPCGradConfig",
    "PINNTrainer",
    "PINNLBFGSTrainer",
    "TrainingConfig",
    "TrainerHistory",
    "compute_shared_gradient_diagnostics",
    "train_pinn",
    "train_step",
    "WPINNTrainer",
    "MO4PINNTrainer",
]
