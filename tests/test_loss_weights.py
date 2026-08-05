"""Tests for total-loss weighting configuration."""

import pytest

from losses.weights import (
    LossWeightSchedule,
    LossWeightStage,
    LossWeights,
    resolve_loss_weights,
)


def test_loss_weights_default_to_one() -> None:
    """All default loss weights should be one."""
    weights = LossWeights()

    assert weights.moment == 1.0
    assert weights.shear == 1.0
    assert weights.equilibrium == 1.0
    assert weights.boundary == 1.0
    assert weights.mxy_consistency == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moment": -1.0},
        {"shear": float("nan")},
        {"equilibrium": float("inf")},
        {"boundary": float("-inf")},
        {"mxy_consistency": -0.1},
    ],
)
def test_loss_weights_reject_invalid_values(kwargs) -> None:
    """Weights must be finite and non-negative."""
    with pytest.raises(ValueError):
        LossWeights(**kwargs)


def test_loss_weight_schedule_selects_epoch_stage() -> None:
    """Scheduled weights should use inclusive stage end epochs."""
    schedule = LossWeightSchedule(
        stages=(
            LossWeightStage(end_epoch=2, weights=LossWeights(equilibrium=20.0)),
            LossWeightStage(end_epoch=4, weights=LossWeights(equilibrium=5.0)),
        )
    )

    assert schedule.for_epoch(1).equilibrium == 20.0
    assert schedule.for_epoch(2).equilibrium == 20.0
    assert schedule.for_epoch(3).equilibrium == 5.0
    assert schedule.for_epoch(5).equilibrium == 5.0


def test_loss_weight_schedule_rejects_unordered_stages() -> None:
    """Stage end epochs must increase strictly."""
    with pytest.raises(ValueError):
        LossWeightSchedule(
            stages=(
                LossWeightStage(end_epoch=2, weights=LossWeights()),
                LossWeightStage(end_epoch=2, weights=LossWeights()),
            )
        )


def test_resolve_loss_weights_accepts_static_schedule_and_callable() -> None:
    """Loss weight sources should resolve to concrete ``LossWeights``."""
    static = LossWeights(equilibrium=3.0)
    schedule = LossWeightSchedule(
        stages=(LossWeightStage(end_epoch=1, weights=LossWeights(equilibrium=4.0)),)
    )

    assert resolve_loss_weights(static, epoch=1).equilibrium == 3.0
    assert resolve_loss_weights(schedule, epoch=1).equilibrium == 4.0
    assert resolve_loss_weights(lambda epoch: LossWeights(equilibrium=float(epoch)), epoch=5).equilibrium == 5.0
