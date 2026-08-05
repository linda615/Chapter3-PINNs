"""Tests for experiment configuration JSON persistence."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from configs import config_to_dict, save_experiment_config
from experiments.clamped_rock_roof_nonuniform_support.config import (
    DEFAULT_CONFIG as ROCK_ROOF_CONFIG,
)
from experiments.simply_clamped_uniform_load.config import (
    DEFAULT_CONFIG as MIXED_EDGE_CONFIG,
)
from experiments.simply_supported_sinusoidal.config import (
    DEFAULT_CONFIG as SINUSOIDAL_CONFIG,
)


@pytest.mark.parametrize(
    "config",
    (
        SINUSOIDAL_CONFIG,
        MIXED_EDGE_CONFIG,
        ROCK_ROOF_CONFIG,
    ),
)
def test_all_experiment_configs_are_json_serializable(config) -> None:
    """Every training configuration should serialize without fallback strings."""
    serialized = config_to_dict(config)
    encoded = json.dumps(serialized, allow_nan=False)

    assert encoded
    assert serialized["results_dir"] == config.results_dir.as_posix()
    assert serialized["training"]["epochs"] == config.training.epochs
    assert serialized["plate"]["D"] == config.plate.D
    assert serialized["plate"]["nu"] == config.plate.nu


def test_save_experiment_config_uses_effective_runtime_values(
    tmp_path: Path,
) -> None:
    """The saved file should reflect command-line-style config overrides."""
    training = replace(
        SINUSOIDAL_CONFIG.training,
        epochs=7,
        interior_points=19,
    )
    config = replace(
        SINUSOIDAL_CONFIG,
        results_dir=tmp_path / "results",
        training=training,
        validation_interval=3,
    )

    saved_path = save_experiment_config(
        config,
        run_entry="test.run",
        metadata={"initial_checkpoint": Path("weights/start.weights.h5")},
    )

    assert saved_path == config.results_dir / "config.json"
    with saved_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    assert document["training"]["epochs"] == 7
    assert document["training"]["interior_points"] == 19
    assert document["validation_interval"] == 3
    assert document["boundary_type"] == config.boundary_type.value
    assert document["_metadata"]["run_entry"] == "test.run"
    assert (
        document["_metadata"]["initial_checkpoint"]
        == "weights/start.weights.h5"
    )
    assert document["_metadata"]["python_version"]
