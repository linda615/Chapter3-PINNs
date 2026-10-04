import pytest

from experiments.simply_clamped_uniform_load.benchmark_4090d import (
    _write_incremental_summary,
    _validate_run_id,
    build_timing_summary,
)


def test_build_timing_summary_uses_training_time_only():
    summary = build_timing_summary(
        [
            {
                "model": "W-PINN",
                "seed": 2051,
                "epochs": 20000,
                "train_seconds": 6000.0,
                "validation_seconds": 90.0,
            },
            {
                "model": "MO4-PINN",
                "seed": 2051,
                "epochs": 20000,
                "train_seconds": 9000.0,
                "validation_seconds": 20.0,
            },
            {
                "model": "Mixed-6B",
                "seed": 2051,
                "epochs": 20000,
                "train_seconds": 1800.0,
                "validation_seconds": 10.0,
            },
        ]
    )

    timings = {row["model"]: row for row in summary["timings"]}
    assert timings["W-PINN"]["train_minutes"] == pytest.approx(100.0)
    assert timings["MO4-PINN"]["train_minutes"] == pytest.approx(150.0)
    assert timings["Mixed-6B"]["train_minutes"] == pytest.approx(30.0)

    comparisons = {
        row["baseline"]: row for row in summary["comparisons"]
    }
    assert comparisons["W-PINN"]["time_reduction_percent"] == pytest.approx(70.0)
    assert comparisons["W-PINN"]["speedup"] == pytest.approx(10.0 / 3.0)
    assert comparisons["MO4-PINN"]["time_reduction_percent"] == pytest.approx(80.0)
    assert comparisons["MO4-PINN"]["speedup"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "run_id",
    ["paper_seed2051_4090d", "rtx4090d-20261004", "benchmark.v1"],
)
def test_validate_run_id_accepts_portable_directory_names(run_id):
    assert _validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["../outside", "contains space", "a/b"])
def test_validate_run_id_rejects_unsafe_directory_names(run_id):
    with pytest.raises(ValueError):
        _validate_run_id(run_id)


def test_incremental_summary_persists_train_and_wall_times(tmp_path):
    result = {
        "model": "W-PINN",
        "seed": 2051,
        "epochs": 20000,
        "train_seconds": 6000.0,
        "validation_seconds": 90.0,
    }
    progress = {
        "model_runs": {
            "W-PINN": {
                "attempts": [
                    {
                        "status": "completed",
                        "process_wall_seconds": 6300.0,
                    }
                ]
            }
        }
    }

    summary = _write_incremental_summary(
        run_root=tmp_path,
        run_id="paper_seed2051_4090d",
        gpu={"name": "NVIDIA GeForce RTX 4090 D"},
        seed=2051,
        results=[result],
        progress=progress,
    )

    row = summary["timings"][0]
    assert summary["primary_timing_field"] == "train_seconds"
    assert row["train_minutes"] == pytest.approx(100.0)
    assert row["process_wall_minutes"] == pytest.approx(105.0)
    assert (tmp_path / "timing_summary.json").is_file()
    assert (tmp_path / "timing_summary.csv").is_file()
