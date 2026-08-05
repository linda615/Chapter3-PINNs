"""Compare runtime summaries produced by the PINN experiment variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RESULTS_DIRS = (
    Path(
        "experiments/simply_clamped_uniform_load/"
        "results_bounded_output_rms_loss_six"
    ),
    Path(
        "experiments/simply_clamped_uniform_load/"
        "results_mixed_all_shared"
    ),
    Path("experiments/simply_clamped_uniform_load/results_w_pinn"),
    Path("experiments/simply_clamped_uniform_load/results_mo4_pinn"),
)


def load_runtime_summaries(paths: list[Path]) -> list[dict]:
    """Load available runtime JSON files from result directories or paths."""
    summaries = []
    missing = []
    for path in paths:
        runtime_path = path if path.name == "runtime.json" else path / "runtime.json"
        if not runtime_path.is_file():
            missing.append(runtime_path)
            continue
        with runtime_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["runtime_path"] = str(runtime_path)
        summaries.append(summary)
    if missing:
        print("missing_runtime_files " + " ".join(str(path) for path in missing))
    if not summaries:
        raise FileNotFoundError("No runtime.json files were found.")
    return summaries


def add_relative_costs(summaries: list[dict]) -> list[dict]:
    """Add optimization and wall-time ratios relative to the fastest run."""
    fastest_step = min(item["mean_train_step_seconds"] for item in summaries)
    fastest_wall = min(item["wall_seconds_per_epoch"] for item in summaries)
    for item in summaries:
        item["train_step_slowdown"] = (
            item["mean_train_step_seconds"] / fastest_step
        )
        item["wall_time_slowdown"] = (
            item["wall_seconds_per_epoch"] / fastest_wall
        )
    return summaries


def save_comparison(path: Path, summaries: list[dict]) -> None:
    """Save the normalized runtime comparison table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model_name",
        "epochs",
        "interior_points",
        "boundary_points_per_side",
        "mean_train_step_seconds",
        "train_step_slowdown",
        "mean_validation_seconds",
        "wall_seconds_per_epoch",
        "wall_time_slowdown",
        "training_wall_time_seconds",
        "runtime_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: summary.get(name, "") for name in fields})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_RESULTS_DIRS),
        help="Result directories or runtime.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/simply_clamped_uniform_load/runtime_comparison.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summaries = add_relative_costs(load_runtime_summaries(args.paths))
    save_comparison(args.output, summaries)
    for item in summaries:
        print(
            "runtime_comparison "
            f"model={item['model_name']} "
            f"mean_train_step_seconds={item['mean_train_step_seconds']:.6f} "
            f"train_step_slowdown={item['train_step_slowdown']:.3f} "
            f"mean_validation_seconds={item['mean_validation_seconds']:.6f} "
            f"wall_seconds_per_epoch={item['wall_seconds_per_epoch']:.6f} "
            f"wall_time_slowdown={item['wall_time_slowdown']:.3f}"
        )
    print(f"runtime_comparison_csv={args.output}")


if __name__ == "__main__":
    main()
