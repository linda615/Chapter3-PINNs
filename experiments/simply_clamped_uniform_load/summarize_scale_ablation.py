"""Summarize the completed SCSC alpha/beta scale experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence

from models import FIELD_NAMES

from .scale_ablation import SEEDS
from .scale_ablation_config import (
    DEFAULT_SCALE_ABLATION_ROOT,
    SCALE_VARIANTS,
)


BASELINE_NAME = "current_alpha_beta"
CHECKPOINTS = ("best", "final")
METRICS = tuple(FIELD_NAMES) + ("six_field_mean",)
DEFAULT_BASELINE_ROOT = Path(
    "experiments/simply_clamped_uniform_load/results_ablation/runs"
)
DEFAULT_BASELINE_SUMMARY = Path("data/paper/scsc_ablation.csv")
DEFAULT_OUTPUT = Path("data/paper/scsc_scale_ablation.csv")


def collect_summary_rows(
    scale_root: Path = DEFAULT_SCALE_ABLATION_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    baseline_summary: Path = DEFAULT_BASELINE_SUMMARY,
    seeds: Sequence[int] = SEEDS,
    include_baseline: bool = True,
) -> List[Dict[str, object]]:
    """Aggregate best/final field errors over the requested random seeds."""
    run_paths: Dict[str, List[Path]] = {
        variant: [
            scale_root / "runs" / f"{variant}_seed{seed}" / "result.json"
            for seed in seeds
        ]
        for variant in SCALE_VARIANTS
    }
    rows: List[Dict[str, object]] = []
    if include_baseline:
        baseline_paths = [
            baseline_root / f"Mixed-6B_seed{seed}" / "result.json"
            for seed in seeds
        ]
        if all(path.exists() for path in baseline_paths):
            rows.extend(
                _aggregate_results(
                    BASELINE_NAME,
                    [_read_result(path) for path in baseline_paths],
                )
            )
        else:
            rows.extend(_read_baseline_summary(baseline_summary, len(seeds)))

    for variant, paths in run_paths.items():
        results = [_read_result(path) for path in paths]
        rows.extend(_aggregate_results(variant, results))
    return rows


def write_summary_csv(
    rows: Iterable[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Write scale-sensitivity statistics as a stable CSV table."""
    fieldnames = (
        "variant",
        "checkpoint",
        "field",
        "mean_percent",
        "sample_sd_percent",
        "runs",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_result(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing completed scale-ablation result: {path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    for checkpoint in CHECKPOINTS:
        if checkpoint not in result:
            raise ValueError(f"{path} does not contain {checkpoint!r} metrics.")
    return result


def _aggregate_results(
    variant: str,
    results: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        for metric in METRICS:
            values = [
                float(result[checkpoint][metric]) for result in results
            ]
            rows.append(
                {
                    "variant": variant,
                    "checkpoint": checkpoint,
                    "field": metric,
                    "mean_percent": mean(values),
                    "sample_sd_percent": (
                        stdev(values) if len(values) > 1 else 0.0
                    ),
                    "runs": len(values),
                }
            )
    return rows


def _read_baseline_summary(
    path: Path,
    run_count: int,
) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            "The raw Mixed-6B baseline runs and their fallback summary are "
            f"both missing; expected fallback {path}."
        )
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("model") != "Mixed-6B":
                continue
            if row.get("checkpoint") not in CHECKPOINTS:
                continue
            if row.get("field") not in METRICS:
                continue
            rows.append(
                {
                    "variant": BASELINE_NAME,
                    "checkpoint": row["checkpoint"],
                    "field": row["field"],
                    "mean_percent": float(row["mean_percent"]),
                    "sample_sd_percent": float(row["sample_sd_percent"]),
                    "runs": run_count,
                }
            )
    expected_rows = len(CHECKPOINTS) * len(METRICS)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Mixed-6B rows in {path}, got "
            f"{len(rows)}."
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-root",
        type=Path,
        default=DEFAULT_SCALE_ABLATION_ROOT,
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--without-baseline",
        action="store_true",
        help="Summarize only the configured scale variants.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the paper-ready scale-sensitivity summary table."""
    args = _parse_args()
    rows = collect_summary_rows(
        scale_root=args.scale_root,
        baseline_root=args.baseline_root,
        baseline_summary=args.baseline_summary,
        include_baseline=not args.without_baseline,
    )
    write_summary_csv(rows, args.output)
    print(f"scale_ablation_summary={args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_NAME",
    "collect_summary_rows",
    "write_summary_csv",
]
