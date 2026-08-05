"""Evaluate the strict all-shared low-order Mixed-PINN baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import tensorflow as tf

from models import FIELD_NAMES

from .all_shared_mixed_config import ALL_SHARED_MIXED_CONFIG
from .evaluate import run_full_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=ALL_SHARED_MIXED_CONFIG.results_dir)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = replace(ALL_SHARED_MIXED_CONFIG, results_dir=args.results_dir)
    checkpoint = (
        config.best_validation_loss_weights_path
        if args.checkpoint is None
        else args.checkpoint
    )
    evaluation = run_full_evaluation(
        checkpoint_path=checkpoint,
        pointwise_csv=config.results_dir / "evaluation_fields.csv",
        rel_l2_csv=config.results_dir / "evaluation_rel_l2.csv",
        grid_size=args.grid_size,
        config=config,
        figures_dir=config.figures_dir if args.save_figures else None,
        batch_size=args.batch_size,
    )
    score = tf.add_n(
        [evaluation.direct_rel_l2[name] for name in FIELD_NAMES]
    ) / float(len(FIELD_NAMES))
    parts = [
        "simply_clamped_uniform_load_all_shared_mixed_evaluation",
        f"validation_score={float(score.numpy()):.6e}",
    ]
    parts.extend(
        f"{name}_rel_l2={float(evaluation.direct_rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    parts.extend(
        f"{name}_from_w_rel_l2={float(evaluation.from_w_rel_l2[name].numpy()):.6e}"
        for name in FIELD_NAMES
    )
    print(" ".join(parts))


if __name__ == "__main__":
    main()
