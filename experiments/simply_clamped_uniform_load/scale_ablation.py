"""Train the SCSC Mixed-6B scale-sensitivity experiments."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from configs import save_experiment_config
from losses import resolve_loss_weights

from . import ablation as common
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
)
from .run import create_learning_rate_schedule, set_reproducible_seed
from .scale_ablation_config import (
    DEFAULT_SCALE_ABLATION_ROOT,
    SCALE_VARIANTS,
    make_scale_ablation_config,
    scale_audit,
)
from .trainer import MixedBoundaryPINNTrainer


SEEDS = (2051, 2052, 2053)
EXPECTED_PARAMETER_COUNT = 33862


def build_experiment(
    variant: str,
    seed: int,
    root: Path,
    pilot: bool = False,
):
    """Build one scale variant using the established SCSC training stack."""
    config = make_scale_ablation_config(
        variant=variant,
        seed=seed,
        root=root,
        pilot=pilot,
    )
    set_reproducible_seed(seed)
    model = create_model(config)
    model(tf.zeros((1, 2), dtype=tf.float32), training=False)
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Scale ablation changed model capacity: expected "
            f"{EXPECTED_PARAMETER_COUNT}, got {model.count_params()}."
        )

    trainer = MixedBoundaryPINNTrainer(
        model=model,
        optimizer=tf.keras.optimizers.Adam(
            create_learning_rate_schedule(config)
        ),
        interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(),
        load_fn=create_load_fn(config),
        parameters=config.plate,
        config=config.training,
        loss_weights=config.loss_weights,
        physics_loss_config=config.physics_loss_config,
        boundary_loss_config=config.boundary_loss_config,
        mxy_consistency_config=config.mxy_consistency_config,
        shared_pcgrad_config=config.shared_pcgrad,
        experiment_config=config,
    )
    # The base trainer initializes this legacy attribute. Restore the exact
    # boundary normalization declared by the experiment configuration.
    trainer.boundary_loss_config = config.boundary_loss_config
    if common.GRAPH_MODE:
        from .graph_step import install

        install(trainer)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    common.dump(
        config.results_dir / "scale_audit.json",
        {
            **scale_audit(config, variant),
            "seed": seed,
            "parameters": model.count_params(),
            "TF32": False,
            "execution_mode": (
                "tf.function" if common.GRAPH_MODE else "eager"
            ),
        },
    )
    save_experiment_config(
        config,
        run_entry="experiments.simply_clamped_uniform_load.scale_ablation",
    )
    return config, model, trainer


def check_variant(variant: str, seed: int, root: Path) -> None:
    """Run a one-step finite-loss smoke test without polluting full results."""
    common.setup()
    config, model, trainer = build_experiment(
        variant,
        seed,
        root,
        pilot=True,
    )
    interior_xy, boundary_batches = trainer.sample_epoch(1)
    losses = trainer.train_step(
        interior_xy,
        boundary_batches,
        loss_weights=resolve_loss_weights(config.loss_weights, 1),
        epoch=1,
    )
    values = {name: float(value.numpy()) for name, value in losses.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(
            f"Non-finite smoke-test loss for {variant}: {values}."
        )
    common.dump(
        config.results_dir / "passed.json",
        {
            "variant": variant,
            "seed": seed,
            "total_loss": values["total_loss"],
            "finite": True,
        },
    )
    print(
        f"SCALE CHECK PASSED variant={variant} seed={seed} "
        f"total_loss={values['total_loss']:.7g}",
        flush=True,
    )


def run_variant(variant: str, seed: int, root: Path) -> None:
    """Train, validate, checkpoint, and independently evaluate one run."""
    common.setup()
    config, model, trainer = build_experiment(variant, seed, root)
    result_path = config.results_dir / "result.json"
    if result_path.exists():
        print(f"scale_ablation_skip completed_result={result_path}")
        return

    checkpoint, manager = common.state_checkpoint(
        model,
        trainer,
        config.results_dir,
    )
    test_xy = common.make_test_grid(101)
    reference = common.reference(test_xy, config)
    first_epoch = int(checkpoint.epoch.numpy()) + 1
    final_epoch = config.training.epochs
    for epoch in range(first_epoch, final_epoch + 1):
        started = time.perf_counter()
        interior_xy, boundary_batches = trainer.sample_epoch(epoch)
        losses = trainer.train_step(
            interior_xy,
            boundary_batches,
            loss_weights=resolve_loss_weights(config.loss_weights, epoch),
            epoch=epoch,
        )
        total_loss = float(losses["total_loss"].numpy())
        if not math.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite total loss at epoch {epoch}."
            )
        checkpoint.train_seconds.assign_add(time.perf_counter() - started)

        if epoch == 1 or epoch % config.training.log_every == 0:
            row = {
                "epoch": epoch,
                "variant": variant,
                "seed": seed,
                "train_seconds": float(checkpoint.train_seconds.numpy()),
                **{
                    name: float(value.numpy())
                    for name, value in losses.items()
                },
            }
            common.append_json(config.results_dir / "training.jsonl", row)
            common.dump(root / "current_run.json", row)
            print(
                f"{variant} seed={seed} epoch={epoch}/{final_epoch} "
                f"loss={total_loss:.7g}",
                flush=True,
            )

        if epoch % config.validation_interval == 0:
            started = time.perf_counter()
            validation = common.errors(
                common.fields(model, test_xy, config, "Mixed-6B"),
                reference,
            )
            score = validation["six_field_mean"]
            if not math.isfinite(score):
                raise FloatingPointError(
                    f"Non-finite validation score at epoch {epoch}."
                )
            if score < float(checkpoint.best.numpy()):
                config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                model.save_weights(
                    str(config.best_validation_loss_weights_path)
                )
                checkpoint.best.assign(score)
                checkpoint.best_epoch.assign(epoch)
            checkpoint.validation_seconds.assign_add(
                time.perf_counter() - started
            )
            checkpoint.epoch.assign(epoch)
            manager.save(checkpoint_number=epoch)
            common.append_json(
                config.results_dir / "validation.jsonl",
                {"epoch": epoch, **validation},
            )
            print(
                f"VALIDATION variant={variant} epoch={epoch} score={score}",
                flush=True,
            )

    model.save_weights(str(config.final_weights_path))
    common.evaluate_final(
        variant,
        config,
        model,
        trainer,
        checkpoint,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=SCALE_VARIANTS + ("all",),
        required=True,
    )
    parser.add_argument(
        "--seed",
        choices=tuple(str(seed) for seed in SEEDS) + ("all",),
        default="2051",
        help="Random seed, or 'all' for the three publication seeds.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CH3_SCALE_OUTPUT_ROOT",
                str(DEFAULT_SCALE_ABLATION_ROOT),
            )
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run requested variants and seeds sequentially on one GPU."""
    args = _parse_args()
    variants = SCALE_VARIANTS if args.variant == "all" else (args.variant,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)
    for seed in seeds:
        for variant in variants:
            if args.check:
                check_variant(variant, seed, args.output_root)
            else:
                run_variant(variant, seed, args.output_root)


if __name__ == "__main__":
    main()
