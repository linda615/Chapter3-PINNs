"""Benchmark 100 optimizer iterations for the three PINN formulations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import tensorflow as tf

from losses import resolve_loss_weights
from trainer import SharedTrunkPCGradConfig

from .comparison_trainers import (
    MixedBoundaryMO4PINNTrainer,
    MixedBoundaryWPINNTrainer,
)
from .config import DEFAULT_CONFIG, ExperimentConfig
from .mo4_config import MO4_PINN_CONFIG
from .problem import (
    create_boundary_sampler,
    create_interior_sampler,
    create_load_fn,
    create_model,
    sample_boundary_groups,
)
from .trainer import MixedBoundaryPINNTrainer
from .w_pinn_config import W_PINN_CONFIG
from .w_pinn_problem import create_w_pinn_model


MODEL_ORDER = ("w_pinn", "mo4_pinn", "mixed_6b")


def benchmark_models(
    iterations: int = 100,
    warmup_iterations: int = 5,
    interior_points: int = 2048,
    boundary_points_per_side: int = 80,
    target_iterations: int = 20000,
    require_gpu: bool = True,
    output_path: Path = Path(
        "experiments/simply_clamped_uniform_load/iteration_benchmark.json"
    ),
) -> list[dict[str, float | int | str]]:
    """Measure synchronized Adam iteration time on identical collocation data."""
    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("iterations must be positive and warmup non-negative.")
    if interior_points <= 0 or boundary_points_per_side <= 0:
        raise ValueError("Collocation point counts must be positive.")
    if target_iterations <= 0:
        raise ValueError("target_iterations must be positive.")

    device_name = _accelerator_device_name()
    if require_gpu and device_name.startswith("CPU"):
        raise RuntimeError(
            "No TensorFlow GPU is available. This benchmark must run on one "
            "GPU so the three timings are comparable."
        )

    # One immutable batch is shared by all formulations. This isolates the
    # automatic-differentiation and optimizer cost from sampler randomness.
    interior_sampler = create_interior_sampler()
    boundary_sampler = create_boundary_sampler()
    interior_xy = interior_sampler.sample(interior_points, seed=31001)
    boundary_groups = sample_boundary_groups(
        sampler=boundary_sampler,
        points_per_side=boundary_points_per_side,
        config=DEFAULT_CONFIG,
        seed=41001,
    )

    results = []
    factories = {
        "w_pinn": _create_w_benchmark,
        "mo4_pinn": _create_mo4_benchmark,
        "mixed_6b": _create_mixed_benchmark,
    }
    for model_name in MODEL_ORDER:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(DEFAULT_CONFIG.seed)
        trainer, config = factories[model_name]()
        weights = resolve_loss_weights(config.loss_weights, 1)

        loss_dict = None
        for _ in range(warmup_iterations):
            loss_dict = trainer.train_step(
                interior_xy=interior_xy,
                boundary_batch=boundary_groups,
                loss_weights=weights,
                epoch=1,
            )
        if loss_dict is not None:
            _synchronize(loss_dict)

        started = time.perf_counter()
        for _ in range(iterations):
            loss_dict = trainer.train_step(
                interior_xy=interior_xy,
                boundary_batch=boundary_groups,
                loss_weights=weights,
                epoch=1,
            )
        if loss_dict is None:
            raise RuntimeError("Benchmark did not execute an optimizer step.")
        final_loss = _synchronize(loss_dict)
        elapsed = time.perf_counter() - started
        seconds_per_iteration = elapsed / iterations
        estimated_total_seconds = seconds_per_iteration * target_iterations
        result = {
            "model_name": model_name,
            "device": device_name,
            "tensorflow_version": tf.__version__,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "interior_points": interior_points,
            "boundary_points_per_side": boundary_points_per_side,
            "t_iterations_seconds": elapsed,
            "seconds_per_iteration": seconds_per_iteration,
            "target_iterations": target_iterations,
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total_hours": estimated_total_seconds / 3600.0,
            "final_loss": final_loss,
        }
        results.append(result)
        print(_format_result(result))

    fastest = min(item["seconds_per_iteration"] for item in results)
    for item in results:
        item["slowdown_vs_fastest"] = item["seconds_per_iteration"] / fastest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(f"iteration_benchmark_json={output_path}")
    return results


def _create_w_benchmark():
    config = _benchmark_config(W_PINN_CONFIG)
    return (
        MixedBoundaryWPINNTrainer(
            model=create_w_pinn_model(config),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            interior_sampler=create_interior_sampler(),
            boundary_sampler=create_boundary_sampler(),
            load_fn=create_load_fn(config),
            parameters=config.plate,
            config=config.training,
            loss_weights=config.loss_weights,
            physics_loss_config=config.physics_loss_config,
            boundary_loss_config=config.boundary_loss_config,
            shared_pcgrad_config=SharedTrunkPCGradConfig(enabled=False),
            experiment_config=config,
        ),
        config,
    )


def _create_mo4_benchmark():
    config = _benchmark_config(MO4_PINN_CONFIG)
    return (
        MixedBoundaryMO4PINNTrainer(
            model=create_model(config),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            interior_sampler=create_interior_sampler(),
            boundary_sampler=create_boundary_sampler(),
            load_fn=create_load_fn(config),
            parameters=config.plate,
            config=config.training,
            loss_weights=config.loss_weights,
            physics_loss_config=config.physics_loss_config,
            boundary_loss_config=config.boundary_loss_config,
            shared_pcgrad_config=SharedTrunkPCGradConfig(enabled=False),
            experiment_config=config,
        ),
        config,
    )


def _create_mixed_benchmark():
    config = _benchmark_config(DEFAULT_CONFIG)
    return (
        MixedBoundaryPINNTrainer(
            model=create_model(config),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            interior_sampler=create_interior_sampler(),
            boundary_sampler=create_boundary_sampler(),
            load_fn=create_load_fn(config),
            parameters=config.plate,
            config=config.training,
            loss_weights=config.loss_weights,
            physics_loss_config=config.physics_loss_config,
            boundary_loss_config=config.boundary_loss_config,
            mxy_consistency_config=config.mxy_consistency_config,
            shared_pcgrad_config=SharedTrunkPCGradConfig(enabled=False),
            experiment_config=config,
        ),
        config,
    )


def _benchmark_config(config: ExperimentConfig) -> ExperimentConfig:
    """Disable diagnostics and PCGrad without changing model or losses."""
    return replace(
        config,
        gradient_diagnostics_enabled=False,
        shared_pcgrad=SharedTrunkPCGradConfig(enabled=False),
    )


def _synchronize(loss_dict: dict[str, tf.Tensor]) -> float:
    """Materialize the final scalar so all queued GPU work has completed."""
    return float(loss_dict["total_loss"].numpy())


def _accelerator_device_name() -> str:
    """Return the active GPU name, or the CPU description as a fallback."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU (TensorFlow host execution)"
    details = tf.config.experimental.get_device_details(gpus[0])
    return str(details.get("device_name", gpus[0].name))


def _format_result(result: dict[str, float | int | str]) -> str:
    return (
        f"iteration_benchmark model={result['model_name']} "
        f"device={result['device']} "
        f"iterations={result['iterations']} "
        f"t_iterations_seconds={result['t_iterations_seconds']:.6f} "
        f"seconds_per_iteration={result['seconds_per_iteration']:.6f} "
        f"estimated_total_hours={result['estimated_total_hours']:.6f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--interior-points", type=int, default=2048)
    parser.add_argument("--boundary-points-per-side", type=int, default=80)
    parser.add_argument("--target-iterations", type=int, default=20000)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU-only benchmark when no TensorFlow GPU is visible.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/simply_clamped_uniform_load/"
            "iteration_benchmark.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark_models(
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        interior_points=args.interior_points,
        boundary_points_per_side=args.boundary_points_per_side,
        target_iterations=args.target_iterations,
        require_gpu=not args.allow_cpu,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
