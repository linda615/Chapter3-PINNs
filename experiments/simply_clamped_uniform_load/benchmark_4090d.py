"""Run the three-model timing benchmark sequentially on one RTX 4090D."""

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


MODEL_NAMES = ("W-PINN", "MO4-PINN", "Mixed-6B")
DEFAULT_SEED = 2051
DEFAULT_EPOCHS = 20000
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results_benchmark_4090d"
MODULE_NAME = "experiments.simply_clamped_uniform_load.ablation"


def _run_command(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        list(arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _query_gpu() -> List[Dict[str, str]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            raise RuntimeError("Unexpected nvidia-smi GPU output: {!r}".format(line))
        rows.append(
            dict(
                zip(
                    ("index", "name", "memory_total_mib", "driver_version", "uuid"),
                    values,
                )
            )
        )
    return rows


def _query_compute_processes(gpu_index: str) -> List[Dict[str, str]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    gpu_rows = _query_gpu()
    selected = next((row for row in gpu_rows if row["index"] == gpu_index), None)
    if selected is None:
        raise ValueError("GPU index {} was not reported by nvidia-smi".format(gpu_index))
    processes = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        if values[0] == selected["uuid"]:
            processes.append(
                dict(
                    zip(
                        ("gpu_uuid", "pid", "process_name", "used_memory_mib"),
                        values,
                    )
                )
            )
    return processes


def _git_value(arguments: Sequence[str]) -> Optional[str]:
    try:
        value = _run_command(["git"] + list(arguments))
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_run_id() -> str:
    return "rtx4090d_seed{}_{}".format(
        DEFAULT_SEED,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "run-id may contain only letters, digits, dots, underscores, and hyphens"
        )
    return run_id


def _result_path(run_root: Path, model_name: str, seed: int) -> Path:
    return run_root / "runs" / "{}_seed{}".format(model_name, seed) / "result.json"


def _read_result(path: Path) -> Dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    required = ("model", "seed", "epochs", "train_seconds", "validation_seconds")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError("{} is missing {}".format(path, ", ".join(missing)))
    return result


def build_timing_summary(
    results: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    rows = []
    by_model = {}
    for result in results:
        model = str(result["model"])
        train_seconds = float(result["train_seconds"])
        row = {
            "model": model,
            "seed": int(result["seed"]),
            "epochs": int(result["epochs"]),
            "train_seconds": train_seconds,
            "train_minutes": train_seconds / 60.0,
            "validation_seconds": float(result["validation_seconds"]),
        }
        rows.append(row)
        by_model[model] = row

    comparisons = []
    mixed = by_model.get("Mixed-6B")
    if mixed is not None:
        for baseline_name in ("W-PINN", "MO4-PINN"):
            baseline = by_model.get(baseline_name)
            if baseline is None:
                continue
            baseline_seconds = float(baseline["train_seconds"])
            mixed_seconds = float(mixed["train_seconds"])
            comparisons.append(
                {
                    "baseline": baseline_name,
                    "candidate": "Mixed-6B",
                    "time_reduction_percent":
                        (baseline_seconds - mixed_seconds) / baseline_seconds * 100.0,
                    "speedup": baseline_seconds / mixed_seconds,
                }
            )
    return {"timings": rows, "comparisons": comparisons}


def _write_summary_csv(path: Path, summary: Mapping[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model",
                "seed",
                "epochs",
                "train_seconds",
                "train_minutes",
                "validation_seconds",
                "process_wall_seconds",
                "process_wall_minutes",
            ),
        )
        writer.writeheader()
        writer.writerows(summary["timings"])


def _stream_process(
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND: {}\n".format(" ".join(command)))
        process = subprocess.Popen(
            list(command),
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, list(command))
    return time.perf_counter() - started


def _read_progress(path: Path, run_id: str, seed: int) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "seed": seed,
        "epochs": DEFAULT_EPOCHS,
        "model_runs": {},
    }


def _begin_attempt(
    progress: Dict[str, object], model_name: str, result_path: Path
) -> Dict[str, object]:
    model_runs = progress["model_runs"]
    model_run = model_runs.setdefault(
        model_name,
        {"status": "pending", "result_path": str(result_path), "attempts": []},
    )
    attempt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    model_run["attempts"].append(attempt)
    model_run["status"] = "running"
    return attempt


def _finish_attempt(
    model_run: Dict[str, object],
    attempt: Dict[str, object],
    status: str,
    process_wall_seconds: float,
) -> None:
    attempt.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "process_wall_seconds": process_wall_seconds,
            "process_wall_minutes": process_wall_seconds / 60.0,
        }
    )
    model_run["status"] = status


def _completed_wall_seconds(model_run: Mapping[str, object]) -> float:
    return sum(
        float(attempt.get("process_wall_seconds", 0.0))
        for attempt in model_run.get("attempts", [])
        if attempt.get("status") == "completed"
    )


def _write_incremental_summary(
    run_root: Path,
    run_id: str,
    gpu: Mapping[str, str],
    seed: int,
    results: Iterable[Mapping[str, object]],
    progress: Mapping[str, object],
) -> Dict[str, object]:
    summary = build_timing_summary(results)
    model_runs = progress.get("model_runs", {})
    for row in summary["timings"]:
        model_run = model_runs.get(row["model"], {})
        process_wall_seconds = _completed_wall_seconds(model_run)
        row["process_wall_seconds"] = process_wall_seconds
        row["process_wall_minutes"] = process_wall_seconds / 60.0
    summary.update(
        {
            "run_id": run_id,
            "gpu": dict(gpu),
            "seed": seed,
            "epochs": DEFAULT_EPOCHS,
            "primary_timing_field": "train_seconds",
        }
    )
    _write_json(run_root / "timing_summary.json", summary)
    _write_summary_csv(run_root / "timing_summary.csv", summary)
    return summary


def _load_or_create_manifest(
    run_root: Path,
    seed: int,
    models: Sequence[str],
    gpu_index: str,
    gpu: Mapping[str, str],
) -> Dict[str, object]:
    manifest_path = run_root / "benchmark_manifest.json"
    requested = {
        "benchmark": "SCSC three-model RTX 4090D timing",
        "seed": seed,
        "epochs": DEFAULT_EPOCHS,
        "models": list(models),
        "gpu_index": gpu_index,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in requested.items():
            if existing.get(key) != value:
                raise ValueError(
                    "Existing manifest has {!r}={!r}, requested {!r}".format(
                        key, existing.get(key), value
                    )
                )
        return existing

    manifest = dict(requested)
    manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "gpu": dict(gpu),
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_status": _git_value(["status", "--short"]),
            "timing_scope": (
                "Per-epoch sampling, first tf.function trace, residual and automatic "
                "differentiation, backpropagation, and optimizer update. Periodic "
                "validation, checkpoint I/O, logging, and final evaluation are excluded."
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def run_benchmark(args: argparse.Namespace) -> Path:
    run_id = _validate_run_id(args.run_id or _default_run_id())
    run_root = args.results_root.resolve() / run_id

    gpu_rows = _query_gpu()
    gpu = next((row for row in gpu_rows if row["index"] == args.gpu), None)
    if gpu is None:
        raise ValueError("GPU index {} was not reported by nvidia-smi".format(args.gpu))
    if "4090 D" not in gpu["name"] and not args.allow_other_gpu:
        raise RuntimeError(
            "Selected GPU is {!r}, not an RTX 4090D. Use --allow-other-gpu only "
            "for a deliberately different benchmark.".format(gpu["name"])
        )

    active = _query_compute_processes(args.gpu)
    if active and not args.allow_busy_gpu:
        raise RuntimeError(
            "GPU {} already has compute processes: {}. Stop them before the timing "
            "benchmark, or use --allow-busy-gpu if the contention is intentional.".format(
                args.gpu, active
            )
        )

    _load_or_create_manifest(
        run_root=run_root,
        seed=args.seed,
        models=args.models,
        gpu_index=args.gpu,
        gpu=gpu,
    )
    progress_path = run_root / "benchmark_progress.json"
    progress = _read_progress(progress_path, run_id=run_id, seed=args.seed)
    _write_json(progress_path, progress)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "CH3_OUTPUT_ROOT": str(run_root),
            "CH3_GRAPH_MODE": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )

    if not args.skip_check:
        for model_name in args.models:
            command = [
                sys.executable,
                "-m",
                MODULE_NAME,
                "--name",
                model_name,
                "--seed",
                str(args.seed),
                "--check",
            ]
            _stream_process(
                command,
                environment,
                run_root / "logs" / "{}_check.log".format(model_name),
            )
        if args.check_only:
            return run_root

    for model_name in args.models:
        result_path = _result_path(run_root, model_name, args.seed)
        if result_path.exists():
            result = _read_result(result_path)
            model_run = progress["model_runs"].setdefault(
                model_name,
                {"result_path": str(result_path), "attempts": []},
            )
            model_run.update(
                {
                    "status": "completed",
                    "train_seconds": float(result["train_seconds"]),
                    "train_minutes": float(result["train_seconds"]) / 60.0,
                    "validation_seconds": float(result["validation_seconds"]),
                }
            )
            _write_json(progress_path, progress)
            completed_results = [
                _read_result(_result_path(run_root, name, args.seed))
                for name in args.models
                if _result_path(run_root, name, args.seed).exists()
            ]
            _write_incremental_summary(
                run_root, run_id, gpu, args.seed, completed_results, progress
            )
            print("{} is already complete; preserving recorded timing.".format(model_name))
            continue

        command = [
            sys.executable,
            "-m",
            MODULE_NAME,
            "--name",
            model_name,
            "--seed",
            str(args.seed),
        ]
        attempt = _begin_attempt(progress, model_name, result_path)
        _write_json(progress_path, progress)
        wall_started = time.perf_counter()
        try:
            process_wall_seconds = _stream_process(
                command,
                environment,
                run_root / "logs" / "{}_seed{}.log".format(model_name, args.seed),
            )
        except BaseException:
            process_wall_seconds = time.perf_counter() - wall_started
            model_run = progress["model_runs"][model_name]
            _finish_attempt(model_run, attempt, "failed", process_wall_seconds)
            _write_json(progress_path, progress)
            raise

        result = _read_result(result_path)
        model_run = progress["model_runs"][model_name]
        _finish_attempt(model_run, attempt, "completed", process_wall_seconds)
        model_run.update(
            {
                "train_seconds": float(result["train_seconds"]),
                "train_minutes": float(result["train_seconds"]) / 60.0,
                "validation_seconds": float(result["validation_seconds"]),
            }
        )
        _write_json(progress_path, progress)
        completed_results = [
            _read_result(_result_path(run_root, name, args.seed))
            for name in args.models
            if _result_path(run_root, name, args.seed).exists()
        ]
        _write_incremental_summary(
            run_root, run_id, gpu, args.seed, completed_results, progress
        )

    results = [
        _read_result(_result_path(run_root, model_name, args.seed))
        for model_name in args.models
    ]
    summary = _write_incremental_summary(
        run_root,
        run_id,
        gpu,
        args.seed,
        results,
        progress,
    )

    print("\nRTX 4090D timing summary")
    for row in summary["timings"]:
        print(
            "{model}: {train_minutes:.2f} min "
            "({train_seconds:.3f} s), seed={seed}".format(**row)
        )
    for comparison in summary["comparisons"]:
        print(
            "Mixed-6B vs {baseline}: reduction={time_reduction_percent:.1f}% "
            "speedup={speedup:.2f}x".format(**comparison)
        )
    print("Results: {}".format(run_root))
    return run_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to run sequentially on the selected GPU.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, choices=(2051,))
    parser.add_argument("--gpu", default="0", help="Physical GPU index before CUDA masking.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-id",
        help="Directory name below results-root. Reuse it to resume an interrupted run.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    if args.check_only and args.skip_check:
        parser.error("--check-only and --skip-check cannot be used together")
    return args


def main() -> None:
    run_benchmark(_parse_args())


if __name__ == "__main__":
    main()
