"""Serialize experiment dataclasses to human-readable JSON files."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import platform
from typing import Any, Dict, Mapping, Optional


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a dataclass configuration into JSON-compatible values."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(
            "config must be a dataclass instance, "
            f"got {type(config).__name__}."
        )
    return {
        field.name: _to_json_value(getattr(config, field.name))
        for field in fields(config)
    }


def save_experiment_config(
    config: Any,
    path: Optional[Path] = None,
    run_entry: str = "training",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the effective experiment configuration to ``config.json``.

    By default, the file is placed directly under ``config.results_dir``.
    Existing files are replaced so the JSON always describes the latest run
    started in that result directory.
    """
    if path is None:
        results_dir = getattr(config, "results_dir", None)
        if results_dir is None:
            raise ValueError(
                "config must define results_dir when path is not provided."
            )
        path = Path(results_dir) / "config.json"
    else:
        path = Path(path)

    document = config_to_dict(config)
    run_metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "run_entry": run_entry,
        "config_class": (
            f"{config.__class__.__module__}.{config.__class__.__qualname__}"
        ),
        "python_version": platform.python_version(),
    }
    if metadata:
        run_metadata.update(_to_json_value(dict(metadata)))
    document["_metadata"] = run_metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            document,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    temporary_path.replace(path)
    return path


def _to_json_value(value: Any) -> Any:
    """Recursively convert supported configuration values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return _to_json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(_to_json_value(key)): _to_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Unsupported configuration value "
        f"{value!r} of type {type(value).__name__}."
    )
