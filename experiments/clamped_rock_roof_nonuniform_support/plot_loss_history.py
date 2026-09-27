"""Plot training losses for the clamped rock-roof experiment."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PLOT_DPI = 300
PLOT_FONT_SIZE = 10.5
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results_bounded_output_rms_loss"
DEFAULT_LOSS_CSV = DEFAULT_RESULTS_DIR / "history" / "loss.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures" / "history"

SeriesSpec = Tuple[str, str, str, str]

OVERALL_SERIES: Tuple[SeriesSpec, ...] = (
    (
        "total_loss",
        r"加权总损失 $\mathcal{L}_{\mathrm{total}}$",
        "#0072B2",
        "-",
    ),
    (
        "weighted_physics_loss",
        r"加权物理损失 $\mathcal{L}_{\mathrm{physics}}^{w}$",
        "#D55E00",
        "--",
    ),
    (
        "weighted_boundary_loss",
        r"加权边界损失 $\mathcal{L}_{\mathrm{bc}}^{w}$（硬约束，恒为0）",
        "#009E73",
        "-.",
    ),
)

PHYSICS_SERIES: Tuple[SeriesSpec, ...] = (
    (
        "moment_loss",
        r"未加权归一化弯矩损失 $\mathcal{L}_{M}$",
        "#0072B2",
        "-",
    ),
    (
        "shear_loss",
        r"未加权归一化剪力损失 $\mathcal{L}_{Q}$",
        "#D55E00",
        "--",
    ),
    (
        "equilibrium_loss",
        r"未加权归一化平衡损失 $\mathcal{L}_{\mathrm{eq}}$",
        "#009E73",
        "-.",
    ),
)

SHEAR_SERIES: Tuple[SeriesSpec, ...] = (
    ("shear_x_loss", r"$x$ 方向剪力损失 $\mathcal{L}_{Q_x}$", "#CC79A7", "-"),
    ("shear_y_loss", r"$y$ 方向剪力损失 $\mathcal{L}_{Q_y}$", "#E69F00", "--"),
)

WEIGHTED_SERIES: Tuple[SeriesSpec, ...] = (
    (
        "weighted_moment_loss",
        r"加权弯矩损失 $\lambda_M\mathcal{L}_{M}$",
        "#0072B2",
        "-",
    ),
    (
        "weighted_shear_loss",
        r"加权剪力损失 $\lambda_Q\mathcal{L}_{Q}$",
        "#D55E00",
        "--",
    ),
    (
        "weighted_equilibrium_loss",
        r"加权平衡损失 $\lambda_{\mathrm{eq}}\mathcal{L}_{\mathrm{eq}}$",
        "#009E73",
        "-.",
    ),
)


def load_loss_history(csv_path: Path) -> Dict[str, List[float]]:
    """Load a finite-valued loss history from CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Loss-history CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'epoch' column: {csv_path}")
        history: Dict[str, List[float]] = {
            field_name: [] for field_name in reader.fieldnames
        }
        for row_number, row in enumerate(reader, start=2):
            for field_name in reader.fieldnames:
                raw_value = row.get(field_name)
                try:
                    value = float(raw_value)  # type: ignore[arg-type]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid value at row {row_number}, column "
                        f"'{field_name}': {raw_value!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value at row {row_number}, "
                        f"column '{field_name}'."
                    )
                history[field_name].append(value)

    if not history["epoch"]:
        raise ValueError(f"Loss-history CSV contains no rows: {csv_path}")
    weighted_physics_columns = (
        "weighted_moment_loss",
        "weighted_shear_loss",
        "weighted_equilibrium_loss",
    )
    missing = [
        column for column in weighted_physics_columns if column not in history
    ]
    if missing:
        raise ValueError(
            "Loss-history CSV cannot reconstruct weighted physics loss; "
            f"missing columns: {missing}."
        )
    history["weighted_physics_loss"] = [
        sum(values)
        for values in zip(
            *(history[column] for column in weighted_physics_columns)
        )
    ]
    return history


def configure_chinese_font() -> None:
    """Configure an installed Chinese font and publication plot defaults."""
    override = os.environ.get("PINN_CHINESE_FONT")
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        Path(override) if override else None,
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for font_path in candidates:
        if font_path is None or not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font_name, "DejaVu Sans"],
                "font.size": PLOT_FONT_SIZE,
                "axes.titlesize": PLOT_FONT_SIZE,
                "axes.labelsize": PLOT_FONT_SIZE,
                "xtick.labelsize": PLOT_FONT_SIZE,
                "ytick.labelsize": PLOT_FONT_SIZE,
                "legend.fontsize": PLOT_FONT_SIZE,
                "axes.unicode_minus": False,
                "savefig.dpi": PLOT_DPI,
            }
        )
        return
    raise RuntimeError(
        "No Chinese font found. Install Microsoft YaHei or set "
        "PINN_CHINESE_FONT to a font file."
    )


def plot_panel(
    axis: plt.Axes,
    history: Mapping[str, Sequence[float]],
    series: Sequence[SeriesSpec],
    caption: str,
    y_label: str,
    y_scale: str,
) -> None:
    """Draw one logarithmic loss panel."""
    epochs = history["epoch"]
    plotted = 0
    for column, label, color, line_style in series:
        if column not in history:
            raise ValueError(f"Loss-history CSV is missing column '{column}'.")
        values = history[column]
        if y_scale == "log" and not any(value > 0.0 for value in values):
            continue
        axis.plot(
            epochs,
            values,
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=1.5,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No positive values are available for '{caption}'.")
    if y_scale == "symlog":
        positive_values = [
            value
            for column, _, _, _ in series
            for value in history[column]
            if value > 0.0
        ]
        linear_threshold = min(positive_values) / 5.0
        axis.set_yscale(
            "symlog",
            linthresh=linear_threshold,
            linscale=0.8,
        )
    else:
        axis.set_yscale(y_scale)
    axis.set_xlabel(f"训练轮次\n\n{caption}")
    axis.set_ylabel(y_label)
    axis.grid(True, which="both", color="#B8B8B8", alpha=0.35, linewidth=0.6)
    axis.legend(frameon=False)


def plot_loss_history(csv_path: Path, output_dir: Path) -> Tuple[Path, ...]:
    """Save the two-panel overview and its standalone loss figures."""
    history = load_loss_history(csv_path)
    configure_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = (
        (
            OVERALL_SERIES,
            "加权损失构成",
            "loss_overall.png",
            "加权损失函数值",
            "symlog",
        ),
        (
            PHYSICS_SERIES,
            "未加权归一化物理残差损失",
            "loss_physics_components.png",
            "未加权归一化损失值",
            "log",
        ),
    )

    combined_path = output_dir / "loss_curves.png"
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    panel_labels = ("（a）", "（b）")
    for axis, panel_label, (series, title, _, y_label, y_scale) in zip(
        axes.flat,
        panel_labels,
        panels,
    ):
        plot_panel(
            axis,
            history,
            series,
            f"{panel_label}{title}",
            y_label,
            y_scale,
        )
    figure.savefig(combined_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

    standalone_paths: List[Path] = []
    for series, title, filename, y_label, y_scale in panels:
        output_path = output_dir / filename
        figure, axis = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
        plot_panel(axis, history, series, title, y_label, y_scale)
        figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(figure)
        standalone_paths.append(output_path)

    return tuple([combined_path] + standalone_paths)


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--loss-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Generate loss-history figures."""
    args = parse_args()
    csv_path = args.loss_csv or args.results_dir / "history" / "loss.csv"
    output_dir = args.output_dir or args.results_dir / "figures" / "history"
    output_paths = plot_loss_history(csv_path, output_dir)
    print(f"loss_history_source={csv_path.resolve()}")
    for output_path in output_paths:
        print(f"saved_figure={output_path.resolve()}")


if __name__ == "__main__":
    main()
