"""Plot training-loss histories for the sinusoidal-load plate experiment."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PLOT_DPI = 300
PLOT_FONT_SIZE = 10.5
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_LOSS_CSV = DEFAULT_RESULTS_DIR / "history" / "loss.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures" / "history"

SeriesSpec = Tuple[str, str, str, str]

OVERALL_SERIES: Tuple[SeriesSpec, ...] = (
    ("total_loss", r"总损失 $\mathcal{L}_{\mathrm{total}}$", "#0072B2", "-"),
    (
        "physics_loss",
        r"物理损失 $\mathcal{L}_{\mathrm{physics}}$",
        "#D55E00",
        "--",
    ),
    (
        "boundary_loss",
        r"边界损失 $\mathcal{L}_{\mathrm{bc}}$",
        "#009E73",
        "-.",
    ),
)

PHYSICS_SERIES: Tuple[SeriesSpec, ...] = (
    ("moment_loss", r"弯矩损失 $\mathcal{L}_{M}$", "#0072B2", "-"),
    ("shear_loss", r"剪力损失 $\mathcal{L}_{Q}$", "#D55E00", "--"),
    (
        "equilibrium_loss",
        r"平衡损失 $\mathcal{L}_{\mathrm{eq}}$",
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
    (
        "weighted_boundary_loss",
        r"加权边界损失 $\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}$",
        "#CC79A7",
        ":",
    ),
)


def load_loss_history(csv_path: Path) -> Dict[str, List[float]]:
    """Load a loss-history CSV and validate that its values are finite."""
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
                        f"Invalid value in {csv_path} at row {row_number}, "
                        f"column '{field_name}': {raw_value!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value in {csv_path} at row {row_number}, "
                        f"column '{field_name}'."
                    )
                history[field_name].append(value)

    if not history["epoch"]:
        raise ValueError(f"Loss-history CSV contains no data rows: {csv_path}")
    return history


def _configure_chinese_font() -> font_manager.FontProperties:
    """Configure an installed Chinese font for publication-ready figures."""
    override = os.environ.get("PINN_CHINESE_FONT")
    candidate_paths: List[Path] = []
    if override:
        candidate_paths.append(Path(override))

    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidate_paths.extend(
        (
            windows_fonts / "msyh.ttc",
            windows_fonts / "simhei.ttf",
            windows_fonts / "NotoSansSC-VF.ttf",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        )
    )

    for font_path in candidate_paths:
        if not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font_properties = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [
                    font_properties.get_name(),
                    "DejaVu Sans",
                ],
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
        return font_properties

    raise RuntimeError(
        "No Chinese font was found. Install Microsoft YaHei, SimHei, or "
        "Noto Sans SC, or set PINN_CHINESE_FONT to a font file path."
    )


def _validate_series(
    history: Mapping[str, Sequence[float]],
    series: Iterable[SeriesSpec],
) -> None:
    """Ensure that all requested columns exist and match the epoch count."""
    epoch_count = len(history["epoch"])
    for column, _, _, _ in series:
        if column not in history:
            raise ValueError(f"Loss-history CSV is missing column '{column}'.")
        if len(history[column]) != epoch_count:
            raise ValueError(
                f"Column '{column}' has {len(history[column])} values; "
                f"expected {epoch_count}."
            )


def _plot_series(
    axis: plt.Axes,
    history: Mapping[str, Sequence[float]],
    series: Sequence[SeriesSpec],
    title: str,
) -> None:
    """Draw one logarithmic loss panel."""
    _validate_series(history, series)
    epochs = history["epoch"]
    plotted_count = 0
    for column, label, color, line_style in series:
        values = history[column]
        if not any(value > 0.0 for value in values):
            continue
        axis.plot(
            epochs,
            values,
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=1.5,
        )
        plotted_count += 1

    if plotted_count == 0:
        raise ValueError(f"No positive loss values are available for '{title}'.")
    axis.set_yscale("log")
    axis.set_xlabel("训练轮次")
    axis.set_ylabel("损失函数值")
    axis.set_title(title)
    axis.grid(True, which="both", color="#B8B8B8", alpha=0.35, linewidth=0.6)
    axis.legend(frameon=False)


def _save_single_panel(
    history: Mapping[str, Sequence[float]],
    series: Sequence[SeriesSpec],
    title: str,
    output_path: Path,
) -> None:
    """Save one loss panel as a standalone figure."""
    figure, axis = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    _plot_series(axis, history, series, title)
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_loss_history(
    csv_path: Path = DEFAULT_LOSS_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, ...]:
    """Generate combined and standalone loss-history figures."""
    history = load_loss_history(csv_path)
    _configure_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "loss_curves.png"
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.2),
        constrained_layout=True,
    )
    panels = (
        (axes[0, 0], OVERALL_SERIES, "总损失与损失类别"),
        (axes[0, 1], PHYSICS_SERIES, "物理损失分量"),
        (axes[1, 0], SHEAR_SERIES, "剪力损失分量"),
        (axes[1, 1], WEIGHTED_SERIES, "加权损失分量"),
    )
    for axis, series, title in panels:
        _plot_series(axis, history, series, title)
    figure.suptitle("四边简支正弦荷载算例训练损失曲线")
    figure.savefig(combined_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

    standalone_specs = (
        (
            OVERALL_SERIES,
            "总损失与损失类别",
            output_dir / "loss_overall.png",
        ),
        (
            PHYSICS_SERIES,
            "物理损失分量",
            output_dir / "loss_physics_components.png",
        ),
        (
            WEIGHTED_SERIES,
            "加权损失分量",
            output_dir / "loss_weighted_components.png",
        ),
    )
    standalone_paths: List[Path] = []
    for series, title, output_path in standalone_specs:
        _save_single_panel(history, series, title, output_path)
        standalone_paths.append(output_path)

    return tuple([combined_path] + standalone_paths)


def resolve_plot_paths(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    loss_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve input and output paths from one selected result directory."""
    results_dir = Path(results_dir)
    resolved_loss_csv = (
        Path(loss_csv)
        if loss_csv is not None
        else results_dir / "history" / "loss.csv"
    )
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else results_dir / "figures" / "history"
    )
    return resolved_loss_csv, resolved_output_dir


def find_available_results() -> Tuple[Path, ...]:
    """Return sinusoidal result directories containing a loss history."""
    return tuple(
        result_dir
        for result_dir in sorted(EXPERIMENT_DIR.glob("results*"))
        if result_dir.is_dir()
        and (result_dir / "history" / "loss.csv").is_file()
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Result directory containing history/loss.csv. Figures are "
            "saved below this directory unless --output-dir is supplied."
        ),
    )
    parser.add_argument(
        "--loss-csv",
        type=Path,
        default=None,
        help="Optional direct path to a loss CSV, overriding --results-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional figure directory, overriding the selected result directory.",
    )
    parser.add_argument(
        "--list-results",
        action="store_true",
        help="List result directories that contain history/loss.csv and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate figures from the command line."""
    args = _parse_args()
    if args.list_results:
        for result_dir in find_available_results():
            print(result_dir.resolve())
        return

    loss_csv, output_dir = resolve_plot_paths(
        results_dir=args.results_dir,
        loss_csv=args.loss_csv,
        output_dir=args.output_dir,
    )
    output_paths = plot_loss_history(
        csv_path=loss_csv,
        output_dir=output_dir,
    )
    print(f"results_dir={args.results_dir.resolve()}")
    print(f"loss_history_source={loss_csv.resolve()}")
    for output_path in output_paths:
        print(f"saved_figure={output_path.resolve()}")


if __name__ == "__main__":
    main()
