"""Plot loss histories for the mixed simply supported/clamped experiment."""

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
PLOT_FONT_SIZE = 10.5  # Chinese size No. 5.
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results_bounded_output_rms_loss_six"

SeriesSpec = Tuple[str, str, str, str]

OVERALL_SERIES: Tuple[SeriesSpec, ...] = (
    ("total_loss", r"总损失 $\mathcal{L}_{\mathrm{total}}$", "#0072B2", "-"),
    (
        "physics_loss",
        r"物理损失 $\mathcal{L}_{\mathrm{physics}}$",
        "#D55E00",
        "--",
    ),
    ("boundary_loss", r"边界损失 $\mathcal{L}_{\mathrm{bc}}$", "#009E73", "-."),
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

MOMENT_SERIES: Tuple[SeriesSpec, ...] = (
    ("moment_x_loss", r"$M_x$ 残差损失", "#0072B2", "-"),
    ("moment_y_loss", r"$M_y$ 残差损失", "#D55E00", "--"),
    ("twisting_moment_loss", r"$M_{xy}$ 残差损失", "#009E73", "-."),
)

SHEAR_SERIES: Tuple[SeriesSpec, ...] = (
    ("shear_x_loss", r"$Q_x$ 残差损失", "#CC79A7", "-"),
    ("shear_y_loss", r"$Q_y$ 残差损失", "#E69F00", "--"),
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
        r"加权平衡损失 $\lambda_E\mathcal{L}_{\mathrm{eq}}$",
        "#009E73",
        "-.",
    ),
    (
        "weighted_boundary_loss",
        r"加权边界损失 $\lambda_B\mathcal{L}_{\mathrm{bc}}$",
        "#CC79A7",
        ":",
    ),
)


def load_loss_history(csv_path: Path) -> Dict[str, List[float]]:
    """Load a scalar loss-history CSV and validate all stored values."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Loss-history CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
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
                        f"{field_name!r}: {raw_value!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value at row {row_number}, column "
                        f"{field_name!r}."
                    )
                history[field_name].append(value)

    if not history["epoch"]:
        raise ValueError(f"Loss-history CSV contains no rows: {csv_path}")
    return history


def configure_chinese_font() -> font_manager.FontProperties:
    """Configure an installed Chinese font and publication figure defaults."""
    override = os.environ.get("PINN_CHINESE_FONT")
    candidate_paths: List[Path] = []
    if override:
        candidate_paths.append(Path(override))

    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidate_paths.extend(
        (
            windows_fonts / "msyh.ttc",
            windows_fonts / "simhei.ttf",
            windows_fonts / "simsun.ttc",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        )
    )

    for font_path in candidate_paths:
        if not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font.get_name(), "DejaVu Sans"],
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
        return font

    raise RuntimeError(
        "No Chinese font was found. Install Microsoft YaHei, SimHei, or "
        "Noto Sans CJK, or set PINN_CHINESE_FONT to a font file path."
    )


def find_transition_epochs(
    history: Mapping[str, Sequence[float]],
) -> Tuple[Tuple[float, ...], Optional[float]]:
    """Find logged loss-weight transitions and the first active PCGrad epoch."""
    epochs = history["epoch"]
    weight_columns = (
        "moment_weight",
        "shear_weight",
        "equilibrium_weight",
        "boundary_weight",
    )
    available = [name for name in weight_columns if name in history]
    transitions: List[float] = []
    if available:
        for index in range(1, len(epochs)):
            if any(
                history[name][index] != history[name][index - 1]
                for name in available
            ):
                transitions.append(epochs[index])

    pcgrad_epoch = None
    if "pcgrad_active" in history:
        for epoch, active in zip(epochs, history["pcgrad_active"]):
            if active > 0.5:
                pcgrad_epoch = epoch
                break
    return tuple(transitions), pcgrad_epoch


def _available_series(
    history: Mapping[str, Sequence[float]],
    series: Iterable[SeriesSpec],
) -> Tuple[SeriesSpec, ...]:
    """Return series that exist and contain at least one positive value."""
    available = []
    epoch_count = len(history["epoch"])
    for spec in series:
        column = spec[0]
        if column not in history:
            continue
        if len(history[column]) != epoch_count:
            raise ValueError(
                f"Column {column!r} has {len(history[column])} values; "
                f"expected {epoch_count}."
            )
        if any(value > 0.0 for value in history[column]):
            available.append(spec)
    return tuple(available)


def _plot_series(
    axis: plt.Axes,
    history: Mapping[str, Sequence[float]],
    series: Sequence[SeriesSpec],
    title: str,
    transition_epochs: Sequence[float],
    pcgrad_epoch: Optional[float],
) -> None:
    """Draw one logarithmic loss panel with training-stage markers."""
    available = _available_series(history, series)
    if not available:
        raise ValueError(f"No positive loss values are available for {title!r}.")

    epochs = history["epoch"]
    for column, label, color, line_style in available:
        axis.plot(
            epochs,
            history[column],
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=1.45,
        )

    for index, epoch in enumerate(transition_epochs):
        axis.axvline(
            epoch,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
            label="损失权重切换" if index == 0 else None,
        )
    if pcgrad_epoch is not None:
        axis.axvline(
            pcgrad_epoch,
            color="#56B4E9",
            linestyle=":",
            linewidth=1.1,
            alpha=0.9,
            label="PCGrad启动",
        )

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
    transition_epochs: Sequence[float],
    pcgrad_epoch: Optional[float],
) -> None:
    """Save one standalone loss figure."""
    figure, axis = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    _plot_series(
        axis,
        history,
        series,
        title,
        transition_epochs,
        pcgrad_epoch,
    )
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_loss_history(
    csv_path: Path,
    output_dir: Path,
) -> Tuple[Path, ...]:
    """Generate combined and standalone loss-history figures."""
    history = load_loss_history(csv_path)
    configure_chinese_font()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_epochs, pcgrad_epoch = find_transition_epochs(history)

    panels = (
        (OVERALL_SERIES, "总损失与损失类别"),
        (PHYSICS_SERIES, "物理损失分量"),
        (MOMENT_SERIES, "弯矩残差损失分量"),
        (SHEAR_SERIES, "剪力残差损失分量"),
    )
    combined_path = output_dir / "loss_curves.png"
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.2),
        constrained_layout=True,
    )
    for axis, (series, title) in zip(axes.flat, panels):
        _plot_series(
            axis,
            history,
            series,
            title,
            transition_epochs,
            pcgrad_epoch,
        )
    figure.suptitle("简支–固支混合边界均布荷载算例训练损失曲线")
    figure.savefig(combined_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

    standalone_specs = (
        (OVERALL_SERIES, "总损失与损失类别", "loss_overall.png"),
        (PHYSICS_SERIES, "物理损失分量", "loss_physics_components.png"),
        (MOMENT_SERIES, "弯矩残差损失分量", "loss_moment_components.png"),
        (SHEAR_SERIES, "剪力残差损失分量", "loss_shear_components.png"),
        (WEIGHTED_SERIES, "加权损失贡献", "loss_weighted_components.png"),
    )
    standalone_paths: List[Path] = []
    for series, title, filename in standalone_specs:
        output_path = output_dir / filename
        _save_single_panel(
            history,
            series,
            title,
            output_path,
            transition_epochs,
            pcgrad_epoch,
        )
        standalone_paths.append(output_path)

    return tuple([combined_path] + standalone_paths)


def resolve_plot_paths(
    results_dir: Path,
    loss_csv: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Path]:
    """Resolve the selected loss CSV and figure output directory."""
    results_dir = Path(results_dir)
    resolved_csv = (
        Path(loss_csv)
        if loss_csv is not None
        else results_dir / "history" / "loss.csv"
    )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else results_dir / "figures" / "history"
    )
    return resolved_csv, resolved_output


def find_available_results() -> Tuple[Path, ...]:
    """Return experiment result directories containing a loss history."""
    return tuple(
        result_dir
        for result_dir in sorted(EXPERIMENT_DIR.glob("results*"))
        if result_dir.is_dir()
        and (result_dir / "history" / "loss.csv").is_file()
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Result directory containing history/loss.csv.",
    )
    parser.add_argument(
        "--loss-csv",
        type=Path,
        default=None,
        help="Direct loss CSV path, overriding --results-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory, overriding results-dir/figures/history.",
    )
    parser.add_argument(
        "--list-results",
        action="store_true",
        help="List result directories containing history/loss.csv and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate loss figures from the command line."""
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
    output_paths = plot_loss_history(loss_csv, output_dir)
    print(f"loss_history_source={loss_csv.resolve()}")
    for output_path in output_paths:
        print(f"saved_figure={output_path.resolve()}")


if __name__ == "__main__":
    main()
