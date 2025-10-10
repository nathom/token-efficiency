#!/usr/bin/env python3
"""Generate the plots described in PLOTS.md."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


# --- Data containers -------------------------------------------------------


@dataclass(frozen=True)
class TokenEfficiencyData:
    shapes: List[str]
    formats: List[str]
    tokenizers: List[str]
    values: Dict[Tuple[str, str], Dict[str, float]]


@dataclass(frozen=True)
class LegibilityResult:
    format: str
    input_nodes: int
    output_nodes: int
    accuracy: float
    jaccard: float
    dictdiff: float
    num_trials: int


@dataclass(frozen=True)
class LegibilityData:
    formats: List[str]
    input_nodes: List[int]
    output_nodes: List[int]
    results: List[LegibilityResult]


# --- Plot styling ----------------------------------------------------------


@dataclass(frozen=True)
class PlotTheme:
    name: str
    text_color: str


LIGHT_THEME = PlotTheme(name="light", text_color="#000000")
"""Light theme uses pure black text as described in PLOTS.md."""


DARK_THEME = PlotTheme(name="dark", text_color="#ffffff")
"""Dark theme uses pure white text as described in PLOTS.md."""


GRID_COLOR = "#d1d5db"
BAR_COLOR = "#4c72b0"

TOKEN_HEATMAP_CMAP = plt.get_cmap("OrRd").copy()
TOKEN_HEATMAP_CMAP.set_bad(color="#eceef1", alpha=0.9)

LEGIBILITY_HEATMAP_CMAP = plt.get_cmap("Greens")

LEGIBILITY_METRICS: Dict[str, str] = {
    "accuracy": "Accuracy",
    "jaccard": "Jaccard",
    "dictdiff": "DictDiff",
}

DEFAULT_FONT_SIZE = float(plt.rcParams.get("font.size", 10.0))
MATRIX_FONT_SIZE = DEFAULT_FONT_SIZE * 2

TOKEN_EFFICIENCY_FORMAT_LABELS: Mapping[str, str] = {
    "json_min": "json",
    "yaml_block": "yaml",
}


def token_format_label(fmt: str) -> str:
    """Return the display label for a token efficiency format."""
    return TOKEN_EFFICIENCY_FORMAT_LABELS.get(fmt, fmt).replace("_", " ")


# --- Data loading ----------------------------------------------------------


def load_token_efficiency_data(path: Path) -> TokenEfficiencyData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    shapes = raw.get("shapes")
    formats = raw.get("formats")
    tokenizers = raw.get("tokenizers")
    observations = raw.get("observations", [])
    if not shapes or not formats or not tokenizers:
        raise ValueError("token efficiency data must define shapes, formats, and tokenizers")

    values: Dict[Tuple[str, str], Dict[str, float]] = {}
    for entry in observations:
        shape = entry["shape"]
        fmt = entry["format"]
        tokenizer = entry["tokenizer"]
        raw_value = entry.get("tokens_per_node")
        if raw_value is None:
            continue
        value = float(raw_value)
        key = (shape, fmt)
        values.setdefault(key, {})[tokenizer] = value

    # Validate coverage
    for shape in shapes:
        for fmt in formats:
            key = (shape, fmt)
            tokens = values.get(key)
            if tokens is None:
                continue
            missing = [tok for tok in tokenizers if tok not in tokens]
            if missing:
                raise ValueError(
                    f"missing tokenizers {missing} for {shape}/{fmt} in tokens_per_node data"
                )

    return TokenEfficiencyData(
        shapes=list(shapes),
        formats=list(formats),
        tokenizers=list(tokenizers),
        values=values,
    )


def load_legibility_data(path: Path) -> LegibilityData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    formats = raw.get("formats")
    input_nodes = raw.get("input_nodes")
    output_nodes = raw.get("output_nodes")
    if not formats or not input_nodes or not output_nodes:
        raise ValueError("legibility data must define formats, input_nodes, and output_nodes")

    results: List[LegibilityResult] = []
    for entry in raw.get("results", []):
        results.append(
            LegibilityResult(
                format=entry["format"],
                input_nodes=int(entry["input_nodes"]),
                output_nodes=int(entry["output_nodes"]),
                accuracy=float(entry["accuracy"]),
                jaccard=float(entry.get("jaccard", entry.get("accuracy", 0.0))),
                dictdiff=float(entry.get("dictdiff", entry.get("jaccard", entry.get("accuracy", 0.0)))),
                num_trials=int(entry["num_trials"]),
            )
        )

    # Ensure there is at least one result for every format/input/output combo
    index = {(r.format, r.input_nodes, r.output_nodes) for r in results}
    for fmt in formats:
        for inp in input_nodes:
            for out in output_nodes:
                if (fmt, inp, out) not in index:
                    raise ValueError(
                        f"missing accuracy entry for format={fmt}, input_nodes={inp}, output_nodes={out}"
                    )

    return LegibilityData(
        formats=list(formats),
        input_nodes=list(input_nodes),
        output_nodes=list(output_nodes),
        results=results,
    )


# --- Helpers ----------------------------------------------------------------


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_theme(fig: plt.Figure, ax: plt.Axes, theme: PlotTheme) -> None:
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.tick_params(colors=theme.text_color)
    ax.title.set_color(theme.text_color)
    ax.xaxis.label.set_color(theme.text_color)
    ax.yaxis.label.set_color(theme.text_color)
    for spine in ax.spines.values():
        spine.set_color(theme.text_color)


def apply_legend_theme(legend, theme: PlotTheme) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor(theme.text_color)
    for text in legend.get_texts():
        text.set_color(theme.text_color)


def auto_text_color(normalized_value: float, cmap) -> str:
    rgba = cmap(normalized_value)
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 0.5 else "#ffffff"


def mean_tokens_per_node(data: TokenEfficiencyData, shape: str, fmt: str) -> float:
    per_tokenizer = data.values[(shape, fmt)]
    return fmean(per_tokenizer.values())


def aggregate_by(data: TokenEfficiencyData, *, axis: str) -> Mapping[str, float]:
    if axis == "format":
        result = {}
        for fmt in data.formats:
            shape_means = []
            for shape in data.shapes:
                key = (shape, fmt)
                if key not in data.values:
                    continue
                shape_means.append(mean_tokens_per_node(data, shape, fmt))
            if shape_means:
                result[fmt] = fmean(shape_means)
        return result
    if axis == "shape":
        result = {}
        for shape in data.shapes:
            format_means = []
            for fmt in data.formats:
                key = (shape, fmt)
                if key not in data.values:
                    continue
                format_means.append(mean_tokens_per_node(data, shape, fmt))
            if format_means:
                result[shape] = fmean(format_means)
        return result
    if axis == "tokenizer":
        result = {tok: [] for tok in data.tokenizers}
        for tokens in data.values.values():
            for tok, value in tokens.items():
                result[tok].append(value)
        return {tok: fmean(values) for tok, values in result.items() if values}
    raise ValueError(f"unsupported aggregation axis: {axis}")


def legibility_lookup(data: LegibilityData) -> Dict[Tuple[str, int, int], LegibilityResult]:
    return {
        (result.format, result.input_nodes, result.output_nodes): result
        for result in data.results
    }


# --- Token efficiency plots -------------------------------------------------


def plot_token_matrix(data: TokenEfficiencyData, theme: PlotTheme, outdir: Path) -> Path:
    ensure_directory(outdir)
    matrix = np.full((len(data.shapes), len(data.formats)), np.nan, dtype=float)
    for row, shape in enumerate(data.shapes):
        for col, fmt in enumerate(data.formats):
            key = (shape, fmt)
            if key not in data.values:
                continue
            matrix[row, col] = mean_tokens_per_node(data, shape, fmt)

    valid_values = matrix[~np.isnan(matrix)]
    if valid_values.size:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = TOKEN_HEATMAP_CMAP

    fig, ax = plt.subplots(figsize=(1.6 * len(data.formats), 0.9 * len(data.shapes) + 2))
    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, aspect="auto", norm=norm)
    apply_theme(fig, ax, theme)
    ax.tick_params(axis="both", labelsize=MATRIX_FONT_SIZE)

    ax.set_xticks(range(len(data.formats)))
    ax.set_xticklabels(
        [token_format_label(fmt) for fmt in data.formats],
        rotation=25,
        ha="right",
        fontsize=MATRIX_FONT_SIZE,
    )
    ax.set_yticks(range(len(data.shapes)))
    ax.set_yticklabels(
        [shape.replace("_", " ") for shape in data.shapes],
        fontsize=MATRIX_FONT_SIZE,
    )
    ax.set_xlabel("Format", fontsize=MATRIX_FONT_SIZE)
    ax.set_ylabel("Shape", fontsize=MATRIX_FONT_SIZE)

    for row, shape in enumerate(data.shapes):
        for col, fmt in enumerate(data.formats):
            value = matrix[row, col]
            if np.isnan(value):
                ax.text(
                    col,
                    row,
                    "N/A",
                    ha="center",
                    va="center",
                    color=theme.text_color,
                    fontsize=MATRIX_FONT_SIZE,
                )
                continue
            text_color = auto_text_color(norm(value), cmap)
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=MATRIX_FONT_SIZE,
            )

    path = outdir / f"tokens_per_node_matrix_{theme.name}.svg"
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def plot_bar(
    values: Mapping[str, float],
    *,
    theme: PlotTheme,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
    label_transform: Callable[[str], str] | None = None,
) -> None:
    labels = list(values.keys())
    heights = [values[label] for label in labels]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels), 4.2))
    apply_theme(fig, ax, theme)

    formatter = label_transform or (lambda value: value.replace("_", " "))
    formatted_labels = [formatter(label) for label in labels]

    bars = ax.bar(range(len(labels)), heights, color=BAR_COLOR)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(formatted_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)

    ymax = max(heights)
    ax.set_ylim(0, ymax * 1.1)
    for bar, value in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            color=theme.text_color,
        )

    fig.tight_layout()
    fig.savefig(outpath, transparent=True)
    plt.close(fig)


def generate_token_efficiency_plots(data: TokenEfficiencyData, outdir: Path) -> List[Path]:
    ensure_directory(outdir)
    outputs: List[Path] = []
    for theme in (LIGHT_THEME, DARK_THEME):
        outputs.append(plot_token_matrix(data, theme, outdir))

        format_values = {
            fmt: value for fmt, value in aggregate_by(data, axis="format").items() if fmt != "csv"
        }
        if format_values:
            plot_bar(
                format_values,
                theme=theme,
                xlabel="Format",
                ylabel="Tokens per node",
                title="Average tokens per node by format",
                outpath=outdir / f"tokens_per_node_by_format_{theme.name}.svg",
                label_transform=token_format_label,
            )
            outputs.append(outdir / f"tokens_per_node_by_format_{theme.name}.svg")

        shape_values = aggregate_by(data, axis="shape")
        plot_bar(
            shape_values,
            theme=theme,
            xlabel="Shape",
            ylabel="Tokens per node",
            title="Average tokens per node by shape",
            outpath=outdir / f"tokens_per_node_by_shape_{theme.name}.svg",
        )
        outputs.append(outdir / f"tokens_per_node_by_shape_{theme.name}.svg")

        tokenizer_values = aggregate_by(data, axis="tokenizer")
        plot_bar(
            tokenizer_values,
            theme=theme,
            xlabel="Tokenizer",
            ylabel="Tokens per node",
            title="Average tokens per node by tokenizer",
            outpath=outdir / f"tokens_per_node_by_tokenizer_{theme.name}.svg",
        )
        outputs.append(outdir / f"tokens_per_node_by_tokenizer_{theme.name}.svg")

    return outputs


# --- Legibility plots -------------------------------------------------------


def _metric_label(metric: str) -> str:
    try:
        return LEGIBILITY_METRICS[metric]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"unsupported metric: {metric}") from exc


def _metric_value(result: LegibilityResult, metric: str) -> float:
    if not hasattr(result, metric):  # pragma: no cover - defensive guard
        raise ValueError(f"LegibilityResult missing metric '{metric}'")
    value = getattr(result, metric)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric '{metric}' must be numeric, got {type(value)!r}")
    return float(value)


def plot_legibility_matrices(
    data: LegibilityData,
    metric: str,
    theme: PlotTheme,
    outdir: Path,
) -> List[Path]:
    ensure_directory(outdir)
    index = legibility_lookup(data)
    outputs: List[Path] = []
    cmap = LEGIBILITY_HEATMAP_CMAP
    sorted_input_nodes = sorted(data.input_nodes)
    metric_label = _metric_label(metric)

    for fmt in data.formats:
        matrix = np.zeros((len(sorted_input_nodes), len(data.output_nodes)), dtype=float)
        for row, inp in enumerate(sorted_input_nodes):
            for col, out in enumerate(data.output_nodes):
                matrix[row, col] = _metric_value(index[(fmt, inp, out)], metric)

        norm = Normalize(vmin=0.0, vmax=1.0)
        fig, ax = plt.subplots(
            figsize=(1.9 * len(data.output_nodes), 1.4 * len(sorted_input_nodes))
        )
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        apply_theme(fig, ax, theme)
        ax.tick_params(axis="both", labelsize=MATRIX_FONT_SIZE)

        ax.set_xticks(range(len(data.output_nodes)))
        ax.set_xticklabels(data.output_nodes, fontsize=MATRIX_FONT_SIZE)
        ax.set_yticks(range(len(sorted_input_nodes)))
        ax.set_yticklabels(sorted_input_nodes, fontsize=MATRIX_FONT_SIZE)
        ax.set_xlabel("Output nodes", fontsize=MATRIX_FONT_SIZE)
        ax.set_ylabel("Input nodes", fontsize=MATRIX_FONT_SIZE)

        for row, inp in enumerate(sorted_input_nodes):
            for col, out in enumerate(data.output_nodes):
                value = matrix[row, col]
                text_color = auto_text_color(norm(value), cmap)
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=MATRIX_FONT_SIZE,
                )

        path = outdir / f"{fmt}_{metric}_matrix_{theme.name}.svg"
        fig.tight_layout()
        fig.savefig(path, transparent=True)
        plt.close(fig)
        outputs.append(path)

    return outputs


def plot_legibility_scatter_fixed_input(
    data: LegibilityData,
    metric: str,
    theme: PlotTheme,
    outdir: Path,
) -> Path:
    ensure_directory(outdir)
    index = legibility_lookup(data)
    fig, axes = plt.subplots(1, len(data.input_nodes), figsize=(4.0 * len(data.input_nodes), 4.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    metric_label = _metric_label(metric)

    for ax, input_nodes in zip(axes, data.input_nodes):
        apply_theme(fig, ax, theme)
        for fmt in data.formats:
            xs = data.output_nodes
            ys = [_metric_value(index[(fmt, input_nodes, out)], metric) for out in data.output_nodes]
            ax.scatter(xs, ys, label=fmt.replace("_", " "))
        ax.set_xlabel("Output nodes")
        ax.set_title(f"Input nodes = {input_nodes}")
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel(metric_label)

    legend = axes[0].legend(loc="best")
    apply_legend_theme(legend, theme)
    path = outdir / f"{metric}_vs_output_nodes_{theme.name}.svg"
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def plot_legibility_scatter_fixed_output(
    data: LegibilityData,
    metric: str,
    theme: PlotTheme,
    outdir: Path,
) -> Path:
    ensure_directory(outdir)
    index = legibility_lookup(data)
    fig, axes = plt.subplots(1, len(data.output_nodes), figsize=(4.0 * len(data.output_nodes), 4.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    metric_label = _metric_label(metric)

    for ax, output_nodes in zip(axes, data.output_nodes):
        apply_theme(fig, ax, theme)
        for fmt in data.formats:
            xs = data.input_nodes
            ys = [_metric_value(index[(fmt, inp, output_nodes)], metric) for inp in data.input_nodes]
            ax.scatter(xs, ys, label=fmt.replace("_", " "))
        ax.set_xlabel("Input nodes")
        ax.set_title(f"Output nodes = {output_nodes}")
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel(metric_label)

    legend = axes[0].legend(loc="best")
    apply_legend_theme(legend, theme)
    path = outdir / f"{metric}_vs_input_nodes_{theme.name}.svg"
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def generate_legibility_plots(data: LegibilityData, outdir: Path) -> List[Path]:
    ensure_directory(outdir)
    outputs: List[Path] = []
    for theme in (LIGHT_THEME, DARK_THEME):
        for metric in LEGIBILITY_METRICS:
            outputs.extend(plot_legibility_matrices(data, metric, theme, outdir))
            outputs.append(plot_legibility_scatter_fixed_input(data, metric, theme, outdir))
            outputs.append(plot_legibility_scatter_fixed_output(data, metric, theme, outdir))
    return outputs


# --- CLI --------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate token efficiency and legibility plots.")
    parser.add_argument(
        "--token-efficiency-data",
        type=Path,
        default=Path("data/token_efficiency.json"),
        help="Path to the token efficiency data JSON file.",
    )
    parser.add_argument(
        "--legibility-data",
        type=Path,
        default=Path("data/legibility.json"),
        help="Path to the legibility data JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot images will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    token_data = load_token_efficiency_data(args.token_efficiency_data)
    legibility_data = load_legibility_data(args.legibility_data)

    token_dir = args.output_dir / "token_efficiency"
    legibility_dir = args.output_dir / "legibility"

    generate_token_efficiency_plots(token_data, token_dir)
    generate_legibility_plots(legibility_data, legibility_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
