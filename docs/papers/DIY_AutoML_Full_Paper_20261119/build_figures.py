"""Regenerate the paper's vector plots from the supplied aggregate data."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
BLUE = "#22607c"
ORANGE = "#a94e19"
GREY = "#646a74"
RED = "#972b50"
INK = "#243340"
GRID = "#dce4e9"


def label_panels(left, right, first, second):
    for ax, label, title in [(left, "A", first), (right, "B", second)]:
        ax.set_title(f"{label}   {title}", loc="left", fontsize=10.7, fontweight="semibold", pad=10)
        ax.grid(axis="y", color=GRID, linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)


def bike_plot(bike, output):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.35), layout="constrained")
    label_panels(left, right, "Performance by month", "Observed class composition")
    months = bike["months"]
    locations = range(len(months))
    ticks = [f"{row['label']}\n(n={row['rows']})" for row in months]
    for field, label, color, marker, line in [
        ("model", "Random forest", BLUE, "o", "-"),
        ("calendar", "Weekday and hour", ORANGE, "s", "--"),
        ("overall", "Overall majority", GREY, "^", ":"),
    ]:
        left.plot(locations, [row[field] for row in months], color=color, marker=marker,
                  linestyle=line, linewidth=1.65, markersize=4.8, label=label)
    left.set_ylabel("Macro F1")
    left.set_ylim(0, 1)
    left.legend(loc="lower left", frameon=False, fontsize=8.4)
    share = [row["busy"] / row["rows"] for row in months]
    right.bar(locations, share, color=BLUE, alpha=.83, width=.63)
    right.axhline(bike["training_busy"] / bike["training_rows"], color=RED, linewidth=1.35,
                  linestyle="--", label="Training share")
    right.set_ylabel("Share labeled busy")
    right.set_ylim(0, 1)
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(loc="lower left", facecolor="white", edgecolor="none", framealpha=.95, fontsize=8.4)
    for ax in (left, right):
        ax.set_xticks(list(locations), ticks)
    fig.savefig(output / "bike_monthly.pdf", bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def scale_plot(scale, output):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.25), layout="constrained")
    label_panels(left, right, "CSV inspection and conversion", "Evaluation after extraction")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_ylim(1, 160)
    inputs = [row["rows"] for row in scale["storage"]]
    for field, title, color, marker, line in [
        ("audit_seconds", "Inspection", BLUE, "o", "-"),
        ("parquet_seconds", "Inspection + Parquet", ORANGE, "s", "--"),
    ]:
        left.plot(inputs, [row[field] for row in scale["storage"]], marker=marker,
                  color=color, linestyle=line, label=title, linewidth=1.7, markersize=5)
    left.set_xticks(inputs, ["10k", "100k", "1M"])
    left.set_xlabel("Input CSV rows")
    left.set_ylabel("Median seconds (log scale)")
    left.legend(loc="upper left", frameon=False, fontsize=8.6)
    training = scale["evaluation"]
    x = range(len(training))
    values = [row["seconds"] for row in training]
    bars = right.bar(x, values, color=BLUE, width=.56)
    right.bar_label(bars, labels=[f"{value:.1f}" for value in values], fontsize=9, padding=3)
    right.set_xticks(list(x), [f'{row["rows"]:,}' for row in training])
    right.set_ylim(0, 13)
    right.set_xlabel("Rows evaluated")
    right.set_ylabel("Median seconds")
    fig.savefig(output / "processing_cost.pdf", bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def main():
    data = json.loads((ROOT / "figure-data.json").read_text(encoding="utf-8"))
    output = ROOT / "images"
    output.mkdir(exist_ok=True)
    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 9.6, "xtick.labelsize": 8.7,
        "ytick.labelsize": 9, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GREY,
        "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
    }):
        bike_plot(data["bike"], output)
        scale_plot(data["scale"], output)


if __name__ == "__main__":
    main()
