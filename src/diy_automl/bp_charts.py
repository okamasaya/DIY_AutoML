"""Show saved classification scores in the business-facing Tk interface."""

import numpy as np
import pandas as pd

from .bp_evaluation import MODEL_LABELS, calendar_label
from .calendar_comparison import BY_MONTH, METHODS


def read_chart_tables(directory, chronological):
    with pd.ExcelFile(directory / "report.xlsx", engine="openpyxl") as book:
        candidates = book.parse("SL_CV_Mean", usecols=["Model", "F1"])
        monthly = book.parse(BY_MONTH, usecols=["Month", "Method", "F1"]) if chronological else None
    return candidates, monthly


def chart_axes():
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.4, 4), layout="constrained", facecolor="white")
    axes = figure.subplots()
    axes.set_axisbelow(True)
    axes.tick_params(length=0, colors="#354d55", pad=6)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["bottom", "left"]].set_color("#b5c6ca")
    return figure, axes


def candidate_figure(candidates, family):
    figure, axes = chart_axes()
    candidates = candidates.sort_values("F1", ascending=False, kind="stable")
    labels = [MODEL_LABELS.get(name, name) if family else name for name in candidates["Model"]]
    positions = range(len(labels))
    bars = axes.barh(positions, candidates["F1"], height=0.56, color="#0a6d60")
    axes.bar_label(bars, fmt="%.3f", padding=5, color="#203740", fontsize=9)
    axes.set_yticks(positions, labels, fontfamily=family or "sans-serif")
    axes.invert_yaxis()
    axes.set(xlabel="macro F1", xlim=(0, 1.12), xticks=np.linspace(0, 1, 6))
    axes.grid(axis="x", color="#e1eaeb")
    return figure


def monthly_figure(monthly, labels, family):
    figure, axes = chart_axes()
    scores = monthly.pivot(index="Month", columns="Method", values="F1").sort_index()
    observed = pd.PeriodIndex(scores.index, freq="M")
    scores.index = observed
    months = pd.period_range(observed[0], observed[-1], freq="M")
    scores = scores.reindex(months)
    # An unobserved month occupies space on the axis and breaks the lines.
    ticks = observed[np.linspace(0, len(observed) - 1, min(7, len(observed)), dtype=int)]
    for method, label, color, line, marker in zip(
        METHODS, labels, ("#0a6d60", "#ad581f", "#526b9c"), ("-", "--", ":"), ("o", "^", "s"), strict=True,
    ):
        axes.plot(range(len(months)), scores[method], label=label, color=color, linestyle=line,
                  marker=marker, markersize=5, linewidth=2)
    axes.set_xticks(ticks.asi8 - months[0].ordinal, ticks.astype(str), rotation=20, ha="right")
    axes.set(ylabel="macro F1", ylim=(-0.04, 1.04), yticks=np.linspace(0, 1, 6))
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
                prop={"family": family or "sans-serif", "size": 9})
    axes.grid(axis="y", color="#e1eaeb")
    return figure


def detach_closed_figure(event):
    # Release the Tk canvas on the UI thread before a training worker can collect it.
    event.canvas.figure.set_canvas(None)


def open_chart_window(parent, directory, result):
    from tkinter import Toplevel, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.font_manager import get_font_names

    calendar = result.get("calendar")
    candidates, monthly = read_chart_tables(directory, calendar is not None)
    fonts = set(get_font_names())
    family = next((name for name in ("Meiryo", "Yu Gothic", "Noto Sans CJK JP", "Noto Sans JP") if name in fonts), None)
    tabs = [(
        "候補モデルの比較", "学習側の5回の検証で得たmacro F1の平均です。",
        candidate_figure(candidates, family),
    )]
    if monthly is not None:
        labels = (result["model"], calendar_label(calendar), "全体の多数決") if family else METHODS
        tabs.append((
            "保留テストの月別推移", "月別のmacro F1です。対象期間と件数は評価Excelの「月別」で確認できます。",
            monthly_figure(monthly, labels, family),
        ))

    window = Toplevel(parent)
    window.title("DIY AutoML | 評価グラフ")
    window.transient(parent.winfo_toplevel())
    window.geometry("800x500")
    window.minsize(600, 380)
    notebook = ttk.Notebook(window, padding=12)
    notebook.pack(fill="both", expand=True)
    for title, note, figure in tabs:
        panel = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        notebook.add(panel, text=title)
        ttk.Label(panel, text=note, style="Quiet.TLabel").pack(anchor="w", pady=(0, 6))
        canvas = FigureCanvasTkAgg(figure, master=panel)
        canvas.mpl_connect("close_event", detach_closed_figure)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    window.bind("<Escape>", lambda _: window.destroy())
    return window
