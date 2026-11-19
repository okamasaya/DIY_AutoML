"""Tk view for the business-facing supervised comparison results."""

from zipfile import BadZipFile

from .bp_evaluation import BP_REPORT, CLASS_HEADERS, calendar_label, interval_caption, overlap_caption, split_caption


class EvaluationPanel:
    def __init__(self, body, back, open_path):
        from tkinter import StringVar, ttk

        self.frame = panel = ttk.Frame(body, style="Card.TFrame", padding=18)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(6, weight=1)
        self.directory = None
        self.result = None
        self.chart_window = None
        self.model = StringVar(panel)
        self.totals = [StringVar(panel) for _ in range(3)]
        self.baseline = StringVar(panel)
        self.interval = StringVar(panel)
        self.overlap = StringVar(panel)

        heading = ttk.Frame(panel, style="Card.TFrame")
        heading.grid(row=0, column=0, sticky="ew")
        ttk.Label(heading, text="評価結果", style="Section.TLabel").pack(side="left")
        ttk.Button(heading, text="点検に戻る", command=back).pack(side="right")

        self.line(ttk, 1, self.model, (4, 9))
        cards = ttk.Frame(panel, style="Card.TFrame")
        cards.grid(row=2, column=0, sticky="ew")
        for index, (label, number) in enumerate(zip(("評価した行", "正解", "誤分類"), self.totals)):
            cards.columnconfigure(index, weight=1, uniform="total")
            card = ttk.Frame(cards, padding=(10, 7))
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 7))
            ttk.Label(card, text=label).pack(anchor="w")
            ttk.Label(card, textvariable=number, style="Value.TLabel").pack(anchor="w")
        self.line(ttk, 3, self.baseline, (9, 2))
        self.line(ttk, 4, self.interval, (2, 2))
        self.line(ttk, 5, self.overlap, (2, 9))

        table_area = ttk.Frame(panel, style="Card.TFrame")
        table_area.grid(row=6, column=0, sticky="nsew", pady=(3, 0))
        columns = ("label", "support", "correct", "missed", "false_positive")
        self.classes = ttk.Treeview(table_area, columns=columns, show="headings", height=4)
        for index, (column, label) in enumerate(zip(columns, CLASS_HEADERS)):
            anchor = "w" if index == 0 else "e"
            self.classes.heading(column, text=label, anchor=anchor)
            self.classes.column(column, width=190 if index == 0 else 87, minwidth=65,
                                anchor=anchor, stretch=index == 0)
        scroll = ttk.Scrollbar(table_area, orient="vertical", command=self.classes.yview)
        self.classes.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.classes.pack(side="left", fill="both", expand=True)

        self.notice = StringVar(panel)
        self.line(ttk, 7, self.notice, (7, 8))
        actions = ttk.Frame(panel, style="Card.TFrame")
        actions.grid(row=8, column=0, sticky="ew")
        ttk.Button(actions, text="グラフを見る…", command=self.show_charts).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="評価Excelを開く", command=lambda: open_path(self.directory / BP_REPORT)).pack(
            side="left", padx=(0, 8),
        )
        ttk.Button(actions, text="保存先を開く", command=lambda: open_path(self.directory)).pack(side="left")

    def line(self, ttk, row, value, padding):
        label = ttk.Label(self.frame, textvariable=value, style="Quiet.TLabel", wraplength=450)
        label.grid(row=row, column=0, sticky="ew", pady=padding)
        label.bind("<Configure>", lambda event: label.configure(wraplength=event.width))

    def show(self, directory, result):
        self.directory, self.result = directory, result
        method = "期間順の5回の比較" if result["confidence"] is None else "学習側の5分割"
        self.model.set(f"{method}で選んだモデル　{result['model']}\n{split_caption(result)}")
        for variable, key in zip(self.totals, ("test_rows", "correct", "incorrect")):
            variable.set(f"{result[key]:,}")
        baseline = (
            f"学習側で最多の「{result['majority_label']}」だけで答えた場合　"
            f"{result['majority_correct']:,}件正解／{result['test_rows'] - result['majority_correct']:,}件誤分類"
        )
        notice = interval_caption(result)
        if calendar := result.get("calendar"):
            baseline = (f"{calendar_label(calendar)}　{calendar['correct']:,}件正解／macro F1 {calendar['f1']:.3f}\n"
                        f"全体の多数決　{result['majority_correct']:,}件正解／macro F1 {calendar['majority_f1']:.3f}")
            notice += " 月別とラベル別の比較は評価Excelにあります。"
        self.baseline.set(baseline)
        f1_caption = f"クラスを同じ重みで平均したF1　{result['f1']:.3f}"
        if result["confidence"] is not None:
            confidence = f"{100 * result['confidence']:g}%"
            f1_caption += f"（{confidence}区間 {result['f1_lower']:.3f}〜{result['f1_upper']:.3f}）"
        self.interval.set(f1_caption)
        overlap = result["test_rows_also_in_train"]
        overlap_text = f"学習側にも同じ入力があった評価行　{overlap:,} 件"
        if overlap:
            overlap_text += f"。{overlap_caption(result)}"
        self.overlap.set(overlap_text)
        self.classes.delete(*self.classes.get_children())
        for label, *counts in result["classes"]:
            self.classes.insert("", "end", values=(label, *(f"{n:,}" for n in counts)))
        self.notice.set(notice)
        self.frame.grid(row=0, column=1, sticky="nsew")

    def show_charts(self):
        from .bp_charts import open_chart_window

        if self.chart_window is not None and self.chart_window.winfo_exists():
            self.chart_window.lift()
            return
        try:
            self.chart_window = open_chart_window(self.frame, self.directory, self.result)
        except (OSError, ValueError, KeyError, BadZipFile):
            self.notice.set("グラフ用の評価表を読み込めませんでした。保存先の report.xlsx を確認してください。")

    def hide(self):
        if self.chart_window is not None and self.chart_window.winfo_exists():
            self.chart_window.destroy()
        self.chart_window = None
        self.frame.grid_remove()
