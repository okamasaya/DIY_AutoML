"""Desktop entry point for previewing the existing BP data audit."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .data_audit import bp_summary_rows, prepare_bp_csv
from .bp_evaluation import evaluate_selection, training_blockers
from .bp_results import EvaluationPanel
from .data_conversion import conversion_blocker
from .data_extract import CSV_ARTIFACT, extract_data
from .data_profile import read_header
from .reporting import file_sha256


ISSUE_LABELS = {
    "incomplete_input": "途中までの読み取り",
    "no_valid_rows": "点検対象のデータ行なし",
    "row_width": "列数が合わない行",
    "duplicate_header": "同じ名前の列",
    "empty_header": "名前のない列",
    "target_not_found": "正解列が見つからない",
    "target_ambiguous": "同じ名前の正解列",
    "target_not_declared": "正解列が未指定",
    "fewer_than_two_classes": "ラベルが2種類未満",
    "classes_under_six_rows": "6件未満のラベル",
    "missing_target": "ラベルの欠損",
    "no_feature_columns": "特徴量の列なし",
    "missing_feature_values": "特徴量の欠損",
    "nonfinite_feature_values": "無限大などの数値",
    "nonnumeric_feature_columns": "文字列を含む特徴量",
    "duplicate_rows": "全列が一致する余分な行",
    "duplicate_features": "入力値が一致する余分な行",
    "conflicting_labels": "同じ入力でラベルが異なる組",
    "missing_identifier": "分割用IDの欠損",
    "repeated_ids": "複数行に現れる分割用ID",
    "time_not_found": "日付列が見つからない",
    "time_ambiguous": "同じ名前の日付列",
    "unparsed_timestamp": "読み取れない日付・時刻",
    "missing_timestamp": "日付・時刻の欠損",
    "mixed_timezones": "タイムゾーン指定あり・なしが混在",
    "metadata_columns": "分割に使う列",
}
ROW_SPLIT = "行単位"
IDENTICAL_SPLIT = "同じ入力をまとめる"
ID_PREFIX = "IDでまとめる　"
TIME_PREFIX = "日付順で分ける　"


def named_columns(header):
    counts = Counter(header)
    return [name for name in header if name.strip() and counts[name] == 1]


def preview_columns(source):
    """Read the same first CSV record as the audit and offer unique, named columns."""
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header, _, error = read_header(csv.reader(stream, strict=True))
    return named_columns(header), error


@dataclass(frozen=True)
class Audit:
    directory: Path
    profile: dict


@dataclass(frozen=True)
class Selection:
    csv: Path
    audit: Audit
    group_identical: bool = False


def result_path(parent):
    name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    return parent / name


def save_audit(source, output, target, compress=False, id_column=None, time_column=None):
    directory = prepare_bp_csv(source, output, target=target, compress=compress,
                               id_column=id_column, time_column=time_column)
    profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    return Audit(directory, profile)


def projection_features(profile):
    scan = profile["scan"]
    if (scan["status"] != "complete" or scan["wrong_width_rows"] or not scan["valid_rows"]
            or profile["target"]["status"] != "counted"):
        return []
    header = [column["name"] for column in profile["columns"]]
    settings = profile["settings"]
    metadata = (settings["target"], settings["id_column"], settings["time_column"])
    return [name for name in named_columns(header) if name not in metadata]


def run_projection(original, features, group_identical=False):
    profile = original.profile
    target = profile["settings"]["target"]
    group_column = profile["settings"]["id_column"]
    time_column = profile["settings"]["time_column"]
    if sum((group_column is not None, time_column is not None, group_identical)) > 1:
        raise ValueError("評価の分割方法を1つ選んでください。")
    available = projection_features(profile)
    if not features or len(features) != len(set(features)) or not set(features).issubset(available):
        raise ValueError("学習に使う列を1つ以上選んでください。")
    chosen = set(features) | {target}
    chosen.update(name for name in (group_column, time_column) if name is not None)
    columns = [column["name"] for column in profile["columns"] if column["name"] in chosen]
    source, digest = Path(profile["settings"]["csv"]), profile["input_sha256"]
    conversion = profile.get("conversion", {})
    parquet = conversion.get("files", {}).get("parquet")
    if parquet and find_spec("duckdb") and not conversion_blocker(profile, ("duckdb",)):
        if file_sha256(source) != digest:
            raise ValueError("元CSVが点検後に変更されました。元CSVを点検し直してください。")
        source, digest = original.directory / parquet["path"], parquet["sha256"]

    output = result_path(original.directory / "selections")
    extracted = extract_data(source, output / "extracted", columns, expected_input_sha256=digest)
    selection = extracted / CSV_ARTIFACT
    audited = save_audit(selection, output / "audit", target, id_column=group_column, time_column=time_column)
    if (audited.profile["target"] != profile["target"]
            or audited.profile["scan"]["valid_rows"] != profile["scan"]["valid_rows"]):
        raise ValueError("抽出前後で行数またはラベルの内訳が一致しません。")
    return Selection(selection, audited, group_identical)


def open_output(path):
    if not path.exists():
        raise FileNotFoundError(path)
    if os.name == "nt":
        os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)], stdin=subprocess.DEVNULL)


def launch_preview(csv=None, output_dir="runs/bp-preview"):
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("画面を開くには Tcl/Tk を含む Python が必要です。") from error
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f"点検画面を開けませんでした。{error}") from error
    app = PreviewWindow(root, Path(output_dir).resolve())
    if csv:
        app.load_csv(Path(csv))
    root.mainloop()


class PreviewWindow:
    def __init__(self, root, output_dir):
        from tkinter import BooleanVar, StringVar, filedialog, ttk

        self.root = root
        self.output_dir = output_dir
        self.filedialog = filedialog
        self.source = self.original = self.selection = self.result = self.evaluation = self.job = None
        self.close_pending = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.path_text = StringVar(root)
        self.target = StringVar(root)
        self.split_mode = StringVar(root, ROW_SPLIT)
        self.compress = BooleanVar(root)
        self.status = StringVar(root)
        self.notes = StringVar(root)
        self.training_hint = StringVar(root)
        self.view_title = StringVar(root, "点検プレビュー")
        self.comparison = StringVar(root)
        self.row_count = StringVar(root)
        self.missing_count = StringVar(root)
        self.duplicate_count = StringVar(root)
        self.label_count = StringVar(root)
        self.counts = (self.row_count, self.missing_count, self.duplicate_count, self.label_count)

        root.title("DIY AutoML | 分類評価")
        root.geometry("1050x660")
        root.minsize(860, 640)
        root.configure(background="#f0f4f5")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.style(ttk.Style(root))

        shell = ttk.Frame(root, padding=(24, 20, 24, 24))
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="DIY AutoML", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(shell, text="データ点検と分類評価", style="Title.TLabel").pack(anchor="w", pady=(4, 14))
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self.build_inputs(ttk, body)
        self.build_results(ttk, body)
        self.evaluation_panel = EvaluationPanel(body, self.show_audit_panel, self.open_path)
        self.target.trace_add("write", self.configure_split_modes)
        self.split_mode.trace_add("write", self.options_changed)
        self.compress.trace_add("write", self.options_changed)
        self.clear_result("CSVファイルを選んでください。")
        self.browse_button.focus_set()

    def style(self, style):
        background, paper, ink, quiet = "#f0f4f5", "#ffffff", "#172c34", "#52666f"
        font = "Meiryo UI" if os.name == "nt" else "TkDefaultFont"
        style.theme_use("clam")
        style.configure(".", font=(font, 10), foreground=ink, background=background)
        style.configure("Brand.TLabel", foreground="#0d6e60", font=(font, 10, "bold"))
        style.configure("Title.TLabel", font=(font, 25, "bold"))
        style.configure("Card.TFrame", background=paper)
        style.configure("Card.TLabel", background=paper)
        style.configure("Section.TLabel", background=paper, font=(font, 13, "bold"))
        style.configure("Quiet.TLabel", background=paper, foreground=quiet)
        style.configure("Value.TLabel", font=(font, 23, "bold"))
        style.configure("Card.TCheckbutton", background=paper)
        style.configure("TButton", padding=(10, 7))
        style.configure("Primary.TButton", background="#0a6d60", foreground=paper)
        style.map("Primary.TButton", background=[("disabled", "#e5ecec"), ("active", "#08594f")],
                  foreground=[("disabled", "#55686b")])
        style.configure("Treeview", background=paper, fieldbackground=paper, rowheight=27)
        style.configure("Treeview.Heading", background="#edf3f4", font=(font, 9, "bold"))

    def build_inputs(self, ttk, body):
        panel = ttk.Frame(body, style="Card.TFrame", padding=18)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        ttk.Label(panel, text="入力ファイル", style="Section.TLabel").pack(anchor="w")
        self.browse_button = ttk.Button(panel, text="CSVを選ぶ…", command=self.choose_csv)
        self.browse_button.pack(fill="x", pady=(10, 6))
        self.path_entry = ttk.Entry(panel, textvariable=self.path_text, width=28, state="readonly")
        self.path_entry.pack(fill="x")
        ttk.Label(panel, text="正解列（任意）", style="Card.TLabel").pack(anchor="w", pady=(19, 5))
        self.columns = ttk.Combobox(panel, textvariable=self.target, width=25, state="disabled")
        self.columns.pack(fill="x")
        ttk.Label(panel, text="評価の分割方法", style="Card.TLabel").pack(anchor="w", pady=(12, 5))
        self.split_picker = ttk.Combobox(panel, textvariable=self.split_mode, width=25, state="disabled")
        self.split_picker.pack(fill="x")
        self.compress_button = ttk.Checkbutton(
            panel, text="作業用Parquetも保存", variable=self.compress, style="Card.TCheckbutton",
        )
        self.compress_button.pack(anchor="w", pady=(18, 16))
        self.run_button = ttk.Button(panel, text="点検する", style="Primary.TButton", command=self.start_audit)
        self.run_button.pack(fill="x")
        self.select_button = ttk.Button(panel, text="使う列を選ぶ…", command=self.choose_projection)
        self.select_button.pack(fill="x", pady=(9, 0))
        self.train_button = ttk.Button(panel, text="5モデルを比較する", command=self.start_evaluation)
        self.train_button.pack(fill="x", pady=(9, 0))
        self.evaluation_button = ttk.Button(panel, text="評価結果を見る", command=self.show_evaluation)
        self.evaluation_button.pack(fill="x", pady=(6, 0))
        self.training_label = ttk.Label(panel, textvariable=self.training_hint, style="Quiet.TLabel", wraplength=174)
        self.training_label.pack(anchor="w", pady=(8, 0))
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 0))
        self.progress.pack_forget()

    def build_results(self, ttk, body):
        self.audit_panel = panel = ttk.Frame(body, style="Card.TFrame", padding=18)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1, uniform="table")
        panel.columnconfigure(1, weight=1, uniform="table")
        panel.rowconfigure(5, weight=1)
        ttk.Label(panel, textvariable=self.view_title, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        views = ttk.Frame(panel, style="Card.TFrame")
        views.grid(row=0, column=1, sticky="e")
        self.original_view = ttk.Button(views, text="元CSV", command=lambda: self.show_result(self.original))
        self.original_view.pack(side="left", padx=(0, 6))
        self.selected_view = ttk.Button(views, text="抽出後", command=lambda: self.show_result(self.selection.audit))
        self.selected_view.pack(side="left")
        status = ttk.Label(panel, textvariable=self.status, style="Card.TLabel", wraplength=525)
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 14))
        status.bind("<Configure>", lambda event: status.configure(wraplength=event.width))

        self.compare_label = ttk.Label(panel, textvariable=self.comparison, style="Quiet.TLabel", wraplength=525)
        self.compare_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.compare_label.bind("<Configure>", lambda event: self.compare_label.configure(wraplength=event.width))
        metrics = ttk.Frame(panel, style="Card.TFrame")
        metrics.grid(row=3, column=0, columnspan=2, sticky="ew")
        metric_labels = ("列数が合う行", "欠損セル", "全列一致の重複", "ラベルの種類")
        for index, (title, value) in enumerate(zip(metric_labels, self.counts)):
            metrics.columnconfigure(index, weight=1, uniform="metric")
            card = ttk.Frame(metrics, padding=(8, 8))
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 7))
            ttk.Label(card, text=title, anchor="w", wraplength=112).pack(fill="x")
            ttk.Label(card, textvariable=value, style="Value.TLabel").pack(anchor="w")

        ttk.Label(panel, text="正解ラベルの内訳", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(18, 7))
        ttk.Label(panel, text="確認する項目", style="Card.TLabel").grid(row=4, column=1, sticky="w", pady=(18, 7))
        self.labels = self.table(ttk, panel, 0, "ラベル")
        self.issues = self.table(ttk, panel, 1, "項目")
        notes = ttk.Label(panel, textvariable=self.notes, style="Quiet.TLabel", wraplength=525)
        notes.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        notes.bind("<Configure>", lambda event: notes.configure(wraplength=event.width))
        actions = ttk.Frame(panel, style="Card.TFrame")
        actions.grid(row=7, column=0, columnspan=2, sticky="ew")
        self.open_report = ttk.Button(actions, text="Excelを開く", command=lambda: self.open_result("report.xlsx"))
        self.open_report.pack(side="left", padx=(0, 8))
        self.open_folder = ttk.Button(actions, text="保存先を開く", command=self.open_result)
        self.open_folder.pack(side="left", padx=(0, 8))
        self.open_selected_folder = ttk.Button(actions, text="抽出CSVの保存先",
                                               command=lambda: self.open_path(self.selection.csv.parent))
        self.open_selected_folder.pack(side="left")

    def table(self, ttk, panel, column, heading):
        frame = ttk.Frame(panel, style="Card.TFrame")
        frame.grid(row=5, column=column, sticky="nsew", padx=(0, 10) if column == 0 else 0)
        table = ttk.Treeview(frame, columns=("name", "count"), show="headings", height=5)
        table.heading("name", text=heading, anchor="w")
        table.heading("count", text="件数", anchor="e")
        table.column("name", width=150, minwidth=92)
        table.column("count", width=62, minwidth=62, anchor="e", stretch=False)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        table.pack(side="left", fill="both", expand=True)
        return table

    def choose_csv(self):
        path = self.filedialog.askopenfilename(
            parent=self.root, title="点検するCSVを選ぶ", initialdir=self.source.parent if self.source else Path.cwd(),
            filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.load_csv(Path(path))

    def load_csv(self, source):
        self.source = None
        self.target.set("")
        self.split_mode.set(ROW_SPLIT)
        self.path_text.set(str(source))
        self.path_entry.xview_moveto(1)
        self.columns.configure(values=())
        try:
            columns, error = preview_columns(source)
        except OSError as exc:
            self.clear_result(f"CSVを開けませんでした。ファイルの場所とアクセス権を確認してください。\n{exc}")
            return
        self.source = source.resolve()
        self.columns.configure(values=("", *columns) if columns else ())
        self.configure_split_modes()
        if error:
            message = "列名を読み取れませんでした。CSVの文字コードや引用符を点検してください。"
        elif not columns:
            message = "選択できる列名がありません。点検でCSVの状態を確認できます。"
        else:
            message = "正解列を選ぶと、ラベルの内訳も確認できます。"
        self.clear_result(message)

    def configure_split_modes(self, *_):
        target = self.target.get()
        columns = [name for name in self.columns.cget("values") if name and name != target]
        choices = (ROW_SPLIT, IDENTICAL_SPLIT, *(TIME_PREFIX + name for name in columns),
                   *(ID_PREFIX + name for name in columns))
        self.split_picker.configure(values=choices)
        if self.split_mode.get() not in choices:
            self.split_mode.set(ROW_SPLIT)
        self.options_changed()

    def split_column(self, prefix):
        mode = self.split_mode.get()
        return mode.removeprefix(prefix) if mode.startswith(prefix) else None

    def options_changed(self, *_):
        if self.result is not None:
            self.clear_result("選択内容を変更しました。点検を実行してください。")

    def update_controls(self):
        busy = self.job is not None
        self.browse_button.configure(state="disabled" if busy else "normal")
        can_choose = not busy and self.source is not None and bool(self.columns.cget("values"))
        self.columns.configure(state="readonly" if can_choose else "disabled")
        self.split_picker.configure(state="readonly" if can_choose else "disabled")
        self.compress_button.configure(state="disabled" if busy else "normal")
        self.run_button.configure(state="normal" if not busy and self.source is not None else "disabled")
        can_project = not busy and self.original is not None and projection_features(self.original.profile)
        self.select_button.configure(state="normal" if can_project else "disabled")
        has_selection = not busy and self.selection is not None
        blockers = training_blockers(self.selection.audit.profile) if self.selection is not None else []
        self.train_button.configure(state="normal" if has_selection and not blockers else "disabled")
        viewing_evaluation = self.evaluation_panel.frame.winfo_manager() == "grid"
        self.evaluation_button.configure(
            state="normal" if not busy and self.evaluation is not None and not viewing_evaluation else "disabled",
        )
        if blockers:
            self.training_hint.set("学習前に確認\n" + "、".join(ISSUE_LABELS[code] for code in blockers[:2]))
        else:
            self.training_hint.set("")
        self.original_view.configure(state="normal" if has_selection and self.result is not self.original else "disabled")
        self.selected_view.configure(state="normal" if has_selection and self.result is self.original else "disabled")
        self.open_selected_folder.configure(state="normal" if has_selection else "disabled")
        output_state = "normal" if not busy and self.result is not None else "disabled"
        self.open_report.configure(state=output_state)
        self.open_folder.configure(state=output_state)

    def clear_result(self, message):
        self.original = self.selection = self.result = self.evaluation = None
        self.view_title.set("点検プレビュー")
        self.compare_label.grid_remove()
        self.status.set(message)
        self.notes.set("")
        for count in self.counts:
            count.set("—")
        for table in (self.labels, self.issues):
            table.delete(*table.get_children())
        self.show_audit_panel()

    def start_audit(self):
        if self.source is None or self.job is not None:
            return
        self.clear_result("CSVを点検しています…")
        self.start_job(
            save_audit, (self.source, result_path(self.output_dir), self.target.get() or None,
                         self.compress.get(), self.split_column(ID_PREFIX), self.split_column(TIME_PREFIX)),
            self.receive_audit, "点検結果を保存できませんでした。入力ファイルと保存先を確認してください。",
        )

    def choose_projection(self):
        from tkinter import Listbox, Toplevel, ttk

        if self.job is not None or self.original is None:
            return
        features = projection_features(self.original.profile)
        if not features:
            return
        target = self.original.profile["settings"]["target"]
        dialog = Toplevel(self.root)
        dialog.title("学習に使う列を選ぶ")
        dialog.transient(self.root)
        dialog.minsize(360, 290)
        width, height = 440, 440
        left = max(0, self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2)
        top = max(0, self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{left}+{top}")
        panel = ttk.Frame(dialog, padding=18)
        panel.pack(fill="both", expand=True)
        details = f"クリックで使う列を切り替えます。\n正解列「{target}」は抽出CSVに含まれます。"
        if group_column := self.original.profile["settings"]["id_column"]:
            details += f"\nID列「{group_column}」は分割に使います。"
        if time_column := self.original.profile["settings"]["time_column"]:
            details += f"\n日付列「{time_column}」は分割に使います。"
        ttk.Label(panel, text=details, wraplength=390).pack(anchor="w")
        body = ttk.Frame(panel)
        body.pack(fill="both", expand=True, pady=(8, 10))
        choices = Listbox(body, selectmode="multiple", exportselection=False, width=37, height=12,
                          font=("Meiryo UI", 10) if os.name == "nt" else "TkDefaultFont")
        scroll = ttk.Scrollbar(body, orient="vertical", command=choices.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=choices.xview)
        choices.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        choices.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        previous = (set(features) if self.selection is None else
                    {column["name"] for column in self.selection.audit.profile["columns"]})
        for index, feature in enumerate(features):
            choices.insert("end", feature)
            if feature in previous:
                choices.selection_set(index)

        actions = ttk.Frame(panel)
        actions.pack(fill="x")
        ttk.Button(actions, text="キャンセル", command=dialog.destroy).pack(side="right")

        def submit():
            selected = [features[index] for index in choices.curselection()]
            if selected:
                dialog.destroy()
                self.start_projection(selected)

        run = ttk.Button(actions, text="抽出して再点検", style="Primary.TButton", command=submit)
        run.pack(side="right", padx=(0, 8))
        choices.bind("<<ListboxSelect>>",
                     lambda _: run.configure(state="normal" if choices.curselection() else "disabled"))
        dialog.bind("<Escape>", lambda _: dialog.destroy())
        dialog.wait_visibility()
        dialog.grab_set()
        choices.focus_set()

    def start_projection(self, features):
        if self.original is None or self.job is not None:
            return
        self.show_audit_panel()
        self.status.set("選んだ列を抽出し、再点検しています…")
        self.start_job(
            run_projection, (self.original, features, self.split_mode.get() == IDENTICAL_SPLIT), self.receive_selection,
            "抽出と再点検に失敗しました。元CSVを編集した場合は点検し直してください。",
        )

    def start_evaluation(self):
        if self.selection is None or self.job is not None or training_blockers(self.selection.audit.profile):
            return
        self.show_result(self.selection.audit)
        self.show_audit_panel()
        self.status.set("CPUで5モデルを比較し、評価結果をまとめています…")
        output = result_path(self.selection.audit.directory.parent / "evaluations")
        self.start_job(
            evaluate_selection, (self.selection, output), self.receive_evaluation,
            "モデル比較を完了できませんでした。抽出したCSVや保存先を確認してください。",
        )

    def start_job(self, task, args, show, error_message):
        self.job = self.executor.submit(task, *args)
        self.update_controls()
        self.progress.pack(fill="x", pady=(12, 0))
        self.progress.start(16)
        self.root.after(80, self.finish_job, show, error_message)

    def finish_job(self, show, error_message):
        if not self.job.done():
            self.root.after(80, self.finish_job, show, error_message)
            return
        job, self.job = self.job, None
        self.progress.stop()
        self.progress.pack_forget()
        try:
            result = job.result()
        except Exception as error:
            self.status.set(f"{error_message}\n{error}")
            self.update_controls()
        else:
            show(result)
        if self.close_pending:
            self.close()

    def receive_audit(self, original):
        self.original = original
        self.show_result(original)

    def receive_selection(self, selection):
        self.selection = selection
        self.evaluation = None
        self.show_result(selection.audit)

    def receive_evaluation(self, evaluation):
        self.evaluation = evaluation
        self.status.set("モデル比較と評価Excelの保存が完了しました。")
        self.show_evaluation()

    def show_audit_panel(self):
        self.evaluation_panel.hide()
        self.audit_panel.grid()
        self.update_controls()

    def show_evaluation(self):
        if self.evaluation is not None and self.job is None:
            self.audit_panel.grid_remove()
            self.evaluation_panel.show(*self.evaluation)
            self.update_controls()

    def show_result(self, result):
        self.result = result
        profile = result.profile
        self.view_title.set("元CSVの点検" if result is self.original else "抽出後の点検")
        for table in (self.labels, self.issues):
            table.delete(*table.get_children())
        self.notes.set("")
        if self.selection is not None:
            self.show_comparison()
        summary = dict(bp_summary_rows(profile))
        for field, name in zip(self.counts, (
            "列数が合ったデータ行", "欠損セル数", "全列が一致する重複行", "正解ラベルの種類数",
        )):
            value = summary[name]
            field.set(f"{value:,}" if isinstance(value, int) else "—")

        target = profile["target"]
        for item in target["classes"]:
            self.labels.insert("", "end", values=(item["label"], f"{item['rows']:,}"))
        if target["other_classes"]:
            self.labels.insert("", "end", values=(f"ほか {target['other_classes']:,} 種類", f"{target['other_rows']:,}"))
        if target["missing"]:
            self.labels.insert("", "end", values=("欠損ラベル", f"{target['missing']:,}"))
        if target["status"] == "not_declared":
            self.labels.insert("", "end", values=("正解列を選ぶと表示します", ""))
        for issue in profile["issues"]:
            code = issue["code"]
            self.issues.insert("", "end", values=(ISSUE_LABELS.get(code, code), f"{issue['count']:,}"))
        if not profile["issues"]:
            self.issues.insert("", "end", values=("今回の点検で該当なし", ""))

        if profile["scan"]["status"] != "complete":
            self.status.set("読み取れた範囲を表示しています。Excelで詳細を確認できます。")
        elif profile["issues"]:
            self.status.set("点検結果を保存しました。確認する項目が見つかりました。")
        else:
            self.status.set("点検結果を保存しました。")
        if result is self.original and profile["target"]["status"] == "not_declared":
            self.notes.set("学習用の抽出には、正解列を選んで点検してください。")
        conversion = profile.get("conversion")
        if conversion:
            if conversion["status"] == "completed":
                original = summary["元CSVのサイズ（バイト）"]
                saved = summary["圧縮後のサイズ（バイト）"]
                ratio = summary["圧縮後のサイズ（元CSV比 %）"]
                self.notes.set(f"作業用Parquetを保存しました。{original:,} → {saved:,} バイト（元CSV比 {ratio:g}%）")
            else:
                self.notes.set("作業用Parquetを保存できませんでした。ExcelのConversionシートに理由があります。")
        self.update_controls()

    def show_comparison(self):
        before, after = self.original.profile, self.selection.audit.profile

        def metrics(profile):
            return (len(profile["columns"]), sum(col["missing"] for col in profile["columns"]),
                    profile["duplicates"]["features"]["conflicting_label_groups"])

        columns, missing, conflicts = zip(metrics(before), metrics(after))
        self.comparison.set(
            f"元CSV → 抽出後    列数 {columns[0]:,} → {columns[1]:,}    "
            f"欠損 {missing[0]:,} → {missing[1]:,}    ラベル不一致 {conflicts[0]:,} → {conflicts[1]:,} 組"
        )
        self.compare_label.grid()

    def open_result(self, name=None):
        if self.result is None:
            return
        directory = self.result.directory
        self.open_path(directory / name if name else directory)

    def open_path(self, path):
        try:
            open_output(path)
        except OSError:
            note = self.evaluation_panel.notice if self.evaluation_panel.frame.winfo_manager() == "grid" else self.notes
            note.set(f"ファイルを開けませんでした。保存先を確認してください。\n{path}")

    def close(self):
        if self.job is not None:
            self.close_pending = True
            self.status.set("処理結果の保存後に画面を閉じます。")
            return
        self.executor.shutdown(wait=False)
        self.root.destroy()
