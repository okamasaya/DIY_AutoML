"""Present the existing supervised comparison in business-facing terms."""

import json

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from threadpoolctl import threadpool_limits

from .calendar_comparison import BY_MONTH, BY_MONTH_CLASS, COMPARISON, METHODS
from .data_audit import excel_value


BP_REPORT = "bp_report.xlsx"
TRAINING_BLOCKERS = {
    "incomplete_input", "no_valid_rows", "row_width", "duplicate_header", "empty_header",
    "target_not_found", "target_ambiguous", "target_not_declared", "fewer_than_two_classes",
    "classes_under_six_rows", "missing_target", "no_feature_columns", "missing_feature_values",
    "nonfinite_feature_values", "nonnumeric_feature_columns",
    "missing_identifier", "time_not_found", "time_ambiguous", "missing_timestamp", "unparsed_timestamp",
    "mixed_timezones",
}
MODEL_LABELS = {
    "Decision Tree": "決定木",
    "Random Forest": "ランダムフォレスト",
    "Logistic Regression": "ロジスティック回帰",
    "SVM": "SVM",
    "k-NN": "k近傍法",
}
CLASS_HEADERS = ("正解ラベル", "評価した行", "正解", "見逃し", "誤検知")


def training_blockers(profile):
    return [issue["code"] for issue in profile["issues"] if issue["code"] in TRAINING_BLOCKERS]


def evaluate_selection(selection, output):
    from .selection_workflow import select_csv

    profile = selection.audit.profile
    if training_blockers(profile):
        raise ValueError("抽出後の点検で、学習に必要な確認項目が見つかりました。")
    with threadpool_limits(limits=1):
        directory = select_csv(
            selection.csv, profile["settings"]["target"], output,
            expected_input_sha256=profile["input_sha256"], bp_summary=True,
            group_column=profile["settings"]["id_column"], group_identical=selection.group_identical,
            time_column=profile["settings"]["time_column"],
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return directory, manifest["bp_summary"]


def summarize_evaluation(data, arrays, test_scores, intervals, manifest, comparison=None):
    train, test = arrays["train"], arrays["test"]
    truth, predictions = arrays["truth"], arrays["predictions"]
    classes = data.classes
    hits = predictions == truth
    support = np.bincount(truth, minlength=len(classes))
    called = np.bincount(predictions, minlength=len(classes))
    matched = np.bincount(truth[hits], minlength=len(classes))
    majority = int(np.argmax(np.bincount(data.y[train], minlength=len(classes))))
    left = pd.MultiIndex.from_frame(pd.DataFrame(data.X[train]).drop_duplicates())
    right = pd.MultiIndex.from_frame(pd.DataFrame(data.X[test]))
    model = manifest["selected_model"]
    score = None if intervals.empty else intervals.set_index("Metric").loc["F1"]
    settings = manifest["settings"]
    result = {
        "model": MODEL_LABELS.get(model, model),
        "train_rows": len(train),
        "test_rows": len(test),
        "correct": int(hits.sum()),
        "incorrect": int((~hits).sum()),
        "majority_label": classes[majority],
        "majority_correct": int(support[majority]),
        "test_rows_also_in_train": int(right.isin(left).sum()),
        "f1": float(test_scores["F1"]),
        "f1_lower": None if score is None else float(score.Lower),
        "f1_upper": None if score is None else float(score.Upper),
        "confidence": settings.get("confidence"),
        "seed": settings["seed"],
        "bootstrap_repeats": settings.get("bootstrap_repeats", 0),
        "group_column": settings.get("group_column"),
        "split": manifest.get("split"),
        "classes": [
            (label, int(total), int(correct), int(total - correct), int(positive - correct))
            for label, total, correct, positive in zip(classes, support, matched, called)
        ],
    }
    if comparison is not None:
        scores = comparison.set_index("Method")
        result["calendar"] = {
            **manifest["calendar_rule"],
            "correct": int(scores.at[METHODS[1], "Correct"]),
            "f1": float(scores.at[METHODS[1], "F1"]),
            "majority_f1": float(scores.at[METHODS[2], "F1"]),
        }
    return result


def split_caption(result):
    split = result.get("split")
    if not split:
        return "行単位の層化80/20分割"
    if split["mode"] == "chronological":
        training, test = split["training"], split["test"]
        zone = "（UTC）" if split["clock"] == "utc" else ""
        return (f"学習 {training['start']}〜{training['end']}／"
                f"評価 {test['start']}〜{test['end']}{zone}")
    unit = (f"ID列「{result['group_column']}」" if split["mode"] == "group_column" else "同じ特徴量の組")
    return f"{unit}で分割（学習 {split['train_groups']:,} 組／保留 {split['test_groups']:,} 組）"


def interval_caption(result):
    if result["confidence"] is None:
        return "時期で分けた評価では、保留期間の性能値を表示します。評価区間は未算出です。"
    if result.get("split"):
        return "固定したモデルを対象に、保留テストをグループ単位で再標本化した区間です。"
    return "固定したモデルを対象に、保留行をクラス内で再標本化した区間です。"


def overlap_caption(result):
    if result.get("split") and result["split"]["mode"] == "chronological":
        return "曜日など、別の日でも同じ入力値になる場合があります。"
    return "各行の由来と評価単位を確認してください。"


def calendar_label(calendar):
    return "曜日・時間帯の多数決" if calendar["uses_hours"] else "曜日別の多数決"


def write_bp_report(directory, result, sheets):
    book = load_workbook(directory / "report.xlsx")
    try:
        overview = book.create_sheet("まず読む", 0)
        model_rows = [
            ("評価する項目", "保留テストの結果"),
            ("選ばれたモデル", result["model"]),
            ("学習行", result["train_rows"]), ("評価した行", result["test_rows"]),
            ("正解", result["correct"]), ("誤分類", result["incorrect"]),
            ("学習側の最多ラベル", result["majority_label"]),
            ("最多ラベルだけで答えた場合の正解", result["majority_correct"]),
            ("学習側にも同じ入力があった評価行", result["test_rows_also_in_train"]),
            ("クラスを同じ重みで平均したF1", result["f1"]),
        ]
        if result["confidence"] is None:
            model_rows.extend([
                ("比較した範囲", split_caption(result)),
                ("モデルの選び方", "学習期間を過去から伸ばし、続く期間で5回比較。"),
                ("統計的なばらつき", interval_caption(result)),
            ])
        else:
            confidence = f"{100 * result['confidence']:g}%"
            model_rows.extend([
                (f"{confidence}区間の下端", result["f1_lower"]),
                (f"{confidence}区間の上端", result["f1_upper"]),
                ("比較した範囲", f"{split_caption(result)}。seed {result['seed']}。学習側の5分割CVで選定。"),
                ("区間の範囲", f"{interval_caption(result)}（{result['bootstrap_repeats']:,}回）"),
            ])
        if calendar := result.get("calendar"):
            label = calendar_label(calendar)
            sources = calendar["source_rows"]
            model_rows.extend([
                (f"{label}の正解", calendar["correct"]),
                (f"{label}のmacro F1", calendar["f1"]),
                ("ルールが使った条件", f"曜日と時間帯 {sources['weekday_hour']:,}行、曜日のみ {sources['weekday']:,}行、"
                                     f"全体の多数決 {sources['overall']:,}行"),
                ("内訳と比較表", "単純ルール比較、月別、月別・ラベル別の各シートを参照。"),
            ])
        else:
            model_rows.append(("内訳と比較表", "ラベル別シートとSL_CV_Meanシートを参照。"))
        if result["test_rows_also_in_train"]:
            model_rows.append(("同じ入力の確認", overlap_caption(result)))
        append_rows(overview, model_rows, (40, 76))
        per_class = book.create_sheet("ラベル別", 1)
        append_rows(per_class, [CLASS_HEADERS, *result["classes"]], (32, 16, 16, 16, 16))
        if calendar:
            write_calendar_sheets(book, sheets, result)
        book.active = 0
        book.save(directory / BP_REPORT)
    finally:
        book.close()


def write_calendar_sheets(book, sheets, result):
    labels = [row[0] for row in result["classes"]]
    methods = dict(zip(METHODS, (result["model"], calendar_label(result["calendar"]), "全体の多数決")))
    scores = {"Method": "予測方法", "Rows": "評価した行", "Correct": "正解", "Accuracy": "正解率", "F1": "macro F1"}
    views = (
        (COMPARISON, "単純ルール比較", scores, (27, 14, 14, 14, 14)),
        (BY_MONTH, "月別", {"Month": "年月", "First date": "評価開始日", "Last date": "評価終了日",
                            "Observed days": "観測日数", **scores}, (12, 15, 15, 14, 27, 14, 14, 14, 14)),
        (BY_MONTH_CLASS, "月別・ラベル別",
         {"Month": "年月", "Method": "予測方法", "Class index": "正解ラベル", "Support": "該当する行",
          "Correct": "正解", "Missed": "見逃し", "False positives": "誤検知", "Recall": "検出率"},
         (12, 27, 20, 15, 14, 14, 14, 14)),
    )
    for index, (source, title, columns, widths) in enumerate(views, start=2):
        frame = sheets[source][list(columns)].copy()
        frame["Method"] = frame["Method"].map(methods)
        if "Class index" in frame:
            frame["Class index"] = [labels[i] for i in frame["Class index"]]
        frame = frame.astype(object).where(frame.notna(), None)
        sheet = book.create_sheet(title, index)
        append_rows(sheet, [tuple(columns.values()), *frame.itertuples(index=False, name=None)], widths)
        sheet.auto_filter.ref = sheet.dimensions
        for column, field in enumerate(columns, start=1):
            if field in ("Accuracy", "Recall", "F1"):
                for cells in sheet.iter_cols(min_col=column, max_col=column, min_row=2):
                    for cell in cells:
                        cell.number_format = "0.000" if field == "F1" else "0.0%"


def append_rows(sheet, rows, widths):
    sheet.freeze_panes = "A2"
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row, start=1):
            cell = sheet.cell(row_index, column, excel_value(value))
            if isinstance(value, str):
                cell.data_type = "s"
            if row_index == 1:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="0A6D60")
