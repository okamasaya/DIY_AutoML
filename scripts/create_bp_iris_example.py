"""Build the BP article's Iris workbook using the existing evaluation APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.page import PageMargins
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score

from diy_automl import select_csv, render_excel_report, evaluate_uncertainty
from diy_automl.validation import validate
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.uncertainty_validation import validate_uncertainty


ROOT = Path(__file__).resolve().parents[1]
NAVY, BLUE, PALE = '183A55', '2465A8', 'EEF4F8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def cell_block(sheet, area, value, size=11, bold=False, background=None, color=NAVY):
    sheet.merge_cells(area)
    cell = sheet[area.split(':')[0]]
    cell.value = value
    cell.font = Font(name='Noto Sans CJK JP', size=size, bold=bold, color=color)
    cell.alignment = Alignment(vertical='center', wrap_text=True)
    if background:
        for row in sheet[area]:
            for item in row:
                item.fill = PatternFill('solid', fgColor=background)


def table(sheet, headings, rows):
    sheet.append(headings)
    for row in rows:
        sheet.append([value.item() if isinstance(value, np.generic) else value for value in row])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(name='Noto Sans CJK JP', bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor=NAVY)
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    sheet.row_dimensions[1].height = 36
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name='Noto Sans CJK JP', size=10)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    for index in range(1, len(headings) + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 23 if index > 1 else 28


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'data/reference/iris.csv'
    input_sha = sha(source)
    frame = pd.read_csv(source)
    iris = load_iris()
    np.testing.assert_array_equal(frame.drop(columns='target').to_numpy(), iris.data)
    np.testing.assert_array_equal(frame.target.to_numpy(), iris.target)
    names = list(iris.target_names)
    shutil.copyfile(source, out / 'input.csv')
    select_csv(out / 'input.csv', 'target', out / 'selected', seed=42,
               bootstrap_repeats=1000, bootstrap_seed=43, confidence=.95)
    validate(out / 'selected', 3)
    render_excel_report(out / 'selected', out / 'illustrated', csv=out / 'input.csv')
    validate_excel_report(out / 'illustrated')
    evaluate_uncertainty(out / 'selected', out / 'uncertainty')
    validate_uncertainty(out / 'uncertainty')
    manifest = json.loads((out / 'selected/manifest.json').read_text())
    with np.load(out / 'selected/arrays.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    truth, prediction = arrays['truth'], arrays['predictions']
    wrong = np.flatnonzero(truth != prediction)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
    uncertainty = pd.read_excel(out / 'uncertainty/report.xlsx', sheet_name='Metric_Uncertainty')
    row = uncertainty[uncertainty.Metric.eq('F1')].iloc[0]
    np.testing.assert_allclose(row.Estimate, f1_score(truth, prediction, average='macro'), atol=1e-12)
    n, correct = len(truth), int((truth == prediction).sum())

    # Keep every existing uncertainty sheet and its embedded chart in a new file.
    book = load_workbook(out / 'uncertainty/report.xlsx')
    summary = book.create_sheet('まず読む', 0)
    mistakes = book.create_sheet('間違えた行', 1)
    conditions = book.create_sheet('入力と条件', 2)
    summary.sheet_view.showGridLines = False
    for column in 'ABCDEFGH':
        summary.column_dimensions[column].width = 12
    for index in range(1, 36):
        summary.row_dimensions[index].height = 20
    cell_block(summary, 'A1:H2', 'Iris｜データを入れて、結果を読む', 20, True, NAVY, 'FFFFFF')
    cell_block(summary, 'A3:H3', 'アヤメの4つの寸法から、3つの種類を見分ける出力例', 11)
    for area, label, value in [('A5:B5', '評価したデータ', n),
                               ('C5:E5', '正しく分けた', correct),
                               ('F5:H5', '確認したい誤り', len(wrong))]:
        cell_block(summary, area, label, 11, True, PALE)
        start, end = area.split(':')
        cell_block(summary, f'{start[0]}6:{end[0]}7', f'{value} 件', 25, True, PALE)
    cell_block(summary, 'A9:H10',
        f'入力{len(frame)}件 → 学習側{len(arrays["train"])}件で5モデルを比較 → 保留した{n}件で最終評価。'
        f'学習側の基準で選ばれたモデルは{manifest["selected_model"]}です。')
    cell_block(summary, 'A12:D12', '本当の種類', 11, True, NAVY, 'FFFFFF')
    cell_block(summary, 'E12:F12', '評価件数', 11, True, NAVY, 'FFFFFF')
    cell_block(summary, 'G12:H12', '別の種類と判定', 11, True, NAVY, 'FFFFFF')
    for index, name in enumerate(names, 13):
        c = index - 13
        cell_block(summary, f'A{index}:D{index}', f'{c}: {name}')
        cell_block(summary, f'E{index}:F{index}', f'{int(matrix[c].sum())} 件')
        cell_block(summary, f'G{index}:H{index}', f'{int(matrix[c].sum()-matrix[c,c])} 件')
    cell_block(summary, 'A17:H17', '今回の成績と、その不確かさ', 12, True, NAVY, 'FFFFFF')
    entries = [('正しく分けた割合', f'{correct/n:.1%}'),
               ('比較用スコア（macro F1）', f'{row.Estimate:.3f}'),
               ('スコアの標準誤差', f'{row["Bootstrap SE"]:.3f}'),
               ('スコアの条件付き95%区間', f'{row.Lower:.3f} 〜 {row.Upper:.3f}')]
    for index, (label, value) in enumerate(entries, 18):
        cell_block(summary, f'A{index}:E{index}', label)
        cell_block(summary, f'F{index}:H{index}', value, 12, True)
    cell_block(summary, 'A23:H24',
        '幅は、固定したモデルと評価データのクラス別件数に条件付けたものです。'
        '学習のやり直しや、別の場所・時期で使うときの変化は含みません。', 10)
    cell_block(summary, 'A26:H26', 'ここから確認したいこと', 12, True, NAVY, 'FFFFFF')
    cell_block(summary, 'A27:H28',
        f'まずは「間違えた行」の{len(wrong)}件を見て、入力・正解・判定を確認できます。'
        '予測との不一致だけで、正解ラベルの誤りとは決めません。')
    cell_block(summary, 'A29:H30',
        f'今回の評価は{n}件です。仕事のデータで使う前に、対象・測り方・正解の意味が'
        '用途に合うかを確認し、別データでも確かめる材料にします。')
    cell_block(summary, 'A32:H33',
        '詳しく見る：入力と条件／Metric_Uncertainty（数値）／Class_Uncertainty（種類別）／'
        'Intervals（区間の図）。元の比較結果と予測も別ファイルで保存しています。', 10)
    cell_block(summary, 'A35:H35', 'DIY AutoML 0.4.0a7｜記事用スクリプトで日本語の要約シートを追加した実物', 9)
    summary.print_area = 'A1:H35'
    summary.sheet_properties.pageSetUpPr.fitToPage = True
    summary.page_setup.orientation = 'portrait'
    summary.page_setup.paperSize = summary.PAPERSIZE_A4
    summary.page_setup.fitToWidth = 1
    summary.page_setup.fitToHeight = 1
    summary.page_margins = PageMargins(left=.3, right=.3, top=.3, bottom=.3, header=.1, footer=.1)
    summary.freeze_panes = 'A9'

    mistake_rows = []
    for i in wrong:
        original = int(arrays['test'][i])
        mistake_rows.append([original + 2, names[int(truth[i])], names[int(prediction[i])],
                              *frame.drop(columns='target').iloc[original].tolist()])
    table(mistakes, ['入力CSVの行番号（ヘッダ=1）', '本当の種類', '予測した種類', *manifest['features']], mistake_rows)
    settings = [('入力件数', len(frame)), ('数値特徴量の数', len(manifest['features'])),
                ('学習側の件数', len(arrays['train'])), ('評価用の件数', n),
                ('選定モデル', manifest['selected_model']), ('分割seed', 42),
                ('入力CSVの欠損セル数', int(frame.isna().sum().sum())),
                ('完全に同じ内容の行の重複数', int(frame.duplicated().sum())),
                ('重複の扱い', 'この見本では削除していません。重複だけで不良とは判定しません。'),
                ('モデルの選び方', '学習側の共通5分割でmacro F1の平均を比較。testで選び直さない。'),
                ('区間の計算', '固定モデル・クラス件数固定の行bootstrap、1,000反復、seed 43。'),
                ('未確認のこと', '新しい場所・時期での性能、入力の測定誤差、数値計算条件の感度、系統的な偏り。'),
                ('入力ファイル', 'input.csv'), ('入力SHA256', input_sha),
                ('元の選定結果', 'selected/report.xlsx'), ('元の図付きExcel', 'illustrated/report_with_figures.xlsx'),
                ('元の誤差評価', 'uncertainty/report.xlsx'), ('再標本化・予測等', 'uncertainty/arrays.npz'),
                ('作成コード', 'scripts/create_bp_iris_example.py')]
    table(conditions, ['確認項目', '今回の内容'], settings)
    conditions.column_dimensions['B'].width = 95
    for row_index in range(2, conditions.max_row + 1):
        conditions.row_dimensions[row_index].height = 32
    book.active = 0
    workbook = out / 'iris_business_report.xlsx'
    book.save(workbook)
    book.close()

    # A new cover must not change the existing numerical results.
    source_book = load_workbook(out / 'uncertainty/report.xlsx')
    result_book = load_workbook(workbook)
    for name in source_book.sheetnames:
        assert list(source_book[name].values) == list(result_book[name].values), name
    assert result_book['間違えた行'].max_row - 1 == len(wrong)
    assert sum(int(matrix[c,c]) for c in range(3)) == correct
    source_book.close(); result_book.close()
    assert sha(source) == input_sha == sha(out / 'input.csv')
    summary_data = dict(status='passed', input_rows=len(frame), features=len(manifest['features']),
        train_rows=len(arrays['train']), test_rows=n, correct=correct, wrong=len(wrong),
        selected_model=manifest['selected_model'], accuracy=correct/n, macro_f1=float(row.Estimate),
        bootstrap_se=float(row['Bootstrap SE']), lower=float(row.Lower), upper=float(row.Upper),
        classes=names, confusion_matrix=matrix.tolist(), mistakes=mistake_rows,
        source_input_sha256=input_sha, script_sha256=sha(__file__),
        original_uncertainty_cells_unchanged=True,
        workbook_sha256=sha(workbook), kind='Article-specific Japanese cover over verified standard outputs; no GUI')
    write_json(out / 'example_summary.json', summary_data)
    print(json.dumps(summary_data, ensure_ascii=False))


if __name__ == '__main__':
    main()
