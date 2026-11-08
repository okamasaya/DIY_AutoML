# 共通関数をNotebookから使う

`notebooks/Local_Comparison.ipynb` はCSVの読込・SL/UL/DLの比較・元の5種類の図・図付きExcelの生成を共通関数で実行する入口です。Notebook内に別の学習・選定ロジックを作りません。元NotebookはoriginalsのZIP内に保全しています。

Python 3.12/Linux CPUの検証環境をそろえる場合:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt -r requirements-notebook.lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

普段使うエディタで、この.venvをNotebookのカーネルに指定します。別途カーネル登録を使う場合は、利用者のJupyter設定に合わせて登録してください。今回の検証ではプロジェクト外の一時prefixに登録した専用カーネルを使い、原稿や共通コードへ環境固有の登録先は埋め込んでいません。

STEP1でCSVと出力先、STEP2〜6のセルでtarget列を指定し、順に実行します。既定例は同梱Wine CSV・target列target・100epochです。Notebook再実行時はtimestamp付きの別フォルダを作り、既存出力を上書きしません。

- 数値結果はOUTPUT/report.xlsx。
- 外部PNG5枚はOUTPUT/figures/。
- 図付きExcelはOUTPUT名に-illustratedを付けた別フォルダのreport_with_figures.xlsx。

ここでのrunは探索的比較です。学習側CVによる選定には、同じカーネルからselect_csv/tune_csvを呼ぶかCLIを使います。Notebookからの選定でもtestを候補探索に使わない規則は変わりません。

DS向け子記事ではNotebook/CLIのどちらも選べる使い方を説明する方針です。現在の親記事・子記事・アプリの位置付けはdocs/AUDIENCE_PLAN.mdを参照してください。ビジネス向けのファイル投入画面は未実装です。
