# 2026年11月19日の作業ログ・再開手順

## 今日の区切り

- **分類評価のマイルストーン**を #2 で `main` に統合した。BP向けに、CSVの点検、任意のParquet圧縮、列を選んで再点検、行・グループ・日付順の分割による5モデル比較、日本語の評価Excelまでつないだ。 [マイルストーン](BP_CLASSIFICATION_MILESTONE_20261119.md) · [PR #2](https://github.com/okamasaya/DIY_AutoML/pull/2)
- **Letterは2版とも保存**した。Full Paper初稿はPDF・編集用TeX・復元用ZIPを入れ、`55a94ac` で `main` に統合した。 [LetterのPR #3](https://github.com/okamasaya/DIY_AutoML/pull/3) · [Full PaperのPR #4](https://github.com/okamasaya/DIY_AutoML/pull/4)
- **Full Paperの改訂稿**では、図1（3ページ目）にCSV点検、Parquet・DuckDBへの任意変換、共通の学習と評価、DS・BP向けの出力分岐を載せた。10万行以上または20列以上は中継ファイルを検討する例とし、再利用と利用者の指定を変換判断に含めた。学習では評価対象の行を原則全件使う。モデル順位と実際の採用判断を分け、差は同じ分割・指標で示す。全12ページを組版・確認し、ZIPから復元した原稿でも同じ本文とページ内容が得られることを確認した。 [改訂稿PDF](papers/DIY_AutoML_Full_Paper_20261119/DIY_AutoML_Full_Paper_20261119.pdf) · [TeXプロジェクト](papers/DIY_AutoML_Full_Paper_20261119/README.md) · [復元用ZIP](papers/DIY_AutoML_Full_Paper_20261119/DIY_AutoML_Full_Paper_20261119_TeX.zip)
- **BP画面の「グラフを見る」**で、候補5モデルの学習側CVスコアを棒グラフにした。日付順の評価では選定モデルと単純ルールの保留テストを月別の折れ線で表示する。KPIは評価画面とExcelの「まず読む」にある。WindowsのPython 3.13.17でIrisの30保留行とBike Sharingの582保留行・5か月を使い、結果の表示、グラフの開き直し、保存ファイルの一致を確認した。関連テストは44件通過。 [画面の起動と使い方](DATA_AUDIT.md#データ点検の画面を使う)
- **自由に作図する機能はTo Do**。Python 3.14.7とPyGWalker 0.5.0.1で、Irisの評価表からローカル用HTMLを生成し、5候補の値を照合した。Streamlitと組み合わせた作図画面、初回のWindowsバッチによる依存導入、BP画面からのワンクリック起動は今後の検証項目。 [詳細グラフのTo Do](ROADMAP.md#bp向けの詳細グラフto-do)

## 保存状態

この区切りの起点はFull Paper初稿を統合した `55a94ac`。利用者の確認後、処理フロー図付きの改訂稿とPDF・TeXの復元用ZIP、BPグラフの実装・テスト、詳細グラフのTo Do、作業ログと引継ぎを `main` への統合対象にまとめた。対応するコミットとマージの記録はGit履歴で確認する。

論文改訂稿の確認先は `docs/papers/DIY_AutoML_Full_Paper_20261119/`。`output/` には初稿作業時の生成物がある。BP画面の実装は `src/diy_automl/bp_preview.py`、`bp_results.py`、`bp_charts.py`。Windowsの動作確認用出力はローカルの `runs/bp-visual-smoke/` にある。

## 次回の再開点

1. `git status --short --branch` と `git log -5 --oneline` で保存状態を照合する。改訂稿とBPグラフの変更は対応するPRまたはGit履歴で確認する。
2. BP画面を確認する場合は、下のコマンドでIrisを開く。正解列に `target` を選び、点検 → 使う列を選ぶ → 5モデルを比較する → グラフを見る。KPIと評価Excelも確認する。日付順の再確認には [Bike Sharingの手順](DATA_AUDIT.md#日付順の分割) を使う。
3. 評価基盤の次の開発は、日付順の保留データでモデルと単純ルールを比較した差の区間から再開する。連続する観測日のまとまり、欠けた観測日、ラベル確定までの遅れを扱う。PyGWalker＋StreamlitはBP向けの追加候補としてロードマップに置く。 [次の推奨順](ROADMAP.md#次の推奨順)

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl bp-preview --csv data/reference/iris.csv
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests/test_bp_evaluation.py tests/test_bp_data_preparation.py tests/test_chronological_selection.py tests/test_grouped_selection.py tests/test_excel_report.py
~~~
