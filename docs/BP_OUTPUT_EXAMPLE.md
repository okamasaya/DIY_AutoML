# BP記事の出力見本と確認範囲

2026年11月12日（日本時間）。対象は `articles/diy-automl-business-three-experiments.md`。「誤差を付けることを義務として説明する」のではなく、何が不確実かを見える形にし、次の判断や確認へつなげる読み物へ更新した。

## Irisの実物

- [日本語要約付きExcel](../validation/business-output-20261112/iris/iris_business_report.xlsx)
- [先頭シートのPDF](../validation/business-output-20261112/iris/summary.pdf)
- [先頭シートを表示したPNG](../images/business-output-20261112/iris-report.png)
- [機械可読な要約](../validation/business-output-20261112/iris/example_summary.json)

現行の `select_csv` → `render_excel_report` → `evaluate_uncertainty` と各検証器を実行し、その誤差評価Excelを元に記事用の日本語シートを追加した。標準出力の既存セルは保持している。一般向けのファイル投入画面や、汎用の業務説明生成を実装した意味ではない。

入力はIris 150件・4特徴量、学習側120件、test30件。学習側CVで選定したSVMは29件正解・1件誤り、Accuracy=0.966667、macro F1=0.966583。F1のbootstrap SE=0.032765、条件付き95%区間=[0.897698, 1.000000]。親記事のMLPとは別の学習例である。

日本語の先頭3シートは「まず読む」「間違えた行」「入力と条件」。誤りはCSVの79行目（ヘッダを1行目と数える）のversicolor→virginicaで、元の寸法も記載した。スコアの高低から正解ラベルの誤りや業務での有用性を自動判定していない。

IrisのCSVとscikit-learnの組込みデータの値・ラベルを完全一致で照合して種類名を付けた。元の誤差評価Excelの全シート・セル値を、新しい要約付きExcelと照合した。元の入力CSVや過去の実験出力は保持している。

## 再作成

Python環境はREADMEのコア依存で実行できる。リポジトリルートから未作成の出力先を指定する。

```sh
MPLCONFIGDIR=/tmp/diy-bp-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/create_bp_iris_example.py --output runs/bp-iris-new
```

元の学習結果・図付きExcel・誤差評価・配列等も出力先へ保存される。作成コードは `scripts/create_bp_iris_example.py`。Iris専用のラベル対応・説明を含む記事用アダプタで、汎用的な品質診断や業務効果の説明を自動化したものではない。

記事のPNGは、実際のxlsxをLibreOfficeで開き、先頭シートをPDFへ書き出してPNGに変換したもの。手描きの見本ではない。今回の環境はLibreOfficeDev 26.8.0.0.alpha0で、PDFはA4・1ページ。Excel本体や他のOffice版での同じピクセル配置までは保証しない。

```sh
mkdir -p runs/bp-iris-new/pdf
soffice -env:UserInstallation=file:///tmp/diy-bp-lo-profile-new --headless --convert-to 'pdf:calc_pdf_Export:{"PageRange":{"type":"string","value":"1"}}' --outdir runs/bp-iris-new/pdf runs/bp-iris-new/iris_business_report.xlsx
pdftoppm -f 1 -singlefile -r 150 -png runs/bp-iris-new/pdf/iris_business_report.pdf runs/bp-iris-new/iris-report
```

## 業務3テーマの差の区間

元の実験のモデルと、事前に決めた参考ルールの保存済み予測を使用した。参考ルールはホテルの日数、カフェ券の習慣と状況、通販の過去2回以上購入。現行パッケージのモデル比較機能を追加したものではなく、記事用の補足集計である。

```sh
MPLCONFIGDIR=/tmp/diy-bp-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/render_bp_comparisons.py --output runs/bp-comparison-new --image runs/bp-comparison-new.png
```

2,000反復・seed 43、正解クラス別件数を固定し、同じ評価行で両方法を比べるpaired percentile bootstrap。差は「モデル−ルール」のmacro F1。結果は [comparison.json](../validation/business-output-20261112/comparison/comparison.json)、反復indexと差は同じフォルダの `paired_bootstrap.npz`。9反復をscikit-learnの独立計算経路で照合した。

| 題材 | 差 | 条件付き95%区間 |
| --- | ---: | --- |
| ホテル | −0.044229 | −0.108544〜＋0.017537 |
| カフェ券 | ＋0.040349 | ＋0.001280〜＋0.078437 |
| 再購入 | ＋0.010542 | −0.041616〜＋0.058960 |

全範囲−1〜1の図を先に生成・確認し、すべての区間が収まる−0.13〜0.13へ拡大した。既存の修正方針に記録された丸め値とも一致する。カフェ券の行の依存、ホテルの顧客/団体、学習・選定のやり直し、別集団・期間、因果効果を含む評価ではない。

## 記事と今後の範囲

3テーマは「今回の結果→差の不確かさ→次に確かめたいこと」で整理した。正解率とmacro F1を区別し、モデルとルールの採用判断や利益を自動で結論しない。通販も、後の期間を正解に使うことと、予測時点をずらす時系列検証を区別した。

過去のトレンド・季節性の整理と未来予測は別課題として、今後の予定へ記載した。アプリ本体の処理・依存は変更していない。文書・出力の検証記録は [editorial-bp-output-20261112.json](../tests/results/editorial-bp-output-20261112.json) に保存する。
