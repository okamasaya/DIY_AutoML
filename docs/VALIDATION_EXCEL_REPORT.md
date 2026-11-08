# 図付きExcelとNotebookの検証（0.4.0a5）

保存済み結果から図付きExcelを生成するreportを追加しました。元の数値Excel・配列・学習manifestを残し、図の用途を入口ごとに分けています。親記事の本文はレビュー済みの0.4.0a4原稿のまま保全し、DS/ビジネス向け子記事の方針を別文書へ記録しました。

## 確定結果

- 開発環境134 passed、独立wheel環境134 passed。失敗・error・skip 0、両環境pip check合格。
- Iris/Wine/Breast Cancer × 6入口の保存結果 × 2環境 = 36個の図付きExcel、計108枚の埋込PNGを検証。
- 対応する描画入力の全配列とPNGバイト列が環境間で一致。
- 全出力で元の数値セル・シート名/順序が不変、画像の配置・大きさ・元PNGとの一致を検証。
- 新たな3データ×2環境のrunはDL100epochを含め実行。全既存モデル配列は初回再実装の保存値と一致し、追加した特徴量/target配列も入力CSVと一致。
- Notebookは専用のipykernelでコード5セルをすべて実行。数値出力、従来の外部PNG5枚、図付きExcel8枚、保存結果検証まで成功。
- NotebookのWine実行はCLIのWine実行と全モデル配列・時間以外の数値表が完全一致。
- TensorFlow未導入のコア環境でも、選定済みSLと保存済みDL反復の図付きExcelを生成・検証できた。

環境はPython 3.12.14 / Linux CPU、matplotlib 3.10.8、openpyxl 3.1.5、Pillow 13.0.0です。DLはTensorFlow CPU 2.20.0 / Keras 3.15.1。Notebook実行ツールの版はrequirements-notebook.lock.txtとnotebook-validation.jsonへ記録しています。

## テストした境界

再学習を禁止した状態でも描画できること、正解とのconfusion matrixと元のスコア、内部SilhouetteだけによるULの表示枠、選定済み1モデルと全seedの表示方針を確認しました。

図付きブックだけを別フォルダへコピーしても、画像データがブックに内包されていることを確認しました。元セルの書換え、画像の削除・表への重なり・描画入力/PNG/表示方針の改変を検知するテストを含みます。

1特徴量/定数特徴量、正解や投影不足、失敗した元結果、描画中の例外、出力上書き拒否も検証しました。凡例生成で使用したmatplotlib引数の互換性問題は開発中のテストで検出・修正し、最終版の全134件は両環境で通過しています。

## 出力サイズと時間

次表は開発環境での保存済み結果検証・図生成・Excel保存・描画結果検証を含む秒数です。元のモデル学習時間は含めません。同時に検証を実行した環境内の実測で、速度比較や時間保証ではありません。

| Dataset | 元の入口 | 図数 | 図付きExcel bytes | 検証を含む秒数 |
| --- | --- | ---: | ---: | ---: |
| iris | run | 8 | 480589 | 2.85 |
| iris | select | 2 | 87821 | 2.08 |
| iris | repeat-ul | 2 | 217154 | 2.10 |
| iris | repeat-dl | 2 | 298408 | 2.48 |
| iris | subsample-ul | 2 | 527113 | 10.17 |
| iris | tune | 2 | 141366 | 4.33 |
| wine | run | 8 | 504312 | 2.78 |
| wine | select | 2 | 89536 | 2.26 |
| wine | repeat-ul | 2 | 223933 | 2.11 |
| wine | repeat-dl | 2 | 272180 | 2.27 |
| wine | subsample-ul | 2 | 550328 | 10.02 |
| wine | tune | 2 | 145358 | 4.13 |
| breast_cancer | run | 8 | 644140 | 2.84 |
| breast_cancer | select | 2 | 96139 | 2.01 |
| breast_cancer | repeat-ul | 2 | 218725 | 2.12 |
| breast_cancer | repeat-dl | 2 | 243620 | 2.36 |
| breast_cancer | subsample-ul | 2 | 575773 | 11.25 |
| breast_cancer | tune | 2 | 153234 | 4.88 |

xlsx全体のバイト列は元結果の時間セルやZIP時刻などで環境間差がありえます。各出力では元の数値表と一致し、図の入力配列とPNGは今回の2環境で一致した、という検証です。フォント・描画ライブラリ・ハードウェアが異なる環境まで同一ピクセル出力を保証するものではありません。Windows/Microsoft Excel実機での表示は未検証です。

## 記録と再実行

- tests/results/0.4.0a5/: JUnit、機械可読な集計、実行済みNotebook、Notebook/CLI一致とTensorFlowなし環境の記録。
- validation/excel-report-0.4.0a5/: 開発環境の18個の図付き結果と元の新規run。
- validation/notebook-0.4.0a5/: Notebookで生成した数値結果・外部図・図付きExcel。

```sh
.venv/bin/python scripts/verify_excel_reports.py --output /新規保存先/development
別wheel環境/bin/python scripts/verify_excel_reports.py --output /新規保存先/wheel --compare /新規保存先/development
```

独立wheel環境はsystem-site-packagesを使用せず、ソースをPYTHONPATHへ追加せずに実行しました。実行済み結果は開発用データの検証であり、未見データでの外部妥当性を示すものではありません。画面アプリ・子記事・Windows/exe・任意グリッドやDLの設定探索は今回の実装範囲に含めていません。
