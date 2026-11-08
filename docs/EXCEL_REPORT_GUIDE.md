# 保存済み結果から図付きExcelを作る（0.4.0a5）

`report` は学習をやり直さず、completedの保存結果から図付きExcelを作る入口です。元の数値Excel・NPZ・学習manifestをコピーして残し、別ファイル `report_with_figures.xlsx` にPNGを埋め込みます。図付きExcel単独でも画像を表示でき、外部PNGへの参照には依存しません。

```sh
.venv/bin/python -m diy_automl select --csv data/reference/wine.csv --target target --output runs/wine-select
.venv/bin/python -m diy_automl report --run runs/wine-select --output runs/wine-select-illustrated
.venv/bin/python scripts/validate_excel_report.py runs/wine-select-illustrated
```

Python入口は `render_excel_report(source, output, max_models=3, csv=None)` です。sourceはmanifest.json/report.xlsx/arrays.npzを含むフォルダ、outputは新規フォルダだけを受け付けます。図の生成に分類器やクラスタモデルの再学習・再予測は行いません。ULの2次元表示には、保存された標準化に基づくPCAを計算します。

## 表示する図と選び方

| 元の入口 | 図 | 既定枚数 |
| --- | --- | ---: |
| run | SLは宣言順の最初3モデルのconfusion matrix、ULは内部Silhouette順の最大3候補の投影と指標、DLは固定1モデルのconfusion matrixと学習曲線 | 最大8 |
| select / tune | 選定済み1モデルのconfusion matrix、全候補の学習側CV平均とfold SD | 2 |
| repeat-ul | KMeans/GMMのTarget ARI/Silhouetteのseed間平均・SD、割当一致の対ごとの平均・SD | 2 |
| repeat-dl | 全学習seedのtrain/validation曲線、全seedのtest4指標 | 2 |
| subsample-ul | 参照/正解との一致の部分集合間平均・SD、共通行の割当一致の対ごとの平均・SD | 2 |

runのSL表示はtestスコアの順位ではありません。ULの最大3件は内部指標による「表示例」であり、モデルの採用・kの決定ではありません。ARIは外部診断として併記します。Silhouetteが同じなら元の候補順で表示枠を割り当てます。SL/DL/ULを混ぜた総合順位はありません。

`--max-models 1` .. `3` はrunのSL/ULの表示例数を制御します。CVの候補や反復のseedはこの引数で間引きません。反復系は全seedを表示し、最高test seedを選びません。表示する量と、不確かさの対象（点・fold・seed・部分集合・依存する対）を図内に記載します。未定義のSDはゼロ幅のエラーバーへ置き換えません。

## 配置と検証

元の数値セル・シート名・順序を維持し、短い表や幅広い表は下、それ以外は右の空き領域へ配置します。複数図の間隔は行高を考慮して決めます。描画用のcaptionはPNG内に含め、データ表のセルへ説明行を追加しません。

出力は次の構成です。

- `report.xlsx` / `arrays.npz` / `manifest.json`: 元の学習結果をそのままコピー。manifestのstatus/version/schemaは学習時のものです。
- `report_with_figures.xlsx`: 数値表を保った図付きExcel。
- `figures/*.png`: Excelに埋め込んだものと同じPNG。
- `figure_data.npz`: confusion matrix、投影座標、ラベル、CV値・SD、学習曲線など実際の描画入力。
- `render_manifest.json`: schema 8。描画の状態、元結果の版/ハッシュ、表示方針、画像の配置・大きさ・ハッシュ、描画ライブラリ版、描画コードのハッシュを記録。

`validate_excel_report.py` は元結果の検証に加え、描画入力が元の予測や表と対応すること、数値セルが変わっていないこと、画像が表や他の図と重ならないこと、Excel内部のPNGが保存PNGと同じことを確認します。PNGの内容そのものをOCRで理解する検証ではありません。コピーした数値結果だけのvalidateと、図付きExcelのvalidateは分けて使ってください。

## 古い結果のCSV補完

0.4.0a5以降のrunは元の特徴量と符号化targetをNPZに保存するので、元CSVが別の場所に移っても描画できます。古いrunはこれらが不足する場合があります。

```sh
.venv/bin/python -m diy_automl report --run validation/current/wine-run --csv data/reference/wine.csv --output runs/legacy-wine-illustrated
```

補完CSVは元manifestのSHA256と一致し、特徴量・ラベルの対応が同じものだけを受け付けます。補完したCSVをsource_input.csvとして出力へコピーするため、検証一式も自己完結します。元パスが見つかる場合はハッシュを確認して自動補完します。古い結果の再描画で元パスが存在しない場合は、保存したsource_input.csv等を明示的に--csvへ渡してください。

正解が不明なconfusion matrixを予測から推測して作ることはありません。対応データがない場合は図を省略して理由を記録します。UL投影が作れない場合は、指標パネルと省略理由を表示します。

## 描画上の範囲

confusion matrixは最大30クラス、投影は最大5,000行・100特徴量です。1特徴量は1次元表示、標準化後に変動がない場合は「PCAの分離なし」と表示します。2次元投影だけからクラスタ妥当性は判断できません。クラス名は図上で短縮する場合がありますが、クラスIDと元manifestのclassesで対応を追えます。

日本語フォントがある場合はNoto Sans CJK JP、なければDejaVu Sansを使い、使用フォント名を記録します。フォントやライブラリが違う環境での同じピクセル出力は保証しません。

失敗した学習結果は描画しません。描画失敗はrender_manifestのfailedとして残し、コピーした元の学習manifestを書き換えません。保存中の障害からの完全復旧・自動再開は未実装です。今回の検証範囲は `docs/VALIDATION_EXCEL_REPORT.md` に記録します。
