# 16 GBのWindows PCで実データを点検・学習する

通信会社の顧客3,150件を収めた公開CSVで、BP向けの点検、任意の圧縮、列の抽出、学習用データの再点検、5モデル比較をCPUで試した。元資料では、最初の9か月の集計から12か月目の解約状態を予測する。 [Iranian Churn（UCI、CC BY 4.0）](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) · [データのDOI](https://doi.org/10.24432/C5JW3Z)

## 再実行する

WindowsのPython 3.13と、模擬データ計測で使用したコア・ストレージ依存を使う。リポジトリのルートから、新しい出力フォルダーを指定する。 [依存と測定環境](../tabular_scale_benchmark/README.md)

~~~powershell
New-Item -ItemType Directory -Path runs -Force | Out-Null
curl.exe -fL -o runs/iranian-churn-source.zip 'https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip'
.\.venv\Scripts\python.exe -X utf8 -m experiments.bp_churn_pilot.run --archive runs/iranian-churn-source.zip --output runs/iranian-churn-pilot-new
~~~

元のCSVはZIPからバイト列のまま保存し、SHA-256を照合する。既存の `prepare-bp`、`extract-data`、`select` を通常のCLI経路で呼ぶ。元データはUCIから再取得し、作業用ファイルとレポートは指定した出力先に保存する。

## 対象と評価

- 元CSVは13特徴量と解約ラベルを持つ。点検で、全列一致の行のうち同じ内容の2行目以降が300行あり、全特徴量が同じでも解約ラベルが異なる組は14組あった。
- 学習入力は、通話失敗数、苦情有無、契約月数、通話秒数、通話回数、SMS回数、異なる通話先数の7特徴量。年齢、契約種別、利用状態、顧客価値等は使う列の範囲から外した。
- 使う列が同じ入力を複数の分割に入れないため、この実験では7特徴量の組でまとめた。ラベルが一致しない20組の128行を外し、一致する組では先頭だけを残して余分な268行を外した。2,754行（解約398・非解約2,356）を使用した。利用実績が同じ別の顧客がいる場合も含む、値の一致による実験用の処理。
- 学習側の層化5分割で5モデルを比較し、選んだRandom Forestを保留551行で評価した。固定したモデルに条件付けたmacro F1は0.869、クラス件数を固定した1,000回のブートストラップによる95%区間は0.824〜0.909。学習データの多数派を常に予測するルールとも同じ保留行で比較した。分割seedは42。別時期の評価に使えるデータは今後の検証対象とする。

## 測定と照合

各工程を3回ずつ順番に実行した。時間は起動から終了までの中央値、メモリはPythonプロセスのピークワーキングセットの最大値。一時ディスクは工程ごとに専用のTEMPと変換・抽出用の作業フォルダーを約10 ms間隔で観測し、ファイルの論理サイズの合計が最も大きい値を取った。観測間に作成・削除されたファイルや、対象外の場所に作られたファイルがある場合は値に反映されない。ダウンロード、実験用の行整理と事後検証の時間は別に扱う。

全15回の出力を既存の検証器で照合した。元CSVから7列への抽出と実験用の行整理を別の集計手順で再構築し、テスト結果の件数・スコア・区間も保存した配列やExcelから確認した。元CSV約132 kB、作業用Parquet約40 kB。全反復とファイルのハッシュは [実測JSON](results-2026-11-19.json)、BP向けの説明は [Zenn下書き](../../articles/backup/diy-automl-bp-churn-pilot.md) に保存した。
