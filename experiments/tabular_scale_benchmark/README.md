# Windowsで表形式データの処理負荷を測る

DIY AutoMLのCLIを使い、学習前のCSV点検・作業用Parquetへの変換・条件抽出と、抽出したデータでの5モデル比較を測る。CSVの行数と特徴量の列数は別々に変えた。 [開発メモ (2) の下書き](../../articles/backup/diy-automl-development-note-02.md)

## 同じ条件で再実行する

リポジトリのルートで実行する。WindowsでPython 3.13の環境と任意依存の `storage` を使う。出力には新しいフォルダーを指定する。

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-core.lock.txt -e ".[storage]" "pyarrow==26.0.1" "duckdb==1.5.6"
.\.venv\Scripts\python.exe -X utf8 experiments/tabular_scale_benchmark/run.py --output runs/tabular-scale-recheck
~~~

実測時のライブラリ版・CPU・Windowsから見えた物理メモリと、全反復の測定値・ファイルのサイズ・SHA-256は [結果JSON](results-2026-11-19.json) に記録した。再実行時のCSV、Parquet、Excel、CLIログ、結果JSONは指定した出力フォルダーに保存する。各条件は既定で3回実行し、`--repeats` で回数を変えられる。

## データと処理

- 固定seed 1842で、0〜9,999の整数を持つ特徴量を28列生成する。特徴量の一部の加重和で二値ラベルを作る。7特徴量のケースは同じ行の先頭7列を使う。数値で欠損のない模擬CSVを使い、各工程の負荷を計測する。
- 1万×7、10万×7、10万×28、100万×7の4条件で、`audit-data`、`audit-data --convert parquet --conversion-types numeric`、`extract-data` を計測する。点検＋Parquetは点検の時間も含む。ParquetはZstandardで圧縮する。DuckDBはParquetを読み、2条件を満たす行を元順で先頭最大1万行出す。
- 100万×7の抽出結果から先頭1,000、5,000、10,000行、10万×28の抽出結果から先頭5,000行を用意する。`select` で同じ5モデルを5分割CVで比較し、選んだモデルの再学習・保留テスト・1,000回のブートストラップ・Excel出力まで測る。5,000行同士は同じ入力行を使い、特徴量の数だけが異なる。

## 計測範囲

ジョブはCPU上で1つずつ実行し、BLAS・OpenMPのスレッド数は1に設定する。時間はCLI用のPythonプロセスの起動から終了まで。入力の模擬CSVを生成する時間、学習用ファイルの行数調整、事後検証の時間は計測外。OSのファイルキャッシュは消去せず、連続実行の3回から時間の中央値を取る。

メモリはCLIを実行したPythonプロセスのピークワーキングセットをWindows APIで読み、3回の最大値を取る。Excel・Parquetを含むCLI処理中の値で、OSやほかのアプリの使用量、プロセスのコミット量、SQLite等の一時ディスク使用量は別の測定対象になる。 [Microsoftのピークワーキングセットの定義](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters)

この結果は特定のPCと数値データに対するもの。カテゴリ列・文字列の長さ・欠損・ラベルの難しさといったデータ特性や、候補モデル・並列数が変わると負荷も変わる。

## 出力の確認

2026年11月19日の計測では、全48ジョブの保存結果を既存の検証器で照合した。元CSVを読み直して、点検の行数・ラベルの件数と、抽出結果の行・順序が一致することを確認した。学習用CSVは抽出結果の所定の先頭行と一致し、5,000行の2条件は同じ行・同じラベルだった。計測と検証の実物はローカルの `runs/tabular-scale-20261119` に保存した。
