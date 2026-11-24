---
title: "DIY AutoML 開発メモ (3) 元CSVを残し、ParquetとDuckDBで学習前のデータを整える"
emoji: "📦"
type: "idea"
topics: ["automl", "python", "データ分析", "parquet", "duckdb"]
published: false
---

データが大きくなるほど、モデルを比べ直すたびに元CSVを読み込み、文字列を解析して必要な列を取り出す作業に時間がかかります。DIY AutoMLでは、点検したデータをParquetやDuckDBの作業用ファイルにして、次の実行でも使えるようにしました。作業用データを扱いやすくするとともに、**ストレージとの読み書き（I/O）とCSVの再解析を減らし、前処理を速く、効率よく進める**のが目的です。

## 元CSVと作業用データを手元に残す

処理の順番は **元CSVを保管する → 点検する → 必要に応じて中継ファイルを作る → 学習に使う列を取り出して再点検する** です。点検では、行数、欠損、重複、正解ラベルの内訳を確認します。中継ファイルを作ったときは、変換した行数と列の型、元CSVを照合するためのSHA-256も記録します。

- **Parquet**は、列単位でデータを格納するファイル形式です。DIY AutoMLではZstandardで圧縮し、ディスクに書き込む量や、再利用時に読み込む量を抑えます。 [Apache Parquetの概要](https://parquet.apache.org/docs/overview/)
- **DuckDB**は、Parquetを直接読み、必要な列に絞って取得するために使います。条件とParquetの統計情報から、対象が含まれないと判断できるデータのまとまりは読み飛ばせます。テーブルを再利用したい場合は、`.duckdb` ファイルとして保存することもできます。 [DuckDBでParquetを直接読む](https://duckdb.org/docs/data/parquet#partial-reading)

PKSHAテックブログでは、DuckDBで同じ分析クエリをCSV・Parquet・ORCに実行し、保存サイズとクエリ時間を比較しています。必要な列だけ読むと、なぜI/Oを減らせるのかも図で紹介されています。 [分析基盤でParquetが選ばれやすい理由をDuckDBで検証してみた](https://zenn.dev/pksha/articles/038bcf47e627c2)

変換の既定値では、読み取ったセルの値を文字列として保存します。たとえば、IDに先頭のゼロが付いていてもその表記を保てます。数値型で保存したいときは `--conversion-types numeric` を指定し、精度を保って変換できる特徴量だけを数値型にします。 [変換時の型と照合の仕様](https://github.com/okamasaya/DIY_AutoML/blob/9b6b22c/docs/DATA_AUDIT.md#parquetとduckdbへの変換)

## Raisinで、行を保ったまま列を選ぶ

レーズンの品種データRaisinの900行を使って、面積（`Area`）、周囲長（`Perimeter`）、正解ラベル（`target`）を取り出します。WindowsのPowerShellで、リポジトリのルートから実行します。仮想環境 `.venv` に変換用の依存がまだなければ、`.\.venv\Scripts\python.exe -m pip install -e ".[storage]"` で準備できます。

```powershell
$python = ".\.venv\Scripts\python.exe"
$work = "runs/zenn-development-note-03-raisin"

& $python -X utf8 -m diy_automl audit-data --csv data/reference/raisin.csv --target target --convert parquet --convert duckdb --output "$work/source-audit"
& $python -X utf8 -m diy_automl extract-data --source "$work/source-audit/converted/data.parquet" --columns Area Perimeter target --output "$work/from-parquet"
& $python -X utf8 -m diy_automl audit-data --csv "$work/from-parquet/data.csv" --target target --output "$work/extracted-audit"
```

最初のコマンドは点検結果と2つの中継ファイルを保存します。片方だけ作る場合は必要な `--convert` だけを指定します。DuckDBファイルから取り出す場合は、`extract-data` の入力を `data.duckdb` に変え、別の出力先を指定します。再実行するときは `$work` に新しいフォルダー名を使います。

どちらの形式から取り出しても、今回選んだ3列の値と行順は元CSVと一致し、**Besni 450行・Kecimen 450行**がそのまま残りました。出力の `data.csv` が学習側に渡すデータです。点検と抽出の `manifest.json` をたどると、どの入力からどの列を取り出したかを確認できます。 [Raisinの出典と再実行の例](https://github.com/okamasaya/DIY_AutoML/blob/9b6b22c/data/reference/README.md)

## 16 GBのPCで、処理ごとに時間を測る

WindowsのCore i5-14400・メモリ16 GBのPCで、数値7列と二値ラベルを持つ模擬データを使いました。**点検と変換は100万行、モデル比較は抽出後の1万行**が対象です。Parquetへの変換時は `--conversion-types numeric` を指定しました。元CSVは **36.2 MB**、圧縮したParquetは **18.9 MB** でした。各工程を3回実行し、時間は中央値、メモリは処理を実行したPythonプロセスのピークワーキングセットの最大値です。

| 工程 | 対象 | 時間（秒） | メモリ（MiB） |
| --- | --- | ---: | ---: |
| CSV点検 | 100万行 | 63.3 | 169 |
| 点検＋Parquet変換 | 100万行 | 91.1 | 186 |
| Parquetから抽出 | 条件に合う1万行 | 2.3 | 186 |
| 5モデルの比較・評価 | 抽出後の1万行 | 10.4 | 195 |

作業用ファイルを作る初回には、点検と変換の時間がかかります。Parquetからの抽出は、変換済みのファイルを再利用するときの計測です。抽出では2つの条件を付け、条件に合う行のうち先頭1万行を取り出しました。同じ10万行で点検とParquet変換をしたときも、特徴量を7列から28列へ増やすと7.4秒から15.6秒になりました。 [計測条件と全反復の記録](https://github.com/okamasaya/DIY_AutoML/tree/9b6b22c/experiments/tabular_scale_benchmark)

普段は、評価したい範囲を決めた上で必要な列を選び、その範囲の行を原則すべて使います。10万行以上または20列以上を一つの目安に、CSVの読み直しや整形を何度行うかも見て、中継ファイルを作るか考えます。学習の時間とメモリは、学習に渡す行数・列数とモデルで見積もります。

次回は、どの特徴量を残すかを考えるために加えた、焼きなまし法（Simulated Annealing）のチェックを紹介します。
