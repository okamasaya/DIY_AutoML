# 学習前にCSVを点検する

`audit-data` は、CSVを整える前に、値・件数・評価条件を確認する入口。学習を起動せず、点検内容を `report.xlsx` と `profile.json`、照合用の `manifest.json` に保存する。CSVの値と行順は保持する。

## 実行例

リポジトリのルートで実行する。`--output` には新しいフォルダを指定する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-data --csv data/reference/raisin.csv --target target --row-unit "レーズン1粒" --prediction-goal "形状から品種を分類" --label-source "UCI Raisinの品種ラベル" --output runs/raisin-data-audit
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_audit.py runs/raisin-data-audit --csv data/reference/raisin.csv
```

分類ラベルは `--target`、グループの識別子は `--id-column`、観測時刻は `--time-column` で指定できる。`--prediction-time` に予測する時点を書き、予測時点で使える列を原資料と照合する。省略した評価条件は `unknown` と表示する。Python APIは `diy_automl.audit_data_csv`。

## ビジネス向けの点検と圧縮

`prepare-bp` は先頭に日本語の「まず読む」シートを付ける。行数、欠損、重複、正解ラベルの種類数、詳しく確認するシートを表示する。`--compress` を付けると作業用の圧縮Parquetを追加し、前後のサイズと元CSV比を表示する。圧縮を使う場合の導入手順は下記の `storage` を参照。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl prepare-bp --csv data/reference/raisin.csv --target target --compress --output runs/bp-raisin
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_audit.py runs/bp-raisin --csv data/reference/raisin.csv
```

元CSVを原本として保管し、作業用ファイルでは先頭ゼロや欠損表記を含めて文字列値を保持する。圧縮後のサイズはデータの内容により変わる。圧縮結果が元CSVより大きい場合も、実際のサイズを表示する。`--compress` を省くと点検だけを行う。Python APIは `diy_automl.prepare_bp_csv`。

## ParquetとDuckDBへの変換

元CSVを原本として保管し、大きいデータを繰り返し処理するときはZstandard圧縮のParquetを作業用に優先する。点検に続けて中継ファイルを作るには、任意依存の `storage` を導入する。DuckDBはParquetを直接読み、クエリで使う列と絞り込み条件を読み出しに反映できる。テーブルをDBファイルに保存する必要がある場合は `--convert duckdb` を指定する。 [Apache Parquetの概要](https://parquet.apache.org/docs/overview/) · [DuckDBによるParquetの直接読み出し](https://duckdb.org/docs/1.5/data/parquet/overview#partial-reading)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[storage]"
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-data --csv data/reference/raisin.csv --target target --convert parquet --conversion-types numeric --output runs/raisin-parquet-work
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_audit.py runs/raisin-parquet-work --csv data/reference/raisin.csv
```

`--convert` は片方だけでも指定できる。出力は `converted/data.parquet` または `converted/data.duckdb`。DuckDBのデータテーブルは `data`。両方を指定した場合は、両ファイルが書き上がり、行数・列名・型・元CSVのSHA-256の照合が済んでからまとめて保存する。Excelの `Conversion` シートとmanifestに結果を記録する。

既定の `preserve` は、CSVとして読み取った値を文字列で保持する。先頭ゼロ、空欄、`NA` などの欠損表記、数値の元の表記もそのまま入る。明示的に `--conversion-types numeric` を付ける場合は、全件が有限数値で精度を保てる特徴量だけを `BIGINT` または `DECIMAL` にする。たとえば `1.20` は文字の並びに代えて、正確な小数値を格納する。`--target`・`--id-column`・`--time-column` で宣言した列、先頭ゼロ付きの整数表記を含む列、欠損や非有限値を含む列、38桁以内の固定小数点に収まらない列は文字列で保持する。

行の形が崩れているCSVは、ExcelとJSONに点検結果と変換を保留した理由を残す。依存や書き込みに問題がある場合も診断レポートは保存し、CLIの終了コードは2になる。元CSVの修正後は新しい出力先で点検と変換を行う。Python APIでは `convert=("parquet", "duckdb")` を指定し、`manifest.json` の `conversion.status` で結果を確認する。

変換はCSVを一定量ずつ読んで書き出す。DuckDBは1スレッド、内部メモリ上限の設定値512 MBで書き込む。Python・Arrowなどを含めたプロセス全体の使用量は、列数や値の長さ、ファイル形式に応じて変わる。

中継ファイルで列や行を絞り込み、集計条件・入力ハッシュと共に次の処理へ渡す。現行の学習CLIはCSV入力なので、学習するデータは必要な行と列を抽出してCSVに用意する。

```python
import duckdb

with duckdb.connect(config={"threads": "1"}) as db:
    rows = db.execute(
        "SELECT Area, Perimeter, target FROM read_parquet(?) WHERE Area >= ? LIMIT 5",
        ["runs/raisin-parquet-work/converted/data.parquet", 100_000],
    ).fetchall()
    print(rows)
```

## 学習用CSVを抽出する

`extract-data` の入力はParquetファイル、またはDuckDBファイル内の `main.data` テーブル。指定した列と行を `data.csv` に出力する。列順は `--columns` の順。分類に使う場合は正解ラベルの列も含める。変換と同じ `storage` 環境で実行できる。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl extract-data --source runs/raisin-parquet-work/converted/data.parquet --columns Area Perimeter target --where Area ge 60000 --limit 250 --output runs/raisin-extracted
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_extract.py runs/raisin-extracted --source runs/raisin-parquet-work/converted/data.parquet
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-data --csv runs/raisin-extracted/data.csv --target target --output runs/raisin-extracted-audit
```

- `--where 列 演算子 値` は繰り返せる。複数指定時はすべての条件を満たす行を使う。比較は `eq`（等しい）、`ne`（異なる）、`lt`・`le`（未満・以下）、`gt`・`ge`（より大きい・以上）。条件だけに使う列も指定できる。
- 文字列と真偽値は `eq`・`ne` を使う。数値の範囲条件には数値型の列を使う。整数・固定小数点のしきい値が列の型に収まることを確認し、条件値が丸められる場合は抽出前に知らせる。実ファイルの列名・型を確認し、条件値はSQLにパラメータとして渡す。 [DuckDBのパラメータの扱い](https://duckdb.org/docs/1.5/clients/python/dbapi#prepared-statements)
- `--limit` は条件に合う行を元の順序で先頭から指定件数まで取る。省略すると該当する全行を出す。入力がクラスなどで並んでいると件数制限後の分布も変わるため、抽出後に `audit-data` でクラス件数を確認してから学習に進む。 [DuckDBの行順序](https://duckdb.org/docs/1.4/sql/dialect/order_preservation)

`manifest.json` に抽出条件、入力ファイルと出力CSVのSHA-256、入力の列型、出力の件数を保存する。点検で作った中継ファイルを変換manifestと一緒に使うと、元CSVのSHA-256もたどれる。Pythonからは `diy_automl.extract_data` を呼べる。

このRaisinの例は、元CSVでは各450行。面積が60,000画素以上の行を件数上限なしで抽出すると、Besni 428行、Kecimen 237行の665行だった。元順で先頭250行を取ると、Besni 13行、Kecimen 237行になる。条件のみの結果は `--limit 250` を省いて再現できる。3段階のExcelは `prepare-bp` で出力し、「まず読む」の対象行数と「Class_Counts」の品種別件数を確認した。250行の再点検と既存の `select` まで実行し、抽出CSVのハッシュが学習記録の入力と一致することも確認した。 [BP向けの解説原稿](../articles/diy-automl-bp-filtered-rows.md)

## 確認する順序

1. `Summary` と `Review` で走査状態、行・列数、列数の合わない行、次に確認する箇所を見る。壊れた引用符・不正なUTF-8などで止まった場合、`partial` として、その時点までに読めた範囲を出す。`complete` は末尾まで読んだ状態で、確認事項は `Review` に残る。
2. `Columns` で列ごとの数値・文字・真偽値の件数、欠損、非有限値、数値の範囲を見る。空白だけのセルや既定の欠損表記を欠損と数える。列型は検出した内容の要約。
3. `Duplicates` で全項目の一致と、候補の特徴量だけの一致を確認する。後者は、ラベル列と解決済みのID・時刻列を外した値の組で比較し、ラベルが異なる組も数える。指標の `Extra rows` は、各組を1行と数えた分を超える行数。文字列としてパースした値の一致をSHA-256で照合するため、`1` と `1.0` は別の値として扱う。各種類の先頭10例は `Examples` に元CSVの物理行番号で表示する。
4. `Class_Counts` と `Evaluation_Context` でクラス別件数、少数クラス、IDの種類と繰り返し、時刻範囲、記入した評価条件を確認する。クラスは件数上位100件を表示し、他のクラス数と行数を集計する。ID件数から独立な標本数を推定するには、IDが何を表すかの確認が必要になる。
5. 必要な修正を別のCSVに保存し、そのCSVで点検をやり直す。学習コマンドには点検済みのCSVを渡す。現行の学習入口は、渡したCSVのラベル以外の列を特徴量に使う。点検でID・時刻と宣言した列を学習に使うかは、目的と予測時点に合わせて入力CSVを用意する。

時刻はISO形式を対象にする。タイムゾーン付きの値はUTCに変換して範囲を出す。タイムゾーンが指定されていない値との混在は `mixed_timezones` とし、一つの範囲を算出する前に確認を促す。

## 保存内容と再現性

`profile.json` は読み取った集計値と条件を保存する。検証器へ元CSVを渡すと、SHA-256を照合して再走査し、JSONとExcelの値を確かめられる。元CSVを省略すると、保存済みのJSON・Excel・ハッシュの整合性を確認する。

走査はCSVを1行ずつ読み、重複・ラベル・ID集計用の一時SQLiteをディスクに置く。メモリには列ごとの集計と表示対象の例を保持する。一時ディスクの使用量は、重複のないレコードや特徴量の組、クラス・IDの種類に応じて増える。

## Windowsでの検証

Python 3.13.17でIrisとRaisinを実行し、元CSVから再走査して出力を照合した。Irisの150行では完全一致の重複が1件あった。Raisinの900行では今回の確認項目は0件で、既存の分類実行と入力SHA-256も一致した。自動テストでは、列数の不一致、欠損・非有限値、重複、ラベルが異なる同一特徴量、少数クラス、ID未指定、壊れた引用符と不正なUTF-8を確認した。

100,000件に列数の合わない行を1件加えた模擬CSV（約4 MB）では、点検とExcel・JSONの出力に約3.2秒かかった。WindowsのPythonプロセスのピークワーキングセットは約156 MiBだった。意図的に入れた欠損10件、列数の不一致1件、ラベルが異なる同一特徴量1組を確認した。実測はこのCSVの規模・構成での値。入力はローカルの `runs/data-audit-example-input.csv`、出力は `runs/data-audit-example-100k` に保存した。

模擬CSVから列数が合わない1行を取り除いた別ファイルを用意し、10万行を `numeric` で両形式に変換した。WindowsのPython 3.13.17、PyArrow 26.0.1、DuckDB 1.5.6で、点検から両形式の出力まで約4.7秒、ピークワーキングセットは約229 MiBだった。CSV約3.95 MBに対し、Parquet約0.46 MB、DuckDB約1.59 MBだった。繰り返しが多い模擬データでのサイズ。行数、列と型、元CSVと変換結果のハッシュを検証した。入力は `runs/storage-example-prepared-100k.csv`、出力は `runs/storage-example-100k`。

この10万行のParquetから、2条件で絞った先頭2,000行・2列を抽出し、元CSVを同じ条件で読んだ結果と全件一致した。Python APIで抽出からCSV・manifestの保存まで約0.08秒だった。出力はローカルの `runs/extract-example-100k`。

同じ10万行に `prepare-bp --compress` を使い、点検と日本語要約、文字列を保持した圧縮Parquetの出力を確認した。入力3,954,471バイト、Parquet 464,192バイト（元CSV比11.7%）。CLI起動から出力まで約5.5秒で、元CSVのSHA-256は実行前後で一致した。出力はローカルの `runs/bp-compression-100k`。
