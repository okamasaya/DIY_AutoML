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

## データ点検の画面を使う

WindowsのTcl/Tk付きPythonで、リポジトリのルートから起動する。CSVの初期選択には `--csv`、保存先の親フォルダーの指定には `--output-dir` を使える。 [PythonのTkinter](https://docs.python.org/3.13/library/tkinter.html)

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl bp-preview
.\.venv\Scripts\python.exe -X utf8 -m diy_automl bp-preview --csv data/reference/raisin.csv
~~~

MacBook Airで確認するときは、python.org版のTcl/Tk付きPythonを使用し、次をリポジトリのルートから実行する。`python -m tkinter` に相当するコマンドで確認用のウィンドウを開き、それを閉じてからBP画面を起動する。以下はmacOS向けの確認手順で、実機での動作と計測は未確認。 [Python.orgのmacOS版Tcl/Tk](https://www.python.org/download/mac/tcltk/) · [Tkinterの確認方法](https://docs.python.org/3/library/tkinter.html)

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m tkinter
.venv/bin/python -X utf8 -m diy_automl bp-preview --csv data/reference/raisin.csv
```

1. 「CSVを選ぶ」でUTF-8のファイルを開く。正解列を選ぶとラベルの内訳も表示する。正解列は空欄でも点検できる。顧客IDや観測日で評価を分けるときは、点検前に「評価の分割方法」で選ぶ。 [グループ単位の分割](#グループ単位の分割) · [日付順の分割](#日付順の分割)
2. 「点検する」で走査し、列数が合う行・欠損セル・全列一致の余分な行・ラベルの種類と、確認する項目を表示する。読み取りが途中で止まった場合は、集計できた範囲を表示する。
3. 「Excelを開く」で詳細を参照する。既定では `runs/bp-preview` の下に点検ごとのフォルダーを作り、Excel、JSON、元CSVの照合用情報を保存する。元CSVは読み取りに使う。
4. 正解列のあるCSVを点検すると「使う列を選ぶ」が使える。学習に使う列を選び、「抽出して再点検」で保存する。正解列は抽出CSVに含まれ、元の列順・セルの値・行順を保つ。
5. 「元CSV」と「抽出後」で、それぞれの点検結果とExcelを切り替えられる。比較欄には列数・欠損セル数・同じ入力でラベルが異なる組の数を表示する。「抽出CSVの保存先」から `data.csv` と抽出時の `manifest.json` を開ける。
6. 抽出後のCSVが学習できる形式なら「5モデルを比較する」で実行できる。対象は欠損のない数値特徴量と分類ラベルで、学習側の各クラスに5分割分の行が必要。点検後に抽出CSVを編集した場合はもう一度抽出と再点検を行う。
7. 「評価結果」で保留テストの正解・誤分類、ラベル別の見逃しと誤検知、学習側の最多ラベルだけで答えた場合の件数を確認できる。「評価Excelを開く」で日本語要約、ラベル別の件数、5モデルの比較表を参照できる。行またはグループで分ける場合はF1などの評価区間も保存する。
8. 「グラフを見る」で学習側の候補モデルの比較を棒グラフで表示する。日付順で評価した場合は「保留テストの月別推移」タブで選定モデルと単純ルールの月別macro F1を確認できる。対象期間や各月の件数は評価Excelの「月別」にある。正解・誤分類などの主要指標は評価画面とExcelの「まず読む」に表示する。

作図には学習時に保存した評価表を使う。図を開き直したり切り替えたりするときに、学習は実行されない。

作業用Parquetを保存する場合は `storage` を導入し、「作業用Parquetも保存」を選ぶ。画面に圧縮前後のサイズを表示し、変換が保留・失敗した場合はExcelの `Conversion` シートから理由を確認できる。

画面は `prepare_bp_csv` と `extract_data` を呼び出す。点検で保存したParquetをDuckDBで読める場合は作業用ファイルを使い、圧縮を選ばない場合は元CSVから直接抽出する。抽出時に点検済みファイルのハッシュを照合する。元CSVを変更したら、もう一度「点検する」から始める。列数の違う行や読み取り途中の問題がある場合は、元CSV側の点検結果で確認する。

Iranian Churnで圧縮なしの画面操作を検証した。特徴量を13列から7列に絞ると、3,150行と正解ラベルの内訳（非解約2,655件、解約495件）を保ったまま、ラベル不一致の組は14組から20組になった。圧縮時の131,728バイトから39,649バイトへの変換もExcel・JSONと照合済み。

### 画面からのモデル比較

BP画面は1件ずつモデルを比較し、教師あり学習・予測中の対応するネイティブスレッドプールの上限を1に設定する。ランダムフォレストも並列数1で実行する。行単位とグループ単位の評価区間は1,000反復で計算する。指定値は `manifest.json` と詳細Excelの `Summary` に記録する。

点検済みの抽出CSVを共通の `select_csv` に渡す。画面の実行では、正解ラベルの文字列表記を保ち、たとえば `001` と `1` はそれぞれのクラスとして扱う。学習側の5分割CVのmacro F1で候補を比較し、選んだモデルを保留テストで評価する。既定の「行単位」では層化80/20分割を使い、固定したモデルの95%区間は保留行をクラス内で再標本化して求める。

評価結果は、選んだ列の実行フォルダーの `evaluations` に保存する。通常の `report.xlsx` と配列に加え、`bp_report.xlsx` は同じ詳細シートの先頭に「まず読む」と「ラベル別」を付ける。学習側にも同じ数値の入力があった保留行は件数を表示する。行の由来や顧客・時期の分割を検討するときの手掛かりになる。

Windowsの画面でIranian Churnの7特徴量・3,150行を使い、保留630行で581正解・49誤分類と表示した。解約99件のうち64件を拾い、35件を見逃し、14件を解約と誤判定した。macro F1は0.839（95%区間0.793〜0.878）。保留の107行は学習側にも同じ入力値があった。これは3,150行での画面操作検証で、別の実験の2,754行への整理とは評価対象が異なる。

2026年11月26日にWindows（Core i5-14400、16 GB）で、画面と同じ点検・列抽出・5モデル比較・評価Excel作成の処理を各1回計測した。時間は処理開始からExcelの保存まで、メモリはPythonプロセスのピークワーキングセット。

| データ | 行数 | 特徴量 | 経過時間 | ピークメモリ |
| --- | ---: | ---: | ---: | ---: |
| UCI Raisin | 900 | 7 | 3.3秒 | 175 MiB |
| UCI Iranian Churn | 3,150 | 7 | 4.1秒 | 178 MiB |

### グループ単位の分割

- 「IDでまとめる」では、同じ顧客や回答者を表すID列を選ぶ。IDは文字列として扱い、先頭ゼロを保つ。抽出CSVには分割用のID列を含め、モデルの特徴量からは除く。IDが欠けている行がある場合は、点検結果を確認する。
- 「同じ入力をまとめる」では、選んだ特徴量を数値として読み込み、そのすべてが一致する行をまとめる。ラベルが異なる行も同じグループに入り、元の行はすべて保持する。一致した入力を学習と評価で共有させないための設定で、顧客や回答者の同一性は確認できない。

どちらも、同じグループが学習と保留テストの両方に入らないようにし、学習側の5分割CVでもグループを維持する。各ラベルが学習・保留テスト・学習側の各検証foldに入る分割を探すため、各ラベルには少なくとも6つのグループが必要。グループの大きさとラベル分布によっては条件を満たせず、評価を中止する。保留の行数は全体の約2割を目安とする。95%区間は固定したモデルの保留テストをグループごとに再標本化して求め、評価Excelの `SL_Test_Intervals` に保存する。 [scikit-learnの層化グループ分割](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

CLIでも同じCPUのモデル比較を実行できる。`--group-column` と `--group-identical` はどちらか一方を指定する。`uncertainty` の追加集計は行単位の再標本化を行うため、グループ分割では比較時に保存した区間を参照する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl select --csv customer-data.csv --target churn --group-column customer_id --output runs/customer-grouped
.\.venv\Scripts\python.exe -X utf8 -m diy_automl select --csv data/reference/raisin.csv --target target --group-identical --output runs/raisin-grouped
```

同じIranian Churnの7特徴量・3,150行で「同じ入力をまとめる」を選ぶと、学習側にも同じ入力があった保留行は107件から0件になった。保留テストは564グループ・628行で、590正解・38誤分類。macro F1は0.869（95%区間0.823〜0.908）だった。今回は保留する行が変わるため、行単位分割とのスコア差にはテスト対象の違いも含まれる。

### 日付順の分割

「評価の分割方法」で「日付順で分ける」と観測日または観測時刻の列を選ぶ。ISO 8601形式（例 `2012-08-07`）を使う。時差指定付きの時刻はUTCの日付に揃え、時差指定なしは入力の暦日を使う。欠損や解釈できない日付、時差指定の有無が混在する場合は点検結果に表示する。分割用の列は抽出CSVに保存し、特徴量からは除く。

観測がある日のうち、最新の2割（日数を切り上げ）を保留する。同じ日付の行は一緒に扱うため、保留行の割合はデータによって変わる。学習側の観測日を6区間に分け、最初の区間から学習を始める。次の区間で検証し、その区間を学習側に加える操作を5回行う。各クラスが最初の学習区間、各検証区間、保留期間に必要で、最初の学習区間には5行以上必要。日付の境界はラベルの分布に合わせて動かさない。

予測時点で使える値を特徴量に選び、各評価期間の開始時点までにそれ以前の正解ラベルが得られていることを確認する。ラベルの確定が遅れる場合の待機期間や、顧客を分けながら日付順も守る分割は今後の対応。日付順では保留期間の性能値と5回の検証結果を保存し、時間依存を考慮した評価区間は未算出と表示する。`uncertainty` の行単位再標本化は使えない。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl select --csv bike-calendar.csv --target target --time-column observed --output runs/bike-by-date
.\.venv\Scripts\python.exe -X utf8 scripts/validate_selection_run.py runs/bike-by-date
```

動作確認にはUCI Bike Sharingの `hour.csv` から8・12・16・20時の記録を抽出し、`dteday` を `observed`、`hr` を `hour`、`mnth` を `month` とした。特徴量は `hour`、`month`、`workingday`。`target` は `cnt` が200件以上なら `busy`、それ以外は `regular` とした。入力順に依存しないことを確かめるため、CSVの行は `random.Random(42).shuffle` でシャッフルした。 [Fanaee-TのBike Sharingデータ](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset)

2,913行のうち2012年8月7日〜12月31日の582行を保留し、すべての分割で学習側の日付が検証・保留側より前であることを確かめた。ランダムフォレストは410件正解、macro F1は0.638。学習側の最多クラス `busy` だけで答える場合は473件正解、macro F1は0.448だった。ランダムフォレストは `regular` 109件のうち80件を拾った。カレンダー上の特徴量の組が学習側と同じ行も、異なる観測日の記録として評価に含まれる。

#### 月別の成績と単純ルール

日付順の評価では、選んだモデル、曜日・時間帯の多数決、学習期間全体の多数決を同じ保留行で比べる。曜日・時間帯の多数決は、観測時刻の曜日と時（例 `08:00`〜`08:59`）が合う学習行の最多ラベルを返す。該当する学習行がなければ曜日だけで集計し、曜日もなければ全体の多数決を使う。時刻がない行は曜日から始める。時差付きの入力はUTCの曜日・時を用いる。同数の場合は保存済みのクラス順の先頭を選ぶ。モデル選定は学習側の5区間の比較で行う。

BPの評価Excelでは「単純ルール比較」「月別」「月別・ラベル別」を参照する。各月に含まれる観測日と件数を表示するので、保留期間が月の途中から始まる場合も確認できる。月別のmacro F1は全クラスを同じ重みで平均し、F1が定義できないクラスを0とする。月別・ラベル別の検出率は、そのラベルの行がない月では空欄にする。通常の評価Excelでは `SL_Test_Comparison`、`SL_Test_By_Month`、`SL_Test_By_Month_Class` に保存する。クラス番号は `manifest.json` の `classes` の順で0から始まる。

上記の自転車貸出データで `observed` に `hour` から時刻を加え（例 `2012-08-07T08:00:00`）、同じ2,913行で比較し直した。保留日とモデルの予測は前回と一致した。曜日・時間帯の多数決は430件正解、macro F1は0.656。ランダムフォレストは8・9月、単純ルールは10〜12月でmacro F1が高かった。`regular` 109件の検出ではランダムフォレストが80件、単純ルールが72件だった。数値はこの保留期間についての点推定。

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

`extract-data` はCSV、Parquet、DuckDBファイル内の `main.data` テーブルから、指定した列を抽出する。`--columns` を使うと指定順で `data.csv` に保存する。分類に使う場合は正解ラベルの列も含める。CSVからCSVへの列抽出は基本依存で実行できる。ParquetやDuckDBを入出力に使う場合は任意依存の `storage` を導入する。

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl extract-data --source data/reference/raisin.csv --columns Area Perimeter target --output runs/raisin-selected-columns
~~~

### JSONで列と保存形式を指定する

DS向けCLIでは、`--columns` と `--recipe` のいずれかを使う。たとえば次の内容をUTF-8の `raisin-columns.json` に保存する。

```json
{
  "columns": ["Area", "Perimeter", "target"],
  "format": "parquet"
}
```

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl extract-data --source data/reference/raisin.csv --recipe raisin-columns.json --output runs/raisin-selected-parquet
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_extract.py runs/raisin-selected-parquet --source data/reference/raisin.csv
```

`columns` は出力する列名の配列で、記載順を使う。`format` は `csv`、`parquet`、`duckdb` のいずれかで、省略時は `csv`。出力フォルダーには選んだ形式の `data.*` と `manifest.json` が入る。DuckDBのテーブル名は `main.data`。Pythonでは `diy_automl.extract_data(source, output, columns, output_format="parquet")` と指定する。

CSV入力のセルは文字列表記を保持する。CSVからParquetやDuckDBを作る場合は、選択列の一時CSVを出力フォルダー内に作り、書き出し後に片付ける。型付きデータのParquet出力で再読込時に列の型が変わる場合は保存を中止する。たとえばDuckDBの128ビット整数 (`HUGEINT`) を保つには `duckdb` を選ぶ。ParquetやDuckDBから現行の学習コマンドに渡すには `format` を `csv` にして抽出する。この場合は値をCSVの表現で出力する。

### 行条件を指定する

`--where` と `--limit` は `--columns` と `--recipe` のどちらにも追加できる。行条件を使う場合はParquetかDuckDBを入力する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl extract-data --source runs/raisin-parquet-work/converted/data.parquet --columns Area Perimeter target --where Area ge 60000 --limit 250 --output runs/raisin-extracted
.\.venv\Scripts\python.exe -X utf8 scripts/validate_data_extract.py runs/raisin-extracted --source runs/raisin-parquet-work/converted/data.parquet
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-data --csv runs/raisin-extracted/data.csv --target target --output runs/raisin-extracted-audit
```

- `--where 列 演算子 値` は繰り返せる。複数指定時はすべての条件を満たす行を使う。比較は `eq`（等しい）、`ne`（異なる）、`lt`・`le`（未満・以下）、`gt`・`ge`（より大きい・以上）。条件だけに使う列も指定できる。
- 文字列と真偽値は `eq`・`ne` を使う。数値の範囲条件には数値型の列を使う。整数・固定小数点のしきい値が列の型に収まることを確認し、条件値が丸められる場合は抽出前に知らせる。実ファイルの列名・型を確認し、条件値はSQLにパラメータとして渡す。 [DuckDBのパラメータの扱い](https://duckdb.org/docs/1.5/clients/python/dbapi#prepared-statements)
- `--limit` は条件に合う行を元の順序で先頭から指定件数まで取る。省略すると該当する全行を出す。入力がクラスなどで並んでいると件数制限後の分布も変わるため、抽出後に `audit-data` でクラス件数を確認してから学習に進む。 [DuckDBの行順序](https://duckdb.org/docs/1.4/sql/dialect/order_preservation)
- 元CSVから直接抽出する場合は、セルの文字列表記を保つ。件数上限を指定した場合も、末尾まで行の形を確認してからファイルを保存する。

`manifest.json` に抽出条件、入力・出力ファイルのSHA-256、入力の列型、出力の件数を保存する。点検で作った中継ファイルを変換manifestと一緒に使うと、元CSVのSHA-256もたどれる。

このRaisinの例は、元CSVでは各450行。面積が60,000画素以上の行を件数上限なしで抽出すると、Besni 428行、Kecimen 237行の665行だった。元順で先頭250行を取ると、Besni 13行、Kecimen 237行になる。条件のみの結果は `--limit 250` を省いて再現できる。3段階のExcelは `prepare-bp` で出力し、「まず読む」の対象行数と「Class_Counts」の品種別件数を確認した。250行の再点検と既存の `select` まで実行し、抽出CSVのハッシュが学習記録の入力と一致することも確認した。 [BP向けの解説原稿](../articles/backup/diy-automl-bp-filtered-rows.md)

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
