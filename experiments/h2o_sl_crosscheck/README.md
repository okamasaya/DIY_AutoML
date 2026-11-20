# H2O AutoMLとDIY AutoMLの保留テストの照合

WindowsのPython 3.14.7、NumPy 2.5.3、H2O 3.46.0.12、Eclipse Temurin Java 17.0.21でIrisとWineを比較した。Javaは作業用に展開した64ビットJREを用い、Javaヒープの上限を1 GiB、スレッドを1本にした。H2Oのサーバーをローカルのループバックで起動し、終了時に停止する。使用したPython・NumPyの組み合わせはこの端末での検証結果。公式資料にはPython 3.7〜3.11とNumPy 2未満の対応記載がある。 [H2O-3の動作要件](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements)

DIY AutoMLの保存済み`rank-sl`を検証し、元CSVのハッシュ、学習行、最初の5分割1回分、保留テスト行を再利用する。H2O AutoMLでは一般化線形モデル、分散ランダムフォレスト、勾配ブースティングから最大5候補を作る。H2Oの交差検証で得た予測から5分割の平均macro F1を計算して選び、同率では学習側のlog lossが小さい候補を選ぶ。保留データは選定後の評価に使う。DIY側はもとの5反復で選定したモデルの予測を使う。 [H2O AutoMLのパラメーター](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

| データ | 保留行 | DIY macro F1 | H2O macro F1 | H2O − DIY（95%区間） |
| --- | ---: | ---: | ---: | --- |
| Iris | 30 | 0.9666 | 1.0000 | +0.0334［0.0000, +0.1023］ |
| Wine | 36 | 0.9710 | 0.9407 | −0.0302［−0.0960, 0.0000］ |

両データで候補は5つそろい、H2O側では一般化線形モデルを選んだ。F1差の区間は、モデルを固定し、正解クラスごとにDIYとH2Oで同じ保留行を再標本化した1,000反復の2.5%点と97.5%点。両データとも区間はゼロを含む。

## 再実行

リポジトリのルートで実行する。既存の`rank-sl`結果を使う場合は、そのフォルダーを`--reference`に指定する。新しく作る場合は次のコマンドを使う。Iris・Wineの既定の保留分割を使い、Wineでは元の比較と同じく追加の比較用分割も確保する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl rank-sl --csv data/reference/iris.csv --target target --output runs/h2o-reference/iris
.\.venv\Scripts\python.exe -X utf8 -m diy_automl rank-sl --csv data/reference/wine.csv --target target --comparison-fraction 0.25 --output runs/h2o-reference/wine
```

JavaをPATHで利用できるようにするか、ポータブルJREのルートを`JAVA_HOME`で指定する。H2Oとコアの依存ライブラリがあるPythonを使う。次はこの端末の構成での実行例で、JREのパスは実際の展開先に合わせる。

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17-jre'
$env:PYTHONPATH = (Resolve-Path src).Path
python3.14 -X utf8 experiments/h2o_sl_crosscheck/run.py --csv data/reference/iris.csv --reference runs/h2o-reference/iris --output runs/h2o-comparison/iris
python3.14 -X utf8 experiments/h2o_sl_crosscheck/run.py --csv data/reference/wine.csv --reference runs/h2o-reference/wine --output runs/h2o-comparison/wine
```

各コマンドでは新しい出力フォルダーを指定する。`report.xlsx`にH2O候補の学習側スコア、保留テストの結果、差の区間を出力する。分割・正解・予測・再標本化の添字は`arrays.npz`、入力ハッシュと依存の版は`manifest.json`にある。実測のローカル出力は`runs/h2o-crosscheck-20261120-validated/`に保存した。
