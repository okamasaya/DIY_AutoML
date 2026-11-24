# 公開時系列のDIY AutoML・AutoGluon・H2O比較

AirPassengers（国際線旅客数）とNYC Taxi（タクシー乗客数）で、直前までの観測から1か月先・30分先をそれぞれ予測する。前回のDIY AutoMLで使った元データのSHA-256、選定区間と保留区間の時刻、実測値を照合した。前回の点予測から、現行 `kalman-track` の選択、季節候補を加えた試験での選択、カルマンを終了まで続けた場合の3列を引き継いだ。 [公開データの出典とDIYの選定条件](../../docs/UNCERTAINTY_DESIGN.md#公開時系列でカルマンを選ばない条件を確かめる)

| 系列 | 学習 | 選定 | 保留 |
| --- | ---: | ---: | ---: |
| AirPassengers | 1949〜1953年の60時点 | 1954〜1956年の36時点 | 1957〜1960年の48時点 |
| NYC Taxi | 2014年7月1日〜8月25日の2,688時点 | 8月26日〜9月22日の1,344時点 | 9月23日〜2015年1月31日の6,288時点 |

AutoGluon・H2Oには同じ特徴量を与えた。月次は直近1・2・3・12か月の実績と月の周期、タクシーは直近1・2・3・48・336時点の実績と時刻・曜日の周期を使い、両系列で通算時点も使った。開始直後のラグは欠測のまま渡す。予測対象時点のカレンダーは既知の値とする。学習後はモデルを固定し、保留期間でも各予測の直前までに実際に観測された値をラグとして使う。

AutoGluon Tabular 1.6.3では線形回帰・Random Forest・Extra Treesと加重アンサンブルを候補にした。H2O AutoML 3.46.0.12ではGLM・DRF・GBMから最大4モデルを作り、`nfolds=0` とした。両者は選定区間のRMSEでモデルを選ぶ。選ばれたのはAutoGluonが両系列とも加重アンサンブル、H2OがAirPassengersでGLM、NYC TaxiでGBM。H2OではAirPassengersの一部の候補が最小行数の条件を満たさなかった。 [前回と同じ比較用環境](../voyager_crosscheck/README.md#再実行と出力) · [H2Oの時系列順の検証に使った設定](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## 結果

| 方式 | AirPassengers 選定RMSE（千人） | AirPassengers 保留RMSE（千人） | NYC Taxi 選定RMSE（人） | NYC Taxi 保留RMSE（人） |
| --- | ---: | ---: | ---: | ---: |
| DIY AutoML 現行・直近値 | 30.76 | 48.04 | 1,660.99 | 1,708.36 |
| DIYの季節候補を含む試験 | 16.14 | 27.89 | 1,660.99 | 1,708.36 |
| AutoGluon Tabular | 19.11 | **27.49** | 907.75 | **1,115.22** |
| H2O AutoML | 24.24 | 42.34 | 998.95 | 1,171.83 |
| カルマンを継続 | 40.80 | 65.69 | 1,901.19 | 1,962.19 |

現行のDIY AutoMLがカルマンから直近値へ切り替えたことで、保留RMSEはAirPassengersで約27%、NYC Taxiで約13%下がった。AirPassengersの追加試験では「前年同月×成長倍率」を選び、保留RMSEはAutoGluonとの差が0.41千人だった。同じ保留区間のMAEは季節候補22.09千人、AutoGluon 22.61千人。NYC TaxiではAutoGluonとH2Oが直近値を上回った。結果は今回の観測間隔、候補集合と分割に対応する。

妥当性の点検では、各方式について選定区間の絶対誤差の90%点を求め、保留誤差がこの参照値以内だった割合を保存した。AirPassengersでは現行DIY 64.6%、季節候補62.5%、AutoGluon 68.8%、H2O 70.8%だった。NYC Taxiでは同じ順で89.4%、89.4%、87.5%、88.8%。選定区間の誤差を保留区間へそのまま当てはめたときの経験的な範囲として読む。

## 再実行

リポジトリのルートで実行する。元の公開データとDIYの予測は `scripts/check_kalman_selection.py` で用意できる。新しい比較先を `--run` に指定する。 [元の予測を作る手順](../../docs/UNCERTAINTY_DESIGN.md#公開時系列でカルマンを選ばない条件を確かめる)

```powershell
$comparison = 'runs/seasonal-forecast-crosscheck-new'
& .\.venv\Scripts\python.exe -X utf8 -m experiments.seasonal_forecast_crosscheck.run prepare --run $comparison
& .\.venv-autogluon\Scripts\python.exe -X utf8 -m experiments.seasonal_forecast_crosscheck.run autogluon --run $comparison
$env:JAVA_HOME = (Resolve-Path 'runs/h2o-temurin17-runtime/jdk-17.0.21+9-jre').Path
python3.14 -X utf8 -m experiments.seasonal_forecast_crosscheck.run h2o --run $comparison
& .\.venv\Scripts\python.exe -X utf8 -m experiments.seasonal_forecast_crosscheck.run report --run $comparison
```

今回の出力は `runs/seasonal-forecast-crosscheck-20261124`。`predictions.csv` に各方式の予測、実測値と絶対誤差を、`metrics.csv` と `comparison.json` に集計を保存した。ラグ、分割、予測時刻を照合し、保存済み予測から集計を再現した。
