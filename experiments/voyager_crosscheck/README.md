# ボイジャー合成軌道でのDIY AutoML・AutoGluon・H2O比較

地球から見た距離と距離変化率から、7日先の探査機の太陽中心位置 `(x, y)` を予測する。作成済みのボイジャー合成データの乱数seed 42を使った。表形式モデルには現在と前回の観測、現在の地球の位置・速度、現在の時刻を与える。軸ごとに回帰器を学習し、2軸を合わせた位置誤差で採点する。DIY AutoMLのEKFには同じ観測列と地球の状態に加え、合成軌道を生成した力学モデルと同型のモデル、観測モデル、出発時点の近似状態を与える。比較には、初期推定を力学モデルだけで進める予測と、時刻から位置を線形外挿する予測も入れた。 [合成軌道の仕様](../../docs/UNCERTAINTY_DESIGN.md#ボイジャーを題材にした合成追跡データ)

## 分割と入力条件

予測対象の日を基準に、21〜350日目の48行を学習、357〜420日目の10行を検証、427〜728日目の44行を保留とした。AutoGluonとH2Oは通常観測で一度学習・選定し、選んだモデルで3条件を評価した。EKFは各時刻の観測を逐次取り込み、次の観測が来る前の位置予測を採点した。

- **通常**は元の合成観測。
- **系統バイアス**は420日目以降の観測に、距離で `+6σ`、距離変化率で `−4σ` が入る条件。
- **過去観測の混入**は420日目以降の距離と距離変化率を、学習期間の観測ペアから乱数seed 42で選んだ値に置換した条件。時刻と正解位置は共通。

AutoGluon Tabular 1.6.3では線形回帰・Random Forest・Extra Treesと加重アンサンブルを候補とし、検証10行を `tuning_data` に使った。H2O AutoML 3.46.0.12ではGLM・DRF・GBMから最大4モデル、`nfolds=0` とし、同じ検証10行を `validation_frame` と `leaderboard_frame` に使った。両方ともCPU 1スレッドで実行した。H2Oでは一部の候補が最小行数の条件を満たさなかった。選ばれたモデルは両軸ともAutoGluonが加重アンサンブル、H2OがGLM。 [AutoGluonの学習API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) · [H2O AutoMLの設定](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## 結果

| 7日先の位置RMSE（km） | 通常 | 系統バイアス | 過去観測の混入 |
| --- | ---: | ---: | ---: |
| DIY AutoML EKF | **41,982** | **502,682** | 964,714,784 |
| 力学モデルのみ | 5,240,893 | 5,240,893 | **5,240,893** |
| 時刻の線形外挿 | 21,536,437 | 21,536,437 | 21,536,437 |
| AutoGluon Tabular | 60,110,119 | 60,077,242 | 121,361,610 |
| H2O AutoML | 13,461,984 | 13,575,428 | 439,672,766 |

力学を使う2手法には、単一軌道を外挿するという今回の設定に対応した事前情報がある。通常ではEKFが観測を取り込むことで力学モデルのみより誤差を下げた。過去観測の混入では、観測を使う3方式すべてで通常時より誤差が増え、力学モデルのみが最も小さかった。結果はこの1軌道と候補・入力条件に対して読む。

入力の点検として、距離の前回との差と、距離変化率を台形則で7日間積分した値との差を計算した。通常観測の検証10行における絶対差の95%点を閾値とすると、保留44行のうち通常は2行、系統バイアスは6行、過去観測の混入は42行が閾値を超えた。モデルの点検では、検証10行の位置誤差90%点を各方式の参照値とした。EKFでこの値に収まったのは通常39/44、系統バイアスと過去観測の混入はいずれも0/44だった。この割合は、短い検証区間から得た参照値に対する経験的な結果として読む。

## 再実行と出力

リポジトリのルートで実行する。AutoGluon・H2Oの環境は既存の比較実験と同じ構成を使う。H2Oはループバックで起動して終了時に停止する。 [AutoGluon環境](../autogluon_sl_crosscheck/README.md#Windowsで再実行する) · [H2O環境](../h2o_sl_crosscheck/README.md#再実行)

```powershell
$run = 'runs/voyager-crosscheck-new'
& .\.venv\Scripts\python.exe -X utf8 experiments/voyager_crosscheck/run.py prepare --run $run --source runs/voyager-synthetic-20261124
& .\.venv\Scripts\python.exe -X utf8 experiments/voyager_crosscheck/run.py diy --run $run
& .\.venv-autogluon\Scripts\python.exe -X utf8 experiments/voyager_crosscheck/run.py autogluon --run $run
$env:JAVA_HOME = (Resolve-Path 'runs/h2o-temurin17-runtime/jdk-17.0.21+9-jre').Path
python3.14 -X utf8 experiments/voyager_crosscheck/run.py h2o --run $run
& .\.venv\Scripts\python.exe -X utf8 experiments/voyager_crosscheck/run.py report --run $run
```

`dataset.csv` に予測時点・予測対象の日・分割を、`input.json` に元データのハッシュと混入観測の元時刻を保存する。`predictions.csv` は予測位置と真値、`metrics.csv` は精度と検証時の参照値、`quality.csv` は入力の点検結果。`comparison.json` は使用バージョンと結果の集計を含む。今回の出力は `runs/voyager-crosscheck-20261124`。分割、予測時刻、集計を照合し、最終時点の観測を変更してもその時点までを対象とするEKFの先行予測が変わらないことを確認した。
