# 時系列とIrisでGCNの辺を評価する

ボイジャー、NYC Taxi、IrisとUCIのPedal Meで、辺から情報を集める価値を調べた。回帰では2層GCNと、同じ入力・レイヤー・学習設定で集約を自己ノードだけにしたネットワークを比べた。Irisにはロジスティック回帰も加えた。回帰の誤差は小さいほど良く、分類のmacro F1は大きいほど良い。

## 辺と時刻の扱い

- **ボイジャー**は合成軌道の7日先の二次元位置。先の比較と同じ48時点で学習し、10時点で選定、44時点で採点する。ノードは予測時点で、現在の入力と1〜3時点前から情報を集める。通常観測で学習したモデルを固定し、通常、系統バイアス、過去観測混入の3条件を比べた。 [元のデータと分割](../voyager_crosscheck/README.md)
- **NYC Taxi**は30分先の乗客数。前の比較と同じ2,688・1,344・6,288時点の学習・選定・保留と同じ特徴量を使う。辺は1・48・336時点前を参照し、30分前・前日同時刻・前週同時刻の情報を集約する。 [元のデータと分割](../seasonal_forecast_crosscheck/README.md)
- **Iris**は `data/reference/iris.csv` の花の分類。学習60、選定18、保留72件。学習行で標準化し、4つの実測特徴だけで5近傍の無向グラフを作る。ノード特徴量を全件参照する既知グラフの設定で、クラスラベルは辺の計算に使っていない。 [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)
- **Pedal Me**はロンドン15地域で翌週の配送件数を予測する。各ノードに同じ地域の実測3週分と地域IDの指示変数を渡し、同じ予測時点の地域間で公開データの近接重みに従って集約する。対象週のインデックス3〜21を学習、22〜27を選定、28〜35を保留とした。自己ループを含む配布重みを行ごとに正規化して使用した。 [UCI Pedal Me Bicycle Deliveries](https://archive.ics.uci.edu/dataset/847/pedal+me+bicycle+deliveries)

回帰の標準化は学習区間だけで推定した。過去ラグの欠測は標準化後の0で埋めた。ボイジャーとタクシーの時系列辺は有向で現在と過去だけを参照する。Pedal Meは同時点の地域間で、予測対象週より前の観測から作った特徴量を共有する。3系列で後の時点の入力を変更しても、先に出した予測が一致することを確認した。

固定した隠れ幅16と学習上限120エポックで、回帰は選定区間RMSE、Irisは選定macro F1によって保存エポックを選んだ。回帰の3回は分割を固定して初期値を変え、Irisは分割と初期値を変えた。表はseed 42・43・44の保留結果の中央値。

## 保留結果

| データと条件 | 指標 | 辺なし | GCN | 比較の基準 |
| --- | --- | ---: | ---: | --- |
| ボイジャー・通常 | 位置RMSE（km） | 151,550,199 | 207,860,502 | EKF 41,982 km |
| ボイジャー・系統バイアス | 位置RMSE（km） | 151,494,514 | 207,787,906 | EKF 502,682 km |
| ボイジャー・過去観測混入 | 位置RMSE（km） | 496,190,718 | 458,505,680 | 力学のみ 5,240,893 km |
| NYC Taxi | RMSE（人） | 2,717.39 | 3,684.80 | 直近値 1,708.36人、AutoGluon 1,115.22人 |
| Iris | macro F1 | 0.8609 | 0.8466 | ロジスティック回帰 0.9722 |
| Pedal Me | RMSE（件） | 18.41 | 23.33 | 前週値 17.06件 |

ボイジャーの過去観測混入では、今回の時間辺で辺なしより位置RMSEが約7.6%低下した。通常観測では辺なしが良く、力学と観測モデルを指定したEKFの誤差が最小だった。NYC Taxiの短期予測とPedal Meの週次予測では、今回の集約条件で直近値の方が良かった。Irisは3分割すべてでロジスティック回帰がGCNを上回った。今回の結果は、既知グラフで引用辺が有効だったCoraとは異なるデータと集約の組み合わせでの結果。 [Coraの比較](../../docs/GRAPH_CLASSIFICATION.md#coraでの確認結果)

## 再実行

前回のボイジャー・NYC Taxiの比較出力を使う。別の場所に準備した場合は `--voyager`・`--taxi` で渡す。Pedal Meの配布ZIPは `356930eabc9b78a3a683295cc83110f70b1d66abc5e86de6c140d1eef27d26a4`（SHA-256）を使った。

```powershell
$graphRun = 'runs/gnn-crosscheck-20261124'
New-Item -ItemType Directory -Path $graphRun -Force
Invoke-WebRequest 'https://archive.ics.uci.edu/static/public/847/pedal+me+bicycle+deliveries.zip' -OutFile "$graphRun/pedal-me-uci.zip"
$env:OPENBLAS_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe -X utf8 -m experiments.gnn_application_checks.run --pedal-zip "$graphRun/pedal-me-uci.zip" --output "$graphRun/new"
```

元の出力は `runs/gnn-crosscheck-20261124/evaluation`。`results.json` に入力ハッシュと各seedの指標を、各CSVに保留時点の実測値と予測を保存した。指標をCSVから再計算し、同じ設定で再実行して全ファイルが一致した。共有学習コードへ整理した後も、Coraのseed 42で分割と予測が以前の結果と一致した。
