# DIY AutoMLで予測モデルの使いどころを検証する

投稿用の技術下書き。ボイジャー合成軌道と公開時系列の比較を一つの記事で扱う。想定読者は、時系列分析でモデルを選び、その判断を予測誤差やデータ品質から説明したい開発者。

## 記事のねらい

- データから予測を出し、選んだ方法がどの条件で有効だったかを、共通の保留区間で比較する。
- 予測誤差に加え、入力の整合性、時点別の誤差、モデル自身が見積もった不確かさを確認する。
- ボイジャーの合成例で力学を利用できる場面を示し、航空旅客数とタクシー乗客数でカルマンを打ち切って別の候補を使う場面を示す。

## 比較1 ボイジャーを題材にした合成軌道

**目的**は、地球からの距離と距離変化率を使い、探査機の7日先の太陽中心位置を予測すること。NASAのミッション解説を題材に、太陽重力だけの二次元軌道と地球の円軌道を生成した。真の軌道と観測値を別々に持つため、観測を汚したときも同じ真値で採点できる。 [NASAのVoyager 1ミッション解説](https://science.nasa.gov/mission/voyager/voyager-1/) · [NASAの深宇宙追跡の解説](https://science.nasa.gov/learn/basics-of-space-flight/chapter18-1/)

**比較条件**は単一の軌道で、学習48時点、選定10時点、保留44時点。DIY AutoMLの拡張カルマンフィルタ（EKF）には、軌道の生成と同型の力学モデル、観測モデル、初期の位置・速度の概算を与えた。AutoGluon TabularとH2O AutoMLは過去の正解位置から軸ごとの回帰器を学習した。通常の観測で選んだ回帰器を固定し、以下の3条件を同じ保留時刻で比較した。 [実験設定と再現手順](../../experiments/voyager_crosscheck/README.md)

- 通常観測。
- 420日目以降の距離に `+6σ`、距離変化率に `−4σ` の系統バイアスを加えた観測。`σ` は各観測のノイズの標準偏差。
- 420日目以降の観測ペアを過去の学習時刻のペアに置き換えた観測。採点する時刻と真値は共通。

| 7日先の位置RMSE（km） | 通常 | 系統バイアス | 過去観測の混入 |
| --- | ---: | ---: | ---: |
| DIY AutoML・EKF | 41,982 | 502,682 | 964,714,784 |
| 力学モデルのみ | 5,240,893 | 5,240,893 | 5,240,893 |
| 時刻からの線形外挿 | 21,536,437 | 21,536,437 | 21,536,437 |
| AutoGluon Tabular | 60,110,119 | 60,077,242 | 121,361,610 |
| H2O AutoML | 13,461,984 | 13,575,428 | 439,672,766 |

| 妥当性の点検 | 通常 | 系統バイアス | 過去観測の混入 |
| --- | ---: | ---: | ---: |
| 距離の変化と距離変化率の整合性で閾値を超えた時点 | 2/44 | 6/44 | 42/44 |
| EKFの公称95%予測楕円に真値が入った時点 | 44/44 | 1/44 | 0/44 |

**考察の材料**。通常観測では、適切な力学モデルに観測を加えたEKFの位置RMSEが最も低い。過去の観測が混ざった条件では、力学モデルのみの予測が最も安定した。系統バイアスでは位置RMSEが通常の約12倍に増え、95%楕円の平均長半径は約30万kmのままだった。入力の整合性点検、真値に対する誤差、予測楕円の被覆率は別々に載せると、各指標がどの異常を捉えたかが伝わる。 [入力の点検と楕円の評価方法](../../experiments/voyager_crosscheck/README.md#ekf自身が見積もった誤差の評価)

## 比較2 航空旅客数とニューヨークのタクシー乗客数

**目的**は、旅客数の公開時系列で、現行DIY AutoMLがカルマンをいつ打ち切り、ほかの候補を加えると結果がどう変わるかを確認すること。AirPassengersは月次144件で1か月先、NYC Taxiは30分集計10,320件で30分先を予測した。 [RのAirPassengers](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html) · [NABのNYC Taxiと集計方法](https://github.com/numenta/NAB/tree/master/data#nab-data-corpus)

**比較条件**。時系列順に学習・選定・保留を分け、AirPassengersは60・36・48時点、NYC Taxiは2,688・1,344・6,288時点を使った。現行の `kalman-track` は先行区間でカルマンのRMSEが直近値より5%以上改善した場合に処理を続ける設定で、両系列で直近値を選んだ。季節候補を含む試験では、先行区間のRMSEで候補を選んだ。AutoGluonとH2Oには、過去の観測のラグとカレンダーから作る共通の特徴量を渡し、同じ保留区間で採点した。 [時系列の分割と候補](../../experiments/seasonal_forecast_crosscheck/README.md)

| 方式 | AirPassengers 保留RMSE（千人） | NYC Taxi 保留RMSE（人） |
| --- | ---: | ---: |
| DIY AutoML 現行・直近値 | 48.04 | 1,708.36 |
| DIY AutoML 季節候補を含む試験 | 27.89 | 1,708.36 |
| AutoGluon Tabular | 27.49 | 1,115.22 |
| H2O AutoML | 42.34 | 1,171.83 |
| カルマンを最後まで実行 | 65.69 | 1,962.19 |

- 現行の選定で直近値へ切り替えると、カルマンを継続した場合に比べて保留RMSEが航空で約27%、タクシーで約13%下がった。
- 航空では「前年同月×過去の成長倍率」が、季節候補を含む試験で選ばれた。保留RMSEはAutoGluonと0.41千人差で、保留MAEは季節候補22.09千人、AutoGluon 22.61千人だった。
- タクシーではAutoGluonとH2Oの全体RMSEがDIY AutoMLの直近値より低い。保留6,288時点を実測値との絶対誤差で一対一に比べると、直近値の方が小さい時点は2,056、H2Oの方が小さい時点は4,232だった。全体スコアと各時点の誤差を併記すると、どの方法がどの時点で有効だったかを追える。 [時点別予測と出力の説明](../../experiments/seasonal_forecast_crosscheck/README.md#再実行)

## 原稿で示す結論と適用範囲

- DIY AutoMLで示した価値は、カルマンを使う条件を切り分け、直近値や季節予測を同じ時点で比較し、結果の妥当性を議論できること。ボイジャーでは、観測が正常な場合に力学モデルとEKFが有効だった。公開時系列では、カルマンを打ち切る選択が有効だった。
- ボイジャーでの成績にはEKFへ与えた力学の事前知識が効いている。公開時系列の結果は各データの一つの時系列分割に対するもの。AutoGluon Tabular 1.6.3とH2O AutoML 3.46.0.12は小規模なCPU設定で比較し、H2Oの一部候補は小標本の最小行数の条件を満たさなかった。記事では対象のモデル、条件と指標を結果の近くに置く。 [AutoGluonの学習API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) · [H2O AutoMLの設定](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## DEV Community English draft outline

**Working title**  Evaluating When to Use a Kalman Filter with DIY AutoML

- **Question.** When does a domain model improve forecasts, and what evidence tells us when to switch to a simpler method? Introduce held-out error, input consistency, and empirical coverage as distinct measurements.
- **Voyager-inspired case.** Use a synthetic outbound trajectory and predict heliocentric position seven days ahead. The EKF has access to a dynamics model of the same form as the simulator. Under nominal observations it reaches an RMSE of 41,982 km. When earlier observations are replayed under later timestamps, the physics-only forecast has the lowest error. The EKF's nominal 95% ellipse contains the true position at 44 of 44 nominal points, one of 44 biased points, and zero of 44 replayed points.
- **Public time series.** Use one-month-ahead AirPassengers and 30-minute-ahead NYC Taxi forecasts with chronological splits. The current DIY workflow selects persistence on both. Adding seasonal candidates in the AirPassengers experiment gives a held-out RMSE of 27.89 thousand passengers, compared with AutoGluon's 27.49 thousand. AutoGluon and H2O lead on NYC Taxi overall. Persistence has a lower absolute error than H2O at 2,056 of the 6,288 held-out timestamps.
- **Takeaway.** Show the input and modeling assumptions beside each result. Include what the comparisons reveal about when a filter, a simple baseline, or a learned tabular model works under these settings.

## ChatGPTへの引継ぎ

上の骨子と表を使い、①なぜこの比較をしたか、②ボイジャーで力学モデルが効く条件、③公開時系列で手法を切り替えた結果、④モデルの有用性を判断する材料、の順で文章化する。数字には単位と保留区間を添える。NYC Taxiでの優劣は全体RMSEと時点別比較の両方を残す。日本語本文の構成と英語見出しは編集時に調整できる。

Gitにある再現手順は、[ボイジャー比較](../../experiments/voyager_crosscheck/README.md) と [公開時系列比較](../../experiments/seasonal_forecast_crosscheck/README.md)。作業端末の `runs/voyager-crosscheck-20261124`、`runs/voyager-crosscheck-uncertainty-20261124`、`runs/seasonal-forecast-crosscheck-20261124` に比較表の元データがある。タクシーの時点別比較では最後のディレクトリの `predictions.csv` を使い、`nyc-taxi` の `holdout` に絞って、同一時刻の `absolute_error` を比較した。
