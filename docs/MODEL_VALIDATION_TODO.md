# 追加モデル・データ検証のTo Do

2026年11月8日（日本時間）の利用者方針を記録します。AutoML本体の整備を優先し、追加モデルの実装・検証は後続作業にします。特に大きなメモリ、GPU、長時間の探索などを要する実験は、今回の完成条件に含めません。この文書の候補は未導入・未検証です。現在の依存関係や既定モデルを変更していません。

## 利用者別の範囲

利用者の追加方針により、0.4.0a7でPyTorch追加の準備として共通のデータ/結果形式とTensorFlow固有処理を分離しました。PyTorchの学習実装や追加依存は今後です。JAXは未経験の検討候補、Julia/Rust版は情勢・必要性を見て考える将来展望としてDS記事へ記載します。大きいモデルやGPU実験の保留方針は維持します。詳細は[DL_BACKENDS.md](DL_BACKENDS.md)です。

- DS向け: 共通のNotebook/CLI入口を保ち、既存モデルから新しいモデルまで任意に比較対象を広げられる構成を目指す。Iris/Wine/Breast Cancerは既存の開発例として維持する。
- ビジネス向け: 意味を理解しやすい小規模な業務データと主要モデルに絞った利用例を優先する。追加モデルやGPUは標準手順の必須条件にしない。

## To Do — 16 GBのWindows環境での運用負荷

2026年11月18日追記。利用者のWindows PC（搭載メモリ16 GB）を負荷検証の基準にする。SeedsとRaisinで主要5モデルをCPU比較した `select` は、それぞれ約3.7秒・154 MiB、約4.1秒・155 MiBだった。メモリ値は小規模サンプルを処理したPythonプロセスのピークワーキングセット。 [実測条件と結果](../data/reference/README.md#windowsでの確認)

- [x] 模擬数値CSVで行数・特徴量数を変え、点検・Parquet変換・抽出と、最大1万行の主要5モデル比較からExcel出力までを測定した。 [Windowsでの手順と結果](../experiments/tabular_scale_benchmark/README.md)
- [x] 実データの小規模例として、Iranian Churnの3,150行で、点検・圧縮・列の抽出・学習にかかる時間、ピークメモリ、一時ファイルの観測最大値を測った。 [Windowsでの手順と結果](../experiments/bp_churn_pilot/README.md)
- [ ] より行・列の多い実データで、同じ工程の負荷と一時ディスクの使用量を測る。
- [ ] 実測結果から、対象データの規模に応じた候補数・CV反復数・並列数の目安を決める。BP向けの初回結果では、正解・誤分類の件数と評価区間を先に示す。
- [ ] Annealing checkはDS向けの任意実行とする。BP向けへの追加は、削減できた入力項目の取得・測定の手間、保留テストでの性能差とその区間、探索にかかる時間とメモリを比較して判断する。 [Annealing checkの検証記録](ANNEALING_CHECK.md)

## モデルの深さ・反復・閾値と過学習の検証計画

学習データに当てはまりすぎる現象と、候補を繰り返し比較するうちに選定データに当てはまりすぎる現象を調べる。まず既存の小規模データと保存済み結果で評価手順を確認し、数値を見て設定を見直す範囲は開発用データに限定する。 [共通の評価設計](UNCERTAINTY_DESIGN.md)

| 順序 | 変える条件 | 比較と確認点 |
| --- | --- | --- |
| 1 | 決定木の `max_depth` | Wineの同じ層化学習・検証分割で `1`, `3`, `5`, `None` を比較する。初回は1分割・4学習。乱数・入力列などを固定し、学習側と検証側のmacro F1、その差、実際の木の深さを記録する。 |
| 2 | 学習のエポック数と初期化seed | まずGCNの `GCN_Training` にある学習lossと選定macro F1を使い、最終エポックと選定エポックの違いを確認する。再学習を行う段階では、エポック上限と初期化seedを別々に変え、分割・学習条件・停止理由を残す。 [GCNの実装とレポート](GRAPH_CLASSIFICATION.md) |
| 3 | アニーリングの探索予算と冷却 | 評価済みのIris全15候補のCVスコアから開始する。同じ初期候補・分割・予算で一様ランダムと比較し、冷却の比較では予算を固定する。探索予算を増やす場合は候補数と学習回数を記録し、CVの上昇と選定後の評価を照合する。 [保存スコアによる探索の再検証](ANNEALING_CHECK.md#irisで探索予算と終点温度を変える) |
| 4 | カルマンの採用閾値 | 保存した選定区間の予測から、直近値に対する最小RMSE改善幅 `0%`, `2%`, `5%`, `10%` の採否を比較する。AirPassengers・NYC Taxiは改善率が負の事例とし、境界の確認には改善率が5%前後の合成例も用意する。候補の比較と閾値の決定は先行区間で行う。 [公開時系列の結果](../experiments/seasonal_forecast_crosscheck/README.md) |

**保留評価。** 深さ、停止点、探索予算、閾値の選定手順を先に固定する。分類では層化した保留行、同一顧客の反復があればグループ単位、時系列では後続期間を使う。前処理は各学習区間で合わせる。既に結果を見ているWine・Iris・公開時系列は開発時の診断に使い、手法の有効性は別データや事前に隔離した期間で確認する。

**実装の入口と出力。** 最初は決定木の比較を `experiments/` の任意実行から始める。既存の分割、指標、保存・検証処理を再利用し、設定、分割ID、seed、候補数、学習・選定・保留指標、単純基準との差、実行時間を既存のExcel・配列・manifestに残す。早期停止の条件、カルマンの採用条件、実行するseed数は独立した設定として扱う。CPUの小規模試験から始め、学習を伴う実行は朝の自動処理が終わってから手動で行う。

### 最初の開発用確認

- **決定木**。Wineを1分割し、4候補を1スレッドで実行した。上限3・5・なしはいずれも実際の深さ3で、検証macro F1は0.9039。初回の保留36件は今後の手順確認用に残した。 [再実行と出力](../experiments/decision_tree_depth/README.md)
- **GCN**。Coraの `GCN_Training` を、manifestにある採用エポックと最終120エポックで照合した。seed 42はエポック84の選定F1が0.7721、最終は0.7639。seed 44はエポック71で0.7966、最終は0.7862。seed 43はエポック118と最終がともに0.8052。3分割とも最終エポックの学習lossは採用時点より低かった。ここでの差はチェックポイント選定に使ったスコアの差で、各チェックポイントの独立した評価は別途行う。 [Coraの分割と保存内容](GRAPH_CLASSIFICATION.md)
- **アニーリング**。固定した分割で探索seedだけを変えた保存結果から、`Selection` のCVと `Test_Differences` の保留F1を比較した。Wineの10 seedのうち、SAが一様ランダムをCVで上回った5件は、保留で2勝・2同点・1敗だった。IrisでもCVで上回った1件は保留で同点だった。SAとランダムの保留F1差の既存の95%区間は20実行すべてで0を含んだ。 [探索条件と区間の定義](ANNEALING_CHECK.md)
- **カルマン**。保存済み選定区間の予測を `assess_pilot` で再判定した。AirPassengersは相対改善−32.64%、NYC Taxiは−14.46%で、閾値0・2・5・10%のすべてで直近値を選んだ。境界例として実測0・直近値予測100を4時点共通にした場合、カルマン予測96・95・94では改善幅が4・5・6%。5%の条件は5%と6%を採用した。改善幅0%の同率は、閾値0%で直近値を選んだ。 [公開時系列の予測と出力](../experiments/seasonal_forecast_crosscheck/README.md)

### 分割と初期化を変えた追加検証

- **決定木**。Raisinの10分割で、検証で選んだ上限付きの木を保留で採点した。上限なしの木は深さ13〜26で、学習macro F1平均1.0000、保留平均0.8056。選定した木は保留平均0.8536で、10分割すべてで上回った。 [深さごとの結果と再実行](../experiments/decision_tree_depth/README.md#raisinで層化分割を10回変えた結果)
- **GCN**。Coraで分割5種と初期化5種を組み合わせた。選定エポックは選定macro F1で平均+0.0037、同じ軌跡の120エポック目と比べ保留では平均−0.0020だった。各組の予測を保存し、両時点のF1を再計算した。 [チェックポイント比較と再実行](../experiments/graph_checkpoint_selection/README.md)

## 軽量な分類での外部クロスチェックと機能差

2026年11月26日。Iris、Wine、Seedsの保存済みの比較を再点検した。両比較で元CSVのハッシュ、学習・保留行、再標本化行、DIYの予測が一致した。H2Oが参照した5分割はAutoGluonとの比較に使った最初の5分割と一致した。保留予測からmacro F1を再計算し、予測ラベルの差を行単位で数えた。 [IrisとWineのAutoGluon比較](../experiments/autogluon_sl_crosscheck/README.md) · [IrisとWineのH2O比較](../experiments/h2o_sl_crosscheck/README.md) · [SeedsとVoyagerの進捗付き比較](../experiments/automl_progress_comparison/README.md)

| データ | DIYの保留macro F1 | AutoGluonの保留macro F1 | H2Oの保留macro F1 | DIYとAutoGluonの予測が異なる行 | DIYとH2Oの予測が異なる行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iris | 0.9666 | 1.0000 | 1.0000 | 1／30 | 1／30 |
| Wine | 0.9710 | 0.9710 | 0.9407 | 2／36 | 1／36 |
| Seeds | 0.8731 | 0.8731 | 0.9277 | 0／42 | 2／42 |

小規模な欠損なしの数値分類の日常開発は、DIYで採点・進捗・出力の回帰を確認する。分割・前処理・採点・候補モデルを変えた際は、必要に応じて外部2方式で保留行をそろえ、予測・指標・選定条件を照合する。モデル内の反復と候補ごとのfoldをそれぞれの記録から確認する。今回の表は固定した小規模データでの記述的な比較。

| 機能 | 現行のDIYと後続の検討 | 参照 |
| --- | --- | --- |
| 欠損値・カテゴリの前処理 | CSV点検で列や欠損を把握できる。表形式の学習入口は欠損のない数値特徴量を受け付ける。補完とカテゴリ変換を任意指定できるようにし、各foldの学習部分で変換を決める作業を優先候補とする。 | [AutoGluonの特徴量処理](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html) · [H2Oの欠損値処理例](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm-faq/missing_values.html) |
| 学習済みモデルの再利用 | `select` と `tune` は設定・分割・評価用の予測を保存する。新しいCSVを繰り返し採点する用途では、学習済み分類器、前処理、列・ラベル定義の保存と読込が候補。 | [AutoGluonの保存・読込API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html) · [H2Oのモデル保存](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/save-and-load-model.html) |
| 探索予算とモデル内の反復 | DIYはCPU並列度・木の深さ・固定グリッドと候補/foldの進捗を扱う。DLとGCNではエポックごとの履歴も保存する。表形式では実行時間上限、停止理由、反復型モデルの曲線を用途に応じて検討する。 | [AutoGluonの学習設定](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) · [AutoGluonの学習曲線](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.learning_curves.html) · [H2Oの探索と停止設定](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) |
| 分類確率・判断閾値・説明 | DIYの混同行列、ラベル別の誤り、入力を変えた感度評価に加え、二値分類の確率と閾値、特徴量の寄与の出力が候補。閾値は学習側で選び、保留での見逃し・誤検知を評価する。クラスの偏りが大きい用途では学習時の重み付けも検討する。 | [AutoGluonの閾値と重要度](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html) · [H2Oのクラス調整](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) · [H2Oのモデル説明](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/explain.html) |
| 回帰・追加モデル・広い時系列予測 | 汎用表形式回帰、勾配ブースティングや積み重ね型のアンサンブル、複数系列・外生変数の予測は後続候補。DIYの1系列の状態推定と先行予測は既に使える。 | [H2O AutoMLの候補と回帰](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) · [AutoGluonの時系列予測](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html) |

手元のAutoGluon Tabular 1.6.3とLightGBM 4.7.0で、IrisをCPU 1スレッドで学習し、`learning_curves=True` から学習側・検証側のmacro F1系列を取得した。出力は `runs/automl-feature-check-20261126/learning-curves.json`。ここで確認したのはログ取得で、比較表とは別実行。

## 追加モデル（すべてTo Do）

| 候補 | 確認したいこと | 実験の扱い |
| --- | --- | --- |
| HistGradientBoosting | 現在のscikit-learnで使える勾配ブースティングと既存モデルの比較 | 本体整備後。CPUでの時間・メモリも記録 |
| LightGBM / XGBoost / CatBoost | 業務向け表形式データでの効果とカテゴリ処理 | 任意依存として後続検討 |
| HDBSCAN等 | 非球状の構造・ノイズを含むクラスタ分析 | ノイズの扱いと評価可能な比較範囲を設計してから |
| TabPFN等の表形式基盤モデル | 事前学習済みモデルの効果・推論負荷 | 版・重み・利用条件を確認し、小規模な実測から。GPU必須とは決めつけない |
| より大きなDL・探索・アンサンブル | 改善幅と計算予算の関係 | 重い実験として保留。自動ダウンロードや大規模実行を開始しない |
| 時系列基盤モデル | 需要予測などへの展開 | 回帰・時系列の分割と評価を整えた後の別範囲 |

強化学習は現在のSL/UL/DL比較の一部ではありません。現時点で追加開発の対象にはせず、必要になったときに環境・行動・報酬・試行単位を扱う別の設計を検討します。

## データ候補と後続検証

海外の企業・大学・ビジネススクールによる公開例を `docs/BUSINESS_DATASETS.md` に調査しました。公開データを優先して検討し、自作ダミーデータは補助の候補にします。公開元の出典・規模・明示された利用条件を調べ、未確認の条件も記録しました。

読者の入口向け2テーマと深掘り向け1テーマの提案は `docs/BUSINESS_SAMPLE_CANDIDATES.md` にあります。利用者の承認後、ホテル予約キャンセル、カフェ券の受取回答、購入履歴からの再購入を、既存SLで小規模に実行しました。結果と評価範囲は `docs/BUSINESS_QUICK_RESULTS.md` に記録しています。追加モデルの保留方針は変更しません。一般的なカテゴリ/欠損処理、回答者/顧客グループ・時間を分離する評価、別期間での確認は後続作業です。

- DS向け追加例: Digits、Dry Bean、形状・密度を制御した人工データ。現在の3データの置換はしない。
- ビジネス向け分類の公開データ候補: UCI Iranian Churn。3,150顧客の利用状況と後の解約を扱う。採用列と符号化されたカテゴリ・算出指標の意味を確認する。購買履歴のCDNOWや、架空教材のIBM Telco Customer Churnも後続候補。
- 自作の説明補助例: 意味の明確な数値列を少数持つ営業案件CSV（架空・説明用）。作る場合は成約前に分かる情報だけで成約有無を予測する。生成条件・seed・架空データであることを明記し、実際の営業効果の根拠にはしない。
- ビジネス向け公開データ候補: [UCI Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)。440顧客の購入額を扱う小規模データ。6購入額列とChannel/Regionを区別し、Channel/Regionをクラスタの真の正解にはしない。現在のULのtarget要件を満たす実験設計か、ラベルなしUL入口の整備が先に必要。
- Bank Marketing、Bike Sharingはカテゴリ処理・回帰/時系列対応などを整えてからの後続候補。Online Retailは限定した顧客集計・再購入分類を実行済みで、汎用入力支援や別期間での評価は今後。

## 実験を始めるときの記録

モデル名だけでなく、版・設定・前処理・計算予算・実行状態・seed・データ出典と加工を対応させます。教師ありの比較では同じ分割で学習側だけを使って選定し、モデル追加のたびに同じtestを見て最良モデルを選び直しません。繰り返し参照したデータを未見の外部検証とは扱いません。

実行時間・メモリ・必要なCPU/GPUを測り、改善幅とともに示します。最新モデルでも大きな改善が保証されるわけではなく、大きく改善する場合もあるため、文献紹介だけで同等性や優劣を結論しません。親記事の参考文献と、このリポジトリで将来測定する結果は分けます。
