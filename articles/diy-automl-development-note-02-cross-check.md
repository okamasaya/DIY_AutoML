---
title: "DIY AutoML 開発メモ (2) TensorFlow・PyTorch・AutoGluon・H2Oで結果を照合する"
emoji: "🔬"
type: "tech"
topics: ["automl", "tensorflow", "pytorch", "h2o", "機械学習"]
published: false
---

前回は、CSVの点検から分類モデルを選び、取り置いたデータで最後に評価するところまでを紹介しました。今回は、その結果を別のライブラリでも確かめてみます。題材はIrisとWineです。

TensorFlowとPyTorchには、同じ構成のニューラルネットワークを渡して、予測ラベルと確率を比べました。AutoGluonでは、選んだモデルをDIY AutoMLと同じ保留テスト行・評価指標で照合しました。H2O AutoMLでの照合は追加検証として掲載します。

## TensorFlowとPyTorchで、分類ラベルと予測確率を比べる

WindowsのCPUとPython 3.13.17で、TensorFlow 2.20.0とPyTorch 2.14.0のCPU版を使いました。分類モデルの隠れ層は32・16・8ユニットです。学習行で求めた標準化、Float32の入力、初期重みとミニバッチの行順を共通にし、学習seedを3つ変えて検証しました。両方へ共通の重みを渡して推論すると、確率を0〜1で表したときの最大絶対差は約 6 × 10⁻⁸ でした。 [実験条件と再実行手順](https://github.com/okamasaya/DIY_AutoML/tree/9b6b22c/experiments/dl_backend_crosscheck)

次に、TensorFlowとPyTorchでそれぞれ学習しました。3つのseedすべてで保留テストの予測ラベルが行ごとに一致し、混同行列とmacro F1も一致しました。各クラスに付けた確率を見比べると、次の差がありました。

| データ | 保留テストの行数 | 学習後の予測確率の最大差 |
| --- | ---: | ---: |
| Iris | 30 | 3.05パーセントポイント |
| Wine | 36 | 0.21パーセントポイント |

たとえば学習seed 42のIrisでは、CSVの58行目（ヘッダー込み）を両者ともクラス2と予測しました。そのクラスに付けた確率はTensorFlowが **56.867%**、PyTorchが **53.814%** でした。正解のクラス1に付けた確率は、TensorFlowが42.232%、PyTorchが45.253%です。混同行列では同じ「クラス1をクラス2と予測した」1件として数えられます。

そこで、正解クラスに付けた確率も見るためにlog lossを集計しました。確率が低いほど、その行の損失は大きくなります。Wineでは、固定した3 seedの平均損失の差（PyTorch − TensorFlow）を同じ保留行で再標本化しました。反復を1,000回から20,000回に増やすと、95%区間の上端は −3.0 × 10⁻⁷ から +1.8 × 10⁻⁷ に動き、区間はごくわずかにゼロを含みました。学習済みモデルと保留テストの36行は固定しています。この程度の差で、どちらが優れているかの判断は保留しました。 [log lossの定義](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) · [保存した予測から区間を再計算する手順](https://github.com/okamasaya/DIY_AutoML/blob/9b6b22c/articles/diy-automl-ds-backend-probabilities.md)

## AutoGluonと、同じデータ分割で照合する

DIY AutoMLの順位付け（`rank-sl`）が保存した交差検証（CV）の分割と最終テストの行番号を、AutoGluon Tabular 1.6.3でも使いました。候補と前処理はそれぞれの設定です。AutoGluon側で実施したのは、ランダムフォレスト、Extra Trees、k近傍法、ロジスティック回帰、加重アンサンブルの5候補です。この端末では、空きメモリの判定でLightGBMを見送りました。

両方とも共通のCVの検証行でmacro F1を計算し、ワークフローごとに首位を選びます。DIYでは両データともSVM、AutoGluonではIrisがk近傍法、Wineがロジスティック回帰でした。こうして選んだ候補を共通の最終テスト行で評価しました。表の点数は、各クラスのF1を同じ重みで平均したテストのmacro F1です。 [比較した候補と再実行手順](https://github.com/okamasaya/DIY_AutoML/tree/9b6b22c/experiments/autogluon_sl_crosscheck)

| データ | DIY | AutoGluon | 差と区間 |
| --- | ---: | ---: | ---: |
| Iris・30行 | 0.9666 | 1.0000 | +0.0334［0.0000, +0.1023］ |
| Wine・36行 | 0.9710 | 0.9710 | 0.0000［−0.0911, +0.0911］ |

差はAutoGluonからDIY AutoMLのスコアを引いた値です。学習済みモデルを固定し、正解クラス内から両者で同じ行を取り直す操作を1,000回繰り返して、差の95%区間を求めました。どちらのデータでも区間がゼロを含むので、優劣の判断は保留しました。

Wineでは、macro F1も混同行列も同じでした。**誤分類した行は、DIYが132行目、AutoGluonが136行目**でした（いずれもCSVのヘッダーを1行目として数えています）。

:::details H2O AutoMLとの追加検証

同じWindows PCで、Python 3.14.7とJava 17を使ってH2O AutoML 3.46.0.12を動かしました。候補は一般化線形モデル・分散ランダムフォレスト・勾配ブースティングから最大5つに絞りました。H2OにはDIY AutoMLが保存した5分割の最初の1反復を渡し、各分割のmacro F1の平均で候補を選びました。比較に使うDIY AutoMLの結果は、前節と同じ5反復で選んだモデルです。 [実験条件と再実行手順](https://github.com/okamasaya/DIY_AutoML/tree/main/experiments/h2o_sl_crosscheck) · [H2O AutoMLの設定](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

H2O側では、どちらのデータでも一般化線形モデルが選ばれました。共通の保留テストで評価した結果です。差はH2OからDIY AutoMLのmacro F1を引いた値で、前節と同じ再標本化行で95%区間を求めました。

| データ | H2Oのmacro F1 | DIYとの差と区間 |
| --- | ---: | ---: |
| Iris・30行 | 1.0000 | +0.0334［0.0000, +0.1023］ |
| Wine・36行 | 0.9407 | −0.0302［−0.0960, 0.0000］ |

IrisではAutoGluonとH2Oが30行すべてを正しく分類しました。WineではDIY AutoMLが36行中35行、H2Oが34行を正しく分類しました。H2Oとの差の区間はどちらもゼロを含み、今回のデータと候補で確かめられた範囲に留めます。

:::

## 既存のライブラリで確かめた上で、自作する理由

評価Excelのスコアに加えて、テストの行番号・正解・予測を保存しておくと、同じ点数になった理由を行単位で確かめられます。TensorFlowとPyTorchの比較では予測確率も保存したので、学習をやり直さず、ラベルの一致から一歩進んでlog lossの差と区間まで確かめられました。

既存のライブラリを使って結果を照合しながら、DIY AutoMLでは元データの点検、分割の理由、見逃しや誤検知、評価値の不確かさを一続きでたどれるようにしています。DS向けには再計算できる形で、BP向けには件数や図で結果を渡せるようにするのが、手元で作っている理由です。 [開発を始めた理由](https://zenn.dev/okamasaya/articles/diy-automl-evaluation-framework)

次回は、比較の出発点になる元CSVを保管しながら、ParquetとDuckDBで学習前の作業を繰り返しやすくした話に進みます。
