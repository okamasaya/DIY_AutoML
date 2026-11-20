---
title: "TensorFlowとPyTorchで分類結果が一致しても、予測確率はどう違うか"
emoji: "🔬"
type: "tech"
topics: ["tensorflow", "pytorch", "機械学習", "python", "automl"]
published: false
---

DIY AutoMLで、小さな分類モデルをTensorFlowとPyTorchの両方で学習しました。IrisとWineの保留テストでは、3つの学習seedすべてで予測ラベルが行ごとに一致しました。そこで、ラベルを決める前の予測確率も保存し、どのくらい差があったかを調べました。

## 同じ分類結果でも、正解クラスに付けた確率は違った

Irisの学習seed 42では、入力CSVの58行目（ヘッダーを1行目とする）で、両者ともクラス2を選びました。この行の正解はクラス1です。

| 予測確率 | TensorFlow | PyTorch |
| --- | ---: | ---: |
| クラス2（予測） | 56.867% | 53.814% |
| クラス1（正解） | 42.232% | 45.253% |

混同行列では、どちらも「クラス1をクラス2と予測した」1件です。Log lossでは、正解クラスに高い確率を付けたPyTorchのほうが、この1行の損失は小さくなります（TensorFlow 0.862、PyTorch 0.793）。

## 重みを共通にした推論と、別々に学習した結果を比べる

WindowsのCPUで、隠れ層が32・16・8のMLPを使いました。データの分割を共通にし、標準化の平均と標準偏差は学習行から求めました。各seed（42、43、44）で同じFloat32入力、初期重み、ミニバッチの行順を使い、バッチサイズ16で100 epoch学習しました。

| データ | 同じ重みで推論した場合 | 別々に学習した後 |
| --- | ---: | ---: |
| Iris（テスト30行） | 約 $6.0 \times 10^{-8}$ | 0.03053 |
| Wine（テスト36行） | 約 $6.0 \times 10^{-8}$ | 0.00211 |

表は、テスト行・クラス・seed全体での予測確率の最大絶対差です。同じ重みでの比較には、共通の初期重みを使った推論と、TensorFlowで学習した重みをPyTorchにも読み込ませた推論の両方を含めました。別々に学習した後の最大差は、Irisで3.05パーセントポイント、Wineで0.21ポイントです。全行でラベルが一致したため、混同行列とmacro F1は各seedで一致しました。 [比較条件と再実行手順](https://github.com/okamasaya/DIY_AutoML/tree/main/experiments/dl_backend_crosscheck)

学習ではAdamの学習率、$\beta_1$、$\beta_2$、$\epsilon$の設定値を揃え、更新はそれぞれの実装を使いました。KerasとPyTorchでは、バイアス補正に対する$\epsilon$の扱いが異なります。今回の比較で観測した差には、最適化や浮動小数点演算の違いが含まれ得ます。どの要因がどれだけ寄与したかを調べるには、更新則を共通にするなど、条件を追加した実験が必要です。 [KerasのAdam](https://keras.io/api/optimizers/adam/) · [PyTorchのAdam](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)

## 同じテスト行を引き直して、log lossの差に区間を付ける

Log lossは、各行で正解クラス $y_i$ に割り当てた確率 $p_{i,y_i}$ を取り、$-\log p_{i,y_i}$ の平均を求めます。正解クラスの確率が小さい行ほど、大きな損失になります。 [scikit-learnのlog loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)

各行で「PyTorch − TensorFlow」の損失差を求め、固定した3学習seedで平均しました。次に、各クラスの行数を固定したまま、クラス内から復元抽出しました。両フレームワークと3 seedに同じ行の抽出を適用し、1,000回の平均差の2.5%点と97.5%点を区間の端にしました。

| データ | 平均log loss差 | 95%区間（1,000回） |
| --- | ---: | ---: |
| Iris | −0.002212 | [−0.005877, +0.000602] |
| Wine | −0.00006439 | [−0.00013799, −0.00000030] |

負の値はPyTorch側の損失が小さい方向です。Irisでは区間がゼロをまたぎます。Wineでは1,000回の区間がゼロのすぐ手前まで来たため、同じ乱数列を20,000回まで延ばしました。すると上端は **−3.0 × 10⁻⁷ から +1.8 × 10⁻⁷** になり、区間がわずかにゼロをまたぎました。

増やしたのは再標本化の回数です。Wineで観測したテスト行は36行で、学習済みモデルも固定しています。区間はこの分割・クラス構成・3学習seedを前提とした、テスト行の標本不確かさの探索的な見積もりです。別の分割や再学習での変動は、分割や学習の段階から繰り返して確かめる必要があります。

正解ラベルのone-hot表現と予測確率の二乗誤差をクラス方向に合計する、多クラスBrier scoreも計算しました。平均差はIrisで−0.001214、Wineで−0.00005093でした。確率を閾値や費用の判断に使う場合は、こうしたスコアに加えて、校正や実際に使う閾値での判断件数も確認します。 [Brier scoreの定義](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) · [確率の校正](https://scikit-learn.org/stable/modules/calibration.html)

## 入力の測定値をずらすと、ラベルも動いた

Irisでは、学習済みモデルを固定し、推論時の花弁幅に一律で+0.5 cmを足すシナリオも試しました。どちらのフレームワークでも各seedで同じ6〜7行のラベルが変わり、macro F1が0.166〜0.232下がりました。この値は、仮定したずれに対する感度です。実測の系統誤差を評価するには、測定器や測定条件の記録が必要になります。

## 保存したlogitから再計算する

比較を実行すると、各フレームワークのNPZにテストlogit、`arrays.npz`に正解ラベルと1,000回分の再標本化indexが保存されます。下のコードをPythonファイルにして、引数にIrisかWineの比較出力フォルダーを渡すと、モデルを学習し直さずにlog loss差と区間を確認できます。 [出力ファイルとWindowsでの再実行手順](https://github.com/okamasaya/DIY_AutoML/tree/main/experiments/dl_backend_crosscheck)

:::details log lossの平均差と、1,000回・20,000回の区間を再計算する
~~~python
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

folder = Path(sys.argv[1])
with np.load(folder / "arrays.npz") as saved:
    truth = saved["truth"]
    stored_draws = saved["test_bootstrap_indices"]


def row_log_loss(backend):
    with np.load(folder / f"{backend}.npz") as saved:
        logits = saved["test_logits"].astype(np.float64)
    correct_logits = logits[:, np.arange(truth.size), truth]
    return logsumexp(logits, axis=-1) - correct_logits


delta = (row_log_loss("pytorch") - row_log_loss("tensorflow")).mean(axis=0)
print(f"平均log loss差: {delta.mean():+.8g}")

# 記事で使った実験と同じ再標本化seed
rng = np.random.default_rng(43)
strata = [np.flatnonzero(truth == label) for label in np.unique(truth)]
draws = np.stack([
    np.concatenate([rng.choice(rows, rows.size, replace=True) for rows in strata])
    for _ in range(20_000)
])
np.testing.assert_array_equal(draws[:len(stored_draws)], stored_draws)

replicates = delta[draws].mean(axis=1)
for count in (1_000, 20_000):
    lower, upper = np.quantile(replicates[:count], [0.025, 0.975])
    print(f"{count:,}回 [{lower:+.8g}, {upper:+.8g}]")
~~~
:::
