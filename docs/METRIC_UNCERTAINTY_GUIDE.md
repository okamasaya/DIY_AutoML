# 評価値の誤差評価（0.4.0a6）

`uncertainty`は、保存済み分類結果のAccuracyとmacro Precision/Recall/F1に、bootstrap標準誤差とpercentile区間を付けます。クラス別Precision/Recall/F1、TP/FP/FN、件数、全反復値も保存します。学習済みモデルは固定し、元のrunを変更せず、新規フォルダにschema 9のExcel・NPZ・PNG・manifestを作ります。

## 最小手順

```sh
.venv/bin/python -m diy_automl select --csv data/reference/wine.csv --target target --output runs/wine-select
.venv/bin/python -m diy_automl uncertainty --run runs/wine-select --output runs/wine-uncertainty
.venv/bin/python scripts/validate_uncertainty_run.py runs/wine-uncertainty
```

開くファイルは `runs/wine-uncertainty/report.xlsx` です。この入口では最初から区間の図を埋め込みます。schema 8を作る`report`コマンドを重ねる必要はありません。

対象はcompletedの`run`（schema 2のSL/DL）、`select`（3）、`repeat-dl`（5）、`tune`（7）です。探索的SLは宣言順の全5モデル、DL反復は全学習seedを保持します。testでモデルやseedを選ぶ入口にはしません。ULの指標は別の性質なので対象外です。ULの初期化/部分抽出の安定性には既存の`repeat-ul`と`subsample-ul`を使います。

過去のschema 2にtarget配列がない場合は、元のCSVの`--csv`を指定します。入力SHA256・特徴量とクラスの対応が保存記録と一致する必要があります。新しいrunはtargetを保存するので、通常この補足は不要です。

```sh
.venv/bin/python -m diy_automl uncertainty --run validation/current/wine-run --csv data/reference/wine.csv --output runs/legacy-wine-uncertainty
```

## 何を誤差評価するか

点推定は、保存されたtest正解と予測から計算します。正解と予測を同じindexで復元抽出するpaired bootstrapで、既定は1,000反復・seed 43・95%です。ここでpairedは正解/予測の対応を意味し、モデル間差の検定ではありません。

- `stratified`（既定）: 正解クラス別件数を固定し、各クラス内で行を復元抽出します。既存select/tuneと同じ条件付きの範囲です。
- `ordinary`: 全test行から同数の行を復元抽出し、クラス別件数も変動します。依然として同じ観測対象からの行単位再標本化です。

```sh
.venv/bin/python -m diy_automl uncertainty --run runs/wine-select --resampling ordinary --bootstrap-repeats 2000 --bootstrap-seed 43 --confidence 0.95 --output runs/wine-ordinary
```

標準誤差（Bootstrap SE）は、bootstrapで得た評価値の標本標準偏差（ddof=1）です。**反復数の平方根では割りません。** SD/√Bはbootstrap平均を有限回の計算で近似する誤差に関係する量で、評価指標そのものの標準誤差とは別です。

区間は上下の経験分位点で、線形補間を使います。非対称になりうるため、点推定±SEを95%区間と置き換えません。点推定がpercentile区間の外に来る場合も、点と区間をそのまま描きます。各モデル・各指標の周辺区間であり、多重比較を補正した同時区間ではありません。

## Excelの7シート

| シート | 内容 |
| --- | --- |
| Metric_Uncertainty | 各モデルの4指標、点推定、bootstrap平均・SE、区間端点・幅、反復数・水準、ゼロ幅フラグ |
| Class_Uncertainty | 各クラスのPrecision/Recall/F1と同じ誤差評価、正解/予測件数、TP/FP/FN |
| Evaluation_Units | train/validation/test件数、観測クラス数、最少クラス件数、注意事項 |
| Bootstrap | 各モデルの4指標の全反復値 |
| Scope | 推定対象、再標本化単位、前提、含まない変動 |
| Summary | 入口・方式・反復数などの概要 |
| Intervals | 点推定と区間の図。保存元の順序を保持 |

macro指標の対象は保存元の全クラスで固定し、`zero_division=0`です。testで観測0件のクラスも対象から勝手に除きませんが、値の0は「性能が確認できた」意味ではありません。クラス別には`class_absent_in_test`、件数シートには注意を残します。通常bootstrapで反復中にクラスが消えた場合も、同じクラス集合と計算規約を使います。

`arrays.npz`には、評価ごとの正解・予測・train/validation/test index、全再標本化index、4指標の反復値、クラス別反復値、confusion matrixを保存します。`source_manifest.json`と`source_report.xlsx`は元ファイルのコピーです。元NPZのSHA256も記録しますが、特徴量やモデル重み全体はコピーしません。

## Python / Notebook

```python
from diy_automl import evaluate_uncertainty

output = evaluate_uncertainty(
    "runs/wine-select", "runs/wine-uncertainty-notebook",
    bootstrap_repeats=1000, confidence=0.95,
    bootstrap_seed=43, resampling="stratified",
)
```

直接配列を渡す場合は`evaluate_metric_uncertainty(truth, predictions, classes, ...)`を使います。正解・予測は整数符号化し、`classes`は元の全クラスに対応する`np.arange(C)`です。戻り値は全体指標DataFrame、クラス別DataFrame、再標本化index等の辞書です。CSVのラベル変換は上位の保存結果入口に任せます。

## 範囲と制約

行が独立で、対象集団を適切に代表するなどの前提があります。依存する顧客・回答者・時系列に対して、この行単位区間をそのまま有効とはしません。学習/選定/再学習の変動、分布の変化、系統的な偏り、数値計算誤差を網羅する区間ではありません。通常bootstrapへ変更しても、この制約は解消しません。

ゼロ幅は未知の不確かさゼロではありません。少ない独立標本、完全予測、観測0クラス、離散的な指標などで退化しえます。反復数が少ない場合や、最少クラス件数が10未満などは注意を表示します。10件以上なら十分という採用基準ではありません。標本が偏っているかどうか、必要件数や業務で採用すべきかは、この区間だけでは判定しません。

上限は評価ごとにtest 10,000行・50クラス、1出力20評価、2〜10,000反復、全評価の反復×test件数合計10,000,000、反復×評価数×(4+3×クラス数)合計5,000,000です。上限を超えたら一部を自動選抜せず拒否します。行は復元抽出し、学習・推論はやり直しません。TensorFlowなしでも保存済みDL結果を扱えます。

検証器は記録seedから全再標本化を再生成し、保存配列・点推定・SE・区間・表・埋込PNGとの対応を確認します。元のCSVやモデルは不要です。元の学習やデータの代表性を認定する機能ではありません。assertを無効にする`python -O`は使いません。
