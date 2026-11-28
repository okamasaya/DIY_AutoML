# UC_AutoMaiLの主要関数と引数　CSVの分類からカルマンフィルタの診断まで

UC_AutoMaiLをPythonから使うときに、どの関数に何を渡すかを確認するためのリファレンスです。Irisの分類と一次元の位置推定を試し、最後にVoyager 1の実験で使ったEKFを動かします。対象はTestPyPIの `0.4.1a0`、Python 3.12以降です。 [試験配布版と導入手順](https://test.pypi.org/project/uc-automail/0.4.1a0/)

## CSVの点検、モデル比較、Excel

| 関数 | 主な引数 | 戻り値と出力 |
| --- | --- | --- |
| `uc_al.audit_data_csv(csv, output, target=None)` | `csv` は入力CSV、`output` は結果を保存するフォルダー。任意の `target` に正解ラベルの列名を渡すと、ラベルの件数も確認できる | 点検結果のフォルダー。主な出力は `profile.json` と `report.xlsx` |
| `uc_al.select_csv(csv, target, output)` | `target` に分類する正解ラベルの列名を指定。評価設定は下の表を参照 | 選定結果のフォルダー。分割、正解、予測は `arrays.npz`、設定は `manifest.json`、スコアは `report.xlsx` に保存 |
| `uc_al.render_excel_report(source, output, charts="summary")` | `source` は `select_csv` が返した結果フォルダー。図の種類は `summary`、`detail`、`none` から選ぶ | レポート用のフォルダー。下の例では `report_with_figures.xlsx` ができる |

`select_csv` でよく使う任意の引数です。

| 引数 | 既定値 | 指定する内容 |
| --- | --- | --- |
| `seed` | `42` | 学習・保留の分割やモデル比較を再現するための乱数の初期値 |
| `metric` | `"F1"` | 候補選定用の指標。分類ではクラスごとのF1を同じ重みで平均する |
| `bootstrap_repeats` | `1000` | 選定済みモデルの保留評価で、スコアの区間を計算するときの反復回数 |
| `confidence` | `0.95` | スコアの区間の水準 |
| `progress` | `"off"` | `"both"` にすると端末への表示と、進捗ログ・Excelへの記録が有効になる |

実行例では、IrisをCSVに保存してから、点検したデータを同じパスで分類に渡します。出力フォルダーには実行時刻を付けています。 [Irisのデータ](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

```python
from datetime import datetime
from pathlib import Path

from sklearn.datasets import load_iris
import uc_automail as uc_al

root = Path("runs") / datetime.now().strftime("iris-api-%Y%m%d-%H%M%S-%f")
root.mkdir(parents=True)
csv = root / "iris.csv"
load_iris(as_frame=True).frame.to_csv(csv, index=False)

audit = uc_al.audit_data_csv(csv, root / "audit", target="target")
selection = uc_al.select_csv(csv, target="target", output=root / "selection", progress="both")
report = uc_al.render_excel_report(selection, root / "report", charts="detail")
print(audit / "report.xlsx")
print(report / "report_with_figures.xlsx")
```

点検結果は最初のExcel、選定と保留評価は次のExcelで確認できます。手元のCSVを使う場合は、`csv` と両関数の `target` を合わせます。

## EKF・UKFに観測とモデルを渡す

`extended_kalman_filter` と `unscented_kalman_filter` は、観測行の順に状態を推定する関数です。ここでは状態が `n` 成分、観測が `m` 成分、観測行数が `N` とします。

| 引数 | 指定する内容 |
| --- | --- |
| `observations` | 時刻1から順に並べた `(N, m)` の配列。一次元観測も `[[1.1], [2.0]]` のように渡す。欠測成分は `NaN` にする |
| `initial_mean`, `initial_covariance` | 時刻0の状態平均 `(n,)` と共分散 `(n, n)`。位置をメートルで表す場合、位置の分散の単位はm² |
| `transition(state, step)` | 状態を1ステップ進めて `(n,)` を返す関数。ステップ番号は1始まり |
| `measurement(state, step)` | 状態から予測される観測を計算し、`(m,)` を返す関数 |
| `process_covariance` | 遷移後の状態に加わる過程ノイズの共分散 `Q`。固定の `(n, n)` または時刻ごとの `(N, n, n)` |
| `observation_covariance` | 観測ノイズの共分散 `R`。固定の `(m, m)` または時刻ごとの `(N, m, m)`。標準偏差0.10 mの一次元センサーなら `[[0.10**2]]` |
| `transition_jacobian`, `measurement_jacobian` | EKFで必須。遷移と観測の関数を状態で微分したヤコビ行列を返す。形はそれぞれ `(n, n)` と `(m, n)` |

初期共分散と `Q` は半正定値、`R` は正定値の対称行列を渡します。どちらの関数も辞書を返します。`forecast` はその時刻の観測を取り込む前の状態、`filtered` は取り込んだ後の状態です。共分散は `forecast_covariance` と `filtered_covariance` に入ります。観測値の予測と共分散は `predicted_observation` と `predicted_observation_covariance` で参照できます。 [状態推定の設計と検証](https://github.com/okamasaya/DIY_AutoML/blob/main/docs/UNCERTAINTY_DESIGN.md)

## 推定の診断に使う2つの関数

| 関数 | 主な引数 | 主な戻り値 |
| --- | --- | --- |
| `uc_al.filter_diagnostics(result, observations)` | `result` はEKF・UKFの戻り値、`observations` はその実行に使った観測。任意の `max_lag=10` で調べるラグの上限、`confidence=0.95` でNISの参照範囲の水準を指定 | `nis`、`autocorrelation`、`autocorrelation_pairs`、`ljung_box_pvalue` など |
| `uc_al.evaluate_filter_consistency(truth, result)` | `truth` に、`result["filtered"]` と時刻・成分・単位を合わせた真値 `(N, n)` を渡す。NEESの参照範囲は `confidence=0.95` | 推定誤差 `error`、成分ごとの標準化誤差 `pull`、状態全体の標準化二乗誤差 `nees` など |

次は、位置が1ステップごとに1 m進むモデルです。3行目の位置観測を欠測にし、EKF・UKFの推定と診断を表示します。

```python
import uc_automail as uc_al

observations = [[1.1], [2.0], [float("nan")], [3.9]]
truth = [[1.0], [2.0], [3.0], [4.0]]
settings = dict(
    initial_mean=[0.0],
    initial_covariance=[[1.0]],
    transition=lambda state, step: state + 1.0,
    measurement=lambda state, step: state,
    process_covariance=[[0.01]],
    observation_covariance=[[0.10**2]],
)

ekf = uc_al.extended_kalman_filter(
    observations,
    transition_jacobian=lambda state, step: [[1.0]],
    measurement_jacobian=lambda state, step: [[1.0]],
    **settings,
)
ukf = uc_al.unscented_kalman_filter(observations, **settings)
for name, result in (("EKF", ekf), ("UKF", ukf)):
    diagnostics = uc_al.filter_diagnostics(result, observations, max_lag=2)
    consistency = uc_al.evaluate_filter_consistency(truth, result)
    print(name, "位置の推定", result["filtered"][:, 0])
    print("ラグごとの観測ペア数", diagnostics["autocorrelation_pairs"][:, 0])
    print("Ljung–Boxのp値", diagnostics["ljung_box_pvalue"])
    print("NEES", consistency["nees"])
```

この条件では欠測行も時刻を進め、観測更新前の予測を推定として引き継ぎます。Ljung–Boxは等間隔の観測を前提とし、すべての観測が揃っていて残差に変動のある成分に近似p値を返します。この例のp値は `NaN` です。自己相関に使えたペア数は、欠測行を含めた元の時刻順で確認できます。

## Voyager 1の比較実験で使ったEKFを動かす

ここからはVoyager 1の比較実験で使った、軌道の遷移、距離・距離変化率の観測モデル、それぞれのヤコビ行列とEKFへの渡し方です。NASA/JPL Horizonsの公開軌道を平面に射影して参照位置を求め、地球からの距離と距離変化率に乱数でノイズを加えて模擬観測を作ります。記事のコードは、その実験の通常観測条件を取り出して実行する例です。 [Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html) · [比較実験の実装と集計](https://github.com/okamasaya/DIY_AutoML/tree/4ea5e255939e6818186cc13b84ec49ec76e07146/experiments/voyager_crosscheck)

### 公開軌道と模擬観測を用意する

`0.4.1a0` を入れた仮想環境を有効にします。次のPowerShellの手順では、GitHubから比較実験のコードを取得し、Voyager 1と地球の公開軌道をJPLからダウンロードします。すでにリポジトリを取得している場合は、そのルートで `$experiment` の行から実行できます。

```powershell
git clone https://github.com/okamasaya/DIY_AutoML.git
Set-Location DIY_AutoML
$experiment = 'experiments/voyager_crosscheck/horizons.py'
$work = "runs/qiita-voyager-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
$source = "$work/input"
$run = "$work/compare"
New-Item -ItemType Directory -Path $source | Out-Null
$urls = @(python -X utf8 $experiment urls)
if ($LASTEXITCODE -ne 0 -or $urls.Count -ne 2) { throw 'Horizons URLs were not generated' }
Invoke-WebRequest -Uri $urls[0] -OutFile "$source/voyager1.json" -ErrorAction Stop
Invoke-WebRequest -Uri $urls[1] -OutFile "$source/earth.json" -ErrorAction Stop
python -X utf8 $experiment prepare --source $source --run $run
if ($LASTEXITCODE -ne 0) { throw 'Voyager input preparation failed' }
```

`dataset.csv` に観測、地球の状態、7日先の参照位置と時系列分割が入ります。`horizons.json` の `prior_mean` は初日の公開軌道の状態に、実験で定めたずれを加えた初期値です。 [軌道取得と模擬観測のソース](https://github.com/okamasaya/DIY_AutoML/blob/4ea5e255939e6818186cc13b84ec49ec76e07146/experiments/voyager_crosscheck/horizons.py)

### 実験で使ったモデルをEKFに渡す

状態は平面上の `(x, y, vx, vy)` で、位置はAU、速度はAU/年です。遷移は太陽の重力を使い、7日間進めます。観測関数には、その時点の地球の位置・速度も渡します。下の `transition` と `measurement`、2つのヤコビ行列は比較実験で使った実装です。 [力学・観測モデルのソース](https://github.com/okamasaya/DIY_AutoML/blob/4ea5e255939e6818186cc13b84ec49ec76e07146/scripts/make_voyager_demo.py) · [EKFと採点のソース](https://github.com/okamasaya/DIY_AutoML/blob/4ea5e255939e6818186cc13b84ec49ec76e07146/experiments/voyager_crosscheck/run.py)

次のコードをリポジトリのルートに `voyager_qiita.py` として保存します。

```python
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import uc_automail as uc_al

sys.path.insert(0, str(Path("experiments/voyager_crosscheck").resolve()))
from run import OBSERVER, SENSORS, score_ekf_uncertainty


SUN_MU = (2 * np.pi) ** 2  # AU^3 per year^2
STEP = 7 / 365.25  # years
PRIOR_STD = np.array([0.005, 0.01, 0.02, 0.03])
PROCESS_STD = np.array([1e-6, 1e-6, 2e-5, 2e-5])
OBSERVATION_STD = np.array([2e-4, 2e-3])


def derivative(state):
    position, velocity = state[:2], state[2:]
    return np.r_[velocity, -SUN_MU * position / np.linalg.norm(position) ** 3]


def transition(state, _step):
    first = derivative(state)
    second = derivative(state + STEP * first / 2)
    third = derivative(state + STEP * second / 2)
    fourth = derivative(state + STEP * third)
    return state + STEP * (first + 2 * second + 2 * third + fourth) / 6


def transition_jacobian(state, step):
    shifts = np.eye(len(state)) * 1e-5
    return np.column_stack([(transition(state + d, step) - transition(state - d, step)) / 2e-5 for d in shifts])


def measurement(state, observer):
    position, velocity = np.split(state - observer, 2)
    distance = np.linalg.norm(position)
    return np.array([distance, position @ velocity / distance])


def measurement_jacobian(state, observer):
    position, velocity = np.split(state - observer, 2)
    distance = np.linalg.norm(position)
    direction = position / distance
    transverse_velocity = velocity - (direction @ velocity) * direction
    return np.array([np.r_[direction, 0, 0], np.r_[transverse_velocity / distance, direction]])


def main(run):
    dataset = pd.read_csv(run / "dataset.csv")
    prior_mean = json.loads((run / "horizons.json").read_text(encoding="utf-8"))["prior_mean"]
    scenario = dataset.loc[dataset["scenario"].eq("nominal")]
    earth = scenario[OBSERVER].to_numpy()
    result = uc_al.extended_kalman_filter(
        scenario[SENSORS].to_numpy(),
        initial_mean=prior_mean,
        initial_covariance=np.diag(PRIOR_STD**2),
        transition=transition,
        transition_jacobian=transition_jacobian,
        measurement=lambda x, k: measurement(x, earth[k - 1]),
        measurement_jacobian=lambda x, k: measurement_jacobian(x, earth[k - 1]),
        process_covariance=np.diag(PROCESS_STD**2),
        observation_covariance=np.diag(OBSERVATION_STD**2),
    )

    # 予測起点の1つ後のステップから、観測更新前の位置と共分散を取る。
    score = scenario.iloc[:-1]
    holdout = score["split"].eq("test").to_numpy()
    rows, _ = score_ekf_uncertainty(
        score.loc[holdout],
        result["forecast"][1:, :2][holdout],
        result["forecast_covariance"][1:, :2, :2][holdout],
    )
    rmse_km = np.sqrt(np.mean(rows["position_error_km"] ** 2))
    print(f"7日先の位置RMSE  {rmse_km:,.0f} km")
    print(f"95%予測楕円に入った参照位置  {rows['within_ellipse'].sum()}/{len(rows)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

```powershell
python -X utf8 voyager_qiita.py $run
```

比較実験と同じ参照軌道では、通常観測の保留44時点で **位置RMSE 884,611 km**、公称95%予測楕円の被覆は **0/44時点** でした。採点に使う `forecast` は予測対象の時刻の観測を取り込む前の値です。遷移・観測モデルと仮定したノイズから求めた予測楕円が、公開軌道とのずれをどれだけ含んだかも確認できます。 [全条件の数値とAutoGluon・H2Oを含む再実行手順](https://github.com/okamasaya/DIY_AutoML/blob/4ea5e255939e6818186cc13b84ec49ec76e07146/experiments/voyager_crosscheck/README.md#jpl-horizonsの公開軌道を参照した比較)
