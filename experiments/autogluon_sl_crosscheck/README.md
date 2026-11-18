# AutoGluon Tabularとのクロスチェック

DIY AutoMLの `rank-sl` が保存した行番号、層化5分割、最終テストを読み込み、AutoGluonを同じデータで評価します。5候補は Random Forest、Extra Trees、k-NN、ロジスティック回帰、AutoGluonの加重アンサンブルです。各foldの学習行の中でAutoGluonが内部検証を行い、外側の検証行を渡す前に `refit_full` で学習行全体に再適合します。両方のCV順位は、同じ外側の検証行に対する予測とmacro F1から計算します。モデル構成、前処理、ライブラリのバージョンは各ワークフローの設定に従います。 [AutoGluonの学習API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) · [refit_full](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.refit_full.html)

## Windowsで再実行する

リポジトリのルートで実行します。Python 3.13.17とAutoGluon Tabular 1.6.3で確認しました。既存の `.venv` を元に比較専用環境を一度作ります。 [AutoGluonの導入ガイド](https://auto.gluon.ai/stable/install.html)

```powershell
.\.venv\Scripts\python.exe -m venv .venv-autogluon
.\.venv-autogluon\Scripts\python.exe -m pip install "autogluon.tabular==1.6.3" -e .
.\.venv-autogluon\Scripts\python.exe -m pip check
```

検証済みの `rank-sl` 出力が手元にある場合は、そのフォルダーを `--reference` に指定します。新しく作成する場合は以下の順に実行します。出力先には新しいフォルダー名を指定します。

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -X utf8 -m diy_automl rank-sl --csv data/reference/wine.csv --target target --comparison-fraction 0.25 --output runs/wine-diy-crosscheck
.\.venv-autogluon\Scripts\python.exe -X utf8 experiments/autogluon_sl_crosscheck/run.py --csv data/reference/wine.csv --reference runs/wine-diy-crosscheck --output runs/wine-autogluon-crosscheck
```

## 結果を読む

- `report.xlsx` の `Crosscheck_Leaders` は、両方のCV首位と最終テストの結果です。最終テストでは、各ワークフローがCVで選んだ1候補を共通の行で評価します。
- `Crosscheck_Test_Delta` は AutoGluon スコアから DIY スコアを引いた値とpaired bootstrap区間です。保存済みのDIY出力と同じ再標本化行を使用します。区間が0を含む場合は優劣を保留します。区間は固定した予測、観測クラス件数、行の独立性を前提とします。
- `CV_Ranking` はAutoGluon内の順位と反復間の変動です。元の `rank-sl` に比較用分割があれば、`Comparison_Differences` にAutoGluon内の各モデルとCV首位の差も記録します。
- `arrays.npz` には行番号、各foldの予測、最終テストの予測と再標本化を保存します。`manifest.json` には依存バージョン、入力・参照ファイルのハッシュ、AutoGluonの評価値との照合件数を記録します。

IrisとWineは小規模な開発用データです。比較結果は今回の分割と設定に対して読みます。

2026年11月18日のWindows実行では、次の結果を確認しました。差は最終テストのmacro F1に対する値です。

| データ | DIYのCV首位 / テストF1 | AutoGluonのCV首位 / テストF1 | AutoGluon − DIY の差（95%区間） |
| --- | --- | --- | --- |
| Iris（テスト30行） | SVM / 0.9666 | k-NN / 1.0000 | +0.0334 [0.0000, +0.1023] |
| Wine（テスト36行） | SVM / 0.9710 | ロジスティック回帰 / 0.9710 | 0.0000 [−0.0911, +0.0911] |

両方で分割と再標本化行を照合し、scikit-learnで1,000反復の差を再計算しました。保存したAutoGluonモデルを読み直し、最終テストの予測が一致することも確認しました。LightGBMは試行時に空きメモリの判定で停止したため、上記の5候補で実施しました。

## 保存モデルで予測する

比較用環境のPythonで次を実行できます。`selected_refit` は、この実験でCV首位になった再学習済みモデルの名前です。 [TabularPredictor.load](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.load.html)

```python
import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

run = Path("runs/wine-autogluon-crosscheck")
metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
predictor = TabularPredictor.load(str(run / "predictor"))
features = pd.read_csv("data/reference/wine.csv", usecols=metadata["features"]).head()
encoded = predictor.predict(features, model=metadata["selected_refit"])
print(encoded.map(dict(enumerate(metadata["classes"]))))
```
