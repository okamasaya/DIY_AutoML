# TensorFlowとPyTorchのCPU比較

DIY AutoMLの共通データ準備と MLP 32-16-8 を使って、2つのフレームワークを比較します。最初に保留テストを確保し、標準化は学習行でfitします。両者に同じFloat32入力、初期重み、各epochの行順、100 epoch、batch 16を渡します。Adamの学習率、beta、epsilonの数値は揃え、更新は各フレームワークの実装で行います。Kerasではepsilonが論文のepsilon-hatに相当します。 [Keras Adam](https://keras.io/api/optimizers/adam/) · [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)

通常の学習入口は `run` と `repeat-dl` です。この比較はWindows CPU、Python 3.13.17、TensorFlow 2.20.0、Keras 3.15.1、PyTorch 2.14.0+cpuで実行します。比較スクリプトはフレームワークごとに別プロセスを順番に起動し、CPUのスレッド数などの設定をmanifestに記録します。

## Windowsで再実行する

リポジトリのルートで比較環境を用意します。 [TensorFlow CPUの配布](https://pypi.org/project/tensorflow-cpu/2.20.0/) · [PyTorchの導入](https://pytorch.org/get-started/locally/)

```powershell
.\.venv\Scripts\python.exe -m venv .venv-dl
.\.venv-dl\Scripts\python.exe -m pip install -r requirements-core.lock.txt -e ".[dl]"
.\.venv-dl\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.14.0+cpu"
.\.venv-dl\Scripts\python.exe -m pip check
```

出力先には新しいフォルダーを指定します。以下のオフセットは運用時の測定ずれを想定した例です。

```powershell
.\.venv-dl\Scripts\python.exe -X utf8 experiments/dl_backend_crosscheck/run.py --csv data/reference/iris.csv --shift "petal width (cm)=0.5" --output runs/iris-dl-backend-comparison
.\.venv-dl\Scripts\python.exe -X utf8 experiments/dl_backend_crosscheck/run.py --csv data/reference/wine.csv --shift "alcohol=0.5" --output runs/wine-dl-backend-comparison
```

## 出力の読み方

- `Forward_Parity` では共通の初期重みと、TensorFlowで学習済みの重みをPyTorchへ移した場合について、両者のlogit・確率・予測ラベルの差を確認します。
- `Test_Intervals` は各学習seedの固定モデルの標本不確かさです。`Backend_Differences` は PyTorch − TensorFlow の差を、共通のテスト行を一緒に再標本化した1,000反復で評価します。
- `Mean_Seed_Difference` は同じ再標本化行をseed間でも共有した平均です。区間の推定対象は、選んだ3 seedで学習した固定モデル群です。
- `Offset_Scenarios` は、学習側を固定したまま指定のずれをテスト入力に加えた場合の影響です。実際の系統的な偏りは未測定として記録します。
- `plan.npz` に分割、標準化済み入力、初期重み、行順が入ります。フレームワーク別のNPZには学習後の重み・履歴・予測を、`arrays.npz` にはテスト指標の再標本化を保存します。Denseの重みは共通の `(入力, 出力)` 順です。

分割seedは42、学習seedは42・43・44を既定とします。IrisとWineは開発用の小規模データであり、結果は今回の条件に対して解釈します。

## Windowsでの確認結果

| データ | 両フレームワークのmacro F1（学習seed 42、43、44） | 学習後のラベル不一致 | 学習後の確率の最大差 |
| --- | --- | --- | --- |
| Iris（テスト30行） | 0.8997、0.9333、0.9666 | 0件 | 0.03053 |
| Wine（テスト36行） | 0.9710、0.9710、0.9710 | 0件 | 0.00211 |

共通の重みを使った予測の最大確率差は、両データ・全seedを通して約 (6.0\times10^{-8}) でした。独立に学習したモデルのラベルも各seedで一致し、固定予測に条件付けたpaired bootstrapでは各反復でF1差が0でした。Irisの花弁幅に+0.5 cmを仮定すると、両フレームワークで同じ6〜7件のラベルが変わり、F1はseedによって0.166〜0.232低下しました。
