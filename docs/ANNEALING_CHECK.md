# CPU版Annealing check

`annealing-check` は、固定した分類器の特徴量選択を必要なときに実行する。既定の分類器はSVM、選定指標はmacro F1、探索予算は各手法32候補。既存のコア環境で動作する。

## 評価手順

1. 約20%を層化して保留テストに確保する。残りの行で共通の層化5分割を使い、既定は2反復とする。候補の標準化は各foldの学習行で行う。
2. 全列ベースライン、Simulated Annealing（SA）、非復元の一様ランダム探索を評価する。`--max-features` は探索候補の列数上限。非空部分集合を候補とし、ランダム探索は許容された各部分集合を等確率で抽出する。SAとランダム探索は同じ探索seedから同じ初期候補を得る。
3. SAは1列の追加・削除または2列の入れ替えで近傍を作る。未評価の近傍から候補を選び、スコアが下がる候補は `exp((新スコア − 現スコア) / 温度)` の確率で移動先にする。既定の温度は0.03から0.002まで幾何的に下げる。未評価の近傍が尽きたときは未評価部分集合から再開する。 [Simulated Annealingの原論文](https://doi.org/10.1126/science.220.4598.671)
4. 各手法でCV平均スコアが最も高い候補を選ぶ。スコアを小数12桁へ丸めて同率の場合は、少ない列数、元の特徴量順で付けたビットマスクの小さい順で決める。各探索の予算は評価した異なる候補数で数える。候補1件あたりの学習回数は `5 × cv_repeats`。
5. 候補を確定してから学習側全体で各手法のモデルを1回ずつ学習し、同じ保留テストで比較する。差の95%区間は、固定した予測と観測されたクラス別件数に条件を置き、テスト行を両手法共通で層化再標本化する。学習や探索のやり直しで生じる変動は別に扱う。

`--budget` は各探索128候補まで。候補空間が予算より小さければ、実際に評価できる個数を使う。`--exhaustive` は255候補までの小さな空間に全探索を追加する。1回の実行あたりCV学習の上限は4,000回で、全列モデルと全探索の分も含む。

## 再実行とレポート

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl annealing-check --csv data/reference/iris.csv --target target --budget 8 --exhaustive --shift 'petal width (cm)=0.5' --output runs/annealing-iris
.\.venv\Scripts\python.exe -X utf8 -m diy_automl annealing-check --csv data/reference/wine.csv --target target --budget 32 --max-features 6 --shift 'alcohol=0.5' --output runs/annealing-wine
.\.venv\Scripts\python.exe -X utf8 scripts/validate_annealing_check.py runs/annealing-iris
```

Pythonからは `annealing_check_csv(csv, target, output, budget=8, exhaustive=True)` を呼ぶ。探索seedを変える場合は `--search-seed` と実行ごとに新しい出力先を指定する。

| 出力 | 確認できる情報 |
| --- | --- |
| `Selection`、`Selected_Features` | CV平均、全列との差、選んだ列、評価候補数、CV学習回数。CV時間には予測も含む。 |
| `Search_History` | CSVの特徴量順のマスク、提案、温度、受理、現在と最良の手順。 |
| `Test_Score`、`Test_Differences` | 保留行の点推定と再学習時間、全列およびSAとランダム探索のペア差。 |
| `Test_Intervals`、`Test_Confusion`、`Test_By_Class` | 手法ごとの区間、混同行列、クラス別のPrecision・Recall・F1。 |
| `Numerical_Metric`、`Numerical_Input`、`Offset_Scenarios` | 有理数とFloat64のスコア差、Float32入力丸めでの変化、指定した仮の入力ずれの感度。除外した列へのずれは `Feature Used` が偽になる。 |
| `arrays.npz`、`manifest.json` | 分割、各候補のCV予測、テスト予測、再標本化、設定、SHA256。保存結果から学習をせずに探索とレポートを照合できる。 |

## IrisとWineでの検証

WindowsのPython 3.13.17で、SVM、分割seed 42、共通5分割の2反復、探索seed 44〜53で実行した。保留行はIris 30件、Wine 36件で、探索seedを変えても同じ行を使う。この10回の結果は固定分割での探索のばらつきとして読む。

| データ | 探索予算と結果 |
| --- | --- |
| Iris | 全15候補を列挙し、CV最良は全4列のF1 **0.9707**。各探索8候補・80学習では、SAが3/10回、ランダム探索が8/10回この最良値に達した。seed 44のSAは花弁幅1列で、全列比の保留F1差は **−0.0333**（95%区間 **−0.1023〜0.0000**）。 |
| Wine | 13列から最大6列を選び、各探索32候補・320学習。CVではSAとランダム探索がそれぞれ5/10回上回った。seed 44のSAは5列で、全13列と保留行の予測が一致した。SA − ランダム探索のF1差は **+0.0558**（95%区間 **0.0000〜+0.1419**）。同じ条件の10 seedのうち、全列と予測が一致したSAは1回。 |

両データの全seedでSA − ランダム探索の95%区間が0を含んだ。各探索で選んだ列と全候補のCVスコアを、保存配列からscikit-learnで再集計した。

seed 44では全手法でFloat32入力丸めによるラベルの変化が0件だった。Irisで花弁幅に+0.5 cmを仮定すると、SAは8/30件、全列は6/30件の予測が変わった。Wineの同seedでSAが除外した `alcohol` への+0.5のずれは、SAから見える入力を変えない。実際の校正ずれは未測定。

検証出力はローカルの `runs/annealing-iris-s44`〜`runs/annealing-iris-s53` と `runs/annealing-wine-s44`〜`runs/annealing-wine-s53` にある。
