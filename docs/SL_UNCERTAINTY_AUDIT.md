# 分類結果の不確かさを点検する

`rank-sl` は、順位表とともにCVで選定したモデルの保留テストを点検します。通常のテスト評価で使う学習済みモデルを、Float32での入力丸めと、指定した校正ずれの感度でも共有します。校正ずれを試すには `--shift` を指定します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl rank-sl --csv data/reference/iris.csv --target target --shift "petal width (cm)=-0.2" --shift "petal width (cm)=0.2" --output runs/iris-rank-for-audit
.\.venv\Scripts\python.exe -X utf8 scripts/validate_sl_ranking.py runs/iris-rank-for-audit
```

保存済みの順位を使って別の校正ずれを試すには、`audit-uncertainty` を使います。元のCSVと学習条件を使い、保存済みのテスト予測を再現した上で点検します。出力先には新しいフォルダーを指定します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl audit-uncertainty --run runs/iris-rank-for-audit --csv data/reference/iris.csv --shift "petal width (cm)=0.5" --output runs/iris-uncertainty-audit
.\.venv\Scripts\python.exe -X utf8 scripts/validate_sl_uncertainty_audit.py runs/iris-uncertainty-audit
```

| シート | 読み方 |
| --- | --- |
| `Test_Intervals` / `Statistical` | 固定モデルのAccuracy・macro Precision・Recall・F1の点、bootstrap標準誤差、区間。独立した点検でも `rank-sl` と同じ正解クラス別件数と再標本化行を使います。 |
| `Test_Confusion` / `Test_By_Class` | `rank-sl` の最終テストで、混同行列は行が正解、列が予測です。クラス別表は正解としての件数（Support）、予測件数、TP・FP・FN、Precision・Recall・F1を示します。予測件数が0のクラスのPrecisionは0とします。 |
| `Numerical_Metric` | 混同行列の整数から有理数で求めた厳密値と、Float64で計算した指標の差。 |
| `Numerical_Input` / `Input_Roundoff` | テスト特徴量をFloat32で丸め、Float64に戻して同じモデルで予測したときの差。列ごとの丸め幅は元の列の単位で表示します。モデル自体の演算精度は固定です。 |
| `Offset_Scenarios` | `--shift` で指定した列のテスト値すべてに一定のずれを加えた場合の予測変化と、指標の差。差の区間は同じテスト行を一緒に再抽出した条件付きbootstrap区間です。 |

実際の系統的な偏りを推定するには、校正記録、基準に沿った再測定、独立したラベルの再判定などが必要です。この出力では実際の偏りを未測定として記録し、利用者が指定したずれに対する感度を示します。推定の対象と前提が異なるため、3つの結果は個別に読みます。 [測定の系統的な効果と不確かさに関するJCGMガイド](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf/cb0ef43f-baa5-11cf-3f85-4dcd86f77bd6?version=1.7)

`arrays.npz` に元の予測とシナリオごとの予測、bootstrapの行番号とスコアを保存します。検証器は保存された配列から区間・点数・数値誤差を再計算します。
