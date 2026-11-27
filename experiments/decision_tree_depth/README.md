# 決定木の深さと学習・検証の差

同じ分割で `max_depth` だけを変え、学習側と検証側のmacro F1を比べる。分類用CSVの読み込み、層化分割、前処理、指標、保存形式にはDIY AutoMLの既存処理を使う。分割seedと決定木のseedはともに既定の42。1分割で4本の木を学習し、数値計算は1スレッドで実行する。

## 再実行

リポジトリのルートから実行し、新しい `--output` を指定する。

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m experiments.decision_tree_depth.run `
  --csv data/reference/wine.csv --target target --output runs/decision-tree-depth-new/wine
```

`report.xlsx` の `Depth_Comparison` に深さの上限、実際の深さ、学習・検証のF1、F1差、学習時間、上限なしとの差を保存する。`arrays.npz` に分割の行番号と学習・検証の予測、`manifest.json` に設定、入力・コード・出力のハッシュを記録する。既定では保留分割の行番号だけを保存する。`--score-holdout` を付けると、検証で上限を選んだ後に、事前に指定した4候補を保留分割で採点する。

## Wineでの初回結果

2026年11月26日。学習113件、検証29件、保留36件。検証macro F1が同点なら、上限が最も小さい候補を選ぶ。

| 深さの上限 | 実際の深さ | 学習macro F1 | 検証macro F1 |
| --- | ---: | ---: | ---: |
| 1 | 1 | 0.5218 | 0.5107 |
| 3 | 3 | 1.0000 | 0.9039 |
| 5 | 3 | 1.0000 | 0.9039 |
| 上限なし | 3 | 1.0000 | 0.9039 |

選定は上限3。上限5と上限なしでも実際の木は深さ3で、深さ3以上の候補の予測は一致した。学習と検証のF1に約0.096の差が見られた。深さを増やしたときに検証スコアが下がるかを調べるには、実際の木の深さが変わる条件で確認する。

今回の出力は `runs/decision-tree-depth-20261126/wine`。保存した予測からscikit-learnで両方のmacro F1を再計算し、Excel、分割、入力と出力のハッシュを照合した。このWineの分割は開発用の確認として扱う。 [関連する検証計画](../../docs/MODEL_VALIDATION_TODO.md#モデルの深さ反復閾値と過学習の検証計画)

## Raisinで層化分割を10回変えた結果

Wineでは深さが3で止まったため、900件のRaisinで同じ4候補を比較した。分割seed 42〜51を使い、決定木のseedは42に固定した。各分割は学習576件、検証144件、保留180件。各木を学習・検証した後に上限を確定し、各候補を同じ保留行で採点した。 [データの出典](../../data/reference/README.md#出典と変換)

| 深さの上限 | 実際の深さ | 学習macro F1平均 | 検証macro F1平均 | 保留macro F1平均 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.8636 | 0.8609 | 0.8541 |
| 3 | 3 | 0.8836 | 0.8545 | 0.8569 |
| 5 | 5 | 0.9155 | 0.8456 | 0.8514 |
| 上限なし | 13〜26 | 1.0000 | 0.8200 | 0.8056 |

検証で選ばれた上限は1が5分割、3が4分割、5が1分割。選定した木の保留macro F1は平均0.8536で、上限なしの0.8056を10分割すべてで上回った。差の平均は+0.0480。分割間で元の観測が重なるため、勝敗数と平均差はこのデータ内の記述的な比較として読む。

```powershell
foreach ($seed in 42..51) {
  & .\.venv\Scripts\python.exe -X utf8 -m experiments.decision_tree_depth.run `
    --csv data/reference/raisin.csv --target target --split-seed $seed --score-holdout `
    --output "runs/decision-tree-depth-new/raisin/seed-$seed"
  if ($LASTEXITCODE -ne 0) { throw "Depth comparison failed for seed $seed" }
}
```

今回の10分割は `runs/decision-tree-depth-20261126/raisin`。`summary.json` に分割別の選定上限と両方式の保留F1を集計した。全候補の学習・検証・保留F1を保存配列から再計算した。
