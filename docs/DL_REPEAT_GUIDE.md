# 固定分割でのDL学習seed反復（0.4.0a2）

`repeat-dl` は同じCSV・同じtrain/validation/test分割・同じ標準化を用いて、MLPの学習seedだけを変えます。分割seedは42、学習seedは42..51の10件、100epoch、batch16が既定です。モデルは32-16-8/ReLU/softmax、Adam(learning_rate=.001)、sparse categorical crossentropyです。既存の単回 `run` も同じ学習関数を使います。

```sh
.venv/bin/python -m diy_automl repeat-dl --csv data/reference/wine.csv --target target --training-seeds 42 43 44 --split-seed 42 --epochs 100 --output runs/wine-repeat-dl
.venv/bin/python scripts/validate_dl_repeat.py runs/wine-repeat-dl
```

Python入口は `diy_automl.repeat_deep_learning_csv(csv, target, output, training_seeds=range(42,52), split_seed=42, epochs=100, batch_size=16)` です。2件以上の重複しない非負の整数seedが必要です。出力先は新規フォルダだけを受け付けます。数値特徴量・欠損なし・分類ラベル付きCSVという既存の制約を引き継ぎます。

## 反復で固定するものと変えるもの

分割は一度だけ作り、scalerもDLのtrainだけで一度だけfitします。全seedへ同じ読取り専用の分割・標準化済み配列を渡します。validation/testはscalerの学習に使いません。概ねtrain64%/validation16%/test20%です。

学習seedは重み初期化と、epochごとの学習データshuffleを同時に変えます。そのため測定対象は「初期重みだけの感度」ではなく「学習seed感度」です。データの再標本化、分割の変更、設定探索は含みません。

全seedについてtest指標を報告しますが、最高test値のseedや重みを選ぶ処理はありません。選定・チューニング用の入口ではなく、固定データと分割に条件付けた記述的な感度分析です。mean/SD/min/maxを計算し、SDはddof=1です。完了seedが1件以下の場合のSDは空欄です。独立test標本やデータセット間の不確かさを表すCI/SEMは作りません。

## schema 5の6シート

| シート | 内容 | 既定10seed・100epochの行数 |
| --- | --- | ---: |
| DL_By_Seed | 指定順のseed、状態、test4指標、最終train/validation指標、時間、エラー | 10 |
| DL_Seed_Summary | 完了seedだけの8指標のcount/mean/SD/min/max | 8 |
| DL_History | seedとepochに対応するtrain/validation accuracy/loss | 1,000 |
| DL_Settings | 各seedに対応するモデル・optimizer・shuffle・分割設定 | 10 |
| DL_Splits | 元CSVのデータ行index（0始まり）、partition、符号化target | 入力行数 |
| Summary | 目的、変動範囲、失敗時処理、解釈上の注意 | 10 |

`arrays.npz` には共通train/validation/test index、scaler mean/scale、全target、test正解を保存します。各完了seedの `seed_<seed>_predictions` と `seed_<seed>_weight_0` .. `weight_7` を保存します。失敗途中で取得できた重みは `partial_weight_` とし、完了済み重みと区別します。NPZの重みだけでoptimizer状態まで復元する機能はありません。学習再開用のKeras checkpointではありません。

manifestには全seed・モデル設定・状態、入力/ソース/出力ハッシュ、TensorFlow/Kerasを含む依存版、実行環境を記録します。

0.4.0a7では、DLのmanifestにdl_backend=tensorflowとdl_result_contract=dense-32-16-8-v1を追加しました。共通のデータ/結果検査とTensorFlow学習処理を分離し、旧manifestも検証できます。現在の学習実装はTensorFlowのみです。PyTorch追加の準備と重みの保存形式は[DL_BACKENDS.md](DL_BACKENDS.md)を参照してください。

## 失敗と途中保存

開始前と各seedの開始・終了時に結果を保存します。実行中はmanifestをrunning、開始したseedもrunningにします。最初の例外で全体をfailedにし、完了済みseedはcompleted、失敗seedはfailed、それ以降はnot_runです。完了seedの履歴・予測・重みと、失敗seedで取得できたepoch履歴・重みを残します。失敗・未実行seedのtest指標は空欄にし、集計へ混ぜません。

```sh
.venv/bin/python scripts/validate_dl_repeat.py runs/failed-repeat --allow-failed
```

通常の検証器はfailedを合格にしません。`--allow-failed` は失敗が解消した意味ではなく、部分結果の対応関係が正しいかを検査します。入力読込や分割前の失敗はmanifestだけになる場合があります。

プロセスの強制終了・環境消失時は最後の保存状態がrunningのまま残る場合があり、完了扱いにはしません。ストレージ障害や保存中の強制終了に対する完全な復旧・自動再開は未実装です。チェックサムと保存結果検証器で不完全な出力を識別します。別ハードウェアやTensorFlow版でのビット一致は保証しません。
