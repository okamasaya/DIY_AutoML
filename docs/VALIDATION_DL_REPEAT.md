# DL反復の検証（0.4.0a2）

元資料からの再実装に、固定分割での学習seed反復を追加した検証です。旧環境の0.4ソースや96件のテストを復元した結果ではありません。初回再実装版0.4.0a1.post1の記録は上書きせず保持しています。

## 確認した結果

- Python 3.12.14 / Linux CPU、TensorFlow CPU 2.20.0 / Keras 3.15.1。
- 開発環境45 passed、独立wheel環境45 passed。両方とも失敗・error・skip 0、pip check合格。
- Iris/Wine/Breast Cancer × 各10学習seed（42..51）×100epochを2環境で実行。計60学習・6,000epoch。
- 分割seed42、batch16、MLP32-16-8、Adam .001。全データは開発用で、外部検証データではない。
- 対応する全保存配列（index/scaler/target/予測/重み）と、時間以外の6シート全表が環境間で完全一致。
- 各データの既定件数はDL_By_Seed 10、DL_Seed_Summary 8、DL_History 1,000、DL_Settings 10、DL_Splitsはデータ件数、Summary 10。
- seed42の分割・scaler・予測・全重み・100epoch履歴は、初回再実装0.4.0a1.post1の保存済み単回DL結果と3データすべてで完全一致。
- 既存schema2/3/4の保存結果10件も現在の検証器で合格。

## 追加テスト

同じ準備済みデータを全seedで共有し、scalerをtrainだけで1回fitすることを検証しました。単回と反復の一致、異なる学習seedでの重みの変化、不正設定の拒否も確認しています。

最初・途中のseedに例外を注入し、completed/failed/not_runと部分履歴・重みの保存を検証しました。実TensorFlowの最初のepoch終了後に例外を注入するテストでも、終了済みepochと重みが取得できました。実行中のcheckpointはrunningとして保存します。

検証器はファイルチェックサムに加え、seed順、分割の重複/欠落、予測とtest指標、履歴の連続性と最終指標、完了seedだけの集計、設定の対応を確認します。チェックサムを更新した改変出力でも、予測・分割・集計・seed順の不整合を検出するテストを含めています。

## 記録と再実行

詳細は `tests/results/0.4.0a2/validation.json`、同ディレクトリのJUnit/XML、`validation/dl-repeat-0.4.0a2/` のExcel/NPZ/manifestにあります。

```sh
.venv/bin/python scripts/verify_dl_repeat.py --output /新規保存先/development
別wheel環境/bin/python scripts/verify_dl_repeat.py --output /新規保存先/wheel --compare /新規保存先/development
```

別wheel環境には、requirements-local.lock.txtの依存とdistでビルドした0.4.0a2 wheelを導入します。今回の独立環境はsystem-site-packagesを使用せず、ソースをPYTHONPATHへ追加せずに実行しました。

単一ハードウェアでの一致確認であり、他OS/GPU/ライブラリ版での完全一致を保証するものではありません。反復の平均・SDは固定データ・固定分割に条件付けた学習seed感度です。最高test seedの選定、データ再標本化、分割変動、学習再開機能は含みません。
