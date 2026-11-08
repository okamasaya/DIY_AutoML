# DLバックエンド追加の準備：検証記録（0.4.0a7）

PyTorchの追加に備え、共通データ・結果の検査とTensorFlow固有の学習処理を分離しました。実際の学習実装はTensorFlowだけです。設計範囲は[DL_BACKENDS.md](DL_BACKENDS.md)、機械可読な結果は[verification.json](../tests/results/0.4.0a7/verification.json)です。

## 実行環境とテスト

Python 3.12.14、Linux CPU、TensorFlow CPU 2.20.0/Keras 3.15.1。同じマシン上で、開発環境とソースを直接参照しないwheel仮想環境を使いました。両方で **200 passed、failure/error/skip 0**、開発・wheel・TensorFlowなしのコア環境でpip checkが通りました。

追加した検査では、分割・scaler・予測・指標・seed・重みの向き/有限値・履歴の不整合を拒否します。共有配列の意図しない変更は、学習成功時と例外時の両方で検出します。過去のimport経路、metadataなしの保存結果、途中失敗の既存処理も確認します。共通処理のimport/データ準備でTensorFlow/Keras/PyTorch/JAXを読み込まないことを検査しています。

## 学習結果と保存結果の照合

正式照合はiris・wine・breast_cancerの3データ、学習seed 42/43、各100epochを両環境で実行した12 fitです。分割seedは42です。

- 0.4.0a2の保存済み10seed結果から該当2seedを取り、分割・scaler・予測・全重み配列を完全一致で照合しました。履歴と設定の値も一致しました。過去のExcelとの比較では、全値が整数になる列の読込型の違いだけを許容しています。
- 開発環境とwheelの6数値出力、6図付きExcel、6誤差評価出力で、全配列・時間以外の表・PNGが一致しました。Excel内の埋込PNGは合計18枚で、各検証器が確認しています。
- TensorFlow学習関数の本文と、共通prepare_data/runの本文は移動前と同一です。関数本文のハッシュとwheel内の全Pythonファイルの照合結果もJSONに記録しました。
- TensorFlow/Keras/PyTorch/JAXのないコア環境で、3つの保存済みDLを検証し、wineの図付きExcelと誤差評価を生成・検証しました。誤差評価の配列とPNGは開発環境と一致しました。

開発環境の出力は[validation/dl-backend-preparation-0.4.0a7](../validation/dl-backend-preparation-0.4.0a7)に保存しています。過去の原本・0.4.0a2の学習結果・0.4.0a6の検証記録は変更していません。新しい学習seedの選定や、test値によるbackend比較は行っていません。

## 再実行

依存を導入したリポジトリのルートから、未使用の出力先を指定します。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_dl_backend_preparation.py --output /tmp/diy-automl-a7-check
```

wheel環境では、ソースのpythonpathを外して`pytest -o pythonpath=`を使いました。照合スクリプトの`--compare`は別環境の出力先との比較用、`--reuse-existing`は同一ソースハッシュの検証済み出力を再利用する検証作業用です。学習途中からの再開機能ではありません。assertを使うためpython -Oは使用しません。

この確認は同じLinux CPU条件でのTensorFlowの変更前後の照合です。異なるframework・GPU・OS・依存版の一致や性能優位性を示すものではありません。PyTorch/JAXの実装・依存追加、Julia/Rust版の制作、PyPI/Zenn公開は今回の範囲に含めていません。
