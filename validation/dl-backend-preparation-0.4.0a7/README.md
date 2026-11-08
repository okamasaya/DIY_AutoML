# 0.4.0a7のTensorFlow変更前後の照合

iris・wine・breast_cancer、固定分割seed 42、学習seed 42/43、各100epochの出力です。各データについて数値結果、`-figures`の図付きExcel、`-uncertainty`の固定モデル誤差評価を保存しています。

予測・重み・履歴を過去の0.4.0a2の該当seedと照合し、独立wheel仮想環境でも同じ配列・時間以外の表・PNGを確認しました。過去の10seed検証を置き換えるものではありません。全seedを宣言順で保持し、test値でseedを選んでいません。

manifestの絶対パスは実行時点の記録です。保存結果は検証用の配列・表を含み、各検証器で照合できます。PyTorch/JAXの学習結果ではありません。

詳細は[検証記録](../../docs/VALIDATION_DL_BACKEND_PREPARATION.md)、[機械可読な記録](../../tests/results/0.4.0a7/verification.json)です。
