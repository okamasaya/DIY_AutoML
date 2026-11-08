# DLバックエンド追加の準備（0.4.0a7）

PyTorchの追加に備え、データの準備・結果の検査と、TensorFlow固有の学習処理を分離しました。現在実装している学習バックエンドはTensorFlowだけです。PyTorch/JAXの学習実装、選択用CLI、追加依存はまだありません。

## 現在の境界

| 場所 | 役割 |
| --- | --- |
| deep_learning.py | 既存のprepare_data / fit_seed / run入口。共通準備とTensorFlowへの接続 |
| dl_contract.py | PreparedDL、失敗時の型、MLPの結果形式と整合性検査。学習frameworkをimportしない |
| tensorflow_backend.py | TensorFlow/KerasのMLP学習、tf.data、履歴・重みの取得。TensorFlowのimportは学習時だけ |
| reporting.py | DLのmanifestにdl_backend=tensorflow、dl_result_contract=dense-32-16-8-v1を記録 |

deep_learning.PreparedDL、TrainingFailure、HISTORY_COLUMNS、MODEL_CONFIG、integerのimport経路を維持します。既存のAPI/CLI、分割・scaler・MLP・学習seed・shuffle・Adam・学習回数は変えず、TensorFlow学習関数の本体は移動前と同じです。単回とrepeat-dlは同じ境界を使います。

schema 2/5の数値表・保存配列の形式は維持します。新しいmetadataがない過去のmanifestも検証できます。新しいmetadataで未対応のbackendを指定したschema 5は、現在の検証器が拒否します。

## 結果形式と検査

今回の形式は参照MLP32-16-8用で、任意のネットワークに対応する汎用形式ではありません。学習処理は `(result_dataframe, history_dataframe, arrays_dictionary)` を返し、共通側で次を検査します。

- 1行の結果と、モデル・層・split/training seed・epoch・batchの対応。
- 指定epoch分の連続した履歴と、accuracy/loss/val_accuracy/val_lossの有限値。
- 共通準備と同じtrain/validation/test index、scaler mean/scale。
- 学習前後の共通準備データ・正解・index・scaler・書込可否の一致。array共有を通じた意図しない変更も検出する。
- 整数クラスのtest予測と、正解・予測から再計算した指標との一致。
- 8つの重み配列の形状と、有限の浮動小数点値。

保存するDense kernelは `(入力次元, 出力次元)`、biasは `(出力次元,)` です。将来のPyTorch実装はネイティブのLinear weightの向き等を明示的に変換します。向きの変換で学習結果まで同一になる意味ではありません。optimizer状態を含む学習再開の形式でもありません。

不整合は停止させ、失敗した出力を採点済みとしません。学習途中のTrainingFailureと部分履歴・重みは既存の保存処理を使います。

## 次にPyTorchを実装するとき

1. まず同じMLPをCPUの任意依存として実装し、大きなモデルやGPU実験と分ける。
2. 共通PreparedDLのindex・標準化済みデータを使い、学習側だけで学習する。
3. 初期化・データ順・乱数生成器・Adamの設定（epsilon等）・dtype・停止条件・依存版と環境を記録する。
4. 正解・予測・履歴・重みを共通形式へ接続し、backend名・結果形式・失敗状態を記録。保存結果検証器も実装に合わせて拡張する。
5. 同じbackendでの再現確認と、異なるbackend間の比較を分ける。同じseed番号だけで初期重みやshuffleが一致するとは扱わず、異なる実装間のビット一致を採用条件にしない。
6. 誤差評価・Excel出力へ接続し、testでbackendやseedを選び直さない。

PyTorchも版・プラットフォーム間で完全な再現を保証していません。[公式の再現性資料](https://docs.pytorch.org/docs/stable/notes/randomness.html)。現在のTensorFlow採用は元Notebook/引継ぎから再実装・検証した経緯で、両者の性能比較で優位性を結論したものではありません。

## JAX・Julia・Rust

| 候補 | 考えている範囲 | 状態 |
| --- | --- | --- |
| JAX | 自動微分・JIT等を使うPython内の別実装 | 利用者は未経験。関心のある将来候補で、プロジェクトでは未実装・未検証 |
| Julia | MLJ等も参考にした別言語版の可能性 | 将来展望。移植・連携・配布方法は未定 |
| Rust | 独立したRust版、またはPython APIを保つ部分的なRust実装 | 将来展望。2案を区別し、実測した必要性から範囲を決める |

参考は[JAX](https://docs.jax.dev/en/latest/)、[JuliaのMLJ](https://juliaai.github.io/MLJ.jl/stable/)、[Python拡張をRustで作るPyO3](https://pyo3.rs/)の公式資料です。特定の言語・frameworkに変えるだけで精度・速度・誤差評価が改善したとはしません。日程や対応版の約束、Julia/Rust版の制作開始、PyPI/Zennへの公開を行ったものではありません。
