# 次の作業

1. 専用ブランチに保存した再実装範囲と検証結果を利用者が確認し、別環境から取得して継続する。
2. DL反復入口は0.4.0a2で実装済み。固定index・train-only scalerを共有し、全seedの設定・状態・指標・履歴・重みを保存する。最高test seedは選ばない。
3. ULの非復元部分抽出による安定性は0.4.0a3で実装済み。操作はdocs/UL_SUBSAMPLE_GUIDE.md、検証はdocs/VALIDATION_UL_SUBSAMPLE.mdを参照。bootstrapなど他方式は未実装。
4. ハイパーパラメータ探索、画像埋込、Web画面、Windows実機検証は独立した追加機能として扱う。

旧資料のSTEP7最優先や時間によるタイブレーク案は、その後の引継ぎ仕様より優先しません。
