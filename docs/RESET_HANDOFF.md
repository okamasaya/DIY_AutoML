# 再開時の状態

## 原本と再実装

元ZIP `ML_Framework_Handoff_20261107.zip` のSHA256は `12a8836bf17aafee0f52886d7162d36bd9e1d135c744f02d467fd6c1b62b4c48` と一致しました。補助MD2点も取得・保全しました。ZIP内Notebook・CSV・Excel・資料は改修前の記録です。ZIP自体を変更せず保存しています。

旧ソースZIP・復旧用ZIP・tar.gz・wheel・旧テスト結果JSON・記事原稿は未取得です。旧環境IDからファイルが移るとは扱いません。作業開始時のGitはworkブランチ、HEADは `b4f6f17 Create .gitkeep` のみでした。元の0.4ソースや96件のテストを復元したわけではありません。

今回のコードは引継ぎにある評価手順から新規実装しました。初回版名は0.4.0a1.post1、DL反復は0.4.0a2、UL部分抽出は0.4.0a3、SL設定探索を追加した現在版は0.4.0a4です。旧版との完全一致や85ファイルの復旧を主張しません。manifestのキー、シート名の詳細、関数引数も新規設計です。

## 実装済み

- CSV検証、分割後のSL/DL標準化、SL5モデル、UL54候補、MLP32-16-8。
- 学習側共通5fold選定、候補順の同率処理、単一test評価、条件付き層化paired bootstrap。
- 固定データのUL反復と階層参考値、全割当・scaler・設定・収束診断・失敗状態保存。
- DLの分割seedと学習seedの独立指定。全分割index・重み・予測・履歴保存。
- CLI、Excel/NPZ/manifest、保存結果検証器、ローカルNotebookと図5種類。
- 0.4.0a2: 固定分割とscalerを共有するrepeat-dl、全seedの状態・指標・履歴・予測・重み、途中失敗の部分結果保存。
- 0.4.0a3: 非復元のsubsample-ul、部分集合ごとの標準化、同じ元行に制限した参照/反復間ARI、index・scaler・割当・失敗状態の保存。

- 0.4.0a4: 学習側CVだけで27設定を選定するtune、部分結果・全fold予測と設定保存、検証器、TensorFlowなしのコア用ロック。
- 新規Zenn紹介記事1本と図を作成。published=falseで保存し、Zennへの投稿は未実施。過去5本の原稿を復元したものではない。

## 未復元・未完了

旧制作物のバイト一致確認、旧wheel比較、旧記事5本は未復元です。bootstrapなど他の再標本化方式によるUL安定性、任意グリッドやDLを含む探索、Excel画像埋込、Django、Windows/exe、ライセンス選定は未完了です。

2026年11月8日（日本時間）、利用者からcommit・pushの実行承認を受け、専用ブランチ `recovery/reimplemented-0.4.0a1-post1` に保存する方針へ変更しました。開始時のmainブランチには変更を加えません。commit・pushの完了はGit履歴とリモート参照の一致で確認してください。再開手順は `docs/COMMIT_GUIDE.md` に記載しています。ZIPの生成・ローカル検証と、利用者がダウンロードできた事実は別に扱います。
