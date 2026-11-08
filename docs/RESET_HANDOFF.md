# 再開時の状態

## 原本と再実装

元ZIP `ML_Framework_Handoff_20261107.zip` のSHA256は `12a8836bf17aafee0f52886d7162d36bd9e1d135c744f02d467fd6c1b62b4c48` と一致しました。補助MD2点も取得・保全しました。ZIP内Notebook・CSV・Excel・資料は改修前の記録です。ZIP自体を変更せず保存しています。

旧ソースZIP・復旧用ZIP・tar.gz・wheel・旧テスト結果JSON・記事原稿は未取得です。旧環境IDからファイルが移るとは扱いません。作業開始時のGitはworkブランチ、HEADは `b4f6f17 Create .gitkeep` のみでした。元の0.4ソースや96件のテストを復元したわけではありません。

今回のコードは引継ぎにある評価手順から新規実装しました。初回版名は0.4.0a1.post1、DL反復は0.4.0a2、UL部分抽出は0.4.0a3、SL設定探索は0.4.0a4、図付きExcelは0.4.0a5、評価値の誤差評価は0.4.0a6、DLバックエンド追加の準備を整えた現在版は0.4.0a7です。旧版との完全一致や85ファイルの復旧を主張しません。manifestのキー、シート名の詳細、関数引数も新規設計です。

## 実装済み

現在の版は0.4.0a7です。PyTorch追加に備え、共通のデータ準備・結果検査（dl_contract.py）と、TensorFlow学習処理（tensorflow_backend.py）を分離しました。既存API/CLIとMLP学習条件は維持し、DLのbackendと結果形式をmanifestへ記録します。PyTorchの学習実装・選択CLI・追加依存は未追加です。JAX・Julia・RustはDS記事の将来展望です。[DL_BACKENDS.md](DL_BACKENDS.md)を参照してください。

今回の検証は開発・独立wheelで各200 passed、失敗/error/skip 0。3データの学習seed 42/43・100epochを保存済み結果と照合し、図付きExcel・誤差評価まで確認しました。[検証記録](VALIDATION_DL_BACKEND_PREPARATION.md)とtests/results/0.4.0a7が根拠です。過去のa6記録は変更せず保持しています。

現在の完了/一部完了/未実装と、次の推奨順の一覧は[ROADMAP.md](ROADMAP.md)です。以下は実装の経過も含む記録です。誤差評価の次の1区切りはデータ点検と評価条件の記録を推奨し、この点検機能は未実装です。

- CSV検証、分割後のSL/DL標準化、SL5モデル、UL54候補、MLP32-16-8。
- 学習側共通5fold選定、候補順の同率処理、単一test評価、条件付き層化paired bootstrap。
- 固定データのUL反復と階層参考値、全割当・scaler・設定・収束診断・失敗状態保存。
- DLの分割seedと学習seedの独立指定。全分割index・重み・予測・履歴保存。
- CLI、Excel/NPZ/manifest、保存結果検証器、ローカルNotebookと図5種類。
- 0.4.0a2: 固定分割とscalerを共有するrepeat-dl、全seedの状態・指標・履歴・予測・重み、途中失敗の部分結果保存。
- 0.4.0a3: 非復元のsubsample-ul、部分集合ごとの標準化、同じ元行に制限した参照/反復間ARI、index・scaler・割当・失敗状態の保存。

- 0.4.0a4: 学習側CVだけで27設定を選定するtune、部分結果・全fold予測と設定保存、検証器、TensorFlowなしのコア用ロック。
- 新規Zenn紹介記事1本と図を作成。published=falseで保存し、Zennへの投稿は未実施。過去5本の原稿を復元したものではない。

- 0.4.0a5: completedの保存結果6入口から図付きExcelを別ファイルへ生成。数値セル・画像入力・埋込PNG・配置を検証。Notebookカーネルで全セル実行を確認。
- 親記事の温度感は利用者確認済み。DS向け（Notebook/CLI）とビジネス向け（アプリでCSV投入→Excel受取）の子記事へ分ける方針を記録。その後、子記事2本も下書きを作成。画面アプリ・操作手順は未実装、3本ともZenn未公開。
- 追加方針: DS向けは任意の広いモデル比較、ビジネス向けは扱いやすい業務データと主要モデルを中心にする。追加モデル・特に計算負荷の高い検証はdocs/MODEL_VALIDATION_TODO.mdで保留。親記事にモデルの改善幅に関する参考文献を追加し、対象版を0.4.0a5へ更新。モデル追加や業務データ検証を完了した意味ではない。
- その後の追加依頼: ホテル・カフェ券・再購入を既存SLで小規模実行。親記事に要約とビジネス向け実験解説下書きを追加。docs/BUSINESS_QUICK_RESULTS.mdとvalidation/business-quick-20261108に条件・実測・限界を保存。汎用前処理や新モデルを本体へ追加したものではない。
- 利用者が示したDIYの主目的は誤差評価を含めたデータ量・品質の議論。docs/UNCERTAINTY_DESIGN.mdに現状と次の設計を記録。区間幅だけで原因は識別できず、学習曲線・再学習/選定の変動・評価単位を優先する方針で、これらの新機能は未実装。
- 誤差評価の範囲を統計・計算・系統的な偏りまで整理。Zennは親・ビジネス実験・DS誤差評価の3下書き（すべてpublished=false）。ビジネス向けは平易な説明と持ち込みデータとの対比、DS向けは推定対象・数式・再標本化単位・保存配列まで扱う。簡易版は計算量と説明を絞る方針で、アプリや追加診断を実装したものではない。
- 0.4.0a6: `uncertainty`（schema 9）で保存済みrun/select/tune/repeat-dlを再学習せず誤差評価。4指標とクラス別指標のSE・percentile区間、全再標本化index/反復値、図付きExcel、検証器。クラス件数固定/通常の行bootstrap。手順はdocs/METRIC_UNCERTAINTY_GUIDE.md。系統/数値誤差や学習手続き全体の変動は未実装。

## 未復元・未完了

旧制作物のバイト一致確認、旧wheel比較、旧記事5本は未復元です。bootstrapなど他の再標本化方式によるUL安定性、任意グリッドやDLを含む探索、Django、Windows/exe、ライセンス選定は未完了です。

2026年11月8日（日本時間）、利用者からcommit・pushの実行承認を受け、まず専用ブランチ `recovery/reimplemented-0.4.0a1-post1` に保存しました。その後の「今のrecoverをメインに移動しても良いかな」という依頼により、mainを変更しない旧方針を更新し、成果物をmainへ反映しました。履歴を維持し、復旧ブランチは移行前の `b7e8a93` として残しています。既定ブランチは元からmainです。再開手順は `docs/COMMIT_GUIDE.md` に記載しています。ZIPの生成・ローカル検証と、利用者がダウンロードできた事実は別に扱います。
