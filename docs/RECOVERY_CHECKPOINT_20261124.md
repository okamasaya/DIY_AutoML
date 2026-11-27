# DIY AutoML 再開用セーブポイント 2026年11月24日

再開時は、まず [投稿準備の資料一覧](PUBLICATION_HANDOFF_20261124.md) を開く。次の作業は、Full Paper、Letter、Zenn、DEV Communityの現行原稿を整理し、それぞれのドラフト案へ入れる範囲を決めること。

## 実装と合意した方針

| 項目 | この時点の状態 |
| --- | --- |
| 時系列 | `main` の `ed14006` に線形カルマン・EKF・UKF・粒子フィルタと、ボイジャー・公開時系列の比較、投稿用技術下書きを統合済み。 |
| GNN | `codex/gnn-node-classification` の `9c889ae` で2層GCNの専用CLIとAPI、`fe082ad` でボイジャー・NYC Taxi・Iris・Pedal Meの比較を保存済み。このチェックポイントはそのブランチ上にある。 |
| DS向けの製品方針 | 入力と目的に適用できる候補を一通り比較し、精度・不確かさ・計算時間・適用条件を記録する。力学・観測モデルやグラフがあるときに対応する専用手法を候補に含める。 |
| BP向けの製品方針 | 1系列の時系列で、直近値・季節の基準とカルマンを比べ、保留誤差と採用理由を表示する。EKF・UKF・粒子フィルタ・GCNは詳細分析に置く。 |
| 今使える入口 | `kalman-track` は1系列の水準・変化率を推定し、先行区間の判定で直近値へ切り替えられる。`gnn-compare` は既知グラフのノード分類。BPの画面は分類データの点検・評価まで検証済み。上のDS・BP標準範囲は今後の製品設計として引き継ぐ。 |
| 将来の評価 | フィジカルAIでは用途ごとのセンサー、時刻、参照値を固定して評価する。LLMでは任せる判断を明示し、入力への根拠、入力変化への感度、判断を保留する条件を定める。 |

## 復元に使うファイル

復元用ZIPは、この作業端末のOneDriveフォルダー内 `runs/recovery-checkpoint-20261124/DIY_AutoML-recovery-20261124.zip`。展開先で `START-HERE.md` を開く。

- `DIY_AutoML.bundle` はローカルブランチとタグのGit履歴。DEV CommunityのIris原稿がある `codex/dev-community-iris`、Full Paper、Letterの履歴も含む。
- `evidence.zip` は主要な検証結果、再実行に使う公開データのスナップショット、作業フォルダーにあった編集用原稿。新規チェックアウトのルートに展開すると、元の相対パスを復元できる。
- `wheels/` と `requirements-windows-py313.txt` はWindows x64のPython 3.13向けの基本環境。チェックポイントのDIY AutoML本体と、必要な基本依存をwheelからインストールできる。Python本体とGitは復元先に用意する。
- `SHA256SUMS.txt` と `evidence-manifest.json` はファイルの照合用。別の端末や保存先へ移した場合も、同梱手順でチェックできる。

現行ブランチの全テストは、TensorFlowを含む作業環境で282件通過した。実行記録は復元資料の `runs/recovery-checkpoint-20261124/full-tests.xml` にある。

復元先のOSが異なる場合は、Gitのコードと資料を復元し、現地のPythonに合わせて `python -m pip install -e ".[dev]"` を実行する。Linux CPU向けの別環境は `requirements-core.lock.txt` と `requirements-local.lock.txt` に記録している。

## 比較用の追加環境

基本環境でカルマン、GCN、保存結果の検証を再開できる。外部モデルの再学習や文書の組版を行うときは該当する環境を別に用意する。

| 用途 | 検証時の環境と再開先 |
| --- | --- |
| TensorFlowを含むテスト | Windows Python 3.13.17、TensorFlow CPU 2.20.0。インストール定義は `.[dev,dl,storage]`。 [DLの検証条件](DL_BACKENDS.md) |
| AutoGluon比較 | Python 3.13.17、AutoGluon Tabular 1.6.3。 [専用環境の作り方](../experiments/autogluon_sl_crosscheck/README.md) |
| H2O比較 | Python 3.14.7、H2O 3.46.0.12、Temurin Java 17.0.21。ループバックで動かした。 [専用環境の作り方](../experiments/h2o_sl_crosscheck/README.md) |
| Full Paper・Letter | TeXと図はGit。Full Paperの組版はXeLaTeXまたはTectonic。作業端末の英語LetterにはTectonic 0.17.0を使用した。 [組版手順](papers/DIY_AutoML_Full_Paper_20261119/README.md) |

## 次に開く資料

1. [投稿準備の資料一覧](PUBLICATION_HANDOFF_20261124.md) と [投稿用の技術下書き](development-notes/time-series-crosscheck-draft.md) を読む。
2. Full PaperとLetterで更新候補の節・段落を指定する。ZennとDEV Communityは既存の連載・初稿を確認し、記事に入れる数字と図を選ぶ。
3. 公開する数値を復元した元の評価ファイルと照合する。実装の対象版、比較条件、予測期間、単位、保留件数を各ドラフトに添える。
