# Full Paper・Letter・Zenn・DEV Communityの投稿準備資料

2026年11月24日時点。4媒体のドラフト案を作る作業の入口。現在の原稿、再利用できる図・実測値、編集するときに残す条件を記録する。

## 現行原稿とドラフト案の範囲

| 媒体 | 編集を始める元資料 | 今回の成果を入れる構成案 |
| --- | --- | --- |
| **Full Paper** | [現行PDF](papers/DIY_AutoML_Full_Paper_20261119/DIY_AutoML_Full_Paper_20261119.pdf) · [TeXの入口](papers/DIY_AutoML_Full_Paper_20261119/main.tex) · [復元用TeX ZIP](papers/DIY_AutoML_Full_Paper_20261119/DIY_AutoML_Full_Paper_20261119_TeX.zip) · [組版手順](papers/DIY_AutoML_Full_Paper_20261119/README.md) | `methods.tex` に状態推定とグラフの評価条件、`experiments.tex` にボイジャーと公開時系列の比較、`discussion.tex` に適用条件、`reproduction.tex` にデータ・コマンド・版を載せる案。現行原稿が参照する `80b25cc` と今回追記する実験の版を明示する。 |
| **Letter** | [日本語改稿PDF](letters/DIY_AutoML_Letter_20261119_review.pdf) · [英語版PDF](letters/DIY_AutoML_Letter_20261120_en.pdf) · [英語版TeX](letters/DIY_AutoML_Letter_20261120_en.tex) · [旧版の制作指示と原稿](DIY_AutoML_Letter_Draft.md)。日本語改稿の編集用Markdownは復元資料の `output/pdf/DIY_AutoML_Letter_20261119_review.md`。 | データ点検、共通の誤差評価、採用判断までを一つの短い事例に絞る。ボイジャーの予測区間が観測条件で崩れた例と、公開時系列で単純な候補を選んだ例を候補にする。日英の主張と数値の対応を揃える。 |
| **Zenn** | [親記事](../articles/diy-automl-evaluation-framework.md) · [DS向け誤差評価](../articles/diy-automl-ds-uncertainty.md) · [BP向け業務例](../articles/diy-automl-business-three-experiments.md) · [開発メモと編集記録](ZENN_EDITORIAL_NOTES.md) · [2つの比較の技術下書き](development-notes/time-series-crosscheck-draft.md) | 開発メモの候補は、ボイジャーと公開時系列の比較、GCNで辺が効く条件の比較。親・DS・BP記事は現行記事と重複する説明を整理し、DSの評価根拠とBPの読み方に必要な実例を選ぶ。 |
| **DEV Community** | `codex/dev-community-iris` の `docs/dev-community/iris-evaluation.en.md`（`3ce4f7a`） · [既存の英語開発メモ5本](development-notes/README.md) · [比較記事の英語骨子](development-notes/time-series-crosscheck-draft.md#dev-community-english-draft-outline) | 初稿のIris評価を確認し、時系列の英語記事案は「いつカルマンを使い、いつ切り替えるか」を中心にする。GCNはCoraと実データ比較の技術メモに分ける案。 |

DEV CommunityのIris初稿はローカルブランチの履歴にある。復元ZIPからクローンした直後は次のように閲覧できる。

```powershell
git show origin/codex/dev-community-iris:docs/dev-community/iris-evaluation.en.md
```

## 根拠として使える比較結果

| 題材 | 原稿に使える数値と条件 | 元の手順と記録 |
| --- | --- | --- |
| **ボイジャー合成軌道** | 7日先の位置RMSEは通常観測のEKFで41,982 km。EKFには合成軌道と同型の力学モデルを与えた。公称95%予測楕円に真値が入った割合は通常44/44、系統バイアス1/44、過去観測混入0/44。観測の整合性検査を閾値越えした件数は2/44、6/44、42/44。 | [比較と楕円の評価](../experiments/voyager_crosscheck/README.md)。復元資料の `runs/voyager-crosscheck-20261124`、`runs/voyager-crosscheck-uncertainty-20261124`。合成軌道図のPNGは `runs/voyager-synthetic-20261124/trajectory.png`。 |
| **AirPassengersとNYC Taxi** | 現行カルマンの継続判定では両方で直近値を採用し、カルマンを最後まで実行した場合より保留RMSEが約27%、約13%下がった。航空で季節候補27.89千人とAutoGluon 27.49千人。タクシーではAutoGluon、H2O、DIY直近値の順に保留RMSEが小さく、同じ時点でDIYの絶対誤差がH2Oより小さいのは2,056/6,288。 | [3方式の比較](../experiments/seasonal_forecast_crosscheck/README.md) · [元の選定条件](UNCERTAINTY_DESIGN.md#公開時系列でカルマンを選ばない条件を確かめる)。復元資料の `runs/seasonal-forecast-crosscheck-20261124`。 |
| **GCN** | Coraの3分割でGCNがロジスティック回帰より高い保留macro F1。seed 42では引用辺あり0.7527、自己ループのみ0.5102。Irisではロジスティック回帰、UCI Pedal Meでは前週値が上回った。ボイジャーの過去観測混入では今回の時間辺で辺なしネットワークよりRMSEが約7.6%低下した。 | [Cora](GRAPH_CLASSIFICATION.md) · [4データの辺あり・なし比較](../experiments/gnn_application_checks/README.md)。復元資料の `runs/gnn-cora-20261124` と `runs/gnn-crosscheck-20261124/evaluation`。 |
| **BPの分類画面と誤差評価** | グループや日付順の分割、単純ルール、クラス別の誤りを併記する。パッケージの実装時点と業務利用時の前提を記録する。 | [既存のBP版マイルストーン](BP_CLASSIFICATION_MILESTONE_20261119.md) · [データの点検手順](DATA_AUDIT.md) · [既存の誤差評価設計](UNCERTAINTY_DESIGN.md)。 |

各ドラフトでは、比較した候補・データ・予測時間・保留区間・単位の組を残す。EKFの楕円の被覆率、モデルの保留RMSE、検証誤差の経験的な90%点はそれぞれの定義で扱う。Coraの既知グラフ評価と、フィジカルAIの実ログによる検証は別の評価段階として書く。採用範囲の合意は [DS・BP向けの方針](AUDIENCE_PLAN.md#時系列とグラフで標準にする範囲) にある。

## 次回の進め方

最初にFull PaperとLetterの現行PDFと編集元を確認し、追記候補と要約の粒度を決める。その後、ZennとDEV Communityで扱うテーマ、記事内の役割、図を決める。最終的な文章化に渡す資料は、対象媒体の元原稿、この資料一覧、対応する技術下書きと検証資料の組とする。 [復元手順](RECOVERY_CHECKPOINT_20261124.md)
