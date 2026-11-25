# DEV Community 投稿原稿

Zenn向けの開発メモ01〜06をもとに、DEV Communityの読者向けに整えた英語原稿です。01〜05は分類と評価の流れ、06は時系列予測への展開です。実験の出典は [GitHubのスナップショット 64b17f1](https://github.com/okamasaya/DIY_AutoML/tree/64b17f120dad1ead6256d963b779c13630457898) です。

| 順番 | 投稿原稿 | 内容 |
| --- | --- | --- |
| 01 | [Audit an Iris CSV before ranking five classifiers](01-workflow.md) | Irisのデータ点検からモデル選定、保留評価、区間推定まで |
| 02 | [Same F1, different mistakes](02-cross-checks.md) | 同じF1でも異なる誤分類、予測確率の差、共通保留データでの比較 |
| 03 | [Keep the CSV and reuse a working copy](03-parquet-duckdb.md) | Parquet・DuckDBによるデータ抽出と100万行の点検・変換の計測 |
| 04 | [Does simulated annealing help pick a smaller feature set?](04-annealing.md) | 同じ候補数でのランダム探索との比較 |
| 05 | [Read a model's mistakes in Excel and a Windows preview](05-business-preview.md) | Excelで見逃しと誤検知を読み、自転車貸出の結果を比較 |
| 06 | [When should you keep a Kalman filter?](06-time-series.md) | 公開時系列での比較と、ボイジャー合成軌道での予測誤差・被覆率・入力品質の点検 |

## 手動で投稿する

1. DEVの [設定](https://dev.to/settings/customization) で **Basic Markdown** を選び、[新規投稿](https://dev.to/new) を開きます。
2. 投稿する.mdを開き、冒頭の `---` から末尾まで貼り付け、下書きとして保存します。タイトル・説明・タグ・連載名をファイルに記載しています。
3. プレビューで表と図を確認し、公開する段階で **Publish** を選びます。全原稿は `published: false` です。

[DEV Editor Guide](https://dev.to/p/editor_guide)

全6本の連載名は **DIY AutoML Development Notes** です。図は公開GitHub上のコミット固定URLを参照するため、手動投稿時に画像の再アップロードは不要です。本文末尾に英語原稿へのAI利用を明記しています。 [DEVのAI利用ガイドライン](https://dev.to/guidelines-for-ai-assisted-articles-on-dev/)

## 英語版の図

以下のリンク先は投稿用のPNGです。追加で記事に埋め込む場合は、画像URLをMarkdownに貼り付けます。

| 用途 | 図 |
| --- | --- |
| 01 ワークフロー | [CSVの点検から保留評価まで](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/audit-to-evaluation.png) |
| 05 業務向け評価 | [Irisの保存済みExcelをもとにした要約](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/iris-report.png) · [Bike Sharingの月別比較](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/bp-monthly-evaluation.png) |
| 06 予測と入力品質 | [真値に対する予測値と不確かさ](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/voyager-forecast-truth-interval.png) · [位置RMSEと予測楕円の被覆](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/voyager-forecast-check.png) |
| 評価・DS向けの補足 | [モデル選定と保留評価の分離](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/selection-flow.png) · [macro F1と条件付き95%区間](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-with-and-without-uncertainty.png) · [学習損失と検証損失の履歴](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-training-history.png) · [precisionとrecallのbootstrap楕円](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-joint-bootstrap.png) |
| 業務3テーマの補足 | [学習側の観測割合と件数](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/business-training-associations.png) · [モデルと参考ルールのmacro F1差](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/model-rule-difference.png) |
