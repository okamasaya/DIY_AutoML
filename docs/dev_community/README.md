# DEV Community 投稿原稿

Zenn向けの評価記事3本と開発メモ01〜06を、DEV Communityの読者向けに英語化した原稿です。評価記事でプロジェクトの目的、計算方法、業務例を説明し、開発メモで個別の実装と検証を紹介します。実験の出典は [GitHubのスナップショット 64b17f1](https://github.com/okamasaya/DIY_AutoML/tree/64b17f120dad1ead6256d963b779c13630457898) です。

## UC_AutoMaiLの新規記事

[Try UC_AutoMaiL from TestPyPI with Iris and Excel](uc-automail-testpypi-quickstart.md) は、Qiitaの紹介編と使い方編をもとにした独立の英語原稿です。TestPyPIから `0.4.1a0` を取得し、`import uc_automail as uc_al` でCSVの点検、分類、図付きExcelの確認まで進めます。下の新規投稿手順で原稿全体を貼り付けられます。 [Qiitaの原稿一覧](../qiita/README.md)

公開wheelと混同行列の画像URLを確認し、記事のPythonコードはWindowsのPython 3.12.16と3.13.17で実行しました。両環境で保留30件中29件正解、25件の交差検証ログ、図4点付きのExcelを確認しました。

## DIY AutoML Evaluation

親記事から入り、関心に応じてDS向けの再現手順とBP向けの業務例に進めます。3本はDEVで同じ連載名を使います。

| 記事 | 内容 |
| --- | --- |
| [What a machine learning score leaves out](diy-automl-evaluation-framework.md) | 親記事。点推定に誤差と学習の途中経過を添える目的、定番3データの実測図 |
| [How to report uncertainty in a machine learning score](diy-automl-ds-uncertainty.md) | DS向け。条件付き区間、対応のある比較、Irisの再計算コード、CLIとNotebookの例 |
| [What do you get back from a model? An Excel report and three business experiments](diy-automl-business-three-experiments.md) | BP向け。IrisのExcelと、ホテル・カフェ券・通販での誤分類と判断の違い |

3本の翻訳元はGitHubの固定版にある日本語原稿です。分類の実測条件を残し、時系列の節は後続の開発メモ06の実績につなげています。11月26日の原稿確認では、掲載値と保存済みの結果、固定版の英語図、Irisの再標本化1,000回、WineのCLI・Notebook・特徴量探索の例を照合しました。

## DIY AutoML Development Notes

01〜05は分類と評価の流れ、06は時系列予測への展開です。

| 順番 | 投稿原稿 | 内容 |
| --- | --- | --- |
| 01 | [Audit an Iris CSV before ranking five classifiers](01-workflow.md) | TestPyPIからの導入、`uc_al` でIrisを点検・分類、図付きExcelで保留評価を読む |
| 02 | [Same F1, different mistakes](02-cross-checks.md) | 同じF1でも異なる誤分類、予測確率の差、共通保留データでの比較 |
| 03 | [Keep the CSV and reuse a working copy](03-parquet-duckdb.md) | Parquet・DuckDBによるデータ抽出と100万行の点検・変換の計測 |
| 04 | [Does simulated annealing help pick a smaller feature set?](04-annealing.md) | 同じ候補数でのランダム探索との比較 |
| 05 | [Read a model's mistakes in Excel and a Windows preview](05-business-preview.md) | Excelで見逃しと誤検知を読み、自転車貸出の結果を比較 |
| 06 | [When should you keep a Kalman filter?](06-time-series.md) | 公開時系列と合成軌道での比較、試験配布版で動くEKFと欠測診断、Voyager 1の公開軌道を使った再現コードへの案内 |

更新する記事は、[DEVのIris記事](https://dev.to/masaya_oka/audit-an-iris-csv-before-ranking-five-classifiers-457) と [DEVの時系列記事](https://dev.to/masaya_oka/when-should-you-keep-a-kalman-filter-three-forecasting-checks-a64) です。導入手順が先に読めるよう、01を更新してから06を更新します。両原稿のコード例はTestPyPIから取得できる `0.4.1a0` を対象にしています。

Windowsの新規Python 3.13.17環境で、TestPyPIからwheelを再取得し、依存をPyPIから導入しました。Irisの分類は保留30件中29件正解で、図4点付きのExcelと進捗ログが出力されました。EKFは欠測行の予測値を引き継ぎ、診断は観測ペア数を返しました。両方のコード例はPython 3.12.16でも確認済みです。

## 手動で投稿する

公開済み記事の更新は既存記事の編集画面を開き、原稿の本文（先頭のFront Matterの直後から末尾まで）と、必要に応じて説明文を差し替えます。公開設定を確認して更新します。

新規記事の場合は次の手順です。

1. DEVの [設定](https://dev.to/settings/customization) で **Basic Markdown** を選び、[新規投稿](https://dev.to/new) を開きます。
2. 投稿する.mdを開き、冒頭の `---` から末尾まで貼り付け、下書きとして保存します。ファイルにはタイトル、説明、タグ、連載記事の場合は連載名が記載されています。
3. プレビューで表と図を確認し、公開する段階で **Publish** を選びます。全原稿は `published: false` です。

[DEV Editor Guide](https://dev.to/p/editor_guide)

図は公開GitHub上のコミット固定URLから読み込めます。本文末尾とFront Matterに英語原稿へのAI利用を明記しています。 [DEVのAI利用ガイドライン](https://dev.to/guidelines-for-ai-assisted-articles-on-dev/)

## 英語版の図

以下のリンク先は投稿用のPNGです。追加で記事に埋め込む場合は、画像URLをMarkdownに貼り付けます。

| 用途 | 図 |
| --- | --- |
| 01 ワークフロー | [CSVの点検から保留評価まで](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/audit-to-evaluation.png) |
| 05 業務向け評価 | [Irisの保存済みExcelをもとにした要約](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/iris-report.png) · [Bike Sharingの月別比較](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/bp-monthly-evaluation.png) |
| 06 予測と入力品質 | [真値に対する予測値と不確かさ](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/voyager-forecast-truth-interval.png) · [位置RMSEと予測楕円の被覆](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/voyager-forecast-check.png) |
| 評価・DS向けの補足 | [モデル選定と保留評価の分離](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/selection-flow.png) · [macro F1と条件付き95%区間](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-with-and-without-uncertainty.png) · [学習損失と検証損失の履歴](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-training-history.png) · [precisionとrecallのbootstrap楕円](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/evaluation-joint-bootstrap.png) |
| 業務3テーマの補足 | [学習側の観測割合と件数](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/business-training-associations.png) · [モデルと参考ルールのmacro F1差](https://raw.githubusercontent.com/okamasaya/DIY_AutoML/8df22d59dbd5378d52edf87f3aeb3ca55938e86e/docs/dev_community/images/model-rule-difference.png) |
