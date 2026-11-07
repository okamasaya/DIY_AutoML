# 新スレッド開始用プロンプト（そのまま貼付可能）

あなたはPython / scikit-learn / TensorFlow・Keras / pandas / Jupyter Notebookで進めてきた「SL/UL/DL統合評価フレームワーク Ver.2」の開発を引き継いでください。

## 最優先

1. このZIPに含まれる `source/Comp_MLmodel_ver2.ipynb` を**現在の正本ソース**として読み、STEP0〜STEP7のコードを実際に確認してください。会話概要だけを根拠に想像で改修しないでください。
2. `docs/02_PROJECT_STATUS_AND_DESIGN.md`、`docs/03_RESULTS_AND_VALIDATION.md`、`docs/04_ISSUES_AND_NEXT_STEPS.md` を確認してください。
3. `reports/` のIris/Wine/Breast Cancerの既存Excel結果を、回帰検証用ベースラインとして用いてください。
4. 原則としてユーザーの指示なしにGitへのcommit/pushを実行しないでください。既存NotebookやExcelも勝手に上書きせず、改修版は別名保存としてください。
5. コードは**STEP番号・セル単位**で提示し、適度にコンパクトな改行にしてください。計算・評価・表示・出力を分離します。

## プロジェクトの目的

任意の分類用CSV（将来的な到達目標）を読み込み、教師あり学習(SL)、教師なし学習(UL)、深層学習(DL)で複数モデルを実行し、精度・クラスタ品質・学習履歴・不確かさ・再現性を評価したうえで、根拠が確認できるExcelレポートを生成する。AutoMLのうち、**自動最適化そのものより、比較評価と妥当性検証**に重点を置く。

## 現状

- STEP0〜STEP7の基本実装とExcelの5シート出力まで動作確認済み。
- Iris（150×4、3クラス）、Wine（178×13、3クラス）、Breast Cancer（569×30、2クラス）で出力済み。
- Excelシートは `SL_Result`、`UL_Result`、`DL_Result`、`DL_History`、`Summary`。
- STEP7の図表埋込と自動Summary評価は未実装。
- **数値が高いことだけで妥当性を宣言しないこと**。ULとSL/DLのスコアを機械的に一つのランキングへ混ぜない。

## 次に取り組みたいこと

**STEP7 Visualization Report拡張。** Notebookには全候補のグラフを出せるようにし、Excelへは各系統で評価上位最大3件の**重要なグラフだけ**を埋め込む。SL: Confusion Matrix、UL: PCA 2D散布図とクラスタ数別Silhouette等、DL: Train/ValidationのAccuracy・Loss曲線とConfusion Matrix。ULの「上位」基準は内部評価（Silhouette等）と外部評価（ARI等）を分け、選定根拠を明記する。DLは現状1設定しかないため、無理に3件を作らない。プロット枚数、描画時間、ファイルサイズをWine等で測定し、負荷と見やすさを検証する。

## 改修前に必ず確認

- STEP2の標準化が全件 `fit_transform` → SL/DLのtrain/test分割へ進むため**データリーク**あり。優先修正。
- KMeansの既存ループ（STEP4-3）と関数呼び出しセル（STEP4-4直後）で最終kが重複記録される。実測ExcelではULが55行で、KMeansのみ10行ある（本来は54行・各モデル9行）。優先修正。
- STEP1はGoogle Colab/Driveの固定パス。CSV入力・ローカルパス・欠損・カテゴリ特徴量対応は今後の機能。
- `DL_config["layers"]` は途中セルで `[32,16,8]` に上書き。元定義 `[16,8]` だけ見て判断しない。
- `summary_report` は説明用3行のみで、`best_model_report` は作成しているがExcelには出力していない。
- Silhouette_SDは**点ごとの分散**、Silhouette_SEは `scipy.stats.sem` による点ごとの標準誤差の計算。いずれもseed変動や独立な反復実験の不確かさをそのまま表すわけではない。

最初に、既存Notebookの現状確認と、STEP7の変更対象セル・追加セル・評価順位と出力図の配置方針を簡潔に提示し、**既存の動作確認済み部分を不用意に壊さない最小改修**から進めてください。
