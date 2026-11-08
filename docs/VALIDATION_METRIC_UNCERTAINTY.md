# 評価値の誤差評価の検証（0.4.0a6）

2026年11月8日（日本時間）、保存済み分類予測へbootstrap SE・percentile区間を付ける`uncertainty`を追加しました。旧ソースの復元ではなく、現在の再実装への追加です。従来のrun/select/tune/repeat-dlの出力を上書きせず、新しいschema 9へ保存します。

## 確認した結果

- Python 3.12.14 / Linux CPU。開発環境・独立wheel環境でそれぞれ **176 passed**、failure/error/skip 0。両環境とTensorFlowなしコア環境でpip check合格。
- 3データ（Iris/Wine/Breast Cancer）×4保存結果family（run/select/repeat-dl/tune）×2方式（stratified/ordinary）×2環境で、**48出力・216個の固定モデル評価・48枚の埋込PNG**を確認。各1,000反復。
- 全再標本化index、正解/予測、全体/クラス別反復値、confusion matrix、全シートの表、PNGバイトが開発/wheel間で一致。
- 各保存元のファイルは検証前後でSHA256不変。SLは宣言順、DLは全seedを保持し、test順のランキングやseed選定は行っていません。
- 既存select/tuneの条件付き区間と新しいstratifiedの点推定・端点が一致（絶対許容値1e-12）。学習・推論のやり直しはありません。
- TensorFlow未導入のコア環境でも、保存済みDL10seedの誤差評価・図付きExcel生成・保存結果検証が完了。

48出力の元モデルは過去に学習済みです。今回48回の新しいモデル学習や、未見の外部検証を行った意味ではありません。同じ環境条件での再現確認で、他OS/GPUや別ライブラリ版での完全一致を保証しません。

## 計算と境界の検証

4指標とクラス別反復値をscikit-learnによる独立計算へ照合しました。4行データの通常bootstrap全256通りを列挙し、Accuracyの分散が独立なベルヌーイ標本平均の式と一致することも確認しました。これは小さい既知の例での計算確認で、実データに独立性があると証明したものではありません。

bootstrap SEは反復指標の標本SD（ddof=1）であり、SD/√Bへ縮めないことを検証しました。正解・予測のpaired抽出、層化方式のクラス件数固定、通常方式の件数変動、固定クラス集合、観測0クラス・完全予測・ゼロ幅の表示も確認しています。

再学習を禁止した状態で4種類の保存結果を処理し、TensorFlow importを拒むプロセスでも保存済みDLを処理しました。現在のSLだけのrunは元CSVなし、過去のtargetなしrunはhash一致CSVで処理できます。不正設定、ULだけ/failedの元結果、hash不一致CSV、上限超過、原本内出力、上書きを拒否します。

描画失敗は出力側のmanifestをfailedにし、元結果を保ちます。保存後の再標本化index、反復値、クラス別値、分割の重複、表、モデル名、scope、埋込画像の不整合を検出します。

## 業務3テーマへの適用

既存のQuick & dirtyの選定済み予測を再利用し、1,000反復のstratifiedで計算しました。元の実験条件は[BUSINESS_QUICK_RESULTS.md](BUSINESS_QUICK_RESULTS.md)を維持します。

| テーマ | macro F1 | bootstrap SE | 条件付き95%区間 |
| --- | ---: | ---: | ---: |
| ホテル | 0.5271 | 0.02627 | 0.47679–0.57645 |
| カフェ券 | 0.6925 | 0.02327 | 0.64750–0.73935 |
| 再購入 | 0.6237 | 0.02545 | 0.57322–0.67239 |

保存先は `validation/metric-uncertainty-0.4.0a6/` です。モデルを変更・選び直した結果ではありません。カフェ券の回答者依存や、ホテル/通販の時点・対象の制約はそのまま残ります。特にカフェ券の行単位区間を、新しい回答者への妥当な区間として解釈しません。1モデルと10seedのPNGを画像として目視し、ラベルと区間の配置を確認しました。

## 保存記録と再実行

JUnit、環境別24出力の結果JSON、ソースハッシュ・wheel SHA256は `tests/results/0.4.0a6/` に記録しています。過去の版の合否・原稿ハッシュ・実験記録は上書きしていません。48出力の全体はこの作業環境の検証用領域に生成し、リポジトリには業務3テーマの実例と再実行スクリプト・検証記録を保存します。

```sh
.venv/bin/python scripts/verify_metric_uncertainty.py --output /新規保存先/development
独立wheel環境/bin/python scripts/verify_metric_uncertainty.py --output /新規保存先/wheel --compare /新規保存先/development
```

手順は[METRIC_UNCERTAINTY_GUIDE.md](METRIC_UNCERTAINTY_GUIDE.md)です。学習・選定・再学習の変動、群/時間依存、分布変化、系統誤差、数値計算誤差の包括的評価は未実装です。今回の追加は固定モデルの評価値に対する行単位の標本不確かさです。
