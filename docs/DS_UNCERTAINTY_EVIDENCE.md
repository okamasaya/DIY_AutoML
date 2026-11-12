# DS向け記事：出典と検証範囲

2026年11月12日（日本時間）。対象は `articles/diy-automl-ds-uncertainty.md`。DS志望者も、誤差の意味・表示・再計算をたどれるように更新した。出典の確認と、DIY AutoMLの実行確認を区別する。

同日、時系列の誤差評価を未実装の課題として追記した。既存の式・コード・図・数値は変更していない。下記のコード実行・公式資料確認のJSONはその追記前の原稿ハッシュを持つ記録として保持し、追記後の原稿・表示の確認は [時系列追記の記録](../tests/results/editorial-timeseries-roadmap-20261112.json) に分ける。

## 公式資料で確認したこと

Web版の案内先は記事内に置き、確認には次の版の公式リポジトリを用いた。取得したファイルのURL・SHA256は [検証記録](../tests/results/editorial-ds-uncertainty-20261112.json) に保存する。最新版全体や市場の調査ではない。

| 確認した資料 | 固定した版とファイル | 記事で裏付ける説明 |
| --- | --- | --- |
| scikit-learn：bias–varianceの例 | [1.8.0 / plot_bias_variance.py](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/examples/ensemble/plot_bias_variance.py) | 回帰の期待二乗誤差におけるbias・variance・noise。測定器の系統誤差とは定義が異なり、F1へ同じ分解をそのまま適用しない |
| scikit-learn：よくある落とし穴 | [1.8.0 / common_pitfalls.rst](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/doc/common_pitfalls.rst) | 予測時に利用できない情報の混入や、前処理fitによるリークが評価を楽観的にしうること |
| scikit-learn：交差検証 | [1.8.0 / cross_validation.rst](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/doc/modules/cross_validation.rst) | 独立性の前提、同じ患者等の群と時系列を考慮する分割が必要になること |
| scikit-learn：置換重要度 | [1.8.0 / permutation_importance.rst](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/doc/modules/permutation_importance.rst) | 特定の学習済みモデルとデータに対する重要度であること、シャッフル反復によるばらつきを求められること |
| AutoGluon：feature_importance | [v1.4.0 / predictor.py](https://github.com/autogluon/autogluon/blob/v1.4.0/tabular/src/autogluon/tabular/predictor/predictor.py#L3376) | `stddev`が置換重要度の反復SD、`n`がシャッフル反復数であること。区間を平均±t値×SD/√nで計算する実装 |
| PyTorch：数値精度 | [v2.9.0 / numerical_accuracy.rst](https://github.com/pytorch/pytorch/blob/v2.9.0/docs/source/notes/numerical_accuracy.rst) | 有限精度・演算順序・計算基盤による差。同じseedや再現性だけでは数値誤差の不在を示さないこと |

「統計・計算・系統」の3分類は本稿の説明上の入口であり、上記資料が独立な3成分の普遍的な分解を定義したと扱わない。「モデルの論文で検証済みでも、任意の手元データ・用途の偏りが評価済みにはならない」という説明は、モデルの仮定と、測定・ラベル・分割等に依存する評価を区別したもの。系統誤差の網羅的な定量化を新たに実施したわけではない。

AutoGluonのAPI説明と区間の実装を確認した範囲に限定する。実際に利用者が見た`fit`ログの意味や、一般利用者の活用率は特定していない。AutoGluon/PyTorchを新たにインストール・実行した比較ではない。また、p値を「特徴が有用である確率」へ読み替える説明は採用していない。

## 記事内コードの確認

- Iris・固定MLP seed 42：掲載コードをそのまま実行し、正解と予測からscikit-learnでmacro F1を1,000反復計算した。
- 再標本化indexの全要素は保存済み結果と完全一致。F1反復値は全1,000件が絶対許容差1e-12以内で一致した。
- 点推定0.933333、bootstrap SE 0.047244、95% percentile区間 [0.829497, 1.000000] を確認した。表示は0.933、0.047、[0.829, 1.000]。
- コードはまず広い軸（0〜1.05）で点と区間を確認する。出力は既存の無視対象 `runs/ds-iris-bootstrap/` に置き、元の保存結果を上書きしていない。
- WineのCLI `select` → 保存結果検証 → `report` → 図付きExcel検証 → `uncertainty` → 誤差評価検証を実行した。
- 掲載したPython APIの `select_csv` も実行し、CLIと時間列を含まない6つの結果シートが完全一致した。
- 初回導入用のclone・仮想環境作成・依存インストールは、この更新で新しい環境へ再実行していない。実行例はこの会話でセットアップ済みのPython 3.12環境で検証した。

## 図と数式の確認

親記事の3図は再利用し、既存の選定フロー図と合わせて記事に4図を掲載する。既存PNGと元の実験出力は変更していない。楕円の経験的な点被覆は90.5%、95.2%、99.1%で、母数の信頼領域とは説明しない。

Zenn CLI 0.5.14とChromium/Playwrightで、1280px/390px幅の4画像、数式31か所、detailsを確認した。KaTeX構文エラーとページ内JavaScriptエラーは0件。狭い画面ではコードを横スクロールでき、本文ページ全体の横はみ出しはなかった。

ローカル検証では2つの環境対応を行った。Chromiumの信頼ストアではプロキシ経由のKaTeX CDN取得に失敗したため、同じURLの内容を継承プロキシとシステムCAでTLS検証したHTTPクライアントで取得し、ブラウザへ渡した。TLS検証は無効化していない。またCLIのflex子要素が長いコードに合わせて広がるため、テスト側だけで `.layout__main { min-width: 0; }` を適用した。記事原稿やアプリへCSSは追加していない。外部の `listen-embed-event.js` は取得制限が残ったが、本記事の数式と4画像は上記方法で表示確認した。

これらの対応と確認結果を機械可読な検証記録に残す。公開Zennサイトでの実機表示や、公開本文への反映を検証したという意味ではない。

## 未実施の範囲

新しいモデル学習、入力からの誤差伝搬、数値精度の実測比較、独立したラベル監査、群・時間外評価、arXiv新着論文の網羅的なレビュー、利用実態の調査は実施していない。アプリ本体や依存宣言は変更していない。
