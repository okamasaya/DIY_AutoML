# Qiitaに投稿する原稿と図

| 投稿順 | 原稿 | タグ候補 |
| --- | --- | --- |
| 1 | [紹介編](uc-automail-alpha-introduction.md) — 開発の目的、α版の機能、評価の不確かさ | `Python` `機械学習` `AutoML` `OSS` |
| 2 | [使い方編](uc-automail-alpha-iris-excel.md) — IrisのCSV化、分類、図付きExcelの確認 | `Python` `機械学習` `scikit-learn` `AutoML` `Excel` |
| 3 | [比較編](uc-automail-iris-voyager-comparison.md) — IrisとVoyager 1でAutoGluon・H2Oと比較 | `Python` `AutoML` `時系列解析` `カルマンフィルタ` `機械学習` |

Qiitaでのパッケージ表記は `UC_AutoMaiL`、Pythonでの読み込みは `import uc_automail as uc_al` とします。開発名とZenn・Dev Communityの記事名には `DIY_AutoML` を使います。 [TestPyPIで配布しているα版](https://test.pypi.org/project/uc-automail/0.4.1a0/) · [Qiitaのタグ設定](https://help.qiita.com/ja/articles/qiita-tag)

## Qiitaエディタへの転記

1. 原稿の先頭見出しをタイトル欄に入れ、次の段落から末尾までを本文欄に貼り付けます。 [Qiitaの記事投稿手順](https://help.qiita.com/ja/articles/qiita-post)
2. 使い方編には混同行列、比較編にはVoyager 1の分割図を添付します。本文の画像パスをそれぞれQiitaでアップロードした画像のURLに置き換えます。 [混同行列のPNG](images/iris-holdout-confusion.png) · [Voyager 1の分割図のPNG](images/voyager-horizons-split.png) · [Qiitaの画像アップロード手順](https://help.qiita.com/ja/articles/qiita-image-upload)
3. 3本の公開URLが揃ったら、関連記事として相互リンクを追加します。

## 実行結果

使い方編のスクリプトは、WindowsのPython 3.12.16と3.13.17で、TestPyPIの `0.4.1a0` を使って実行しました。scikit-learnはいずれも1.9.1です。両環境でSVMが選ばれ、保留30件中29件正解、macro F1は0.967で、図4点付きのExcelが出力されました。

掲載する混同行列はPython 3.13.17で作成した `SL_Test` の図です。元のExcelは作業端末の `PyPI/runs/qiita-iris-20261127T131534687Z/runs/iris-20261127-221620-899467/report/report_with_figures.xlsx` にあります。

比較編のIrisは、開発版 `0.4.0a7` の共通保留30件での比較に基づきます。保存済みの行番号、正解、予測から採点と区間を照合しました。 [AutoGluon比較記録](../../experiments/autogluon_sl_crosscheck/README.md) · [H2O比較記録](../../experiments/h2o_sl_crosscheck/README.md)

比較編のVoyager 1は、JPL Horizonsの公開軌道を参照位置とし、模擬観測を入力した追加実験です。公開軌道の座標から参照位置を再計算し、保留44時点の位置RMSEと予測楕円の内外を照合しました。 [再実行手順](../../experiments/voyager_crosscheck/README.md#jpl-horizonsの公開軌道を参照した比較) · [数値と取得条件](../../experiments/voyager_crosscheck/horizons-results.json)
