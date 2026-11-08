# Zenn紹介記事の下書き

新規下書きは `articles/diy-automl-evaluation-framework.md` です。過去に作成したとされる5本の原稿は未取得であり、それらを復元したものではありません。今回のコードと実測記録を基に、リポジトリ紹介を1本新しく作成しています。

利用者の追加依頼により、3業務テーマのQuick & dirty実験を親記事へ追記し、`articles/diy-automl-business-three-experiments.md` をビジネス向けの実験解説下書きとして追加しました。両方published=falseです。図はimages/business-quick-20261108/associations.png、加工・評価条件はdocs/BUSINESS_QUICK_RESULTS.md、再実行はexperiments/business_quick_20261108/README.mdにあります。アプリ画面の利用手順とは区別します。

親記事の目的を、利用者が示した「実行結果に誤差評価を含め、データ量・品質について議論したい」という軸で追記しました。現行の条件付き区間と、未実装の学習曲線・再学習/選定の変動は区別します。他のAutoMLにも不確かさを扱う機能があるため、未調査の市場全体について有無を断定せず、評価対象・由来・前提・記録を説明する設計を重視します。詳しくはdocs/UNCERTAINTY_DESIGN.mdです。

続く依頼により、親記事へ統計的な不確かさ、学習手順の変動、数値計算誤差、系統的な偏りの区別を追記しました。ビジネス記事は誤差の平易な説明と公開例/持ち込みデータの対比、DS向けの新規下書き `articles/diy-automl-ds-uncertainty.md` は推定対象・式・評価単位・計算条件・保存結果を扱います。現在の下書きは3本で、すべてpublished=falseです。簡易版は計算量と説明を絞る方針で、重要な前提を省略したり、価格や画面機能を決定した意味ではありません。

Front MatterはZenn公式の形式に合わせ、`published: false` としています。記事の投稿、ZennとのGitHub連携、公開予約、公開操作は行いません。この設定はZenn上の公開設定です。ファイル自体は公開GitHubリポジトリの作業ブランチで閲覧できます。

公式ガイド: https://zenn.dev/zenn/articles/zenn-cli-guide

画像は公式の `/images/` 形式に合わせています（https://zenn.dev/zenn/articles/deploy-github-images）。フロー図はimages/diy-automl/selection-flow.png、編集用ソースはdocs/figures/selection-flow.dotと.mmdです。DOTから生成したPNGで、外部の埋込描画サービスを使わずローカルプレビューでも確認できます。

## レビューとプレビュー

```sh
npx --yes zenn-cli@0.5.14 preview
```

リポジトリのルートで実行し、表示されたローカルURLで下書きを確認します。Pythonの学習環境にNode依存を追加する必要はありません。Webエディタを使う場合は、タイトル・絵文字・topicsを画面側に設定し、Front Matterを除いた本文を貼り付けます。Webエディタを使う場合はPNGをZenn側へアップロードし、本文中の画像URLを差し替えます。GitHub連携の `/images/` 指定をそのままWebエディタへ貼る手順とは区別してください。

## 現在の案内先

2026年11月8日（日本時間）、利用者の依頼により復旧ブランチの成果物をmainへ反映し、記事のリンクとcloneコマンドもmainへ統一しました。既定ブランチは元からmainで、設定の変更は不要でした。`recovery/reimplemented-0.4.0a1-post1` は移行前の履歴として残しています。この移行でZennの公開設定は変更せず、3記事ともpublished=falseです。

温度感を確認した0.4.0a4原稿はGit履歴に残し、今回の親記事は実装済みの図付きExcelに合わせ0.4.0a5へ更新しました。利用者の追加方針に従い、新しいモデルでの改善幅について2022年の木系モデル比較、2025年のTabArena、2025年のTabPFN論文を引用しています。従来手法が有力な条件と新手法が改善した条件の両方を扱い、本リポジトリで最新モデルを比較済みとは説明しません。

DS向けは既存から新しいモデルまで任意の比較範囲、ビジネス向けは扱いやすい業務データと主要モデルを中心にします。ビジネス向け実験解説とDS向け誤差評価の下書きは追加しました。DS向け全コマンドの利用ガイド、画面アプリ・その手順は今後です。方針はdocs/AUDIENCE_PLAN.md、追加モデル・特に重い実験のTo Doはdocs/MODEL_VALIDATION_TODO.mdに記録しています。

データセットは繰り返し参照した開発用、モデル/設定の選定は学習側CVだけ、test区間は固定モデルに条件付き、と区別しています。未見データでの一般的な性能保証や、旧0.4ソースの復元を主張していません。

ライセンスは未選定のままで、記事でも自由利用可能なOSSという説明はしていません。今回の作業でライセンスや配布先、リポジトリ公開範囲を変更していません。

0.4.0a4原稿の手順検証・ローカルプレビューの確認結果は `tests/results/0.4.0a4/`、図付きExcelの実行検証は `tests/results/0.4.0a5/` に保存しています。これらに含まれる記事のハッシュは当時の原稿の記録で、今回の編集後の記事と同一という意味ではありません。記事をZennへ投稿した事実や、利用者が編集後の公開内容を最終確認した事実とも区別します。

3テーマの実験と原稿変換の記録は `tests/results/business-quick-20261108/`、今回の目的・設計の追記は `tests/results/editorial-uncertainty-20261108.json` です。記事や文書のハッシュは各記録作成時点のスナップショットであり、後の編集で過去の検証記録を書き換えません。

利用者別の誤差評価説明を反映した3原稿と関連文書の今回の変換・整合性確認は `tests/results/editorial-uncertainty-audiences-20261108.json` に記録します。新しいモデル学習や、追加診断機能の検証を行った記録ではありません。

0.4.0a6では`uncertainty`を実装し、親・DS記事の対象版を更新しました。bootstrap SE・クラス別区間・通常/層化の2方式・保存結果入口を追記し、ビジネス記事には3テーマで実際に計算したSEと出力先を追加しました。今回の原稿の変換・数値照合は `tests/results/0.4.0a6/editorial.json`、実装の検証は `tests/results/0.4.0a6/verification.json` です。3本ともpublished=falseを維持します。
