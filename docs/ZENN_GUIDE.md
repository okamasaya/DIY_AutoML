# Zenn紹介記事の下書き

新規下書きは `articles/diy-automl-evaluation-framework.md` です。過去に作成したとされる5本の原稿は未取得であり、それらを復元したものではありません。今回のコードと実測記録を基に、リポジトリ紹介を1本新しく作成しています。

Front MatterはZenn公式の形式に合わせ、`published: false` としています。記事の投稿、ZennとのGitHub連携、公開予約、公開操作は行いません。この設定はZenn上の公開設定です。ファイル自体は公開GitHubリポジトリの作業ブランチで閲覧できます。

公式ガイド: https://zenn.dev/zenn/articles/zenn-cli-guide

画像は公式の `/images/` 形式に合わせています（https://zenn.dev/zenn/articles/deploy-github-images）。フロー図はimages/diy-automl/selection-flow.png、編集用ソースはdocs/figures/selection-flow.dotと.mmdです。DOTから生成したPNGで、外部の埋込描画サービスを使わずローカルプレビューでも確認できます。

## レビューとプレビュー

```sh
npx --yes zenn-cli@0.5.14 preview
```

リポジトリのルートで実行し、表示されたローカルURLで下書きを確認します。Pythonの学習環境にNode依存を追加する必要はありません。Webエディタを使う場合は、タイトル・絵文字・topicsを画面側に設定し、Front Matterを除いた本文を貼り付けます。Webエディタを使う場合はPNGをZenn側へアップロードし、本文中の画像URLを差し替えます。GitHub連携の `/images/` 指定をそのままWebエディタへ貼る手順とは区別してください。

## 現在の案内先

mainは初期状態のため、記事のリンクとcloneコマンドは `recovery/reimplemented-0.4.0a1-post1` ブランチへ統一しています。mainへのマージやdefault branch変更は行っていません。公開時に案内先を変更した場合は、リンクとcloneコマンドを一緒に見直してください。

親記事の原稿は利用者が確認した0.4.0a4時点の実装範囲を保全しています。対象版の更新とDS/ビジネス向けへの分割はAutoML本体の整備後に行います。方針はdocs/AUDIENCE_PLAN.mdに記録しています。データセットは繰り返し参照した開発用、モデル/設定の選定は学習側CVだけ、test区間は固定モデルに条件付き、と区別しています。未見データでの一般的な性能保証や、旧0.4ソースの復元を主張していません。

ライセンスは未選定のままで、記事でも自由利用可能なOSSという説明はしていません。今回の作業でライセンスや配布先、リポジトリ公開範囲を変更していません。

記事の手順検証・ローカルプレビューの確認結果は `tests/results/0.4.0a4/` に保存します。記事をZennへ投稿した事実や、利用者が公開内容を最終確認した事実とは区別します。
