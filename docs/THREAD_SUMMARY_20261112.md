# 2026年11月12日のスレッドまとめ・再開手順

このスレッドでは、DIY AutoMLの親・DS・BP記事を、誤差の意味、途中経過の保存、実物出力、データ品質の議論がつながる構成へ改稿した。利用者の確認を受け、記事・図・出力見本・根拠をGitHubのmainへ保存・統合済み。この文書は、前の会話や一時環境がなくても再開するための記録である。

## 最初に確認する状態

| 項目 | 状態 |
| --- | --- |
| リポジトリ | https://github.com/okamasaya/DIY_AutoML |
| 通常の取得先 | `main`。新しい作業は別ブランチで進める |
| アプリの版 | **0.4.0a7**。このスレッドの記事編集で版・アプリ本体・依存を変更していない |
| 記事と実物出力の統合地点 | `57412b5973391b8aff5f85806e81e13799568dca`（BP改稿のmerge） |
| BP改稿の内容コミット | `da7128677afa27eae34f310b203bc9bc8a0083b3` |
| その時点の確認 | リモートmainとローカルmainのSHA一致、別フォルダへのcloneで変更38ファイルのバイト一致、成果物28ファイルのハッシュ一致、作業ツリーに未保存変更なし |
| 本まとめの保存 | BP統合の後に文書として追加。最新のコミット・merge・リモート反映はGit履歴と接続時の状態で確認する |

上のSHAは記事・出力をまとめた地点であり、以後のmainを固定する指定ではない。

## 新しいスレッドの読み順

1. [HANDOFF.md](HANDOFF.md)と本書で、方針・直近の完了・残件を読む。
2. [修正方針](REVISION_POLICY_20261110.md)と[ROADMAP](ROADMAP.md)で、記事の解釈と次の開発候補を確認する。
3. 記事作業なら[編集メモ](ZENN_EDITORIAL_NOTES.md)、DSの根拠なら[DSの裏付け](DS_UNCERTAINTY_EVIDENCE.md)、BPの見本なら[BPの出力手順](BP_OUTPUT_EXAMPLE.md)へ進む。
4. 実行が必要な場合に限り、[README](../README.md)と[Git取得・環境の再開手順](COMMIT_GUIDE.md)を読む。文書を読むだけならモデルの再学習は不要。

まだcloneしていない環境では、通常の取得先を使う。

```sh
git clone --branch main --single-branch https://github.com/okamasaya/DIY_AutoML.git
cd DIY_AutoML
git status --short --branch
git log -5 --oneline
```

既存の作業コピーでは、先に未保存変更と接続先を確認する。変更があれば保全し、勝手にreset・上書き・force pushしない。

```sh
git status --short --branch
git fetch origin main
git log -5 --oneline --decorate
git rev-parse main origin/main
```

mainが古い場合は、作業が保全されていることを確認してから`git switch main`、`git merge --ff-only origin/main`を使う。分岐していれば履歴を調べて対応する。

## このスレッドで合意した考え方

- 誤差評価の目的は、**何が不確実なのかを表現し、次に何を確かめるか考えること**。BP向けには「必要だから必ずやれ」という指導の口調にしない。
- 点数だけでなく、入力・分割・選定・予測・学習履歴・反復・計算条件を残す。自作の利点は、自分が必要とする途中経過を保存し、問題の箇所へ戻れることにもある。
- Iris等の定番データは仕組みを確かめる入口。その後、性質を十分に知らないデータでも、統計量・欠損・重複・記録の意味・誤差を調べ、データ品質を議論できる方向へ進む。区間幅だけから品質不良やデータ不足を自動診断しない。
- 統計的な不確かさ、数値計算誤差、系統的な偏りを入口にし、学習seed・分割・選定の変動も区別する。独立な3成分に普遍的に分解・合算できると説明しない。系統的な偏りは、モデルの論文だけで手元の測定・ラベル・用途について評価済みにはならない。
- 既存AutoMLにも不確かさを扱う機能がある。「他にはないから作る」「普段の利用者は使っていない」と市場全体や利用実態を断定しない。自分が確認したい対象・条件・過程を一続きで記録することを動機にする。
- 図は**最初に広い軸範囲で全体を確認し、その後にズームする**。輪郭や外れ値の見切れを確認し、拡大範囲を明記する。親記事の相関図の最終表示は両軸0.80〜1.05。0〜1で楕円を切り落とす制限は戻さない。
- 素粒子実験で誤差を気にする視点は導入に使うが、著者の経歴・実験経験を創作しない。誤差伝搬の公式は今回は採用せず、今後の検討とする。
- 記事へのリンクはZenn、コード・手順・Excel・検証根拠はGitHubへ。記事用補足集計、実装済み機能、将来構想を明確に分ける。
- 計算済みの条件付き区間を、再学習・別集団・別期間・系統/数値誤差まで含む品質保証や、個々の予測が正しい確率と説明しない。区間が0をまたぐ差を、同等性の証明にしない。

## 3記事で完了したこと

| 原稿 | 変更と読む場所 |
| --- | --- |
| [親記事](../articles/diy-automl-evaluation-framework.md) | 誤差の重要性→途中経過→定番サンプルの結果→誤差を添えた読み方→データ品質へ再構成。点のみ/区間付き、学習履歴、2指標の連動を実測図で示す |
| [DS記事](../articles/diy-automl-ds-uncertainty.md) | 誤差の分類・偏り、既存ライブラリの根拠、図の表記・数式・再計算を整理。DS志望者向けの用語説明と、時系列の誤差評価の今後の課題を追加 |
| [BP記事](../articles/diy-automl-business-three-experiments.md) | 実際に受け取るIrisのExcelを先に示す。3業務例を件数・差の不確かさ・次の確認で説明。トレンド分析・時系列への対応は今後の予定へ |

利用者から指定された掲載先は以下。Git原稿の`published: false`は保持している。

- 親：https://zenn.dev/okamasaya/articles/diy-automl-evaluation-framework
- DS：https://zenn.dev/okamasaya/articles/diy-automl-ds-uncertainty
- BP：https://zenn.dev/okamasaya/articles/diy-automl-business-three-experiments

親記事は利用者から公開完了の共有があった。DS/BPについても掲載先は共有済みだが、各改稿のZenn側への反映完了を自動で推定しない。公開本文の取得は環境のプロキシ403で照合できていない。GitHubへの保存・mergeとZennの公開操作は別で、Zenn側の更新は利用者が行う。

## 図・出力・再現の入口

### 親記事の3図

- 描画コード：[render_evaluation_article_figures.py](../scripts/render_evaluation_article_figures.py)。保存済み3データ・全2学習seedの結果を使用する。
- 図：[区間比較](../images/diy-automl/evaluation-with-and-without-uncertainty.png)、[学習履歴](../images/diy-automl/evaluation-training-history.png)、[2指標の相関](../images/diy-automl/evaluation-joint-bootstrap.png)。
- 条件・入力ハッシュ：[evaluation-article-data.json](figures/evaluation-article-data.json)。

相関図はPrecision/Recallのbootstrap反復点を約90/95/99%含む経験的な楕円であり、母数の信頼領域ではない。広い0.75〜1.25で輪郭を確認後、利用者指定の0.80〜1.05へズームした。楕円が指標上限1を超える部分は点群の幾何学的要約で、その値の指標を観測した意味ではない。

### DSの根拠

[DS_UNCERTAINTY_EVIDENCE.md](DS_UNCERTAINTY_EVIDENCE.md)に、scikit-learn 1.8.0・AutoGluon v1.4.0・PyTorch v2.9.0の公式資料6件と確認範囲を記録した。

AutoGluonで裏付けた`stddev`は、`feature_importance()`の置換反復のSDである。利用者が実際に見たfitログの項目を特定したわけではない。平均の区間のSD/√nと、DIY AutoMLのbootstrap SE（反復スコアの標本SD）を混同しない。

### BPのIris見本

- [日本語要約付きExcel](../validation/business-output-20261112/iris/iris_business_report.xlsx)／[先頭シートのPDF](../validation/business-output-20261112/iris/summary.pdf)／[入力CSV](../validation/business-output-20261112/iris/input.csv)。
- [実際のExcelを表示した画像](../images/business-output-20261112/iris-report.png)。LibreOfficeでxlsxからA4・1ページのPDFへ書き出した実物で、画面のモックではない。
- [要約とハッシュ](../validation/business-output-20261112/iris/example_summary.json)、[生成コード](../scripts/create_bp_iris_example.py)、[再作成手順](BP_OUTPUT_EXAMPLE.md)。

150件のうち120件を学習側に使って5モデルを比較し、選んだSVMを保留した30件で評価。29件正解・1件誤り、Accuracy=96.7%、macro F1=0.966583、bootstrap SE=0.032765、条件付き95%区間=[0.897698, 1.000000]。誤りはヘッダを1行目と数えるCSVの79行目、versicolor→virginica。

先頭3シートは「まず読む」「間違えた行」「入力と条件」。既存の誤差評価表・区間図は保持した。日本語要約を加える部分はIris専用の記事用アダプタであり、一般CSV向けのGUI・汎用の説明生成は未実装。親・DSの固定MLPのIris例とは別の結果なので、数値を混ぜない。この見本のCSVはGitの改行自動変換を抑止し、元のハッシュを保持している。

### BPのモデルと参考ルールの差

[元の3テーマの結果](../validation/business-quick-20261108/)を保持し、保存済み予測から同じ行を使うpaired bootstrapを記事用に追加した。[計算条件](../validation/business-output-20261112/comparison/comparison.json)、[反復index・差](../validation/business-output-20261112/comparison/paired_bootstrap.npz)、[全体図](../validation/business-output-20261112/comparison/overview.png)、[拡大図](../images/business-output-20261112/model-rule-difference.png)、[コード](../scripts/render_bp_comparisons.py)を保存している。

差はmacro F1の「モデル−参考ルール」。固定モデル・正解クラス別件数固定・2,000反復・seed 43で、学習や選定からはやり直していない。

| 題材 | 差 | 条件付き95%区間 | 読み方 |
| --- | ---: | --- | --- |
| ホテル | −0.044229 | −0.108544〜＋0.017537 | 点はルールが上。差の方向を明確には判断できない |
| カフェ券 | ＋0.040349 | ＋0.001280〜＋0.078437 | 行単位の仮定で正の差。同じ回答者の依存を扱えておらず、新しい人への優位は未確認 |
| 通販 | ＋0.010542 | −0.041616〜＋0.058960 | 優位も同等性も確認できていない |

まず差の全範囲−1〜1を確認し、−0.13〜0.13へ拡大した。ホテルの顧客/団体等の依存も未評価。パッケージ本体の汎用モデル比較機能を追加した意味ではない。

## 実施した検証と、その限界

- 親記事：保存済み学習3出力・誤差評価3出力を既存検証器で確認。点推定と一部反復をscikit-learnで24照合し、入力不変を確認。ローカルプレビューで1280px/390px・3画像を確認。
- DS記事：Irisの1,000反復indexが完全一致、全F1は絶対許容差1e-12以内。CLI/APIの実行・結果照合、4画像・数式31か所の表示を確認。[検証JSON](../tests/results/editorial-ds-uncertainty-20261112.json)と[時系列追記の確認](../tests/results/editorial-timeseries-roadmap-20261112.json)を分けて保存。
- BP記事：選定schema 3、図付きExcel schema 8、誤差評価schema 9の検証器を実行。scikit-learnで指標・誤り行を照合。元シートのセル値と埋込み区間PNGの保持、PDFの数値、補足比較のindex・区間・独立9反復を確認。[検証JSON](../tests/results/editorial-bp-output-20261112.json)に記録。1280px/390pxで全3画像・リンク・横はみ出し・JavaScriptエラーを確認。
- ローカルZenn確認はCLI 0.5.14とChromium/Playwright。DSではCDN取得とCLIの幅に検証環境側の対応を行ったため、詳細は裏付け文書を読む。BPではCSS補正を使っていない。公開Zennそのものの表示検証とは分ける。
- アプリ本体と依存を変更していないため、この編集でパッケージ全テストを再実行したとは記録しない。既存版の全テスト記録は[0.4.0a7](../tests/results/0.4.0a7/verification.json)等を参照する。
- 過去の検証JSONの原稿ハッシュは当時のスナップショット。後の文書編集に合わせて過去の記録を書き換えない。

## 次に残ること

1. 利用者がZenn掲載版へ必要な改稿を反映する。Webエディタを使う場合は[Zenn手順](ZENN_GUIDE.md)に従い、画像をアップロードして本文の`/images/`参照を掲載先のURLへ置き換える。GitHubのExcel/PDFリンクも確認する。
2. 次の開発候補は[ROADMAP](ROADMAP.md)の「データ点検と評価条件の記録」。このスレッドでは汎用機能を実装していない。新しいスレッドで開発を始める際は、最新の利用者の依頼と対象範囲を確認してから進める。
3. その先は群・時間を考慮する分割と評価、学習曲線、学習・選定からの変動、数値条件と系統的な偏り、業務費用・閾値等。Irisの記事用要約や差の図を、そのまま汎用機能の完成と扱わない。

時系列については、過去のトレンド・季節性の整理、未来の値の予測区間、性能指標の区間、施策の因果効果を分ける。台風の予報円は理解の入口だが「最大でこの範囲内に収まる」保証とはしない。rolling-origin、先の長さ別の被覆率・幅、依存を保つ再標本化、時点別と経路全体の被覆が課題で、機能・実験・期限は未確定。

PyTorch/JAX、Julia/Rust版、新しい大規模モデル、GPU実験、回帰、汎用GUI、PyPI/TestPyPI公開、価格・ライセンスは今後の範囲。このまとめの作成によって新たに実装・公開を開始した意味ではない。

## 実行環境の再開に必要なこと

この会話ではPython 3.12/Linux CPUの`.venv`を使用した。新しい環境に`.venv`やブラウザ、LibreOffice、Node、`/tmp`等の一時ファイルが残っているとは限らない。リポジトリ内の入力・出力・コード・手順を基準に再作成する。クラウドの環境IDや認証を前の会話から流用しない。

Python環境は[README](../README.md)の手順を使用する。SL/ULのコアのみなら`requirements-core.lock.txt`、TensorFlowを含む実装検証なら`requirements-local.lock.txt`を使う。新規学習・描画の出力先には未作成のディレクトリを指定し、保存済み検証結果を上書きしない。

記事のローカル表示はリポジトリルートで`npx --yes zenn-cli@0.5.14 preview`。BPのExcel/PDF再作成で使う追加ツールとコマンドは[BP_OUTPUT_EXAMPLE.md](BP_OUTPUT_EXAMPLE.md)へ記録済み。

GitHub操作はこの環境でnative Gitによるfetch/push/cloneが成功した。GitHub API/ghはForbiddenだったため使用していない。新しい環境では接続・認証・ネットワーク条件を再確認する。秘密情報や一時パスを記事へ追加しない。

## このスレッドの主な保存履歴

| 内容 | 内容コミット | mainへのmerge |
| --- | --- | --- |
| 親記事・実測図・記事リンク | `81feaa9` | `1d2d65a` |
| 楕円図を広く表示・切落し解除 | `968199b` | `9f684c4` |
| 全体確認後、0.80〜1.05へズーム | `46fe48c` | `ba0bb66` |
| DS記事の根拠・数式・再計算 | `6b764e0` | `69f63ca` |
| 時系列の誤差評価を今後の課題へ | `2a67377` | `988983c` |
| BP記事・Iris実物・差の図 | `da71286` | `57412b5` |

引継ぎの継続方針は`99d8523`、修正方針の文書は`e7d7e07`にある。本まとめは`docs/thread-handoff-20261112`で保存し、mainへ統合する。対応する最新SHAは`git log --oneline -- docs/THREAD_SUMMARY_20261112.md docs/HANDOFF.md`で確認できる。

## 新しいスレッドへ渡す短い文

> okamasaya/DIY_AutoMLのmainから再開してください。最初にdocs/HANDOFF.mdとdocs/THREAD_SUMMARY_20261112.mdを読んで、作業ツリー・最新コミット・リモートを確認してください。親・DS・BP記事の改稿とIrisの実物Excel/PDFはmainへ統合済みです。何が不確実なのかと途中経過を残す方針、グラフは広い範囲から確認してズームする方針を継続してください。時系列・汎用のデータ点検・GUI等は今後の課題です。次の作業はこの引継ぎと最新の依頼に合わせて進めてください。
