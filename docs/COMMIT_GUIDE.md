# Gitへの保存と次の環境での再開

2026年11月8日（日本時間）、利用者から「codexでgitへのcommitをやってみてください」と承認を受け、まず専用の復旧ブランチへ保存しました。その後の「今のrecoverをメインに移動しても良いかな」という依頼により、mainを変更しない旧方針を更新し、成果物をmainへ反映しました。現在の通常の取得先はmainです。

- リポジトリ: https://github.com/okamasaya/DIY_AutoML
- 現在の案内・取得先: `main`
- 移行前の保管ブランチ: `recovery/reimplemented-0.4.0a1-post1`（`b7e8a93`）
- 開始時のmain: `b4f6f176182c7c10001ca5c82cbed7fe269c1f8e`（.gitkeepのみ）
- mainの初期commitから復旧ブランチまでは分岐がなく、履歴をそのまま引き継ぐfast-forwardで反映しました。案内先の修正はその上に文書commitとして追加しています。
- 復旧ブランチは残し、履歴の書換え・force push・削除は行いません。既定ブランチは元からmainです。次の機能追加は別の作業ブランチで進めます。

commitはローカルGitへの記録、pushはGitHubへの転送です。mainへの反映後、`git rev-parse main` と `git ls-remote origin refs/heads/main` のハッシュが一致することを確認します。さらに別ディレクトリへmainをcloneして、同じファイルを取得できるか検証します。手元へのダウンロード成功とは別の確認です。

## 新しい環境での再開

```sh
git clone --branch main --single-branch https://github.com/okamasaya/DIY_AutoML.git
cd DIY_AutoML
```

すでにこのリポジトリがある環境では、未保存の作業を確認・保全してからmainを取得します。以前の復旧ブランチだけのsingle-branch cloneでも取得できるよう、mainを追跡対象へ追加しています。

```sh
git remote set-branches --add origin main
git fetch origin
git switch main
git pull --ff-only origin main
```

mainがローカルにない場合、通常は`git switch main`がorigin/mainを追跡するブランチを作ります。ローカル独自の変更でfast-forwardできない場合は、作業を保全して差分を確認します。resetやforce pushで上書きする手順ではありません。

最初にREADME.md、docs/RESET_HANDOFF.md、docs/ROADMAP.md、docs/UL_REPEAT_GUIDE.md、docs/VALIDATION_UL_REPEAT.md、tests/TEST_PLAN.mdを読みます。コードはsrc、原本はoriginals、検証結果はtests/results、実行済み出力はvalidation/currentにあります。旧0.4の復元ではなく、元資料からの再実装です。

現在の版は0.4.0a7です。DLバックエンド追加の準備と検証はdocs/DL_BACKENDS.md、docs/VALIDATION_DL_BACKEND_PREPARATION.md、tests/results/0.4.0a7にあります。0.4.0a6当時の開発・独立wheel環境の各176 passedと48出力の照合記録はtests/results/0.4.0a6、誤差評価の手順はdocs/METRIC_UNCERTAINTY_GUIDE.mdです。mainへの移行自体は文書とGitの参照先の変更でした。以下の23 passed等は初回再実装時点の記録です。

Python 3.12/Linux CPU環境を作る場合:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

DLはこの実行環境でTensorFlow CPU 2.20.0/Keras 3.15.1による100epochの学習を確認しました。別環境でのDL実行可否は、その環境の依存導入と実行結果で確認してください。今回の検証時点では23 passed、独立wheel環境でも23 passedです。詳細はtests/results/validation.jsonに保存しています。0.4.0a2のDL反復に関する追加結果はtests/results/0.4.0a2とdocs/VALIDATION_DL_REPEAT.mdを参照してください。0.4.0a3のUL部分抽出の追加結果はtests/results/0.4.0a3とdocs/VALIDATION_UL_SUBSAMPLE.mdを参照してください。0.4.0a4のSL探索と記事手順の検証はtests/results/0.4.0a4、記事下書きはarticlesに保存しています。0.4.0a5の図付きExcel/Notebook実行と記事分割方針はdocs/EXCEL_REPORT_GUIDE.md、docs/VALIDATION_EXCEL_REPORT.md、docs/AUDIENCE_PLAN.mdを参照してください。このJSON内のcommit_pushは検証実行時点の記録であり、その後のGit保存状況は実際のGit履歴で確認してください。

.venv、キャッシュ、build、distは.gitignoreで除外しています。原本と参照CSVは.gitattributesで改行変換を無効にし、ハッシュを保ちます。wheelはソースから再作成できます。バックアップを作る場合:

```sh
.venv/bin/python scripts/create_backup.py --output /保存先/DIY_AutoML-reimplemented.zip
```

ZIPにはGitメタデータ・仮想環境を入れず、全ファイルのSHA256を持つBACKUP_MANIFEST.jsonを含めます。利用者が実際に取得できたことは別途確認が必要です。
