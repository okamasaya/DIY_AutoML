# Gitへの保存と次の環境での再開

2026年11月8日（日本時間）、利用者から「やってみましょう」「codexでgitへのcommitをやってみてください」と承認を受けました。以前のcommit/push保留方針を変更し、現在のコード・原本・資料・検証結果を専用ブランチへ保存する手順です。

- リポジトリ: https://github.com/okamasaya/DIY_AutoML
- 保存ブランチ: `recovery/reimplemented-0.4.0a1-post1`
- 開始時のmain: `b4f6f176182c7c10001ca5c82cbed7fe269c1f8e`（.gitkeepのみ）
- mainへのマージ、force push、他のブランチの変更は行いません。

commitはローカルGitへの記録、pushはGitHubへの転送です。実行後、`git rev-parse HEAD` と `git ls-remote origin refs/heads/recovery/reimplemented-0.4.0a1-post1` のハッシュが一致することを確認します。さらに別ディレクトリへcloneして、同じファイルを取得できるか検証します。手元へのダウンロード成功とは別の確認です。

## 新しい環境での再開

```sh
git clone --branch recovery/reimplemented-0.4.0a1-post1 --single-branch https://github.com/okamasaya/DIY_AutoML.git
cd DIY_AutoML
```

すでにこのリポジトリがある環境では、未保存の作業を確認・保全してから、通常のfetchとbranch切り替えを行ってください。mainのままでは今回の成果物は表示されません。

最初にREADME.md、docs/RESET_HANDOFF.md、docs/ROADMAP.md、docs/UL_REPEAT_GUIDE.md、docs/VALIDATION_UL_REPEAT.md、tests/TEST_PLAN.mdを読みます。コードはsrc、原本はoriginals、検証結果はtests/results、実行済み出力はvalidation/currentにあります。旧0.4の復元ではなく、元資料からの再実装です。

Python 3.12/Linux CPU環境を作る場合:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

DLはこの実行環境でTensorFlow CPU 2.20.0/Keras 3.15.1による100epochの学習を確認しました。別環境でのDL実行可否は、その環境の依存導入と実行結果で確認してください。今回の検証時点では23 passed、独立wheel環境でも23 passedです。詳細はtests/results/validation.jsonに保存しています。0.4.0a2のDL反復に関する追加結果はtests/results/0.4.0a2とdocs/VALIDATION_DL_REPEAT.mdを参照してください。0.4.0a3のUL部分抽出の追加結果はtests/results/0.4.0a3とdocs/VALIDATION_UL_SUBSAMPLE.mdを参照してください。このJSON内のcommit_pushは検証実行時点の記録であり、その後のGit保存状況は実際のGit履歴で確認してください。

.venv、キャッシュ、build、distは.gitignoreで除外しています。原本と参照CSVは.gitattributesで改行変換を無効にし、ハッシュを保ちます。wheelはソースから再作成できます。バックアップを作る場合:

```sh
.venv/bin/python scripts/create_backup.py --output /保存先/DIY_AutoML-reimplemented.zip
```

ZIPにはGitメタデータ・仮想環境を入れず、全ファイルのSHA256を持つBACKUP_MANIFEST.jsonを含めます。利用者が実際に取得できたことは別途確認が必要です。
