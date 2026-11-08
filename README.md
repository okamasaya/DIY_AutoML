# DIY AutoML 再実装版

旧クラウド環境の0.4.0a1ソースは未取得です。このリポジトリは元資料ZIPと引継ぎ仕様から新たに実装しました。初回再実装は **0.4.0a1.post1**、DL反復を追加した版は **0.4.0a2**、UL部分抽出は **0.4.0a3**、SLの設定探索を追加した現在版は **0.4.0a4** です。旧実装とのAPI・数値・manifest互換性は保証していません。schema番号2/3/4は引継ぎの用途区分を採用したもので、旧validatorとの互換を意味しません。過去の96 passedは今回の検証結果ではありません。

Python 3.12、数値特徴量・欠損なし・分類ラベル付きCSVが対象です。ULも外部診断のためtarget列が必要です。元資料は `originals/` に変更せず保存しています。ライセンスは未選定です。

## セットアップ

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

ロックは今回のPython 3.12/Linux CPU環境のものです。旧環境のロックを復元したものではありません。TensorFlowを使わない場合は `pip install -e .` でコアのみ導入できます。DL用の追加依存は `pip install -e '.[dl]'` です。

## 実行

```sh
.venv/bin/python -m diy_automl run --csv data/reference/wine.csv --target target --output runs/wine
.venv/bin/python -m diy_automl select --csv data/reference/wine.csv --target target --output runs/wine-select
.venv/bin/python -m diy_automl repeat-ul --csv data/reference/wine.csv --target target --seeds 42 43 44 --output runs/wine-repeat
.venv/bin/python scripts/validate_selection_run.py runs/wine-select
```

出力フォルダは新規のみ使用し、既存出力を上書きしません。Excel、配列NPZ、入力・ソース・出力ハッシュ付きmanifestを保存します。失敗時はmanifestをfailedにします。UL反復では完了・失敗・未実行を記録し、部分結果も保存します。

`run` は探索的比較です。SL5モデル、UL54候補、DL単一MLPを実行します。`--methods sl ul` でDLを省略できます。SLはtestをモデル選定に使いません。モデル選定が目的なら `select` を使用してください。

`select` は学習側の共通層化5foldのmacro F1平均で選定し、選定した1モデルだけtest評価します。数値同率はatol=1e-12、rtol=0で残し、候補順を採用します。testスコア・時間では解消しません。層化paired percentile bootstrapは正解クラス別件数を固定した1,000反復、95%、seed=43が既定です。固定済みモデル・観測クラス件数に条件付きであり、学習・選定の変動を含みません。ゼロ幅も未知の不確かさゼロを意味しません。

DLはtrainだけでscalerをfitします。`--split-seed` と `--training-seed` を独立に指定でき、省略時は両方とも `--seed`（42）を使います。分割は概ね64/16/20、MLP32-16-8、100epoch、batch16です。test最高のseedを選ぶ入口はありません。

ULは固定データ最大5,000行、KMeans n_init=auto、GMM n_init=1/full、階層4linkage、k=2..10です。UL反復のTarget ARIとAssignment ARIは別の量です。45 seed対を独立標本としたSEM/CIや自動選定は行いません。fold SD、点ごとのSilhouette SD/SE、seed SD、test区間も区別します。

`notebooks/Local_Comparison.ipynb` は共通関数を呼ぶ5シートと図5枚のローカル入口です。Excel画像埋込は未実装です。組込み3データセットは開発用で、未見の外部検証データではありません。SL/DLとULを総合順位にしません。

復旧範囲と今回の検証結果は `docs/RESET_HANDOFF.md` と `docs/VALIDATION_UL_REPEAT.md` を参照してください。利用者の承認により、今回の成果物を専用ブランチ `recovery/reimplemented-0.4.0a1-post1` にcommit・pushする方針へ変更しました。mainブランチには変更を加えません。

Notebookを使う場合は、必要に応じて `.venv/bin/python -m pip install ipykernel` を追加し、エディタでこの.venvをカーネルに指定してください。Notebookカーネルを通した実行は今回未確認ですが、共通関数での学習・出力と描画関数での図5枚生成は確認しています。

Gitからの再開手順と、ローカルコミット・pushの確認方法は `docs/COMMIT_GUIDE.md` に記載しています。

## DLの学習seed反復（0.4.0a2）

固定した分割とtrain-only scalerを共有し、全学習seedの指標・履歴・予測・重みを保存する `repeat-dl`（schema 5）を追加しました。初期化とshuffleの感度を記述し、test最高のseedは選びません。

```sh
.venv/bin/python -m diy_automl repeat-dl --csv data/reference/wine.csv --target target --training-seeds 42 43 44 --output runs/wine-repeat-dl
.venv/bin/python scripts/validate_dl_repeat.py runs/wine-repeat-dl
```

詳細は `docs/DL_REPEAT_GUIDE.md`、今回の検証は `docs/VALIDATION_DL_REPEAT.md` を参照してください。初回再実装の検証記録は上書きせず保持しています。

## ULの部分抽出による安定性（0.4.0a3）

`subsample-ul`（schema 6）は80%の行を非復元抽出し、各部分集合だけで標準化と学習をやり直します。抽出seedとモデルseedを分離し、全データ参照とのARI、反復間の共通行だけのARI、元行index・scaler・全割当を保存します。少ない共通行や単一クラスタの比較は状態を明示して未採点とします。自動選定やSEM/CIはありません。

```sh
.venv/bin/python -m diy_automl subsample-ul --csv data/reference/wine.csv --target target --sample-seeds 42 43 44 --output runs/wine-subsample
.venv/bin/python scripts/validate_ul_subsample.py runs/wine-subsample
```

新入口の上限は2,000行・100特徴量・参照を含む2,000候補fitです。詳細は `docs/UL_SUBSAMPLE_GUIDE.md`、検証結果は `docs/VALIDATION_UL_SUBSAMPLE.md` を参照してください。旧run/repeat-ulの動作と上限は維持しています。

## SLの設定探索と紹介記事（0.4.0a4）

`tune`（schema 7）は宣言した27設定を学習側の共通5foldで比較し、選定した1設定だけを再学習・test評価します。各foldのtrain-only scaler、同率時の候補順、条件付きbootstrapを使います。詳細は `docs/SL_TUNING_GUIDE.md`、検証結果は `docs/VALIDATION_SL_TUNING.md` に記載しています。

```sh
.venv/bin/python -m diy_automl tune --csv data/reference/wine.csv --target target --output runs/wine-tune
.venv/bin/python scripts/validate_tuning_run.py runs/wine-tune
```

TensorFlowを導入せずSL/ULを試す場合は、`requirements-core.lock.txt` を導入してから `pip install --no-deps -e .` を実行できます。DLを含む全テストは従来の `requirements-local.lock.txt` を使用してください。

Zenn用の新規紹介記事は `articles/diy-automl-evaluation-framework.md`、図は `images/diy-automl/selection-flow.png` に保存しています。`published: false` の下書きで、Zennへの投稿・公開は行っていません。操作と確認範囲は `docs/ZENN_GUIDE.md` を参照してください。
