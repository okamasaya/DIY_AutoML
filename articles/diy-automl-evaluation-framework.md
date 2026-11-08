---
title: "PythonでSL・UL・DLの比較と検証を追える自作評価ツールを作る"
emoji: "🧪"
type: "tech"
topics: ["python", "scikitlearn", "machinelearning", "automl", "codex"]
published: false
---

## 比較結果を、あとから説明できるようにしたい

分類モデルをいくつか実行すると、AccuracyやF1を並べた表は作れます。ただ、その値を見て判断するには「どこで標準化したか」「何を基準に選んだか」「どのデータを評価に使ったか」も必要です。

Notebookで進めていた教師あり学習（SL）・教師なし学習（UL）・深層学習（DL）の比較を、Codexと作業しながらPythonパッケージにまとめています。名前は **DIY AutoML** です。

[GitHubの実装ブランチはこちらです](https://github.com/okamasaya/DIY_AutoML/tree/recovery/reimplemented-0.4.0a1-post1)。

「一番よさそうなモデルを自動で出す」ことより、**比較の手順と判断材料を残すこと**に重点を置いています。Excelに加えて、分割index・予測・クラスタ割当などのNPZと、設定・版・ハッシュを記録するmanifestを保存します。

この記事は開発中の **0.4.0a4** を対象としています。現在の実装は上記の専用ブランチにあります。mainは初期状態のため、下のcloneでもブランチを指定しています。元のクラウド環境のソース一式は取得できず、手元に残った元資料と引継ぎ仕様から再実装したものです。以前の制作物をそのまま復元したという扱いにはしていません。

## 入口を目的ごとに分ける

| コマンド | 確認したいこと |
| --- | --- |
| `run` | SL・UL・DLを探索的に比較する |
| `select` | SLの固定した5モデルから、学習側CVで1モデルを選ぶ |
| `tune` | SLの27設定から、学習側CVで1設定を選ぶ |
| `repeat-ul` | 同じデータでKMeans/GMMの初期化seedを変えたときの違い |
| `repeat-dl` | 分割を固定してMLPの学習seedを変えたときの違い |
| `subsample-ul` | データの一部を取り除いたとき、クラスタ構造が保たれるか |

SLはDecision Tree、Random Forest、Logistic Regression、SVM、k-NNです。ULはKMeans、GMM、階層法4種類で、既定k=2〜10の54候補。DLは32-16-8のMLPを使っています。

SL/DLの予測指標とULのクラスタ指標を混ぜた総合ランキングは作りません。たとえばSilhouetteが高いことと、正解ラベルによく一致することは、別の性質だからです。

## モデルを選ぶ前にtestを分ける

`select` と `tune` では、最初にtestを約20%分離します。残りの学習側だけで、共通の層化5foldを使います。各foldのStandardScalerはそのfoldの学習部分だけでfitし、validationにはtransformだけを適用します。

![学習側CVで設定を選び、保留したtestは最後の1モデルだけ評価する流れ](/images/diy-automl/selection-flow.png)

図は`tune`の27設定を示しています。`select`では同じ手順で固定5モデルを比較します。

主指標は既定でmacro F1です。5foldの値を単純平均し、差が絶対許容値1e-12以内なら同率として残し、宣言した候補順で採用します。testの値や学習時間で同率を解消しません。

`tune` の探索範囲は固定・小規模です。木の深さや葉の最小件数、Logistic Regression/SVMのC、SVMのkernel、k-NNの近傍数・重み付けを変える計27設定で、全パラメータを出力します。Random Forestの木の数は100本に固定しています。あらゆる設定を探索する汎用最適化器ではありません。

探索に使ったCVの最大値は、選定後の楽観的な値になりえます。CV平均自体を「未知データへの不偏な性能推定」として扱わないようにしています。

## 「ばらつき」を同じ意味にしない

testの区間は、固定した学習済みモデルに対して、正解クラス別の観測件数を固定したpaired percentile bootstrapで計算します。既定は1,000反復・95%・seed=43です。点推定と区間でmacro平均の対象クラスをそろえています。

この区間には、学習データやモデル選定をやり直したときの変動は含まれません。区間がゼロ幅でも、未知の不確かさがゼロという意味にはなりません。

ULの反復では、正解との一致である **Target ARI** と、割当どうしの一致である **Assignment ARI** を分けています。10seedから得る45対は互いにseedやデータを共有するので、独立標本としてSEMや信頼区間を作りません。

`subsample-ul` は80%の行を重複なしで抽出し、その部分集合だけで標準化と学習をやり直します。元の行indexで対応付け、反復間は共通行だけを比較します。比較範囲が単一クラスタになるケースや、共通行が少ないケースは状態を明示して未採点にします。非復元抽出なのでbootstrapとは呼びません。

`repeat-dl` はtrain/validation/testとscalerを共有し、初期化とshuffleを制御する学習seedだけを変えます。最高test値のseedを採用する処理はありません。

## TensorFlowなしで試す

まずSL/ULだけを動かせるよう、コア用のロックを分けました。以下はPython 3.12/Linuxで確認する手順です。PyPIからのパッケージ名によるインストールではなく、リポジトリを取得して導入します。

```sh
git clone --single-branch --branch recovery/reimplemented-0.4.0a1-post1 https://github.com/okamasaya/DIY_AutoML.git
cd DIY_AutoML
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
```

同梱のWine CSVで、まず固定モデルの選定を試します。

```sh
.venv/bin/python -m diy_automl select --csv data/reference/wine.csv --target target --output runs/wine-select
.venv/bin/python scripts/validate_selection_run.py runs/wine-select
```

設定探索を試す場合も、入力と出力の指定は同じです。

```sh
.venv/bin/python -m diy_automl tune --csv data/reference/wine.csv --target target --output runs/wine-tune
.venv/bin/python scripts/validate_tuning_run.py runs/wine-tune
```

`report.xlsx` には候補の設定、各foldの結果、平均・SD、選定した1モデルのtest結果、bootstrapと区間などが入ります。`arrays.npz` と `manifest.json` も一緒に残すことで、保存後に分割の混入や指標・設定の対応を検査できます。出力先は新規フォルダだけを使い、既存の実行を上書きしません。

探索的なSL/UL比較だけなら `run --methods sl ul` も使えます。DLも実行する場合は、READMEにある全依存用のロックでTensorFlow CPU/Kerasを導入します。

## どこまで確認したか

Iris、Wine、Breast Cancerの同梱CSVを、繰り返し参照する開発用データとして使っています。これらを未見の外部検証データとは扱っていません。

SLの探索では、開発環境と独立したwheel環境で、全候補のCV結果・選定・予測・保存配列などを照合しています。テスト件数や実行条件は、版ごとの検証資料と機械可読な記録に残しています。

[SL探索の検証記録](https://github.com/okamasaya/DIY_AutoML/blob/recovery/reimplemented-0.4.0a1-post1/docs/VALIDATION_SL_TUNING.md)

[DL反復の検証記録](https://github.com/okamasaya/DIY_AutoML/blob/recovery/reimplemented-0.4.0a1-post1/docs/VALIDATION_DL_REPEAT.md)

[UL部分抽出の検証記録](https://github.com/okamasaya/DIY_AutoML/blob/recovery/reimplemented-0.4.0a1-post1/docs/VALIDATION_UL_SUBSAMPLE.md)

同じCPU環境での一致確認であり、他OS・GPU・ライブラリ版での完全一致を保証するものではありません。また、指標が高いという理由だけで、そのモデルを実用途に使えると判断する機能はありません。

## 今後整えたいところ

現在はラベル付き・欠損なし・数値特徴量の分類用CSVが対象です。回帰、欠損やカテゴリ特徴量への一般的な前処理、ラベルなしULには対応していません。Excelへの画像埋込、Web画面、Windows実機・exe化、ライセンスの選定も残っています。

開発中の小さな道具ですが、学習・評価・表示を分け、結果の根拠を後から追える形にする過程も含めて紹介していきたいと思います。
