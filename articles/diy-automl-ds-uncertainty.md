---
title: "DIY AutoMLの誤差評価を検査する：選定・条件付き区間・反復の設計"
emoji: "📐"
type: "tech"
topics: ["python", "statistics", "machinelearning", "scikitlearn", "automl"]
published: false
---

## スコアと一緒に、推定対象を記録する

DIY AutoMLは、比較結果に誤差評価を含め、データ量・品質・学習の不安定さについて議論できる道具を目指しています。[親記事](https://github.com/okamasaya/DIY_AutoML/blob/main/articles/diy-automl-evaluation-framework.md)から、ここでは推定対象と実装条件まで掘り下げます。[ビジネス向け](https://github.com/okamasaya/DIY_AutoML/blob/main/articles/diy-automl-business-three-experiments.md)と同じ根拠を使い、説明の粒度を変えています。

対象は開発中の **0.4.0a6** です。`select/tune`のtest区間は、学習済みモデルと観測クラス別件数に条件付けています。新しい`uncertainty`は保存済みSL/DLへSE・クラス別区間を広げ、クラス件数固定/通常の行bootstrapを選べます。選定を含む学習手続き全体の不確かさや、系統誤差を網羅した区間は未実装です。現在の仕様と拡張設計を分けて記載します。

## DS版の完成形は、PyPIで配布するPythonパッケージ

DS版は、Pythonから駆動するパッケージとして整え、**PyPIへ公開してpipで導入できる形**を目指します。Notebook・CLI・Python APIのどれを使っても、共通の前処理・学習・評価・保存処理につながる構成です。

完成条件は、公開版を新しい環境へ導入し、ドキュメントの手順に沿って、モデル比較・誤差評価・Excel/NPZ/manifestの保存と結果検証まで進められることです。利用者が呼ぶAPI、版更新時の互換性方針、対応環境、誤差評価の対象と限界も明示します。

ビジネス向けは、同じ評価コアに接続する使い勝手重視の廉価版アプリを目指します。主要モデル・小規模な計算・少ない操作に絞り、持ち込んだデータを平易な説明と対比しながらExcelへ出す入口です。DS向けでは、設定・反復・計算条件・全保存値を詳しく検査できる入口を用意します。提供方法や価格は未確定です。

現在、wheelの作成と独立環境での導入・検証まで確認しています。PyPI/TestPyPIへの公開は今後です。配布名候補の`diy-automl`がPyPIで利用可能かは未確認で、コードのライセンスも未選定です。本稿の実行例は、引き続きGitHubから導入する開発版を対象にします。

配布は次の段階で進める方針です。

1. 公開名、ライセンス、配布メタデータ、対応環境、APIとサンプル手順を整える。
2. コアの導入と、TensorFlowを任意依存として追加する導入を確認する。
3. TestPyPIでクリーンな環境への導入と、評価・出力・検証の手順を確認する。
4. PyPIへα版を公開し、宣言した機能・API・配布手順が安定した段階で正式版へ進む。

最新モデルをすべてそろえることを公開条件にはしません。各版で対応する評価とその前提を明示し、追加の重いモデルは引き続き後続のTo Doとして扱います。

## 評価したい量と、動かすものを分ける

| 問い | 変化させるもの | 現状 |
| --- | --- | --- |
| 固定モデルのtest指標をどの程度精密に推定できるか | クラス件数固定、または通常のtest行の再標本化 | select/tuneの区間と、uncertaintyのSE・全体/クラス別区間を実装 |
| 学習手続きから得る性能はどれほど変動するか | 学習データ、前処理fit、候補選定、再学習 | 手続き全体の反復評価は今後 |
| 同じデータでも学習結果が変わるか | 初期化・shuffle等の学習seed | DL/ULの指定した反復を実装 |
| 数値計算条件へ敏感か | dtype、収束許容値、ソルバ、計算基盤 | 一部の診断・条件保存まで。感度評価は今後 |
| 実際に使う集団や時期で評価できているか | 顧客・施設・時期、測定・ラベルの照合 | 個別実験で制約を記載。汎用の監査・分割は今後 |

予測誤り、性能推定の標本変動、アルゴリズムの乱数、数値計算誤差、系統的な偏りは同じ量ではありません。この表も、それらを独立な成分へ完全分解したものではありません。相互作用や、データだけでは識別できない要因が残ります。

## 選定と最終評価の境界

`select`は固定5モデル、`tune`は宣言した27設定が対象です。約20%を層化testとして最初に分離し、学習側で共通の層化5foldを使います。各foldのStandardScalerはfoldの学習部分だけでfitします。既定主指標はmacro F1です。

候補$j$のfoldスコアを$s_{jk}$とすると、選定に使う値は次です。

$$
\bar{s}_j=\frac{1}{5}\sum_{k=1}^{5}s_{jk}
$$

プールしたout-of-fold予測のF1とは一般に一致しません。最大値との差が絶対許容値$10^{-12}$以内（`rtol=0`）の候補を同率として残し、宣言した候補順の先頭を採用します。testや時間で同率を解消しません。選定後は学習側全体でscalerとモデルをfitし直し、その1モデルだけをtestで評価します。

fold SDは標本SD（`ddof=1`）を保存します。foldは学習データを共有しているため、SDを$\sqrt{5}$で割って、そのまま性能のSEMや信頼区間にはしません。選定に使ったCV最大値には選定による楽観性もあります。独立したtestによる評価と、探索過程の記述を分けます。[nested CVの参考資料](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

## 条件付きpaired percentile bootstrapの実装

testの正解・予測を$(y_i,\hat{y}_i)$、クラス$c$の行index集合を$I_c$、その件数を$n_c$とします。各反復$b$では、各$I_c$から$n_c$行を復元抽出し、それらを結合した$I^{*(b)}$を使います。

$$
T^{*(b)}=T\left(\{(y_i,\hat{y}_i):i\in I^{*(b)}\}\right)
$$

正解と予測は同じindexで取り出します。ここでのpairedはその対応であり、モデル間差の検定を実装した意味ではありません。既定$B=1000$、seed=43、信頼水準95%で、区間端点は次の経験分位点です。

$$
\left[Q_{0.025}(T^*),\;Q_{0.975}(T^*)\right]
$$

現在の実装はpandasの既定の線形補間による分位点を使います。Accuracyとmacro Precision/Recall/F1を計算し、macro対象クラスを点推定と反復でそろえ、`zero_division=0`とします。これはクラス別F1の平均であり、macro Precision/Recallの調和平均ではありません。

`uncertainty`は同じ線形補間をNumPyで計算し、bootstrap SEも保存します。

$$
\widehat{\mathrm{SE}}_{\mathrm{boot}}(T)=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(T^{*(b)}-\bar{T}^{*}\right)^2}
$$

これは評価指標の標準誤差のbootstrap近似で、SD/√Bではありません。後者はbootstrap平均を有限反復で近似するMonte Carlo誤差に関係する量です。反復数を増やすことと評価標本を増やすことは別です。クラス別Precision/Recall/F1も計算し、すべての再標本化indexとクラス別反復値をNPZへ保存します。

`--resampling ordinary`では各反復で全test行から同数を復元抽出し、クラス件数も変動します。macroのクラス集合と`zero_division=0`は固定です。反復内でクラスが消える場合もあり、通常方式にすればクラス不足や行の依存が解消するわけではありません。各指標の周辺区間であり、複数モデルや指標に対する同時被覆を主張しません。

主な解釈条件は次です。

- **学習済みモデルを固定**。学習・選定・再学習をやり直した変動は含みません。
- **観測クラス別件数を固定**。クラス割合が変わる変動は含みません。あるクラスのtest観測が0なら、そのクラスの情報は復元できません。
- **行単位の再標本化**。顧客や回答者の繰返し、時系列依存へ自動対応していません。必要な再標本化単位が行とは限りません。
- **有限反復のpercentile法**。被覆率が常に名目95%になる保証はなく、離散性・小標本・退化によるゼロ幅も起こりえます。反復数を増やしても、独立した評価データは増えません。

この区間を個別予測の確率、将来の観測値の予測区間、学習手続き全体の区間と読み替えません。1つの区間を計算した後で「真の値が95%の確率でここに入る」とする解釈も使いません。

実装は[selection.py](https://github.com/okamasaya/DIY_AutoML/blob/main/src/diy_automl/selection.py)、分割・保存との接続は[selection_workflow.py](https://github.com/okamasaya/DIY_AutoML/blob/main/src/diy_automl/selection_workflow.py)で確認できます。

## seed反復と数値計算の検証を混同しない

`repeat-dl`ではtrain/validation/testのindexとtrainだけでfitしたscalerを共有し、学習seedを変えます。初期化やshuffleに対する感度の観察であり、最高test値のseedを採用しません。`repeat-ul`は固定データのKMeans/GMM初期化感度、`subsample-ul`は部分集合ごとに標準化からやり直すデータ摂動です。

ULでは正解とのTarget ARIと、割当どうしのAssignment ARIを分けます。10seedから得る45対はseedを共有するため、45独立標本としてSEM/CIを作りません。高い割当一致だけで、クラスタの業務上の有用性も決まりません。

数値計算については別の検証が必要です。今後は、同じ入力・分割・学習seedで、比較可能な範囲の計算精度や収束許容値を変え、目的関数・収束状態・予測・指標の変化を対応付ける方針です。異なるソルバは収束先も変えうるため、観測差をすべて丸め誤差と呼びません。現在は一部の収束診断と失敗状態の保存までで、この比較は実装していません。

開発環境と独立wheel環境の配列一致は、実装・配布の再現確認です。同じ問題設定に共通する数値誤差や偏りの不在、他OS/GPUでの一致まで証明するものではありません。

## 系統的な偏りは、照合する基準も必要になる

リーク、測定の校正ずれ、ラベルの誤り、選択された集団、期間・施設の違いを点検します。モデルと正解ラベルの不一致だけを、そのままラベル誤りと判定しません。原票、追加測定、独立した再判定などが必要になる場合があります。未確認の偏りを「0」として誤差伝播へ入れません。

公開データの3テーマでも、具体的な制約がありました。

| データ | 現在の制約 | 次に必要になる検証 |
| --- | --- | --- |
| ホテル | 到着前日の情報抽出。行分割では予約受付時点・別施設での妥当性は未確認 | 予測時点で利用可能な列を確定し、施設や期間を分ける |
| カフェ券 | 同一回答者の複数回答だが回答者IDがない | 回答者単位で分離できるデータ。実購買を問うなら別の正解も必要 |
| 通販 | 1顧客1行、1つの観測・追跡期間 | 予測時点をずらす時間外評価。取消を除いた正の売上という集計定義も点検 |

カフェ券のF1=0.692、行単位の条件付き区間0.647〜0.739を、新しい回答者に対する妥当な95%区間とは扱いません。[実験条件と結果](https://github.com/okamasaya/DIY_AutoML/blob/main/docs/BUSINESS_QUICK_RESULTS.md)に、取得元・加工・評価単位を残しています。

## NotebookでもCLIでも、同じ評価経路を使う

環境の導入は[親記事の手順](https://github.com/okamasaya/DIY_AutoML/blob/main/articles/diy-automl-evaluation-framework.md)を使います。以下はリポジトリのルートから実行し、出力先は未作成のフォルダにします。

```sh
.venv/bin/python -m diy_automl select --csv data/reference/wine.csv --target target --output runs/ds-select
.venv/bin/python scripts/validate_selection_run.py runs/ds-select
.venv/bin/python -m diy_automl report --run runs/ds-select --output runs/ds-select-figures
.venv/bin/python scripts/validate_excel_report.py runs/ds-select-figures
```

保存済み予測の誤差評価は、次の入口で別出力します。`uncertainty`の`report.xlsx`には最初から区間図を埋め込みます。

```sh
.venv/bin/python -m diy_automl uncertainty --run runs/ds-select --output runs/ds-uncertainty
.venv/bin/python scripts/validate_uncertainty_run.py runs/ds-uncertainty
```

Notebookで同じ選定を呼ぶ場合は、同じ環境をkernelにして、リポジトリのルートを作業ディレクトリにします。

```python
from diy_automl import select_csv

result = select_csv(
    "data/reference/wine.csv", target="target", output="runs/ds-select-notebook",
    seed=42, metric="F1", bootstrap_repeats=1000,
    bootstrap_seed=43, confidence=0.95,
)
```

CLIの`select`もこの関数を呼びます。上の例は同じ設定ですが、今回の原稿編集に伴って再実行したコードではありません。既存の[実行済み比較Notebook](https://github.com/okamasaya/DIY_AutoML/blob/main/notebooks/Local_Comparison.ipynb)は探索的なSL/UL/DL比較で、`select`のCV選定とは入口が異なります。

`select`のschema 3では、7シートの数値Excel、分割・foldのindex、test正解と予測、scaler情報、設定とハッシュを保存します。bootstrapの全反復値はExcelへ残します。`report`は別のschema 8成果物として図付きExcelを作り、元の評価結果を上書きしません。検証器は保存結果の整合性を確認するもので、標本の代表性や業務での妥当性まで認定するものではありません。`python -O`でassertを無効にせず実行します。

## 次に拡張する評価

優先するのは、評価単位とデータ監査、固定モデルの学習曲線、選定を含む手続き全体の反復、グループ・時間を考慮した評価です。学習曲線では最終testを使わず、学習部分集合ごとに前処理をfitします。候補選定からやり直す場合は、外側の評価と内側の選定を分けます。

低い性能で曲線が頭打ちになっても、品質不良と一意に診断はできません。特徴量・モデルの適合、ラベル、本来的な予測困難さなどを照合します。確率を業務の判断へ使う場合は、[校正](https://scikit-learn.org/stable/modules/calibration.html)と[判定閾値](https://scikit-learn.org/stable/modules/classification_threshold.html)も別に評価します。Brier/log lossは校正だけの尺度ではありません。現行では確率校正・費用・閾値の最適化は未実装です。

これらの設計は[UNCERTAINTY_DESIGN.md](https://github.com/okamasaya/DIY_AutoML/blob/main/docs/UNCERTAINTY_DESIGN.md)に記録しています。重い追加モデルは引き続きTo Doです。比較対象を広げる際も、推定対象・分割・計算予算をそろえ、testを見て目的・候補・seedを選び直しません。
