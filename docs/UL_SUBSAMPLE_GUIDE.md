# データの部分抽出によるUL安定性（0.4.0a3）

`subsample-ul` は、元データから毎回80%の行を重複なしに抽出し、前処理とクラスタリングをやり直す入口です。全データの参照モデルとの一致と、部分集合どうしの共通行での一致を測ります。bootstrapではありません。既存repeat-ulの「同じデータで初期化seedを変える」実験とは別です。

```sh
.venv/bin/python -m diy_automl subsample-ul --csv data/reference/wine.csv --target target --output runs/wine-subsample
.venv/bin/python scripts/validate_ul_subsample.py runs/wine-subsample
```

小さい試行やkを限定する場合:

```sh
.venv/bin/python -m diy_automl subsample-ul --csv data/reference/wine.csv --target target --sample-seeds 42 43 44 --fraction .8 --model-seed 42 --clusters 2 3 --min-overlap 10 --output runs/wine-small
```

Python入口は `subsample_clustering_csv(csv, target, output, sample_seeds=range(42,52), fraction=.8, model_seed=42, clusters=range(2,11), min_overlap=10)` です。各seedは0..2**32-1の整数、抽出seedは2件以上・重複なし、kは2..10の重複しない整数です。既存出力の上書きは行いません。

## 固定・変更する条件

- 抽出はtargetで層化しない一様非復元抽出です。各回の件数はfloor(全件数×fraction)です。実際の割合も保存します。
- 抽出後の元行indexを昇順に並べ、ランダムな行順変更の影響を持ち込みません。同じseedは同じindexを再現します。異なるseedでも同じ部分集合になる可能性はあります。
- StandardScalerは各部分集合だけで独立にfitします。全データのscalerを流用しません。参照モデルだけは全データでfitします。
- モデルの初期化seedは42で固定します。KMeans n_init=auto、GMM n_init=1/full、階層ward/complete/average/single、既定k=2..10の54候補です。
- seedを固定しても、入力データの変更で初期中心の選択が変わります。初期値そのものを固定した実験ではなく、前処理・学習手順を含めた行欠落への感度です。

## 3種類のARIと比較不能なケース

Target ARIは正解ラベルとの診断です。抽出・候補選定には使いません。現在のCSV仕様は分類ラベル付きの数値特徴量・欠損なしで、ラベルなしULは未対応です。

Reference ARIは全データの参照割当を部分集合の元行indexへ制限し、同じ行で部分集合側の割当と比較します。反復対のAssignment ARIは、その2つの部分集合に共通する元行indexだけに限定します。別々の配列の「先頭から同じ位置」を比較してはいけません。クラスタ番号の置換はARIに影響しません。

比較範囲の行数と両側の観測クラスタ数を記録します。共通行がmin_overlap未満ならinsufficient_overlap、どちらかが単一クラスタならdegenerateとし、比較ARIを空欄にします。全候補の学習が終わっていても、これらの比較はvalidとして集計しません。小さすぎる共通集合や単一クラスタの数値1.0を「安定」と解釈するのを避けます。

平均・SD・min/maxは記述統計です。各指標のCountを示し、SDはddof=1、Countが1以下なら空欄です。反復間の対は互いにデータを共有するため、独立標本としたSEM/CIを作りません。自動的な最良モデル・k・seedの選定もありません。組込みデータは開発用であり、未知データの外部検証ではありません。

## schema 6：9シート

| シート | 内容 | 既定の行数 |
| --- | --- | ---: |
| UL_Reference | 全データの54候補、指標・収束・警告・設定・状態 | 54 |
| UL_By_Subsample | 抽出seed×候補の指標・Reference ARIと比較状態 | 540 |
| UL_Sample_Summary | 候補ごとの完了数、比較状態、指標別count/mean/SD/min/max | 54 |
| UL_Overlap_Pairs | 同じ候補の抽出seed対、共通行数、比較状態、Assignment ARI | 2,430 |
| UL_Overlap_Summary | 候補ごとの対数・比較状態とvalidなARIの記述統計 | 54 |
| UL_Samples | 抽出seed、件数、実際の割合、処理状態 | 10 |
| UL_Sample_Indices | 抽出seed、部分集合内の位置、元データ行index（0始まり） | floor(N×.8)×10 |
| UL_Settings | 候補ID、手法、k、モデルseed、全パラメータ | 54 |
| Summary | 実験の目的と解釈 | 10 |

Reference ARIのCountはvalidな比較だけです。他の指標も計算不能な値は除きます。`UL_By_Subsample.Silhouette_SD/SE` は各学習結果の点ごとの量です。`UL_Sample_Summary.Silhouette_Across_Subsamples_SD` は各部分集合のSilhouette平均の間の標本SDです。量と列名を区別し、どちらからも独立反復の信頼区間は作りません。

## NPZと検証

元CSVから読み取った全特徴量配列featuresと符号化truthを含め、参照scaler・全参照割当、全抽出index、部分集合ごとのscaler・全割当を保存します。キーは `reference_labels_<候補ID>`、`sample_<seed>_indices`、`sample_<seed>_scaler_mean/scale`、`sample_<seed>_labels_<候補ID>` です。Candidateと元行indexで対応を追えます。

検証器はファイルハッシュに加え、抽出indexの再現、部分集合内だけのscaler fit、target診断指標、参照/共通行でのARI、集計、候補設定・状態の対応を確認します。クラスタモデルの再学習はしません。Silhouette等の全内部指標を再計算する検証ではありません。

## 制限・失敗時の保存

この入口の上限は2,000行、100特徴量、参照を含む合計2,000候補fitです。元のrun/repeat-ulの上限とは別です。部分集合の件数は最大kを超え、min_overlap以上である必要があります。Silhouetteの全件計算・反復対の集計・Excel保存を含むため、大きい設定では時間がかかります。各候補のTrain Time (s)は学習と指標計算の経過時間で、標準化・比較表集計・途中保存を含むコマンド全体の所要時間とは別です。

開始、参照の処理区切り、各部分集合の処理区切りでcheckpointを保存します。最初の例外で停止し、完了済み候補はcompleted、失敗候補はfailed、後続はnot_runです。標準化で失敗した場合はその段階と部分集合側のfailedを記録し、未学習候補をfailedとは偽りません。GMMの収束フラグだけをConvergedに記録し、非収束はfalseと警告を残します。フラグを公開しないKMeansと、反復収束が該当しない階層法では空欄にします。KMeansの反復回数と警告も保存し、存在しないフラグをtrueとして扱いません。非収束の診断と学習処理の例外は区別します。例外なく算出できた非収束GMMの指標・割当も保存し、completed側の記述集計に含めます。集計値だけで判断せず、各行のフラグと警告も確認してください。

```sh
.venv/bin/python scripts/validate_ul_subsample.py runs/failed --allow-failed
```

通常の検証はfailedを合格にしません。allow-failedは部分結果の整合性を確認するためのものです。入力/上限検査で失敗した場合はmanifestのみのことがあります。強制終了時は最後のcheckpointがrunningのまま残る場合があり、完了扱いにしません。保存中の強制終了・ストレージ障害からの完全復旧や自動再開は未実装です。
