# 学習側だけのSLハイパーパラメータ探索（0.4.0a4）

`tune` は宣言した27設定を、学習側の共通層化5foldで比較します。testは最初に約20%分離し、選定した1設定だけを学習側全体で再学習してtest評価します。既存の固定5モデルのselectはそのまま残しています。

```sh
.venv/bin/python -m diy_automl tune --csv data/reference/wine.csv --target target --output runs/wine-tune
.venv/bin/python scripts/validate_tuning_run.py runs/wine-tune
```

Python入口は `tune_csv(csv, target, output, seed=42, metric='F1', bootstrap_repeats=1000, bootstrap_seed=43, confidence=.95)` です。CLIではmetric・seed・bootstrap-repeats・bootstrap-seed・confidenceも指定できます。既存出力は上書きしません。上限は5,000行・100特徴量、bootstrapは最大10,000反復です。任意グリッドを指定する入口は未実装です。

## 宣言した探索範囲と順序

| 手法 | 変更する設定（記載順） | 設定数 |
| --- | --- | ---: |
| Decision Tree | max_depth=None,3,5 × min_samples_leaf=1,3 | 6 |
| Random Forest | max_depth=None,5 × min_samples_leaf=1,3 | 4 |
| Logistic Regression | C=1,.1,10 | 3 |
| SVM | kernel=rbf,linear × C=1,.1,10 | 6 |
| k-NN | n_neighbors=5,3,7,9 × weights=uniform,distance | 8 |

Candidate IDは0..26です。手法順は表の順、各直積は右側の値を先に変えます。各手法の最初が従来の既定設定で、IDは0/6/10/13/19です。Random Forestの木の数100、Logistic Regression max_iter=1000、SVM gamma=scale等は固定し、全パラメータを保存します。乱数はseedで固定し、seed自体は探索しません。

共通のStratifiedKFold(n_splits=5,shuffle=True,random_state=seed)を全候補で使います。各foldのPipelineはStandardScalerをfoldの学習部分だけでfitします。候補間で同じfoldのscalerが一致することも実行中に確認します。

選定は既定macro F1の5fold単純平均です。atol=1e-12、rtol=0で同率を残し、候補順の最初を採用します。test、時間、fold SDでは同率を解消しません。選定CV値は探索の楽観バイアスを含みうるため、未知データへの不偏な性能推定と解釈しません。

## testと条件付き区間

選定後に学習側全体で再学習し、その1設定だけでtestを1回予測します。既定の区間は、正解クラス別の観測件数を固定したpaired percentile bootstrap 1,000反復・95%・seed43です。点推定と区間のmacro平均には、同じ全符号化クラス集合を使います。testで観測0件のクラスは再標本化でも0件のままで、平均から暗黙には除きません。この整合性修正は既存selectにも適用しました。

区間は固定した学習済み・選定済みモデルと観測クラス件数に条件付きであり、学習・選定・データ/分割の変動を含みません。ゼロ幅は未知の不確かさゼロを意味しません。fold SDも独立反復のCIではありません。

## schema 7：8シートと配列

| シート | 内容 | 既定行数 |
| --- | --- | ---: |
| SL_Search_Settings | Candidate ID、手法、変更設定と全パラメータ | 27 |
| SL_Search_Folds | 全候補×共通5foldの指標・状態・fit時間・警告/エラー | 135 |
| SL_Search_Mean | 5fold完了時だけの平均、完了fold数、状態 | 27 |
| SL_Search_SD | 5fold間の標本SD（ddof=1） | 27 |
| SL_Test | 選定した1設定だけのtest4指標 | 1 |
| SL_Bootstrap | 固定test予測の条件付き再標本化結果 | 1,000 |
| SL_Test_Intervals | 4指標の点推定とpercentile区間 | 4 |
| Summary | 評価手順と解釈 | 10 |

NPZには読み取った特徴量・符号化target、train/test index、全fold indexとvalidation正解・scaler、各候補foldのvalidation予測、選定後のscalerとtest予測を保存します。manifestには設定・選定候補・同率候補、入力/ソース/出力ハッシュと依存版を残します。未選定候補のtest予測は保存も計算もしません。学習済み分類器のpickle配布や推論APIは未実装です。

検証器は学習せずに分割とfoldの再現、train-only scaler、予測と指標、平均/SD、同率と選定、bootstrapの再現を検査します。StandardScalerの統計は保存特徴量から再計算します。

## 失敗時

最初のfold/refitエラー、またはConvergenceWarningで全体をfailedとして停止します。候補を黙って除外して残りから選び直すことはしません。CVの完了済みfoldはcompleted、例外のfoldはfailed、後続はnot_runです。5fold未完了候補の平均/SDは空欄にします。再学習やtest段階で失敗した場合は、その段階を記録します。

```sh
.venv/bin/python scripts/validate_tuning_run.py runs/failed --allow-failed
```

allow-failedは部分結果の整合性確認であり、学習成功の意味ではありません。入力・分割検査などで失敗した場合はmanifestだけのことがあります。処理区切りでcheckpointを残しますが、強制終了・保存中の障害からの完全復旧や学習再開は未実装です。実行中の状態をcompletedとは扱いません。
