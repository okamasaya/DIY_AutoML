# プロジェクト概要・設計仕様

## 1. 目的・設計理念

単一データセットで高精度を出すことではなく、既知のデータセットで実験手順を再現・検証したうえで、別データでも同じ評価手続きを適用し、出力指標からモデルの適用妥当性を検討できるようにする。将来はCSV入力型の「簡易AutoML評価フレームワーク」を自作ライブラリ化する。

基本フロー：`Dataset → Preprocessing → SL/UL/DL → DataFrame → 評価・比較 → Excel Report`。

## 2. 現行STEP（Notebook実装から確認）

| STEP | 内容 | 状態 |
| --- | --- | --- |
| 0 | numpy, pandas, matplotlib, seaborn, sklearnライブラリ読込 | 実装済 |
| 1 | sklearn組込みDataset選択、出力先 `output_summary` 指定 | 実装済（Colab Drive固定パス） |
| 2 | StandardScalerで `X` 作成 | 実装済（ただしリーク注意） |
| 3 | SL（5モデルの学習・macro評価） | 実装済 |
| 4 | UL（KMeans、階層4 linkage、GMM、k=2..10） | 実装済（重複登録あり） |
| 5 | DL（設定駆動MLP、学習・評価・history） | 実装済 |
| 6 | SL/UL/DLの集計、best行取得、Assessment | 実装済（探索的） |
| 7 | 5シートExcel出力 | 実装済（画像埋込と自動妥当性コメントなし） |

## 3. 入出力変数（実コード準拠）

- `dataset`：load_iris / load_wine / load_breast_cancer の sklearn Bunch。
- `output_summary`：出力Excelへの保存パス（元コードはDrive上の実在フォルダを想定）。
- `X_raw`、`y`、`feature_names`、`target_names`：STEP1で取得。
- `X`：標準化済み配列。現行では全件標準化してからSL/DLを分割。
- `result_df_SL`：Model, Accuracy, Precision, Recall, F1, Train Time (s)。
- `result_df_UL`：Model, Cluster, Silhouette, Silhouette_SD, Silhouette_SE, Davies-Bouldin, Calinski-Harabasz, ARI, AMI, NMI, Train Time (s)。
- `result_df_DL`：Model, Layers, Accuracy, Precision, Recall, F1, Epoch, Batch Size, Train Time (s)。
- `history_df_DL`：accuracy, loss, val_accuracy, val_loss, Epoch。
- `best_SL`、`best_UL`、`best_DL`：現行の選定済行。選定規則は `Accuracy最大 / Silhouette最大 / Accuracy最大`。
- `summary_report`：Method, Purpose, Evaluationの3行。`best_model_report`は現行Excelには未保存。

## 4. モデル・実行条件

### SL
DecisionTree, RandomForest(n_estimators=100), LogisticRegression(max_iter=1000), SVC, KNeighborsClassifier(n_neighbors=5)。train/test=80/20、random_state=42、stratify=y。Accuracy/Precision/Recall/F1、後3つはmacro平均。学習時間はfit周辺のみ測定。

### UL
KMeans、AgglomerativeClustering(linkage=ward/complete/average/single)、GaussianMixture、k=2〜10。内部評価はSilhouette、Davies-Bouldin（小さいほど良い）、Calinski-Harabasz（大きいほど良い）。外部評価はARI、AMI、NMIで、現状は`y`を使うためラベル付きデータ専用。Silhouette_SDはサンプルごとのばらつき、Silhouette_SEはそのSEM（独立性仮定の解釈に注意）。現行STEP4の図はKMeans中心：Silhouetteエラーバー、縦3段評価図、ARI/NMI比較。

### DL
TensorFlow/Keras Sequential MLP。`Input(shape=(input_dim,))`、Dense隠れ層、softmax出力、Adam、sparse_categorical_crossentropy。`DL_config` の元設定：`layers=[16,8]`, activation=relu, optimizer=adam, learning_rate=0.001, epochs=100, batch_size=16, verbose=0。後続セルで `layers=[32,16,8]` に上書きされる。train/test=80/20(stratify, random_state=42)、trainの20%をvalidation_split。結果は単一MLP設定、学習履歴100epoch。`optimizer`キーは設定にあるが生成関数ではAdam固定。

## 5. Excelレポートの現行構成

- `SL_Result`：モデルごとに1行。
- `UL_Result`：モデル×クラスタ数ごとに1行（現行は重複1行あり）。
- `DL_Result`：MLPの設定とtest指標。
- `DL_History`：Epochごとの学習履歴。
- `Summary`：現在はMethod/Purpose/Evaluationの説明用3行だけ。**Best model・数値スコア・判断コメントはまだExcelに入っていない**。

## 6. 開発時の取り決め

- Notebook内でSTEP番号とセル単位の役割を維持する。
- コードの不要な空行を減らし、実行できる単位で短く提示する。
- モデル学習・評価・グラフ・Excel出力は役割分離する。
- SL/DLのAccuracyとULのSilhouetteを直接比較した「総合ランキング」は作らない。
- 実験条件と評価指標の意味、分割・seed・選定方法を必ず記録する。
- 新スレッドでは現行成果を保全し、必要な部分だけ差分改修する。
