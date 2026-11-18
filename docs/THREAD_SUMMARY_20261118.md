# 2026年11月18日の作業ログ・再開手順

GitHubの `main` をWindowsの作業フォルダーへ取得し、分類モデルの順位付け、不確かさと分類品質の点検、AutoGluonおよびDLバックエンドとの比較を進めた。開始時の `main` は `542f5df`、今回の作業ブランチは `codex/sl-ranking`。再開時の最新コミットとリモート反映はGit履歴で確認する。

## 実装と検証

| 項目 | 完了した範囲 |
| --- | --- |
| Windows環境とExcel | Python 3.13.17の専用 `.venv` を使用。図付きExcelのPNG相対パスを `/` 表記で保存し、検証器は以前のWindows形式も受け付ける。 |
| `rank-sl` | 固定5分類器を共通の層化5分割で繰り返し比較する。平均macro F1による順位、分割を変えたときの順位、fold外予測を保存する。比較用分割を指定した場合はCV首位との差をpaired bootstrapで評価する。最終テストはCVで選んだモデル1つに使う。 |
| 誤差評価 | `rank-sl` のレポートに標本のbootstrap区間、厳密な有理数とFloat64の算出値の差、入力のFloat32丸めを追加した。`--shift` を指定すると、学習済みモデルで仮の校正ずれへの感度も計算する。既存結果を点検する `audit-uncertainty` も用意した。 |
| 分類品質 | `Test_Confusion` と `Test_By_Class` を追加。IrisとWineで、混同行列とクラス別Precision・Recall・F1をscikit-learnで照合した。過去の順位付け出力（schema 10、12）も検証できる。 |
| AutoGluon | Tabular 1.6.3で、Iris・Wineの同一CV分割と保留テストを再評価。4分類器と加重アンサンブルを比較した。LightGBMの試行は空きメモリの判定で停止した。 |
| TensorFlowとPyTorch | CPU上で共通のMLP、初期重み、ミニバッチ順を使い、2データ×3 seedで比較した。予測、学習履歴、重み、再標本化、校正ずれシナリオを保存した。 |

設定と再実行の入口は [README](../README.md#教師あり5モデルの繰り返し比較)、[誤差評価の手順](SL_UNCERTAINTY_AUDIT.md)、[AutoGluon比較](../experiments/autogluon_sl_crosscheck/README.md)、[DLバックエンド比較](../experiments/dl_backend_crosscheck/README.md)。

WindowsのPython 3.13.17で、TensorFlowによる実際の学習を含むpytest **207件が通過**した。両比較の保留テストは保存済み配列から照合した。実験入力のIrisとWineは小規模な開発用データで、結果の適用範囲は記録した分割と設定に限る。

## 結果の要点

- **DIY AutoMLの順位付け**。IrisとWineのCV首位はSVMで、平均macro F1はそれぞれ0.9647、0.9905。Wineの比較用36行では、他4候補とCV首位との差の95%区間がすべて0を含んだ。
- **選定モデルの不確かさ**。テストmacro F1はIris 0.9666（標準誤差0.0328、95%区間0.8977〜1.0000）、Wine 0.9710（0.0283、0.9089〜1.0000）。両データでFloat32丸めによるラベル変化は0件。Irisの花弁幅に+0.5 cmを仮定すると、30件中6件でラベルが変わりF1が0.7341へ低下した。実際の系統的な偏りは未測定。
- **AutoGluonとの比較**。Irisの保留テストでAutoGluon − DIYのF1差は+0.0334（95%区間0.0000〜+0.1023）。Wineでは差0.0000（−0.0911〜+0.0911）。Wineの混同行列は一致し、誤分類したCSV行はDIYが132行目、AutoGluonが136行目だった（ヘッダー込み）。
- **DLバックエンド比較**。独立に学習したTensorFlowとPyTorchの分類ラベルとF1は3 seedとも一致。学習後のテスト予測確率の最大差はIrisで3.05パーセントポイント、Wineで0.21ポイント。共通重みでの最大確率差は約6 × 10⁻⁸。Irisの花弁幅に+0.5 cmを仮定した際の分類結果も両方で一致した。
- **DLの確率スコアの追加点検**。保存済みlogitから計算した、固定3 seedでの平均Log loss差（PyTorch − TensorFlow）はIris −0.00221（95%区間−0.00588〜+0.00060）、Wine −0.000064（−0.000138〜−0.0000003）。Brier scoreの差は順に−0.00121、−0.000051。区間は共通のテスト行を再標本化した探索的な結果。

## 環境と出力

- 分類コアはPython 3.13.17の `.venv` と `requirements-core.lock.txt`。Windowsでは `.\.venv\Scripts\python.exe -X utf8` を実行入口にする。
- 比較環境は `.venv-autogluon` と `.venv-dl`。後者で用いた版は TensorFlow CPU 2.20.0、Keras 3.15.1、PyTorch 2.14.0+cpu。環境の構築コマンドは各比較のREADMEにある。
- 当日のレポートはローカルの `runs/classification-quality-20261118-173941`、`runs/uncertainty-audit-20261118-161108`、`runs/autogluon-crosscheck-20261118-155129`、`runs/dl-backend-crosscheck-20261118-181801` に保存した。`runs/` と仮想環境はGitの管理対象外。別環境ではGitにある入力CSVと手順から再生成できる。

## 次に着手すること

文章の更新に先立ち、任意に実行する **CPU版Simulated Annealingによる特徴量選択**を実装・検証する。Irisでは小さい探索空間を全探索で照合し、WineではSAとランダム探索に同じ学習回数を割り当てる。選定列、探索履歴、実際の学習回数、時間を保存し、探索後に保留テストで性能差と区間を確認する。記事にはこの実測を反映する。QUBOとQuantum Annealingは将来テーマとして扱う。 [ロードマップ](ROADMAP.md#annealing-check)
