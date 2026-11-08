# SL設定探索と紹介記事の検証（0.4.0a4）

初回再実装に、学習側だけで行う27設定のSL探索を追加しました。過去のソースを復元したという扱いではなく、以前の検証記録も上書きしていません。

## 確定した検証結果

- Python 3.12.14 / Linux CPU、scikit-learn 1.8.0。
- 開発環境110 passed、独立wheel環境110 passed。失敗・error・skip 0、両環境pip check合格。
- Iris/Wine/Breast Cancerで27設定×5foldと選定した1設定の再学習を2環境で実行。計810 CV fitと6 refit。
- 全保存配列と時間列以外の8シート全表が環境間で完全一致。既定bootstrapは1,000反復。
- 探索に含めた既定5設定のCV値は、初回再実装版の保存済みselect結果と3データすべてで完全一致。
- 既存の保存結果16件も検証器に合格。今回の3件を含む保存結果19件を検証。

test行に固有IDを持たせたデータで、全135fold fitと最終refitのscalerにtestが入らないこと、testへのpredictが最後の1回だけであることを実測しました。test特徴量を変えても学習側の探索結果が変わらないことも確認しています。

途中foldと再学習の例外、ConvergenceWarningを注入し、停止と部分結果保存を確認しました。保存後の分割、scaler、CV予測、選定、test指標、bootstrap、チェックサムの不整合も検出します。

点推定とbootstrapには全符号化クラスを明示するようにしました。testで正解の観測が0件のクラスがあっても、macro平均の分母を暗黙に変更しません。この補正はselectにも適用し、対象クラスがすべてtestに現れる既存3データの値は変わりません。

## 記事手順と表示の確認

`requirements-core.lock.txt` だけを導入した独立venvで、TensorFlowが未導入であることを確認しました。その環境で記事にあるWineのselect/tune、保存結果検証器、pip checkを実行して合格しました。

Zenn CLI 0.5.14とsystem Chromiumによるローカル表示で、タイトル・topics・published=false（下書き）・本文・フロー図PNGを確認しました。図は170,770bytesで、Zenn公式のimagesパスと3MB制限に適合します。外部カードやMermaidの埋込iframeは使わず、手順の図をリポジトリ内のPNGとして用意しました。

この環境ではZenn補助スクリプト `embed.zenn.studio/js/listen-embed-event.js` へのリクエストがネットワーク制約で失敗しました。記事内の外部iframeは0で、本文・PNG表示と下書き設定の確認に支障はありませんでした。ブラウザ上のページ例外は0です。この失敗もpreviewのJSONに記録しています。Zenn本番への投稿・公開後の表示は未検証です。

## 記録と再実行

機械可読な記録とJUnitは `tests/results/0.4.0a4/`、実行済み出力は `validation/sl-tuning-0.4.0a4/`、紹介記事は `articles/diy-automl-evaluation-framework.md` にあります。

```sh
.venv/bin/python scripts/verify_tuning.py --output /新規保存先/development
別wheel環境/bin/python scripts/verify_tuning.py --output /新規保存先/wheel --compare /新規保存先/development
```

独立wheel環境はsystem-site-packagesを使用せず、ソースをPYTHONPATHへ追加せずに実行しました。今回の環境内での完全一致であり、他OS/GPU/ライブラリ版での完全一致を保証しません。

選定CV値の楽観バイアス、固定した学習済みモデルに条件付きのtest区間、開発用データを繰り返し参照していることを区別します。任意グリッド・DLの設定探索や、選定変動を含む不確かさまでは実装・検証していません。
