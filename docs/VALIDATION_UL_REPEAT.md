# 今回の検証

実行結果は `tests/results/validation.json` に記録します。過去の96 passedや旧wheelとの一致結果とは別です。

開発環境とクリーンなwheel環境で、Iris/Wine/Breast Cancerの既定10seedを実行し、NPZ全配列を照合する手順を `scripts/verify_reimplementation.py` に実装しています。旧0.3/0.4wheelは入手できていないため比較していません。時間列の完全一致は検証対象ではありません。

## 今回の確定結果

- Python 3.12.14 / Linux CPU、TensorFlow CPU 2.20.0、Keras 3.15.1。
- 開発環境23 passed・独立wheel環境23 passed、いずれも失敗/skip 0。pip check両環境合格。
- Iris/Wine/Breast Cancer × run/select/repeat-ul × 2環境 = 18実行。runのDLは100epoch、selectはbootstrap1,000反復、repeat-ulは10seed。
- 9実行の対応する全NPZ配列と、時間列を除く全Excel表が環境間で完全一致。
- UL反復の既定6シート件数は180/18/810/18/36/9。Wineの3seed実行も検証器合格。
- Wineの共通描画関数からPNG5枚を生成。Notebookはコードセルの構文を検証したが、Notebookカーネルを通した実行は未実施。
- 原本ZIPの外側SHA256、内部チェックサム全件、ZIP CRCを確認。

保存済みの再実装結果は `validation/current/`、詳細は `tests/results/validation.json` とJUnitにある。旧0.4ソースがないため、旧版との数値一致を確認した結果ではない。
