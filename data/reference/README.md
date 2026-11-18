# 分類サンプルCSV

Iris、Wine、Breast Cancer Wisconsinに加え、CPUで短時間の分類比較を試せるUCIのデータを2件追加した。候補はトライフィールズの一覧から選び、データ本体はUCIから取得した。 [候補を探した一覧](https://www.trifields.jp/uci-machine-learning-repository-datasets-956)

| CSV | 1行の意味 | 数値特徴量 | `target` |
| --- | --- | ---: | --- |
| [`seeds.csv`](seeds.csv) | 小麦の粒1個、210行 | 7列 | 品種ID 1・2・3、各70件 |
| [`raisin.csv`](raisin.csv) | レーズン1個、900行 | 7列 | 品種Besni・Kecimen、各450件 |

どちらも欠損・非有限値0件、全行および特徴量ベクトルの完全一致重複0件。品種、収穫ロット、撮影条件が変わる場面での評価には、対象を分けた追加データが必要になる。

## 出典と変換

- **Seeds**。Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., & Lukasik, S. (2010). *Seeds* [Dataset]. UCI Machine Learning Repository. 公式ZIP内の `seeds_dataset.txt` の空白区切りをCSVへ変換し、面積、周囲長、コンパクトネス、粒の長さ、粒の幅、非対称係数、溝の長さに対応する見出しを補った。数値と品種ID、行順は配布ファイルのまま。 [UCI原典](https://archive.ics.uci.edu/dataset/236/seeds) · [DOI](https://doi.org/10.24432/C5H30K) · [公式ZIP](https://archive.ics.uci.edu/static/public/236/seeds.zip)
- **Raisin**。Çinar, İ., Koklu, M., & Tasdemir, S. (2020). *Raisin* [Dataset]. UCI Machine Learning Repository. 公式ZIP内の `Raisin_Dataset.zip` にある `Raisin_Dataset/Raisin_Dataset.arff` の `@data` 部分に列見出しを付け、`Class` を `target` にした。面積や輪郭、形状などの7列の値と品種名、行順は配布ファイルのまま。 [UCI原典](https://archive.ics.uci.edu/dataset/850/raisin) · [DOI](https://doi.org/10.24432/C5660T) · [配布ファイル指定の原著](https://doi.org/10.30855/gmbd.2020.03.03) · [公式ZIP](https://archive.ics.uci.edu/static/public/850/raisin.zip)

両CSVは、上記の形式変換を施してUCIの配布条件に従い同梱する。 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| データ | 変換元ファイルのSHA-256 | 同梱CSVのSHA-256 |
| --- | --- | --- |
| Seeds | `1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f` | `f6c7cf2fdeb414c126a5046a6491aab9986cad0697dc612d8951ba15cb46cfc3` |
| Raisin | `8a33803b5961650225150740fa09fa31ee8875266277c7ac3291e2a3ee351814` | `149c45ab1c3a767386a6f68855ae43ae7a7d13a36b60816d8b9aee2335f5ea62` |

## CPUで分類を試す

リポジトリのルートで、既存のコア環境を使う。`select` は固定5モデルを学習側の共通層化5分割で比較し、選んだ1モデルを保留テストで評価する。`--csv` を差し替えるとSeedsも同じ手順で実行できる。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl select --csv data/reference/raisin.csv --target target --output runs/raisin-select
.\.venv\Scripts\python.exe -X utf8 scripts/validate_selection_run.py runs/raisin-select
```

混同行列やクラス別指標を見る場合は、次の手順を使う。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m diy_automl rank-sl --csv data/reference/raisin.csv --target target --cv-repeats 2 --output runs/raisin-ranking
.\.venv\Scripts\python.exe -X utf8 scripts/validate_sl_ranking.py runs/raisin-ranking
```

### Windowsでの確認

Python 3.13.17の既存コア環境で、CPUの数値計算スレッドを1に設定して実行した。`select` は25回のCV学習と1回の再学習、`rank-sl --cv-repeats 2` は50回のCV学習と1回の再学習。両方で保留行は共通とし、分割seed 42、再標本化seed 43、1,000反復の既定値を使った。

| データ | 両手順で選んだモデル | 保留テストの正解数 | macro F1（95%区間） | `select` の実行時間とピークメモリ |
| --- | --- | --- | --- | --- |
| Seeds | ロジスティック回帰 | 37/42 | 0.8731（0.7532〜0.9521） | 約3.7秒、154 MiB |
| Raisin | ロジスティック回帰 | 160/180 | 0.8885（0.8486〜0.9332） | 約4.1秒、155 MiB |

実行時間はCLI起動からExcel出力まで。メモリは同じ手順をPython APIから再実行したときのPythonプロセスのピークワーキングセット。区間は固定したモデルと保留行のクラス別件数に条件を置く。混同行列では、RaisinのBesni 90件のうち15件がKecimenに、Kecimen 90件のうち5件がBesniに分類された。

4実行とも保存結果を検証し、F1と混同行列をscikit-learnで照合した。ローカルの出力は `runs/seeds-uci-select`、`runs/raisin-uci-select`、`runs/seeds-uci-ranking`、`runs/raisin-uci-ranking` にある。
