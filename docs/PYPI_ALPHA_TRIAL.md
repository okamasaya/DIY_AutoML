# UC_AutoMaiL 0.4.0a8 試用手順

PyPI公開に先立ち、現在の機能を同じパッケージから試す版です。分類、時系列の状態推定、グラフ分類をCPUで実行できます。ストレージとニューラルネットワークは既存の `storage`、`dl` 追加依存で導入します。

## Windowsで試す

リポジトリのルートからPowerShellで実行します。`$wheel` に試用wheelのパスを指定してください。`$python313` は、この作業端末で使っているPythonです。別の端末では、Python 3.12以降の `python.exe` に置き換えます。

```powershell
$python313 = '.\runs\python-runtimes\cpython-3.13\python.exe'
$wheel = '.\runs\pypi-uc-automail-0.4.0a8-20261127\dist\uc_automail-0.4.0a8-py3-none-any.whl'
$trial = "runs/alpha-local-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
& $python313 -m venv "$trial/.venv"
$python = "$trial/.venv/Scripts/python.exe"
$cli = "$trial/.venv/Scripts/uc-automail.exe"
& $python -m pip install $wheel
& $python -m pip check
& $cli --version
& $cli audit-data --csv data/reference/iris.csv --target target --output "$trial/iris-audit"
& $cli select --csv data/reference/iris.csv --target target --progress both --output "$trial/iris-selection"
& $cli report --run "$trial/iris-selection" --charts detail --output "$trial/iris-report"
```

点検は `iris-audit` にあります。評価と詳細図は `iris-report/report_with_figures.xlsx` を開き、`SL_CV_Mean`、`SL_Test`、`SL_Test_Intervals`、`Progress` の順に確認できます。学習中に記録した `iris-selection/progress.jsonl` も保存されます。同梱のIrisから手元のCSVに変える場合は、`--csv` と、正解ラベル列を指定する `--target` を差し替えてください。

Windowsの画面も試す場合は、Tcl/Tkが有効なPythonで `& $cli bp-preview --csv data/reference/iris.csv --output-dir "$trial/bp"` を実行します。日本語の画面で `target` を選び、点検、列の選択、5モデル比較へ進みます。 [画面の操作手順](DATA_AUDIT.md#データ点検の画面を使う)

## ColabやローカルJupyterで試す

[試用ノートブック](../notebooks/Alpha_Trial.ipynb) と同じwheelを使います。Colabのファイルメニューからノートブックをアップロードし、CPUランタイムで開きます。最初のセルを実行するとwheelを選べます。ローカルJupyterでは、起動したフォルダーにwheelを置くか、ノートブックの `wheel` へ絶対パスを指定します。 [Colabのノートブック読み込み手順](https://research.google.com/colaboratory/faq.html?hl=en)

ノートブックは試用専用の仮想環境を作ります。Irisの分類、選定の進捗ログ、詳細図付きExcelと、1系列のカルマンを確認できます。必要に応じてEKF・UKF・粒子フィルタの比較とGCNのセルを実行できます。最後のセルで入力とExcel、NPZ、manifest、ログをZIPにまとめます。

時系列では `kalman/report.xlsx` の `Decision` で採用理由、`Metrics` で誤差と区間被覆を確認します。グラフでは `graph/report.xlsx` の `Comparison` に選定用と保留のスコアがあります。

## 試用後に機能範囲を決める

分類と1系列の時系列で、導入からExcelを読めるまでにかかった時間、入力に必要だった情報、採用理由の分かりやすさを確認します。詳細分析では、力学や観測モデル、グラフの辺を用意する手間と、追加の評価が判断に役立った場面を確認します。この結果を基にDS・BPの標準の入口と、詳細分析の配置を決めます。 [DS・BP向けの方針](AUDIENCE_PLAN.md#時系列とグラフで標準にする範囲)

## 2026年11月25日の試用結果

Windows x64のPython 3.13.17で、新しい環境にwheelを導入しました。ローカルに取得済みの依存wheelからの導入は約70秒、導入後の環境は約365 MiBでした。IrisのCLI実行はSVMを選定し、保留30件中29件正解、macro F1は0.9666でした。

同じWindowsのローカルJupyterカーネルで、ノートブックの全セルを実行しました。分類、線形カルマン、EKF・UKF・粒子フィルタの比較、GCNの結果を検証し、ZIPにまとめられました。インストール後の各コマンドは保存結果の検証込みで約5〜6秒でした。Colabのホスト環境での動作は未確認です。

wheelとソース配布物は `twine check --strict` を通過し、TensorFlowを含む開発環境で全282テストが通過しました。

## 2026年11月27日の試用結果

MITライセンス付きのUC_AutoMaiL候補を使い、wheelを入れたWindowsのPython 3.12.16と、ソース配布物を入れた3.13.17で、Irisの進捗ログと詳細図4点を含むExcelを確認しました。同じwheelでローカルJupyterの全5コードセルを実行し、分類、時系列、グラフの結果を含む31ファイルを最終ZIPで照合しました。Colabのホスト環境での動作確認は残っています。 [配布物の場所と検証結果](PYPI_RELEASE.md#2026年11月27日のuc_automail候補)

## 配布の準備

公開候補のビルド、TestPyPIからの導入確認、本番PyPIへのアップロードは公開手順にまとめています。 [PyPIへの公開手順](PYPI_RELEASE.md)
