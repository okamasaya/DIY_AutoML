# UC_AutoMaiLをPyPIへ公開する手順

表示名は **UC_AutoMaiL**、配布名とCLIは `uc-automail`。配布候補 `0.4.1a1` は `import uc_automail as uc_al` からEKFの1ステップ処理、制御入力、拡張した整合性診断を使える。公開元は既存の [okamasaya/DIY_AutoML](https://github.com/okamasaya/DIY_AutoML)。既存の記事やコードは `import diy_automl` と `diy-automl` を引き続き利用できる。Python 3.12以降が対象で、`storage` と `dl` は追加依存として選べる。 [PyPIの名前の正規化](https://packaging.python.org/en/latest/specifications/name-normalization/)

- [配布候補 `0.4.1a1` の配布物とSHA-256](candidates/0.4.1a1/) · [TestPyPIの公開版 `0.4.1a0`](https://test.pypi.org/project/uc-automail/0.4.1a0/) · [0.4.1a0の検証記録](candidates/0.4.1a0/release.json) · [0.4.0a9の検証記録](candidates/0.4.0a9/release.json) · [0.4.0a8の検証記録](candidates/0.4.0a8/release.json)
- [PyPIに掲載する英語説明](README.pypi.md) · [Windows・Colabでの試用手順](TRIAL.md) · [試用ノート](Alpha_Trial.ipynb)
- [ビルド設定](../pyproject.toml) · [ソース配布物の収録設定](../MANIFEST.in) · [ライセンス](../LICENSE)

配布候補 `0.4.1a1` は、両配布物の `twine check --strict`、収録した67モジュールとMITライセンスの照合、回帰テスト312件を通過した。下の2環境でインストールした版を確認し、制御入力を渡したEKFの一括・1ステップ出力、欠測がある場合の診断とNIS・NEES、IrisのSVM評価（保留30件中29件正解）と図4点のExcelを確認した。

| Windowsの新規環境 | 導入と追加確認 |
| --- | --- |
| Python 3.12.16 | wheelを導入し、依存をPyPIから取得 |
| Python 3.13.17 | ソース配布物からビルドし、取得済み依存で `storage` とParquet・DuckDB出力も確認 |

試用ノートはWindowsのローカルJupyterで全5コードセルを実行し、生成されたZIPの31ファイルを照合した。Colab上の試用は次の確認項目。作業端末の検証ログは `PyPI/runs/verify-0.4.1a1/` にあり、配布物のSHA-256と検証結果はリリース記録に残した。 [0.4.1a1のリリース記録](candidates/0.4.1a1/release.json)

配布物に対応するソースはGitの `3820ecf` にある。ソース配布物の72入力をGitの内容と照合し、Python 3.12.16と3.13.17に導入した候補で欠測・遅着の実験を実行した。 [対応するソース](https://github.com/okamasaya/DIY_AutoML/commit/3820ecf81b5d7d669bc6aa91da83d8021780c8c6) · [カルマン系の検証](../experiments/kalman_sensitivity/README.md#欠測と時刻の遅れを同じ系列で比べる)

公開済みの `0.4.1a0` を試す場合は、wheelをTestPyPIから取得し、依存ライブラリはPyPIからインストールする。 [TestPyPIから試す](TRIAL.md#testpypiから試す) · [配布候補をWindowsで試す](TRIAL.md#windowsで試す)

`0.4.1a0` の両配布物をTestPyPIから再取得し、公開候補とSHA-256が一致した。回帰テスト310件と、Python 3.12・3.13での修正版wheelの整合性診断を確認した。 [整合性診断の使い方](../docs/UNCERTAINTY_DESIGN.md#ekfukfの整合性を確認する)

TestPyPIに公開した `0.4.0a9` のwheelとソース配布物を再取得し、`PyPI/candidates/0.4.0a9/` の候補とSHA-256が一致することを確認した。取得したwheelをWindowsのPython 3.13環境に入れ、`import uc_automail`、`import diy_automl`、`python -m uc_automail --version` を確認した。初回配布版 `0.4.0a8` もTestPyPIから再取得して検証済み。 [初回のTestPyPI公開版](https://test.pypi.org/project/uc-automail/0.4.0a8/)

パッケージの自作コードと配布用ファイルにはMITライセンスを適用する。`pyproject.toml` の `license` と `license-files` でSPDX識別子と本文の同梱を指定する。サンプルデータの出典と条件はデータごとに記録している。 [ライセンスの本文と適用範囲](../LICENSE) · [サンプルデータの利用条件](../data/reference/README.md) · [パッケージのライセンスメタデータ](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license-and-license-files)

## 配布物を作って確認する

リポジトリのルートで、`build` と `twine` を入れたPython環境から実行する。新しい出力先を使い、公開に使うwheelとソース配布物を一組で管理する。

```powershell
$name = 'uc-automail'
$version = '0.4.1a1'
$dist = "PyPI/dist/$version-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
python -m build --outdir $dist
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$wheel = "$dist/uc_automail-$version-py3-none-any.whl"
$sdist = "$dist/uc_automail-$version.tar.gz"
python -m twine check --strict $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'Package metadata check failed' }
Get-FileHash -Algorithm SHA256 -LiteralPath $wheel,$sdist
```

新しくビルドした配布物の出力先は `PyPI/dist/`。PyPIの英語説明は `PyPI/README.pypi.md` を使う。配布の対象はパッケージのソースとメタデータで、論文、元の引継ぎZIP、データ、実験出力、リポジトリ向けのテストはGit側から参照する。公開用の配布物には、採用したライセンスの本文とメタデータを反映する。 [パッケージの作成手順](https://packaging.python.org/en/latest/tutorials/packaging-projects/) · [ライセンスのメタデータ](https://packaging.python.org/en/latest/specifications/license-expression/)

### 検証済みの配布候補から再開する

`0.4.1a1` は、下の手順で `PyPI/candidates/0.4.1a1/` のSHA-256を照合する。以前の配布物の照合には、`$releaseRoot` にそのバージョンのフォルダーを指定する。

初回配布版 `uc-automail==0.4.0a8` は、Gitの `4ff386a` からビルドした。検証時のソースと試用ノートは `7692334` で確認できる。配布物のSHA-256とコミットは `release.json` に記録している。 [初回配布物と検証記録](candidates/0.4.0a8/)

初回配布物は `twine check --strict` を通過し、配布した64個のPythonモジュールをGitのソースと照合した。開発環境の全304テストが通過した。WindowsのPython 3.12.16にwheel、3.13.17にソース配布物を新規導入し、どちらもIrisで保留30件中29件正解、25件のCVログ、図4点のExcelを確認した。試用ノートはローカルJupyterで全5セルが通り、出力ZIPの31ファイルを照合した。

リポジトリのルートで次を実行する。初回配布物の詳細ログは元の開発フォルダーの `runs/pypi-uc-automail-0.4.0a8-ready/` にある。

```powershell
$releaseRoot = 'PyPI/candidates/0.4.1a1'
$release = Get-Content -Raw -LiteralPath "$releaseRoot/release.json" | ConvertFrom-Json
$name = $release.name
$version = $release.version
$wheel = Join-Path $releaseRoot ($release.files | Where-Object kind -eq 'bdist_wheel').filename
$sdist = Join-Path $releaseRoot ($release.files | Where-Object kind -eq 'sdist').filename
foreach ($file in $release.files) {
    $actual = Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $releaseRoot $file.filename) -ErrorAction Stop
    if ($actual.Hash -ne $file.sha256) { throw "SHA-256 mismatch for $($file.filename)" }
}
```

### 公開候補の確認記録

作業端末で作った配布候補は `runs/pypi-release-0.4.0a8/dist`、SHA-256は同じ作業フォルダーの `SHA256SUMS.txt` にある。両配布物で `twine check --strict` が通り、収録ファイルとメタデータを照合した。

| Windows x64の新規環境 | 導入方法 | Irisの結果 |
| --- | --- | --- |
| Python 3.12.16 | wheel、依存はPyPIから取得 | SVMを選定、保留30件中29件正解 |
| Python 3.13.17 | ソース配布物からビルド、取得済み依存でオフライン導入 | SVMを選定、保留30件中29件正解 |

両環境で `pip check` と、Irisの点検・比較結果の検証が通った。ここでの記録は作業端末から配布候補を導入した結果。TestPyPIから取得する試験は次の手順で行う。

### 2026年11月26日の回帰確認

木の深さ・CPU並列度の指定とJSONによる列抽出を含む11月26日時点の候補は `runs/pypi-regression-0.4.0a8-20261126/dist`。同じ作業フォルダーに `SHA256SUMS.txt`、`build.log`、`pytest.xml`、環境別の `smoke.log` を保存した。wheelとソース配布物は `twine check --strict` を通過し、収録した63個のPythonモジュールは作業ツリーと一致した。開発環境の全300テストが通過した。

Windowsの新規Python 3.12.16環境にwheelを、3.13.17環境にソース配布物を導入した。PyPIへ接続できない実行環境だったため、依存パッケージは取得済みのものを新しい環境に複製した。両方で `pip check`、Irisの既定評価（SVM、保留30件中29件正解）、JSONでの列抽出、木の深さとCPUの指定値の照合が通った。3.13.17環境では `storage` のParquetとDuckDBへの抽出も確認した。

### 2026年11月27日の統合版

GitHubの `main`（`0d53cd7`）と、進捗・詳細図の実行例を加えた英語の `README.pypi.md` から、ライセンスを確定する前の試用候補を作成した。元の開発フォルダーの `runs/pypi-release-0.4.0a8-20261127/dist/` に当時のwheelとソース配布物がある。`SHA256SUMS.txt`、`source-verification.json`、`twine.log` は同じ作業フォルダーに保存した。両配布物で `twine check --strict` が通過し、64個のPythonモジュールと英語説明をソースと照合した。

| Windows x64の新規環境 | 導入と検証 |
| --- | --- |
| Python 3.12.16 | wheelを導入し、依存を本番PyPIから取得。選定、JSON列抽出、木の深さとCPU設定を確認 |
| Python 3.13.17 | 本番PyPIから取得した依存wheelを使い、ソース配布物を `storage` 付きでビルド・導入。ParquetとDuckDBへの抽出も確認 |

両環境で `pip check` が通り、インストールしたCLIのIris評価はSVMで30件中29件正解。`--progress both` で25件のCVログを保存し、`--charts detail` で図4点を含むExcelを作成した。検証器でログ、図の入力、元のセル値を照合した。環境別の `install.json` と `smoke.log` に導入した版と結果を残した。 [試用ノートブックの入口](TRIAL.md)

同じwheelでWindowsのローカルJupyter試用ノートを全セル実行した。分類と詳細図、線形カルマン、EKF・UKF・粒子フィルタ、GCNを通し、最終ZIPの31ファイルを確認した。結果は作業フォルダーの `notebook-local/` に保存した。TestPyPIへのアップロードとColabホストでの試用は未実施。

### 2026年11月27日のMITライセンス付き候補

`diy-automl==0.4.0a8` にMITライセンスを反映して隔離環境でwheelとソース配布物を作り直した。元の開発フォルダーの `runs/pypi-licensed-0.4.0a8-20261127/dist/` に当時の検証用配布物がある。

両配布物で `twine check --strict` が通過した。Core Metadata 2.4の `License-Expression: MIT` と `License-File: LICENSE`、同梱した本文、64個のPythonモジュール、英語説明、公開元URLを照合した。ハッシュと確認結果は同じ作業フォルダーの `SHA256SUMS.txt` と `source-verification.json` にある。

Windowsの新規Python 3.12.16にwheelを、3.13.17にソース配布物を `storage` 付きで導入し、導入に使ったファイルのSHA-256と、インストール先のMITメタデータ・本文を照合した。両方で `pip check` が通り、IrisはSVMで保留30件中29件正解。進捗ログ、図4点のExcel、JSONによる列抽出、木の深さとCPU設定を検証し、3.13.17ではParquet・DuckDB出力も確認した。TestPyPIと本番PyPIへのアップロードは未実施。

### 2026年11月27日のUC_AutoMaiL候補

不確かさ（Uncertainty）の評価から始まった開発の経緯を反映して、公開名を **UC_AutoMaiL** に決定した。元の開発フォルダーの `runs/pypi-uc-automail-0.4.0a8-20261127/dist/` に、新名称で隔離ビルドしたwheelとソース配布物がある。

両方で `twine check --strict` が通過した。配布メタデータの名称、MITライセンス本文、公開元URL、英語説明、64個のPythonモジュールを照合した。ハッシュとソースの確認記録は同じ作業フォルダーの `SHA256SUMS.txt` と `source-verification.json` に保存した。CLIは `uc-automail` と旧記事の `diy-automl` を同じ入口に接続している。

Windowsの新規Python 3.12.16環境にwheelを、3.13.17環境にソース配布物を `storage` 付きで導入した。配布物のSHA-256、`pip check`、両CLIのバージョン表示、IrisのSVM評価（保留30件中29件正解）、25件のCVログ、図4点のExcelを確認した。3.13.17ではParquetとDuckDBへの出力も検証した。

同じwheelを使ったローカルJupyterで試用ノートの全5コードセルを実行し、最終ZIPの31ファイルを照合した。実行結果は同じ作業フォルダーの `notebook-local/` に保存した。Colabホストでの試用と、TestPyPI・本番PyPIへのアップロードは未実施。 [試用手順](TRIAL.md)

## TestPyPIで導入を確認する

TestPyPIは本番PyPIとはアカウントとAPIトークンが別。TestPyPI側のAPIトークンを、Twineが求めたときに端末で入力する。登録が反映されたら、wheelとソース配布物の両方を取得し、選んだローカル候補とSHA-256が一致することを確認する。 [TestPyPIのアカウント作成](https://test.pypi.org/account/register/) · [TestPyPIの公式手順](https://packaging.python.org/en/latest/guides/using-testpypi/) · [配布ファイルのJSON API](https://docs.pypi.org/api/json/#get-a-release)

```powershell
python -m twine upload --repository-url https://test.pypi.org/legacy/ --username __token__ $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'TestPyPI upload failed' }

$trial = "PyPI/runs/testpypi-$version-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
python -I PyPI/scripts/verify_pypi_downloads.py --index testpypi --name $name --version $version --wheel $wheel --sdist $sdist --output "$trial/download"
if ($LASTEXITCODE -ne 0) { throw 'TestPyPI download verification failed' }
$download = Join-Path "$trial/download" (Split-Path -Leaf $wheel)
python -m venv "$trial/.venv"
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$testPython = "$trial/.venv/Scripts/python.exe"
& $testPython -m pip --isolated install --index-url https://pypi.org/simple/ $download
if ($LASTEXITCODE -ne 0) { throw 'Install failed' }
& $testPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $testPython -I PyPI/scripts/smoke_pypi_install.py --version $version --csv data/reference/iris.csv --output "$trial/iris" --kalman
if ($LASTEXITCODE -ne 0) { throw 'Package smoke test failed' }
```

`--kalman` は、`0.4.1a1` の公開APIを使って1ステップEKFと一括処理を照合し、欠測を含む診断を確認するオプション。確認用スクリプトは、点検、5モデル比較、進捗ログ、詳細図付きExcel、JSONでの列抽出、木の深さ・CPU並列度を指定した評価も行い、保存結果を検証する。`storage` を導入した環境では `--storage` でParquet・DuckDB出力の確認を追加できる。Colabでは同じバージョンのwheelを試用ノートブックで選べる。 [Colabでの試用](TRIAL.md#colabやローカルjupyterで試す)

取得検証器は、2026年11月27日にTestPyPIと本番PyPIの `sampleproject` で実際に両配布物を取得して確認した。DIY AutoMLのローカル候補を使った模擬応答では、両方が一致する場合と、ソース配布物の欠落、登録ハッシュの不一致、取得したファイルの不一致を確認した。照合を通過したファイルだけ指定の取得先に残る。 [取得検証器](scripts/verify_pypi_downloads.py)

### Windowsの制限付き実行で確認した症状

- **Jupyterの起動待ち**。制限付き実行では、`127.0.0.1` のTCP通信が `WinError 10013` で拒否された。Jupyterはカーネル起動前にPythonのイベントループを初期化するため、そこで止まっていた。同じPCの通常のWindows権限では、最小ノートと試用ノート全5セルを実行できた。
- **ソース配布物のビルド時にpipが `WinError 5` を返す**。既定のwheelキャッシュ内への一時ファイル作成に失敗した。`pip --isolated` では `PIP_NO_CACHE_DIR` と `PIP_CACHE_DIR` の環境変数が無視される。コマンドに `--no-cache-dir` を渡す場合と、`--cache-dir` で作業フォルダー内を指定する場合の両方で、同じソース配布物からwheelを作れた。 [pipのオプション](https://pip.pypa.io/en/stable/cli/pip/#general-options)
- **joblibのCPU検出警告**。開発環境の1.5.3は `Get-CimInstance` でCPU情報の照会を拒否され、代替の `wmic` も見つからず、論理16コアに切り替えた。通常のWindows権限では1.5.3も物理10コアを取得できた。配布候補の新規導入で解決された1.6.0はWindows APIを使い、制限付き実行でも警告なしで物理10コアを取得できた。 [WMICの非推奨化](https://learn.microsoft.com/en-us/windows/whats-new/deprecated-features)

比較に使ったログは作業端末の `PyPI/runs/verify-0.4.1a1/diagnosis/` に保存した。

## 本番PyPIに公開する

配布候補のソースをGitHubの `main` に反映し、英語説明からリンクしたライセンスと手順を読める状態にする。TestPyPIで確認したwheelとソース配布物を、本番PyPIのアカウントとトークンでアップロードする。公開したファイル名は再利用できないため、内容を修正した場合は新しいバージョンでビルドする。 [本番PyPIの公開手順](https://docs.pypi.org/publishing/publishing-to-pypi/) · [同じファイル名での再アップロード](https://pypi.org/help/#file-name-reuse)

```powershell
python -m twine upload --repository-url https://upload.pypi.org/legacy/ --username __token__ $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'PyPI upload failed' }
```

公開後は同じ検証器を `--index pypi` と新しい取得先で実行し、両配布物を照合する。その後、別の新規環境で `python -m pip install "$name==$version"` を実行し、同じIrisとEKFの確認を行う。GitHubのソースと公開版が対応するコミットをリリース記録に残す。
