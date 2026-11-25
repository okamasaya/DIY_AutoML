# DIY AutoMLをPyPIへ公開する手順

初回の配布候補名は `diy-automl`、バージョンは `0.4.0a8`。Pythonのimport名は `diy_automl`、コマンド名は `diy-automl`。分類、状態推定、GCNを同じパッケージから使う。Python 3.12以降が対象で、`storage` と `dl` は追加依存として選べる。 [試用手順](PYPI_ALPHA_TRIAL.md)

候補名の公開メタデータは、2026年11月25日に本番PyPIとTestPyPIの両方で404だった。名前が利用できるかは最初の登録時に確定する。 [PyPIの名前選定](https://docs.pypi.org/project-management/choosing-a-name/)

## 配布物を作って確認する

リポジトリのルートで、`build` と `twine` を入れたPython環境から実行する。新しい出力先を使い、公開に使うwheelとソース配布物を一組で管理する。

```powershell
$name = 'diy-automl'
$version = '0.4.0a8'
$dist = "dist/pypi-$version-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
python -m build --outdir $dist
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$wheel = "$dist/diy_automl-$version-py3-none-any.whl"
$sdist = "$dist/diy_automl-$version.tar.gz"
python -m twine check --strict $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'Package metadata check failed' }
Get-FileHash -Algorithm SHA256 -LiteralPath $wheel,$sdist
```

PyPIの英語説明は `README.pypi.md` を使う。配布の対象はパッケージのソースとメタデータで、論文、元の引継ぎZIP、データ、実験出力、リポジトリ向けのテストはGit側から参照する。公開用の配布物には、採用したライセンスの本文とメタデータを反映する。 [パッケージの作成手順](https://packaging.python.org/en/latest/tutorials/packaging-projects/) · [ライセンスのメタデータ](https://packaging.python.org/en/latest/specifications/license-expression/)

### 公開候補の確認記録

作業端末で作った配布候補は `runs/pypi-release-0.4.0a8/dist`、SHA-256は同じ作業フォルダーの `SHA256SUMS.txt` にある。両配布物で `twine check --strict` が通り、収録ファイルとメタデータを照合した。

| Windows x64の新規環境 | 導入方法 | Irisの結果 |
| --- | --- | --- |
| Python 3.12.16 | wheel、依存はPyPIから取得 | SVMを選定、保留30件中29件正解 |
| Python 3.13.17 | ソース配布物からビルド、取得済み依存でオフライン導入 | SVMを選定、保留30件中29件正解 |

両環境で `pip check` と、Irisの点検・比較結果の検証が通った。ここでの記録は作業端末から配布候補を導入した結果。TestPyPIから取得する試験は次の手順で行う。

## TestPyPIで導入を確認する

TestPyPIは本番PyPIとはアカウントとAPIトークンが別。TestPyPI側のAPIトークンを、Twineが求めたときに入力する。 [TestPyPIの公式手順](https://packaging.python.org/en/latest/guides/using-testpypi/)

```powershell
python -m twine upload --repository-url https://test.pypi.org/legacy/ --username __token__ $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'TestPyPI upload failed' }

$trial = "runs/testpypi-$version-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
python -m venv "$trial/.venv"
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$testPython = "$trial/.venv/Scripts/python.exe"
& $testPython -m pip --isolated download --no-cache-dir --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: --dest "$trial/download" "$name==$version"
if ($LASTEXITCODE -ne 0) { throw 'TestPyPI download failed' }
$download = Join-Path "$trial/download" (Split-Path -Leaf $wheel)
if ((Get-FileHash -LiteralPath $download).Hash -ne (Get-FileHash -LiteralPath $wheel).Hash) {
    throw 'The downloaded wheel differs from the checked wheel'
}
& $testPython -m pip --isolated install --index-url https://pypi.org/simple/ $download
if ($LASTEXITCODE -ne 0) { throw 'Install failed' }
& $testPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $testPython -I scripts/smoke_pypi_install.py --version $version --csv data/reference/iris.csv --output "$trial/iris"
if ($LASTEXITCODE -ne 0) { throw 'Iris smoke test failed' }
```

確認用スクリプトは、インストール先から対象バージョンを読み、点検と5モデル比較を実行する。続けてExcel・配列・manifestの整合を検証する。ColabではTestPyPIから取得した同じwheelを試用ノートブックで選べる。 [Colabでの試用](PYPI_ALPHA_TRIAL.md#colabやローカルjupyterで試す)

## 本番PyPIに公開する

TestPyPIで確認したwheelとソース配布物を、本番PyPIのアカウントとトークンでアップロードする。公開したファイル名は再利用できないため、内容を修正した場合は新しいバージョンでビルドする。 [本番PyPIの公開手順](https://docs.pypi.org/publishing/publishing-to-pypi/) · [同じファイル名での再アップロード](https://pypi.org/help/#file-name-reuse)

```powershell
python -m twine upload --repository-url https://upload.pypi.org/legacy/ --username __token__ $wheel $sdist
if ($LASTEXITCODE -ne 0) { throw 'PyPI upload failed' }
```

公開後は別の新規環境で `python -m pip install "diy-automl==0.4.0a8"` を実行し、同じIrisの確認を行う。GitHubのソースと公開版が対応するコミットをリリース記録に残す。
