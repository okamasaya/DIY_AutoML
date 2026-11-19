# DIY AutoML Full Paper

2026年11月19日の日本語技術論文初稿。比較した実装の参照点は `80b25cc` です。原稿を変更する際は `main.tex` と `sections/` を編集してください。

## 組版

プロジェクトのフォルダーでXeLaTeXを2回実行します。Tectonicを使う場合は `tectonic main.tex` で参照の解決まで行います。

~~~sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

日本語の書体は原ノ味、游明朝・游ゴシック、Noto CJKのうち利用できるものを使います。図ファイルは `images/` に同梱しています。

## 図の再生成

`figure-data.json` は、Bike Sharingの保留期間の月別評価表と、Windowsの処理負荷計測を元にした集計値です。MatplotlibのあるPython環境で次を実行すると、2点のベクターPDFを作り直せます。

~~~sh
python build_figures.py
~~~

モデル比較やデータ整形の再実行条件は、論文の付録と参考文献を参照してください。
