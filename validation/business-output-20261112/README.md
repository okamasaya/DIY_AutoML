# BP記事の実物出力と補足集計

入口は [iris/iris_business_report.xlsx](iris/iris_business_report.xlsx)。日本語の先頭3シートと、標準の誤差評価表・区間図を含む。先頭シートのみを読む場合は [PDF](iris/summary.pdf)。

- `iris/input.csv`：入力したIris 150件。
- `iris/selected`、`illustrated`、`uncertainty`：現行APIで生成・検証した選定、図付きExcel、誤差評価の元出力。
- `iris/example_summary.json`：要約の数値とハッシュ。評価30件中29件正解、1件誤り。
- `comparison`：既存の業務3テーマの予測から、同じ行を取り出して求めたモデル−参考ルールのmacro F1差。反復値・index・条件・広い軸範囲の図を保存。

日本語要約とモデル差の集計は記事用で、汎用GUIやパッケージの新機能ではない。Irisだけを新規学習し、過去の業務3テーマの実験出力は変更していない。再作成方法と条件付き評価の範囲は [BP_OUTPUT_EXAMPLE.md](../../docs/BP_OUTPUT_EXAMPLE.md) を参照。
