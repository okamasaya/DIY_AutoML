# UL初期化感度

`repeat-ul` は標準化済みの同一データに対してKMeansとGMMだけをseed反復します。階層法は1回だけです。既定seed42..51、k2..10です。全体を最大5,000行に制限しています。

6シートの既定行数: UL_By_Seed 180、UL_Seed_Summary 18、UL_Seed_Pairs 810、UL_Pair_Summary 18、UL_Reference 36、Summary 9。

Target ARIは正解ラベルとの一致です。Assignment ARIはseed間割当一致です。seed対のmean/std/min/maxは依存した対の記述統計であり、独立標本の推測統計ではありません。SEM/CIを付けず、自動的に最良候補を選びません。

scalerと各seed・各候補の全ラベルをNPZに保存します。Excelにはモデルパラメータ、収束、反復回数、警告も残します。例外を起こした候補はfailed、その後はnot_run、既に終了した候補はcompletedです。全体manifestはfailedになります。非収束警告は例外とは区別し、Converged=falseと警告を明示して結果を保存します。
