# GCNで既知グラフのノードを分類する

`gnn-compare` は、Cora形式のノード特徴量だけを使うロジスティック回帰と、特徴量と辺を使う2層GCNを共通分割で比較する。Pythonの入口は `diy_automl.classify_graph_nodes(nodes, edges, output, ...)`。実装は既存のNumPy・SciPyとscikit-learnで動く。 [GCNの原論文](https://arxiv.org/abs/1609.02907)

## 入力と評価条件

- ノードファイルはヘッダーなしのTSVで、`ノードID → 1列以上の数値特徴量 → クラスラベル` の順。ノードIDは識別に使い、特徴量は両手法で行ごとにL1正規化する。
- 辺ファイルはヘッダーなしのTSVで、1行に既知のノードIDを2つ書く。辺は無向化し、重複をまとめる。GCNの集約では各ノードに自己ループを1本加え、次数で対称正規化する。
- 初版は既知グラフ内の分類（transductive設定）を扱う。全ノードの特徴量と辺が見える条件で、学習用ラベルから重みを更新する。選定用ラベルでロジスティック回帰の `C`（1・10・100）、GCNの保存エポック、手法をmacro F1で選ぶ。保留ラベルは両手法の最終採点に使う。
- 初期値は各クラス20ノードを学習、残りの約20%をクラスごとに選定、残りを保留に割り当てる。GCNは隠れ層16、120エポック、Adamの学習率0.01、第1層のL2係数0.0005。コマンドから分割seed、学習件数、選定の割合、隠れ層の幅、エポック数を変更できる。

## Coraでの確認結果

LINQSの `cora.tgz` を使った。入力は論文2,708件、特徴量1,433列、7クラス、引用の入力辺5,429行。今回の無向化・重複整理後は5,278辺。各分割の学習・選定・保留は140・514・2,054件。使用したアーカイブのSHA-256は `0d4ed463d1627bb7f3e8420effe8f5545fd492ae8f88dab44ce86cee7b26d7e8`。 [LINQSの公開データ](https://linqs.org/datasets/) · [Coraアーカイブ](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

| seed | ロジスティック回帰 保留macro F1 | GCN 保留macro F1 | ロジスティック回帰 学習・選定（秒） | GCN 学習・選定（秒） |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.5555 | 0.7527 | 0.16 | 0.53 |
| 43 | 0.5708 | 0.8068 | 0.17 | 0.45 |
| 44 | 0.5672 | 0.7771 | 0.14 | 0.40 |

3分割すべてで選定区間がGCNを選び、保留macro F1もGCNが上回った。同じseed 42の分割とGCN設定で引用エッジを外し、自己ループだけにすると、保留macro F1は0.5102だった。処理時間はこのWindows環境での実測で、入力読み込み・Excel保存を除く。評価対象は一つの既知グラフ。

## 実行と保存結果

```powershell
$python = '.\.venv\Scripts\python.exe'
$cora = 'runs/gnn-cora'
New-Item -ItemType Directory -Path $cora -Force
Invoke-WebRequest 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' -OutFile "$cora/cora.tgz"
tar -xzf "$cora/cora.tgz" -C $cora cora/cora.content cora/cora.cites
& $python -X utf8 -m diy_automl gnn-compare --nodes "$cora/cora/cora.content" --edges "$cora/cora/cora.cites" --output "$cora/seed-42" --seed 42
& $python -X utf8 -c "from diy_automl.validation import validate; print(validate('runs/gnn-cora/seed-42')['selected_model'])"
```

出力先には新しいディレクトリを指定する。エッジを外す比較では、`--edges` に空のTSVを渡し、同じseedと新しい出力先を指定する。`report.xlsx` に指標、選定の経過、保留ノード別の予測と計算時間、`arrays.npz` に分割と予測、`manifest.json` に設定・ファイルのハッシュを保存する。ローカル実測は `runs/gnn-cora-20261124` にある。
