"""Render the training-only descriptive associations used in both article drafts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
font = 'Noto Sans CJK JP' if any(f.name == 'Noto Sans CJK JP' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': font, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
panels = [
    ('hotel', 'lead_time', 'ホテル：予約から宿泊まで', ['0–7日', '8–30日', '31–90日', '91日以上'], 'キャンセル率'),
    ('coupon', 'coffee_monthly', 'カフェ券：普段の来店頻度', ['月1回未満', '月1回以上'], '受取回答率'),
    ('retail', 'monetary_gbp', '顧客：過去の購入金額', ['上位25%以外', '上位25%'], '9–10月の再購入率')]
data = []
for ax, (name, feature, title, labels, ylabel) in zip(axes, panels):
    table = pd.read_csv(args.results / name / 'training_associations.csv')
    table = table[table.Feature == feature].reset_index(drop=True)
    ax.bar(np.arange(len(table)), table.Rate * 100, color=['#2563a6', '#3290a8', '#e5a14b', '#cc6c3b'][:len(table)], width=.63)
    ax.set(xticks=np.arange(len(table)), xticklabels=labels, ylim=(0, 100), ylabel=ylabel + '（%）', title=title)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    for i, row in table.iterrows():
        ax.text(i, row.Rate * 100 + 2, f'{row.Rate:.1%}\n{int(row.Positive)}/{int(row.N)}', ha='center', fontsize=10)
        data.append({'dataset': name, **row.to_dict(), 'label': labels[i]})
fig.suptitle('事前に決めた見方と照合：学習側データでの記述的な割合', fontsize=15, y=.98)
fig.text(.5, .035, '各棒は陽性件数/対象件数。要因の因果効果や独立した外部検証を示す図ではありません。', ha='center', fontsize=10)
fig.tight_layout(rect=(0, .08, 1, .92))
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160, facecolor='white')
plt.close(fig)
output = Path(__file__).parent
pd.DataFrame(data).to_csv(output / 'figure_data.csv', index=False)
(output / 'figure_manifest.json').write_text(json.dumps({'kind': 'training_association_figure', 'font': font,
    'source_manifest_sha256': hashlib.sha256((args.results / 'experiment_manifest.json').read_bytes()).hexdigest(),
    'plot_data_sha256': hashlib.sha256((output / 'figure_data.csv').read_bytes()).hexdigest(),
    'png_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2) + '\n')
print(args.output)
