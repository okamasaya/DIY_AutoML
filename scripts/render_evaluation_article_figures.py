"""Render article figures from frozen results; never fit or select a model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'diy-automl-mpl'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from diy_automl.dl_repeat_validation import validate_dl_repeat
from diy_automl.uncertainty_validation import validate_uncertainty


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'validation/dl-backend-preparation-0.4.0a7'
DATASETS = ('iris', 'wine', 'breast_cancer')
NAMES = {'iris': 'Iris', 'wine': 'Wine', 'breast_cancer': '乳がん'}
COLORS = ('#2465a8', '#b05c00', '#277a57')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'images/diy-automl')
    parser.add_argument('--metadata', type=Path,
                        default=ROOT / 'docs/figures/evaluation-article-data.json')
    args = parser.parse_args()
    inputs = [Path(__file__).resolve()]
    for dataset in DATASETS:
        inputs.append(ROOT / f'data/reference/{dataset}.csv')
        for directory in (BASE / dataset, BASE / f'{dataset}-uncertainty'):
            inputs.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    records, arrays, histories = [], {}, {}
    for dataset in DATASETS:
        validate_dl_repeat(BASE / dataset)
        manifest = validate_uncertainty(BASE / f'{dataset}-uncertainty')
        assert manifest['settings']['resampling'] == 'stratified'
        table = pd.read_excel(BASE / f'{dataset}-uncertainty/report.xlsx',
                              sheet_name='Metric_Uncertainty')
        histories[dataset] = pd.read_excel(BASE / f'{dataset}/report.xlsx',
                                           sheet_name='DL_History')
        with np.load(BASE / f'{dataset}-uncertainty/arrays.npz', allow_pickle=False) as saved:
            arrays[dataset] = {key: saved[key] for key in saved.files}
        data = arrays[dataset]
        for index, seed in enumerate((42, 43)):
            prefix = f'evaluation_{index}__'
            truth, pred = data[prefix + 'truth'], data[prefix + 'predictions']
            samples = data[prefix + 'samples']
            row = table[table.Evaluation.eq(f'evaluation_{index}') & table.Metric.eq('F1')].iloc[0]
            labels = np.arange(len(manifest['classes']))
            np.testing.assert_allclose(row.Estimate,
                f1_score(truth, pred, labels=labels, average='macro', zero_division=0), atol=1e-12)
            # Check stored draws against an independent sklearn scoring path.
            for draw_index in (0, 500, 999):
                draw = data[prefix + 'indices'][draw_index]
                np.testing.assert_allclose(samples[draw_index, 3],
                    f1_score(truth[draw], pred[draw], labels=labels,
                             average='macro', zero_division=0), atol=1e-12)
            records.append(dict(dataset=dataset, training_seed=seed, test_rows=len(truth),
                estimate=float(row.Estimate), se=float(row['Bootstrap SE']),
                lower=float(row.Lower), upper=float(row.Upper),
                bootstrap_repeats=int(row.Replicates), bootstrap_seed=43))

    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name):
        path = args.output_dir / name
        fig.savefig(path, dpi=180, facecolor='white')
        plt.close(fig)
        outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharex=True, sharey=True)
    positions = (5.5, 4.6, 3.2, 2.3, .9, 0.)
    for ax in axes:
        ax.set(xlim=(.79, 1.015), ylim=(-.7, 6.2), xlabel='test macro F1')
        ax.set_xticks((.8, .85, .9, .95, 1.))
        ax.grid(axis='x', alpha=.2)
    for i, (record, y) in enumerate(zip(records, positions)):
        value = record['estimate']
        for ax in axes:
            ax.scatter(value, y, color=COLORS[i // 2], s=48, zorder=3)
        axes[1].hlines(y, record['lower'], record['upper'], color=COLORS[i // 2], lw=2.4)
        axes[1].plot([record['lower'], record['upper']], [y, y], '|',
                     color=COLORS[i // 2], markersize=10)
        axes[0].annotate(f'{value:.3f}', (value, y), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=10)
    axes[0].set_yticks(positions, [f"{NAMES[r['dataset']]} / seed {r['training_seed']}\n"
                                 f"test {r['test_rows']}件" for r in records])
    axes[0].set_title('A：点推定だけを見る')
    axes[1].set_title('B：同じ点推定に、条件付き95%区間を添える')
    fig.suptitle('同じ保存済みMLPの結果でも、誤差表示で伝わる情報が変わる', fontsize=14)
    fig.text(.5, .025, '固定モデル・観測クラス件数固定の行bootstrap：1,000反復、seed 43\n'
             '学習・選定のやり直し、群や時間の依存、数値・系統誤差は含まない。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save(fig, 'evaluation-with-and-without-uncertainty.png')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    history_summary = []
    for ax, seed in zip(axes, (42, 43)):
        data = histories['breast_cancer'].query('`Training Seed` == @seed')
        ax.plot(data.Epoch, data.loss, label='学習 loss', color=COLORS[0], lw=2)
        ax.plot(data.Epoch, data.val_loss, label='検証 loss', color=COLORS[1], lw=2)
        ax.set(title=f'学習 seed {seed}', xlabel='epoch', xlim=(1, 100))
        ax.grid(alpha=.2)
        ax.legend()
        best = data.loc[data.val_loss.idxmin()]
        final = data.iloc[-1]
        history_summary.append(dict(training_seed=seed, minimum_validation_epoch=int(best.Epoch),
            minimum_validation_loss=float(best.val_loss), final_training_loss=float(final.loss),
            final_validation_loss=float(final.val_loss)))
    axes[0].set_ylabel('交差エントロピー損失')
    fig.suptitle('途中経過を残す：乳がんデータの学習・検証損失', fontsize=14)
    fig.text(.5, .025, '分割seed 42・100 epoch固定。全2学習seedを表示。\n'
             'この図を見てepochやseedを選び直した結果ではない。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save(fig, 'evaluation-training-history.png')

    points = arrays['breast_cancer']['evaluation_0__samples'][:, [1, 2]]
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    assert (eigenvalues > 0).all()
    delta = points - center
    distances = np.einsum('ij,jk,ik->i', delta, np.linalg.inv(covariance), delta)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    fig, ax = plt.subplots(figsize=(6.5, 6.4))
    ax.scatter(*points.T, s=15, color=COLORS[0], alpha=.25, edgecolors='none')
    ellipse_records = []
    for fraction, color in zip((.99, .95, .90), ('#ab463b', '#b05c00', '#2465a8')):
        threshold = float(np.quantile(distances, fraction, method='higher'))
        width, height = 2 * np.sqrt(eigenvalues[::-1] * threshold)
        coverage = float(np.mean(distances <= threshold))
        ellipse = Ellipse(center, float(width), float(height), angle=float(angle),
                          fill=False, edgecolor=color, lw=1.7,
                          label=f'反復点の約{fraction:.0%}を囲む楕円')
        ax.add_patch(ellipse)
        ellipse_records.append(dict(fraction=fraction, actual_fraction=coverage,
                                     squared_distance_threshold=threshold))
    point = next(r for r in records if r['dataset'] == 'breast_cancer' and r['training_seed'] == 42)
    ax.set(xlabel='macro Precision', ylabel='macro Recall', xlim=(.75, 1.25), ylim=(.75, 1.25),
           title='2指標の連動：同じ行を再標本化した1,000点')
    ax.set_xticks(np.linspace(.75, 1.25, 6))
    ax.set_yticks(np.linspace(.75, 1.25, 6))
    ax.axvline(1, color='#777777', ls='--', lw=1, label='指標の上限（1）')
    ax.axhline(1, color='#777777', ls='--', lw=1)
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', fontsize=9)
    fig.text(.5, .025, f"乳がん・固定MLP seed 42・test {point['test_rows']}件・クラス件数固定\n"
             '指標は0〜1。楕円は1を超える部分も表示する。\n'
             '反復点の要約であり、母数の95%信頼領域ではない。',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, 1))
    save(fig, 'evaluation-joint-bootstrap.png')

    assert before == {str(p.relative_to(ROOT)): digest(p) for p in inputs}, 'Inputs changed'
    evidence = dict(status='passed', retrained=False, source_files_unchanged=True,
        input_sha256=before, versions={name: importlib.metadata.version(name)
            for name in ('diy-automl', 'numpy', 'pandas', 'matplotlib', 'scikit-learn')},
        validated_saved_runs=6, independent_sklearn_checks=24, estimates=records,
        breast_cancer_history=history_summary,
        joint_bootstrap=dict(dataset='breast_cancer', training_seed=42,
            axis_limits=dict(x=[.75, 1.25], y=[.75, 1.25]),
            ellipse_clipped_to_metric_domain=False,
            correlation=float(np.corrcoef(points, rowvar=False)[0, 1]),
            method='Empirical quantiles of squared Mahalanobis distances around bootstrap mean; '
                   'descriptive point coverage, not parameter confidence coverage',
            ellipses=ellipse_records),
        output_sha256={path.name: digest(path) for path in outputs})
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(f'Validated 6 saved runs; rendered {len(outputs)} figures without retraining.')
    print(args.metadata)


if __name__ == '__main__':
    main()
