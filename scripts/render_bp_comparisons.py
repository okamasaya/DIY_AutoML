"""Article supplement: paired differences from frozen business predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from diy_automl.metric_uncertainty import resample_indices, score_samples
from diy_automl.validation import validate


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('hotel', 'coupon', 'retail')
LABELS = ('ホテル', 'カフェ券*', '再購入')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    if args.image.exists():
        raise FileExistsError(args.image)
    args.output.mkdir(parents=True, exist_ok=False)
    records, stored, inputs = [], {}, {}
    for dataset in DATASETS:
        source = ROOT / 'validation/business-quick-20261108' / dataset
        validate(source / 'selected', 3)
        paths = [source / 'comparison_predictions.npz', source / 'comparisons.csv',
                 source / 'selected/arrays.npz', source / 'selected/manifest.json']
        inputs.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
        with np.load(source / 'comparison_predictions.npz', allow_pickle=False) as saved:
            truth, pred, rule = [saved[k] for k in ('truth', 'selected_predictions', 'reference_1')]
            with np.load(source / 'selected/arrays.npz', allow_pickle=False) as selected:
                np.testing.assert_array_equal(truth, selected['truth'])
                np.testing.assert_array_equal(pred, selected['predictions'])
                np.testing.assert_array_equal(saved['test_indices'], selected['test'])
        indices = resample_indices(truth, repeats=2000, seed=43, resampling='stratified')
        point, _, samples, _, _ = score_samples(truth, pred, np.array([0, 1]), indices)
        reference, _, reference_samples, _, _ = score_samples(truth, rule, np.array([0, 1]), indices)
        differences = samples[:, 3] - reference_samples[:, 3]
        estimate = float(point[3] - reference[3])
        lower, upper = map(float, np.quantile(differences, [.025, .975], method='linear'))
        comparisons = pd.read_csv(source / 'comparisons.csv')
        np.testing.assert_allclose(point[3], comparisons.iloc[0].F1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference[3], comparisons.iloc[2].F1, rtol=0, atol=1e-12)
        for index in (0, 999, 1999):
            draw = indices[index]
            expected = (f1_score(truth[draw], pred[draw], average='macro')
                        - f1_score(truth[draw], rule[draw], average='macro'))
            np.testing.assert_allclose(differences[index], expected, rtol=0, atol=1e-12)
        records.append(dict(dataset=dataset, test_rows=len(truth), model_f1=float(point[3]),
            rule_f1=float(reference[3]), difference=estimate, lower=lower, upper=upper,
            reference_name=str(comparisons.iloc[2].Method)))
        stored[f'{dataset}_indices'] = indices
        stored[f'{dataset}_differences'] = differences
    assert all(sha(ROOT / p) == digest for p, digest in inputs.items())
    np.savez_compressed(args.output / 'paired_bootstrap.npz', **stored)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11,
                         'axes.unicode_minus': False})

    def draw(limits, destination):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
        for ax in axes:
            ax.axvline(0, color='#666666', ls='--', lw=1)
            ax.set(xlim=limits, ylim=(-.6, 2.8), xlabel='比較用スコアの差（macro F1）')
            ax.set_yticks([2, 1, 0], LABELS)
            ax.grid(axis='x', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        for y, record in zip([2, 1, 0], records):
            value = record['difference']
            for ax in axes:
                ax.scatter(value, y, s=55, color='#2465a8', zorder=3)
            axes[0].annotate(f'{value:+.3f}', (value, y), xytext=(0, 12),
                             textcoords='offset points', ha='center')
            axes[1].hlines(y, record['lower'], record['upper'], color='#2465a8', lw=2)
            axes[1].plot([record['lower'], record['upper']], [y, y], '|', color='#2465a8', ms=10)
        axes[0].set_title('点の差だけを見る')
        axes[1].set_title('同じ差に、条件付き95%区間を添える')
        fig.suptitle('学習モデル − 参考ルール：0より右がモデル側の高い点数', fontsize=14)
        fig.text(.5, .025, '固定モデル・クラス件数固定・対応を保つ行bootstrap：2,000反復、seed 43\n'
                 '* カフェ券は同一回答者の依存を扱えていない。区間は新しい回答者への保証ではない。',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .14, 1, .93))
        fig.savefig(destination, dpi=180, facecolor='white')
        plt.close(fig)

    # Inspect the full possible difference range before choosing the close view.
    draw((-1, 1), args.output / 'overview.png')
    limit = float(np.ceil(max(abs(r[k]) for r in records for k in ('lower', 'upper')) * 1.15 * 100) / 100)
    assert all(-limit < r['lower'] <= r['upper'] < limit for r in records)
    args.image.parent.mkdir(parents=True, exist_ok=True)
    draw((-limit, limit), args.image)
    report = dict(status='passed', records=records, repeats=2000, seed=43,
        scope='Conditional fixed-model paired class-count-preserving row bootstrap; excludes training, group/time dependence and causal effects',
        article_supplement_not_package_feature=True, overview_limits=[-1, 1], display_limits=[-limit, limit],
        input_sha256=inputs, script_sha256=sha(__file__), image_sha256=sha(args.image),
        arrays_sha256=sha(args.output / 'paired_bootstrap.npz'), source_files_unchanged=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
