"""Uniform sampling without replacement and index-aligned clustering comparisons."""
from itertools import combinations
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

MAX_ROWS = 2000
MAX_FEATURES = 100
MAX_FITS = 2000
COMPARE_COLUMNS = ['Shared Rows', 'Left Clusters', 'Right Clusters', 'Comparison Status', 'Assignment ARI']
AGGREGATE_METRICS = ['Target ARI', 'Reference ARI', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']


def integer(value, name, minimum=0, maximum=2**32-1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def configuration(sample_seeds, fraction, model_seed, clusters, min_overlap):
    seeds = [integer(s, 'sample seed') for s in sample_seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct sample seeds are required')
    ks = [integer(k, 'cluster count', 2, 10) for k in clusters]
    if not ks or len(set(ks)) != len(ks):
        raise ValueError('Cluster counts must be distinct')
    if isinstance(fraction, bool) or not isinstance(fraction, Real) or not 0 < fraction < 1:
        raise ValueError('fraction must be strictly between 0 and 1')
    if 6 * len(ks) * (len(seeds)+1) > MAX_FITS:
        raise ValueError(f'At most {MAX_FITS} candidate fits are supported')
    return seeds, float(fraction), integer(model_seed, 'model seed'), ks, integer(min_overlap, 'min_overlap', 2)


def sample_plan(X, seeds, fraction, clusters, min_overlap):
    n, width = X.shape
    if n > MAX_ROWS or width > MAX_FEATURES:
        raise ValueError(f'Subsampling supports at most {MAX_ROWS} rows and {MAX_FEATURES} features')
    size = int(np.floor(n*fraction))
    if size <= max(clusters) or size < min_overlap:
        raise ValueError('Subsample must exceed every cluster count and meet min_overlap')
    # Sorted source indices separate row omission from random row-order changes.
    return {seed: np.sort(np.random.default_rng(seed).choice(n, size, replace=False)) for seed in seeds}


def compare_assignments(left_indices, left_labels, right_indices, right_labels, min_overlap=10):
    shared, a, b = np.intersect1d(left_indices, right_indices, return_indices=True)
    left = np.asarray(left_labels)[a]
    right = np.asarray(right_labels)[b]
    lc, rc = len(np.unique(left)), len(np.unique(right))
    if len(shared) < min_overlap:
        status, ari = 'insufficient_overlap', np.nan
    elif min(lc, rc) < 2:
        status, ari = 'degenerate', np.nan
    else:
        status, ari = 'valid', float(adjusted_rand_score(left, right))
    return dict(zip(COMPARE_COLUMNS, [len(shared), lc, rc, status, ari]))


def sample_summary(results):
    rows = []
    for candidate, group in results.groupby('Candidate', sort=False):
        done = group[group.Status == 'completed']
        row = {'Candidate': candidate, 'Model': group.Model.iloc[0], 'Cluster': group.Cluster.iloc[0],
               'Completed': len(done), 'Failed': int(group.Status.eq('failed').sum()),
               'Not Run': int(group.Status.eq('not_run').sum()),
               'Running': int(group.Status.eq('running').sum())}
        for status in ['valid', 'degenerate', 'insufficient_overlap']:
            row['Reference '+status] = int(done['Reference Status'].eq(status).sum())
        for metric in AGGREGATE_METRICS:
            values = done[metric].dropna().astype(float)
            for label, value in [('Count',len(values)), ('Mean',values.mean()), ('SD',values.std(ddof=1)),
                                 ('Min',values.min()), ('Max',values.max())]:
                suffix = 'Across_Subsamples_SD' if label == 'SD' else label
                row[f'{metric}_{suffix}'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_tables(results, arrays, seeds, min_overlap, settings):
    pairs = []
    completed = {(int(row['Sample Seed']), int(row.Candidate)) for _,row in results.iterrows()
                 if row.Status == 'completed'}
    for a,b in combinations(seeds,2):
        for _,candidate in settings.iterrows():
            cid = int(candidate.Candidate)
            if (a,cid) not in completed or (b,cid) not in completed:
                continue
            pairs.append({'Candidate':cid,'Model':candidate.Model,'Cluster':int(candidate.Cluster),
                          'Seed A':a,'Seed B':b,
                          **compare_assignments(arrays[f'sample_{a}_indices'],arrays[f'sample_{a}_labels_{cid}'],
                                                arrays[f'sample_{b}_indices'],arrays[f'sample_{b}_labels_{cid}'],min_overlap)})
    frame = pd.DataFrame(pairs, columns=['Candidate','Model','Cluster','Seed A','Seed B',*COMPARE_COLUMNS])
    summary = []
    for _,candidate in settings.iterrows():
        group = frame[frame.Candidate == candidate.Candidate]
        values = group.loc[group['Comparison Status'] == 'valid','Assignment ARI'].astype(float)
        summary.append({'Candidate':int(candidate.Candidate),'Model':candidate.Model,'Cluster':int(candidate.Cluster),
                        'Pairs':len(group),'Valid':len(values),
                        'Degenerate':int(group['Comparison Status'].eq('degenerate').sum()),
                        'Insufficient Overlap':int(group['Comparison Status'].eq('insufficient_overlap').sum()),
                        'Mean':values.mean(),'SD':values.std(ddof=1),'Min':values.min(),'Max':values.max()})
    return frame,pd.DataFrame(summary)
