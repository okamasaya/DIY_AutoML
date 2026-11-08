"""Check saved results and reconstruct sampled inputs from pinned raw records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.selection import choose
from diy_automl.supervised import metrics
from diy_automl.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--raw-dir', type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Do not run with python -O')
    root = args.directory
    m = json.loads((root / 'experiment_manifest.json').read_text())
    assert m['status'] == 'completed' and set(m['cases']) == {'hotel', 'coupon', 'retail'}
    assert hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() == m['protocol_sha256']
    for name, expected in m['artifacts'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    for name, result in m['cases'].items():
        folder = root / name
        frame = pd.read_csv(folder / 'input.csv')
        rows = pd.read_csv(folder / 'row_mapping.csv')
        selected = validate(folder / 'selected', 3)
        validate_excel_report(folder / 'illustrated')
        assert len(frame) == len(rows) == result['sample_rows'] <= 2000
        assert frame.columns.tolist() == result['features'] + ['target']
        assert np.isfinite(frame.to_numpy()).all()
        scores = pd.read_excel(folder / 'selected/report.xlsx', sheet_name='SL_CV_Folds')
        means = scores.groupby('Model', sort=False)[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()
        index, _ = choose(means)
        assert means.iloc[index].Model == selected['selected_model'] == result['selected_model']
        with np.load(folder / 'selected/arrays.npz') as a, np.load(folder / 'comparison_predictions.npz') as b:
            train, test = a['train'], a['test']
            assert sorted(np.concatenate([train, test])) == list(range(len(frame)))
            assert not set(train) & set(test)
            assert np.array_equal(a['truth'], frame.target.iloc[test].to_numpy())
            assert np.array_equal(a['truth'], b['truth']) and np.array_equal(a['predictions'], b['selected_predictions'])
            assert np.array_equal(test, b['test_indices'])
            assert np.allclose(a['scaler_mean'], frame[result['features']].iloc[train].mean(), atol=1e-10, rtol=0)
            assert np.allclose(a['scaler_scale'], frame[result['features']].iloc[train].std(ddof=0).replace(0, 1), atol=1e-10, rtol=0)
            for key, value in metrics(a['truth'], a['predictions'], [0, 1]).items():
                assert np.isclose(result['selected_test'][key], value, atol=1e-12, rtol=0)
            assert confusion_matrix(a['truth'], a['predictions'], labels=[0, 1]).ravel().tolist() == [result['selected_test'][key] for key in ['tn', 'fp', 'fn', 'tp']]
            for i, reference in enumerate(result['comparisons'][1:]):
                for key, value in metrics(b['truth'], b[f'reference_{i}'], [0, 1]).items():
                    assert np.isclose(reference[key], value, atol=1e-12, rtol=0)
            association = pd.read_csv(folder / 'training_associations.csv')
            for _, group in association.groupby('Feature'):
                assert group.N.sum() == len(train) and group.Positive.sum() == frame.target.iloc[train].sum()
                assert np.allclose(group.Rate, group.Positive / group.N)
        if args.raw_dir is not None:
            from run import load_raw, prepare
            protocol = json.loads((root / 'protocol.json').read_text())
            rebuilt, rebuilt_rows, _ = prepare(name, load_raw(args.raw_dir, name), protocol)
            assert np.allclose(rebuilt.to_numpy(), frame.to_numpy(), rtol=1e-12, atol=1e-10)
            pd.testing.assert_frame_equal(rebuilt_rows.astype(str), rows.astype(str), check_dtype=False)
            if name == 'retail':
                raw = pd.read_excel(args.raw_dir / 'Online_Retail.xlsx')
                positive = raw[raw.CustomerID.notna() & (raw.Country == 'United Kingdom') &
                               (raw.Quantity > 0) & (raw.UnitPrice > 0) &
                               ~raw.InvoiceNo.astype(str).str.upper().str.startswith('C')].drop_duplicates()
                future_ids = set(positive.loc[(positive.InvoiceDate >= '2011-09-01') &
                                             (positive.InvoiceDate < '2011-11-01'), 'CustomerID'].astype(int))
                assert np.array_equal(rows.customer_id.isin(future_ids).astype(int), frame.target)
                assert rows.customer_id.is_unique
                assert set(rows.customer_id.iloc[train]).isdisjoint(rows.customer_id.iloc[test])
                for _, sampled in rows.iterrows():
                    history = positive[(positive.CustomerID == sampled.customer_id) &
                                       (positive.InvoiceDate >= '2011-06-01') & (positive.InvoiceDate < '2011-09-01')]
                    i = sampled.name
                    assert len(history) > 0
                    assert frame.loc[i, 'frequency_invoices'] == history.InvoiceNo.nunique()
                    assert np.isclose(frame.loc[i, 'monetary_gbp'], (history.Quantity * history.UnitPrice).sum())
        raw_scope = 'raw reconstruction checked' if args.raw_dir is not None else 'raw reconstruction not requested'
        print(name + ': saved metrics, split, scaler, references and images passed; ' + raw_scope, flush=True)


if __name__ == '__main__':
    main()
