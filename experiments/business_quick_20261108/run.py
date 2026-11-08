"""Bounded, numeric-only exploratory study; leaves the package defaults unchanged."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from diy_automl import __version__
from diy_automl.excel_report import render_excel_report
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.reporting import source_hashes, write_json
from diy_automl.selection_workflow import select_csv
from diy_automl.supervised import metrics
from diy_automl.validation import validate


ROOT = Path(__file__).resolve().parent
SOURCES = {
    'hotel': dict(file='hotel_bookings.csv', rows=119390, columns=32,
        sha256='7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06',
        git_blob='1dabe01810f742e874a73346d2ae9f28d7e13327',
        url='https://raw.githubusercontent.com/rfordatascience/tidytuesday/d75aaa0d31596ad6487ae0db20138067d301e031/data/2020/2020-02-11/hotels.csv',
        original='https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/'),
    'coupon': dict(file='coupon.csv', rows=12684, columns=26,
        sha256='87e841f5d41cdaf31a09d0a0770187b2fc9cabe4171c30bf7ea2182dd074abaa',
        git_blob='07b54ba1695ae1b4b7d64cc2df8e9fd5d9e49f06',
        url='https://raw.githubusercontent.com/artessari/MLProject_In-VehicleCouponRecommendation/c1e183fc5b2de34b5cd75759eb322aaab246bbc2/in-vehicle-coupon-recommendation.csv',
        original='https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation'),
    'retail': dict(file='Online_Retail.xlsx', rows=541909, columns=8,
        sha256='43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d',
        git_blob='a58636c3714c4cca4e2e1cfe9a1fcb28af30494e',
        url='https://raw.githubusercontent.com/SarahMestiri/online-retail-case/ba0c6609ad283ce1aa08e627dcd10654358a9af5/Online%20Retail.xlsx',
        original='https://archive.ics.uci.edu/dataset/352/online+retail')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(raw_dir, name):
    info = SOURCES[name]
    path = raw_dir / info['file']
    content = path.read_bytes()
    assert digest(path) == info['sha256'], 'Unexpected downloaded bytes'
    assert hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == info['git_blob']
    frame = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
    assert frame.shape == (info['rows'], info['columns']), 'Unexpected raw dimensions'
    frame.index.name = 'raw_row'
    return frame


def prepare(name, raw, protocol):
    stats = {'raw_rows': len(raw), 'exact_duplicate_rows': int(raw.duplicated().sum())}
    d = raw.drop_duplicates().copy()
    stats['rows_after_exact_deduplication'] = len(d)
    if name == 'hotel':
        fields = protocol[name]['features']
        good = d[fields + ['is_canceled']].notna().all(axis=1)
        stats['selected_field_missing_exclusions'] = int((~good).sum())
        d = d.loc[good]
        x = d[fields].copy()
        x['target'] = d.is_canceled.astype(int)
        rows = d[['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].reset_index()
    elif name == 'coupon':
        d = d[d.coupon == 'Coffee House'].copy()
        stats['coffee_house_rows_after_deduplication'] = len(d)
        required = ['CoffeeHouse', 'expiration', 'destination', 'passanger', 'temperature', 'has_children',
                    'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same', 'Y']
        good = d[required].notna().all(axis=1)
        stats['selected_field_missing_exclusions'] = int((~good).sum())
        d = d.loc[good]
        assert set(d.CoffeeHouse).issubset({'never', 'less1', '1~3', '4~8', 'gt8'})
        x = pd.DataFrame({
            'coffee_monthly': d.CoffeeHouse.isin(['1~3', '4~8', 'gt8']).astype(int),
            'expires_1day': (d.expiration == '1d').astype(int),
            'no_urgent_destination': (d.destination == 'No Urgent Place').astype(int),
            'with_friends': (d.passanger == 'Friend(s)').astype(int),
            'with_kids': (d.passanger == 'Kid(s)').astype(int),
            'temperature_F': d.temperature,
            'has_children': d.has_children,
            'detour_ge15min': d.toCoupon_GEQ15min,
            'detour_ge25min': d.toCoupon_GEQ25min,
            'same_direction': d.direction_same,
            'target': d.Y.astype(int)})
        rows = pd.DataFrame({'raw_row': d.index})
    else:
        filters = [('missing_customer_id', d.CustomerID.notna()),
                   ('non_uk', d.Country == 'United Kingdom'),
                   ('cancellation', ~d.InvoiceNo.astype(str).str.upper().str.startswith('C')),
                   ('nonpositive_quantity_or_price', (d.Quantity > 0) & (d.UnitPrice > 0))]
        # Sequential exclusions give non-overlapping counts.
        for label, mask in filters:
            mask = mask.reindex(d.index)
            stats['excluded_' + label] = int((~mask).sum())
            d = d.loc[mask]
        d['amount'] = d.Quantity * d.UnitPrice
        d['day'] = d.InvoiceDate.dt.normalize()
        spec = protocol[name]
        history = d[(d.InvoiceDate >= spec['observation_start_inclusive']) &
                    (d.InvoiceDate < spec['observation_end_exclusive'])]
        future = d[(d.InvoiceDate >= spec['outcome_start_inclusive']) &
                   (d.InvoiceDate < spec['outcome_end_exclusive'])]
        cutoff = pd.Timestamp(spec['observation_end_exclusive'])
        grouped = history.groupby('CustomerID', sort=True)
        x = grouped.agg(frequency_invoices=('InvoiceNo', 'nunique'), monetary_gbp=('amount', 'sum'),
                        units=('Quantity', 'sum'), distinct_products=('StockCode', 'nunique'),
                        active_purchase_days=('day', 'nunique'), last_purchase=('InvoiceDate', 'max'),
                        first_purchase=('InvoiceDate', 'min'))
        x['recency_days'] = (cutoff - x.last_purchase.dt.normalize()).dt.days
        x['observed_tenure_days'] = (cutoff - x.first_purchase.dt.normalize()).dt.days
        x['target'] = x.index.isin(future.CustomerID.unique()).astype(int)
        x = x[spec['features'] + ['target']]
        rows = pd.DataFrame({'customer_id': x.index.astype('int64')})
        stats.update(positive_observation_lines=len(history), positive_outcome_lines=len(future),
                     eligible_customers=len(x), observation_start=spec['observation_start_inclusive'],
                     observation_end_exclusive=spec['observation_end_exclusive'],
                     outcome_start=spec['outcome_start_inclusive'], outcome_end_exclusive=spec['outcome_end_exclusive'])
    assert list(x.columns) == protocol[name]['features'] + ['target']
    assert not x.isna().any().any() and np.isfinite(x.to_numpy()).all()
    assert set(x.target) == {0, 1}
    stats['eligible_rows'] = len(x)
    stats['eligible_positive_rows'] = int(x.target.sum())
    positions = np.sort(np.random.default_rng(protocol['sample_seed']).choice(len(x), min(len(x), protocol['sample_rows_max']), replace=False))
    x = x.iloc[positions].reset_index(drop=True)
    rows = rows.iloc[positions].reset_index(drop=True)
    stats.update(sample_rows=len(x), sample_positive_rows=int(x.target.sum()), sample_positive_rate=float(x.target.mean()))
    return x, rows, stats


def score(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    return {**metrics(y, pred, [0, 1]), 'positive_precision': float(p[0]),
            'positive_recall': float(r[0]), 'positive_f1': float(f[0]),
            'tn': int(((y == 0) & (pred == 0)).sum()), 'fp': int(((y == 0) & (pred == 1)).sum()),
            'fn': int(((y == 1) & (pred == 0)).sum()), 'tp': int(((y == 1) & (pred == 1)).sum())}


def associations(name, train):
    result = []
    def add(feature, groups):
        for group, part in train.groupby(groups, observed=True, sort=True):
            result.append({'Feature': feature, 'Group': str(group), 'N': len(part),
                           'Positive': int(part.target.sum()), 'Rate': float(part.target.mean()), 'Scope': 'training sample only'})
    if name == 'hotel':
        add('lead_time', pd.cut(train.lead_time, [-1, 7, 30, 90, 1000000], labels=['0-7d', '8-30d', '31-90d', '91+d']))
        add('is_repeated_guest', train.is_repeated_guest)
    elif name == 'coupon':
        for feature in ['coffee_monthly', 'expires_1day', 'detour_ge15min', 'with_friends', 'no_urgent_destination']:
            add(feature, train[feature])
    else:
        for feature in ['monetary_gbp', 'frequency_invoices', 'recency_days']:
            if feature == 'frequency_invoices':
                add(feature, pd.Series(np.where(train[feature] >= 2, '2+ invoices', '1 invoice'), index=train.index))
            elif feature == 'recency_days':
                add(feature, pd.Series(np.where(train[feature] <= 30, '<=30 days', '>30 days'), index=train.index))
            else:
                q = float(train[feature].quantile(.75))
                add(feature, pd.Series(np.where(train[feature] >= q, 'top25% training threshold', 'below top25% threshold'), index=train.index))
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Do not run with python -O')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    (args.output / 'protocol.json').write_bytes((ROOT / 'protocol.json').read_bytes())
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), version=__version__,
                    python=platform.python_version(), protocol_sha256=digest(ROOT / 'protocol.json'),
                    script_sha256=digest(__file__), source_hashes=source_hashes(), sources=SOURCES,
                    source_note='Public mirrors pinned by commit and byte hashes; official UCI direct downloads returned HTTP 403; exact upstream bytes not independently verified',
                    cases={})
    write_json(args.output / 'experiment_manifest.json', manifest)
    overview = []
    try:
        for name in SOURCES:
            start = time.perf_counter()
            folder = args.output / name
            folder.mkdir()
            x, rows, stats = prepare(name, load_raw(args.raw_dir, name), protocol)
            x.to_csv(folder / 'input.csv', index=False)
            rows.to_csv(folder / 'row_mapping.csv', index=False)
            select_csv(folder / 'input.csv', 'target', folder / 'selected', seed=protocol['split_seed'])
            selected = validate(folder / 'selected', 3)
            with np.load(folder / 'selected' / 'arrays.npz') as a:
                train, test, pred, truth = (a[key].copy() for key in ['train', 'test', 'predictions', 'truth'])
                assert np.allclose(a['scaler_mean'], x.drop(columns='target').iloc[train].mean().to_numpy(), rtol=0, atol=1e-10)
            assert np.array_equal(truth, x.target.iloc[test].to_numpy())
            train_frame, test_frame = x.iloc[train], x.iloc[test]
            association = associations(name, train_frame)
            association.to_csv(folder / 'training_associations.csv', index=False)
            majority = int(train_frame.target.value_counts().idxmax())
            rules = [('Training majority', np.full(len(test), majority))]
            thresholds = {'majority_label': majority}
            if name == 'hotel':
                threshold = float(train_frame.lead_time.median())
                thresholds['lead_time_median'] = threshold
                rules.append(('Lead time >= training median', (test_frame.lead_time >= threshold).to_numpy().astype(int)))
            elif name == 'coupon':
                rules.append(('Published coffee context rule', ((test_frame.coffee_monthly == 1) &
                              (((test_frame.no_urgent_destination == 1) & (test_frame.with_kids == 0)) |
                               (test_frame.expires_1day == 1))).to_numpy().astype(int)))
            else:
                rules.append(('At least two prior invoices', (test_frame.frequency_invoices >= 2).to_numpy().astype(int)))
                threshold = float(train_frame.monetary_gbp.quantile(.75))
                thresholds['monetary_training_q75'] = threshold
                rules.append(('Past spend >= training Q75', (test_frame.monetary_gbp >= threshold).to_numpy().astype(int)))
            comparisons = [{'Method': selected['selected_model'], 'Role': 'selected by training CV', **score(truth, pred)}]
            baseline_arrays = {'test_indices': test, 'truth': truth, 'selected_predictions': pred}
            for i, (label, values) in enumerate(rules):
                comparisons.append({'Method': label, 'Role': 'prespecified reference, not model selection', **score(truth, values)})
                baseline_arrays[f'reference_{i}'] = values
            pd.DataFrame(comparisons).to_csv(folder / 'comparisons.csv', index=False)
            np.savez_compressed(folder / 'comparison_predictions.npz', **baseline_arrays)
            render_excel_report(folder / 'selected', folder / 'illustrated')
            rendered = validate_excel_report(folder / 'illustrated')
            means = pd.read_excel(folder / 'selected' / 'report.xlsx', sheet_name='SL_CV_Mean')
            cv = means[means.Model == selected['selected_model']].iloc[0]
            intervals = pd.read_excel(folder / 'selected' / 'report.xlsx', sheet_name='SL_Test_Intervals')
            result = dict(status='completed', **stats, train_rows=len(train), test_rows=len(test),
                          test_positive_rows=int(truth.sum()), features=list(x.columns[:-1]),
                          selected_model=selected['selected_model'], selected_cv_macro_f1=float(cv.F1),
                          selected_test=comparisons[0], comparisons=comparisons,
                          thresholds=thresholds, intervals=intervals.to_dict('records'),
                          figure_count=rendered['figure_count'], seconds=time.perf_counter()-start)
            write_json(folder / 'summary.json', result)
            manifest['cases'][name] = result
            overview.extend([{'Dataset': name, **row} for row in comparisons])
            write_json(args.output / 'experiment_manifest.json', manifest)
            print(json.dumps({'case': name, 'model': result['selected_model'], 'test_f1': result['selected_test']['F1'], 'rows': len(x)}, ensure_ascii=False), flush=True)
        pd.DataFrame(overview).to_csv(args.output / 'comparisons.csv', index=False)
        with pd.ExcelWriter(args.output / 'overview.xlsx', engine='openpyxl') as writer:
            pd.DataFrame(overview).to_excel(writer, sheet_name='Comparisons', index=False)
            pd.DataFrame([{'Dataset': key, 'Sample': v['sample_rows'], 'Train': v['train_rows'], 'Test': v['test_rows'],
                           'Positive_in_test': v['test_positive_rows'], 'Model': v['selected_model'],
                           'CV_macro_F1': v['selected_cv_macro_f1'], 'Scope': protocol[key]['evaluation_scope']}
                          for key, v in manifest['cases'].items()]).to_excel(writer, sheet_name='Scope', index=False)
        manifest.update(status='completed')
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        write_json(args.output / 'experiment_manifest.json', manifest)
        raise
    manifest['artifacts'] = {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob('*'))
                             if p.is_file() and p.name != 'experiment_manifest.json'}
    write_json(args.output / 'experiment_manifest.json', manifest)


if __name__ == '__main__':
    main()
