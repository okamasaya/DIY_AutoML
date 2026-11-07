"""Training-seed sensitivity conditional on one fixed data split."""
import time

import numpy as np
import pandas as pd

from .data import load_csv
from .deep_learning import (HISTORY_COLUMNS, MODEL_CONFIG, TrainingFailure, fit_seed,
                            integer, prepare_data)
from .reporting import fail, prepare, save, summary
from .supervised import METRICS

FINAL_METRICS = {'accuracy': 'Final Train Accuracy', 'loss': 'Final Train Loss',
                 'val_accuracy': 'Final Validation Accuracy', 'val_loss': 'Final Validation Loss'}
SUMMARY_METRICS = [*METRICS, *FINAL_METRICS.values()]
RESULT_COLUMNS = ['Training Seed', 'Split Seed', 'Status', 'Model', 'Layers', 'Epoch',
                  'Batch Size', *SUMMARY_METRICS, 'Train Time (s)', 'Error']


def seed_summary(results):
    completed = results.loc[results.Status == 'completed']
    rows = []
    for metric in SUMMARY_METRICS:
        values = completed[metric].astype(float)
        rows.append({'Metric': metric, 'Count': len(values), 'Mean': values.mean(),
                     'SD': values.std(ddof=1), 'Min': values.min(), 'Max': values.max()})
    return pd.DataFrame(rows)


def repeat_deep_learning_csv(csv, target, output, training_seeds=range(42, 52),
                             split_seed=42, epochs=100, batch_size=16):
    seeds = [integer(s, 'training_seed', maximum=2**32-1) for s in training_seeds]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError('At least two distinct training seeds are required')
    split_seed = integer(split_seed, 'split_seed', maximum=2**32-1)
    epochs = integer(epochs, 'epochs', minimum=1)
    batch_size = integer(batch_size, 'batch_size', minimum=1)
    settings = dict(csv=str(csv), target=target, training_seeds=seeds, split_seed=split_seed,
                    epochs=epochs, batch_size=batch_size, methods=['dl'], model=MODEL_CONFIG)
    out, manifest = prepare(output, 5, settings)
    rows = [{**dict.fromkeys(RESULT_COLUMNS, np.nan), 'Training Seed': seed,
             'Split Seed': split_seed, 'Status': 'not_run', 'Model': 'MLP',
             'Layers': '32-16-8', 'Epoch': epochs, 'Batch Size': batch_size, 'Error': ''}
            for seed in seeds]
    histories = []
    try:
        data = load_csv(csv, target)
        fixed = prepare_data(data.X, data.y, split_seed)
        arrays = {**fixed.arrays(), 'target': data.y, 'truth': data.y[fixed.test]}
        manifest.update(classes=data.classes, features=data.feature_names, rows=len(data.y),
                        protocol='Fixed split and train-only scaler shared by all training seeds; no selection',
                        uncertainty_scope='Conditional training-seed sensitivity; no data resampling, SEM or CI')
        partitions = np.empty(len(data.y), dtype='<U10')
        for name in ['train', 'validation', 'test']:
            partitions[getattr(fixed, name)] = name
        splits = pd.DataFrame({'Index': np.arange(len(data.y)), 'Partition': partitions,
                               'Encoded Target': data.y})
        configurations = pd.DataFrame([
            {'Training Seed': seed, 'Split Seed': split_seed, 'Epochs': epochs,
             'Batch Size': batch_size, **{k: str(v) for k, v in MODEL_CONFIG.items()}}
            for seed in seeds])

        def checkpoint():
            results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
            history = (pd.concat(histories, ignore_index=True) if histories else
                       pd.DataFrame(columns=['Training Seed', *HISTORY_COLUMNS]))
            manifest['seed_states'] = [{'training_seed': r['Training Seed'], 'status': r['Status']}
                                       for r in rows]
            sheets = {'DL_By_Seed': results, 'DL_Seed_Summary': seed_summary(results),
                      'DL_History': history, 'DL_Settings': configurations, 'DL_Splits': splits,
                      'Summary': summary([
                          ('Purpose', 'Training-seed sensitivity on one fixed split'),
                          ('Split seed', split_seed), ('Training seeds', str(seeds)),
                          ('Scaling', 'Fit once on DL training samples only'),
                          ('Variation', 'Initialization and training shuffle vary together'),
                          ('Summary', 'Completed seeds only; sample SD; no SEM or confidence interval'),
                          ('Scope', 'Fixed dataset and split; excludes data/split uncertainty'),
                          ('Selection', 'None; never choose a seed using test scores'),
                          ('Failures', 'Stop after first failure; retain completed/failed/not_run'),
                          ('Reproducibility', 'Deterministic ops in recorded environment; cross-platform equality not guaranteed'),
                      ])}
            save(out, manifest, sheets, arrays, finalize=False)

        checkpoint()
        for i, seed in enumerate(seeds):
            rows[i]['Status'] = 'running'
            checkpoint()
            started = time.perf_counter()
            try:
                result, history, trained = fit_seed(fixed, seed, epochs, batch_size)
                row = result.iloc[0].to_dict()
                row.update({label: float(history.iloc[-1][key]) for key, label in FINAL_METRICS.items()})
                rows[i].update(row, Status='completed', **{'Train Time (s)': time.perf_counter()-started})
                histories.append(history.assign(**{'Training Seed': seed})[['Training Seed', *HISTORY_COLUMNS]])
                arrays.update({f'seed_{seed}_{k}': v for k, v in trained.items()
                               if k == 'predictions' or k.startswith('weight_')})
            except Exception as error:
                rows[i].update(Status='failed', Error=f'{type(error).__name__}: {error}',
                               **{'Train Time (s)': time.perf_counter()-started})
                if isinstance(error, TrainingFailure):
                    histories.append(error.history.assign(**{'Training Seed': seed})[['Training Seed', *HISTORY_COLUMNS]])
                    arrays.update({f'seed_{seed}_partial_weight_{j}': w for j, w in enumerate(error.weights)})
                manifest.update(status='failed', error=rows[i]['Error'])
                checkpoint()
                raise
            checkpoint()
        manifest['status'] = 'completed'
        checkpoint()
        return out
    except Exception as error:
        fail(out, manifest, error)
        raise
