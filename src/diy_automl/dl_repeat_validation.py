"""Schema 5 validation, including predictions and partial failure records; no TensorFlow."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .deep_learning_repeat_workflow import SUMMARY_METRICS, seed_summary
from .supervised import metrics

SHEETS = {'DL_By_Seed', 'DL_Seed_Summary', 'DL_History', 'DL_Settings', 'DL_Splits', 'Summary'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_dl_repeat(directory, allow_failed=False):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    require(manifest['schema_version'] == 5, 'Unexpected schema')
    status = manifest['status']
    require(status == 'completed' or (allow_failed and status == 'failed'), 'Run did not complete')
    for name in ['arrays.npz', 'report.xlsx']:
        digest = hashlib.sha256((root/name).read_bytes()).hexdigest()
        require(digest == manifest['artifacts'][name], f'Artifact checksum mismatch: {name}')
    sheets = pd.read_excel(root/'report.xlsx', sheet_name=None)
    require(set(sheets) == SHEETS, 'Sheet names differ')
    require({k: len(v) for k,v in sheets.items()} == manifest['sheets'], 'Sheet row counts differ')
    settings = manifest['settings']
    seeds = settings['training_seeds']
    epochs = settings['epochs']
    rows = sheets['DL_By_Seed']
    require(rows['Training Seed'].tolist() == seeds, 'Seed order differs')
    require(len(seeds) >= 2 and len(seeds) == len(set(seeds)), 'Invalid seed list')
    require(rows['Split Seed'].eq(settings['split_seed']).all(), 'Split seed differs')
    require(rows['Epoch'].eq(epochs).all() and rows['Batch Size'].eq(settings['batch_size']).all(), 'Training configuration differs')
    states = rows.Status.tolist()
    require(set(states) <= {'completed', 'failed', 'not_run'}, 'Unknown seed state')
    require(manifest['seed_states'] == [{'training_seed': s, 'status': state} for s,state in zip(seeds,states)], 'Manifest states differ')
    completed = rows[rows.Status == 'completed']
    if status == 'completed':
        require(len(completed) == len(seeds), 'Incomplete seeds in completed run')
    else:
        require(states.count('failed') == 1, 'Expected exactly one failed seed')
        failed = states.index('failed')
        require(states[:failed] == ['completed']*failed and states[failed+1:] == ['not_run']*(len(seeds)-failed-1), 'Invalid failure order')
        require(pd.notna(rows.iloc[failed]['Error']), 'Missing failure reason')
    configs = sheets['DL_Settings']
    require(configs['Training Seed'].tolist() == seeds, 'Configuration seeds differ')
    require(configs['Split Seed'].eq(settings['split_seed']).all(), 'Configuration split differs')
    require(configs.Epochs.eq(epochs).all() and configs['Batch Size'].eq(settings['batch_size']).all(), 'Configuration epochs/batch size differ')
    # Excel converts numeric/bool configuration strings back to their native types.
    for key,value in settings['model'].items():
        if isinstance(value, (int, float, bool)):
            require(configs[key].eq(value).all(), f'Model configuration differs: {key}')
        else:
            require(configs[key].astype(str).eq(str(value)).all(), f'Model configuration differs: {key}')
    history = sheets['DL_History']
    require(set(history['Training Seed']) <= set(seeds), 'Unknown history seed')
    try:
        pd.testing.assert_frame_equal(seed_summary(rows).reset_index(drop=True),
                                      sheets['DL_Seed_Summary'].reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError('Seed summary differs from completed results') from error
    with np.load(root/'arrays.npz', allow_pickle=False) as arrays:
        groups = [arrays[name] for name in ['train', 'validation', 'test']]
        all_indices = np.concatenate(groups)
        n = manifest['rows']
        require(all(np.issubdtype(g.dtype, np.integer) for g in groups), 'Indices must be integers')
        require(np.array_equal(np.sort(all_indices), np.arange(n)), 'Split coverage or overlap is invalid')
        split_frame = sheets['DL_Splits']
        require(split_frame.Index.tolist() == list(range(n)), 'Split table indices differ')
        target = arrays['target']
        require(len(target) == n and np.array_equal(target[arrays['test']], arrays['truth']), 'Stored truth differs')
        require(np.array_equal(split_frame['Encoded Target'].to_numpy(), target), 'Split table targets differ')
        for name in ['train', 'validation', 'test']:
            require(set(split_frame.loc[split_frame.Partition == name, 'Index']) == set(arrays[name]), 'Split table differs from arrays')
        width = len(manifest['features'])
        require(arrays['scaler_mean'].shape == (width,) and arrays['scaler_scale'].shape == (width,), 'Scaler dimensions differ')
        require(np.isfinite(arrays['scaler_mean']).all() and np.isfinite(arrays['scaler_scale']).all() and (arrays['scaler_scale'] > 0).all(), 'Invalid scaler')
        classes = np.arange(len(manifest['classes']))
        require(np.array_equal(np.unique(target), classes), 'Encoded target classes differ')
        dimensions = [width, *settings['model']['layers'], len(classes)]
        for seed,state in zip(seeds,states):
            seed_history = history[history['Training Seed'] == seed]
            require(seed_history.Epoch.tolist() == list(range(1,len(seed_history)+1)), 'History epochs not contiguous')
            if state == 'not_run':
                require(seed_history.empty, 'Unrun seed has history')
            if state == 'completed':
                require(len(seed_history) == epochs, 'Completed seed has incomplete history')
                require(np.isfinite(seed_history[['accuracy','loss','val_accuracy','val_loss']]).all().all(), 'Non-finite completed history')
                pred = arrays[f'seed_{seed}_predictions']
                require(pred.shape == arrays['truth'].shape and np.isin(pred,classes).all(), 'Invalid predictions')
                row = rows.loc[rows['Training Seed'] == seed].iloc[0]
                for metric,estimate in metrics(arrays['truth'],pred,classes).items():
                    require(np.isclose(row[metric],estimate,atol=1e-12,rtol=0), 'Test metric differs from predictions')
                final = seed_history.iloc[-1]
                for key,label in [('accuracy','Final Train Accuracy'), ('loss','Final Train Loss'),
                                  ('val_accuracy','Final Validation Accuracy'), ('val_loss','Final Validation Loss')]:
                    require(np.isclose(row[label],final[key],atol=1e-12,rtol=0), 'Final history metric differs')
                for j,(left,right) in enumerate(zip(dimensions,dimensions[1:])):
                    require(arrays[f'seed_{seed}_weight_{2*j}'].shape == (left,right), 'Weight shape differs')
                    require(arrays[f'seed_{seed}_weight_{2*j+1}'].shape == (right,), 'Bias shape differs')
            else:
                require(len(seed_history) <= epochs, 'Too many failed history epochs')
                require(f'seed_{seed}_predictions' not in arrays.files, 'Incomplete seed has test predictions')
                require(rows.loc[rows['Training Seed'] == seed,SUMMARY_METRICS].isna().all().all(), 'Incomplete seed has scored metrics')
    return manifest
