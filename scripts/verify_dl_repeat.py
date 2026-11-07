"""Default DL repeats and cross-environment comparisons on development datasets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from diy_automl.validation import validate

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path.cwd())
p.add_argument('--output', type=Path, required=True)
p.add_argument('--compare', type=Path)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--training-seeds', nargs='+', type=int, default=list(range(42,52)))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
records = []
for dataset in ['iris','wine','breast_cancer']:
    dest = a.output/dataset
    start = time.perf_counter()
    command = [sys.executable,'-m','diy_automl','repeat-dl',
               '--csv',str(a.root/'data/reference'/f'{dataset}.csv'), '--target','target',
               '--output',str(dest), '--epochs',str(a.epochs),
               '--training-seeds',*[str(seed) for seed in a.training_seeds]]
    proc = subprocess.run(command,text=True,capture_output=True)
    (a.output/f'{dataset}.log').write_text(proc.stdout+'\n'+proc.stderr)
    if proc.returncode:
        raise RuntimeError(f'{dataset}: see log')
    m = validate(dest,5)
    record = {'dataset':dataset,'status':'passed','seconds':time.perf_counter()-start,
              'sheets':m['sheets'],'source_hashes':m['source_hashes']}
    if a.compare:
        other = a.compare/dataset
        with np.load(dest/'arrays.npz') as x,np.load(other/'arrays.npz') as y:
            if set(x.files) != set(y.files): raise ValueError('Array names differ')
            for key in x.files:
                np.testing.assert_array_equal(x[key],y[key],err_msg=f'{dataset}/{key}')
        left = pd.read_excel(dest/'report.xlsx',sheet_name=None)
        right = pd.read_excel(other/'report.xlsx',sheet_name=None)
        for sheet in left:
            pd.testing.assert_frame_equal(left[sheet].drop(columns='Train Time (s)',errors='ignore'),
                                          right[sheet].drop(columns='Train Time (s)',errors='ignore'),
                                          check_exact=True)
        record['all_arrays_and_non_timing_tables_exactly_equal'] = True
    records.append(record)
    print(dataset+': passed',flush=True)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
