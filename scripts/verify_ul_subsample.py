"""Benchmark default subsampling, validate artifacts and compare environments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from diy_automl.validation import validate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path.cwd())
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
records=[]
for dataset in ['iris','wine','breast_cancer']:
    out=a.output/dataset
    started=time.perf_counter()
    process=subprocess.run([sys.executable,'-m','diy_automl','subsample-ul','--csv',
                            str(a.root/'data/reference'/f'{dataset}.csv'),'--target','target',
                            '--output',str(out)],text=True,capture_output=True)
    (a.output/f'{dataset}.log').write_text((process.stdout+'\n'+process.stderr).rstrip('\n')+'\n')
    if process.returncode: raise RuntimeError(f'{dataset}: see log')
    fitted_seconds=time.perf_counter()-started
    m=validate(out,6)
    record={'dataset':dataset,'status':'passed','run_seconds':fitted_seconds,
            'run_and_validation_seconds':time.perf_counter()-started,
            'report_bytes':(out/'report.xlsx').stat().st_size,
            'arrays_bytes':(out/'arrays.npz').stat().st_size,
            'sheets':m['sheets'],'source_hashes':m['source_hashes']}
    # Existing full-data UL results from the initial reimplementation.
    baseline=a.root/'validation/current'/f'{dataset}-run/arrays.npz'
    with np.load(out/'arrays.npz') as x,np.load(baseline) as y:
        np.testing.assert_array_equal(x['reference_scaler_mean'],y['ul_scaler_mean'])
        np.testing.assert_array_equal(x['reference_scaler_scale'],y['ul_scaler_scale'])
        for cid in range(54): np.testing.assert_array_equal(x[f'reference_labels_{cid}'],y[f'ul_labels_{cid}'])
    record['reference_matches_initial_reimplementation']=True
    if a.compare:
        other=a.compare/dataset
        with np.load(out/'arrays.npz') as x,np.load(other/'arrays.npz') as y:
            if set(x.files)!=set(y.files): raise ValueError('Array keys differ')
            for key in x.files: np.testing.assert_array_equal(x[key],y[key],err_msg=f'{dataset}/{key}')
        left=pd.read_excel(out/'report.xlsx',sheet_name=None)
        right=pd.read_excel(other/'report.xlsx',sheet_name=None)
        for sheet in left:
            pd.testing.assert_frame_equal(left[sheet].drop(columns='Train Time (s)',errors='ignore'),
                                          right[sheet].drop(columns='Train Time (s)',errors='ignore'),check_exact=True)
        record['all_arrays_and_non_timing_tables_exactly_equal']=True
    records.append(record)
    print(f'{dataset}: passed ({fitted_seconds:.1f}s; report {record["report_bytes"]} bytes)',flush=True)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
