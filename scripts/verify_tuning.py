"""Default SL grid checks, unchanged default-candidate CV results and wheel comparisons."""
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
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
for dataset in ['iris','wine','breast_cancer']:
    out=a.output/dataset;start=time.perf_counter()
    result=subprocess.run([sys.executable,'-m','diy_automl','tune','--csv',
                           str(a.root/'data/reference'/f'{dataset}.csv'),'--target','target','--output',str(out)],
                          capture_output=True,text=True)
    (a.output/f'{dataset}.log').write_text((result.stdout+'\n'+result.stderr).rstrip('\n')+'\n')
    if result.returncode: raise RuntimeError(f'{dataset}: see log')
    m=validate(out,7)
    tables=pd.read_excel(out/'report.xlsx',sheet_name=None)
    defaults=tables['SL_Search_Folds'].query('Candidate in [0,6,10,13,19]')[['Model','Fold','Accuracy','Precision','Recall','F1']].reset_index(drop=True)
    previous=pd.read_excel(a.root/'validation/current'/f'{dataset}-select/report.xlsx',sheet_name='SL_CV_Folds')
    pd.testing.assert_frame_equal(defaults,previous,check_exact=True)
    record={'dataset':dataset,'status':'passed','seconds':time.perf_counter()-start,'sheets':m['sheets'],
            'source_hashes':m['source_hashes'],'selected_candidate':m['selected_candidate'],
            'default_candidates_CV_matches_initial_select':True}
    if a.compare:
        other=a.compare/dataset
        with np.load(out/'arrays.npz') as x,np.load(other/'arrays.npz') as y:
            if set(x.files)!=set(y.files): raise ValueError('Array keys differ')
            for key in x.files: np.testing.assert_array_equal(x[key],y[key],err_msg=f'{dataset}/{key}')
        other_tables=pd.read_excel(other/'report.xlsx',sheet_name=None)
        for sheet in tables:
            pd.testing.assert_frame_equal(tables[sheet].drop(columns='Fit Time (s)',errors='ignore'),
                                          other_tables[sheet].drop(columns='Fit Time (s)',errors='ignore'),check_exact=True)
        record['all_arrays_and_non_timing_tables_exactly_equal']=True
    records.append(record);print(dataset+': passed',flush=True)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
