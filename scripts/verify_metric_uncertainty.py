"""Exercise all classification result families and compare independent installations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diy_automl import __version__, evaluate_uncertainty
from diy_automl.uncertainty_validation import validate_uncertainty

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path.cwd())
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
records=[]
for dataset in ['iris','wine','breast_cancer']:
    sources={
        'run':a.root/'validation/current'/f'{dataset}-run',
        'select':a.root/'validation/current'/f'{dataset}-select',
        'repeat-dl':a.root/'validation/dl-repeat-0.4.0a2'/dataset,
        'tune':a.root/'validation/sl-tuning-0.4.0a4'/dataset,
    }
    for family,source in sources.items():
        before={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in source.rglob('*') if f.is_file()}
        for mode in ['stratified','ordinary']:
            name=f'{dataset}-{family}-{mode}'
            out=a.output/name;started=time.perf_counter()
            evaluate_uncertainty(source,out,resampling=mode,
                                 csv=a.root/'data/reference'/f'{dataset}.csv' if family=='run' else None)
            m=validate_uncertainty(out)
            assert len(m['evaluations'])=={'run':6,'select':1,'repeat-dl':10,'tune':1}[family]
            frame=pd.read_excel(out/'report.xlsx',sheet_name='Metric_Uncertainty')
            if mode=='stratified' and family in ('select','tune'):
                old=pd.read_excel(source/'report.xlsx',sheet_name='SL_Test_Intervals')
                np.testing.assert_allclose(frame[['Estimate','Lower','Upper']],old[['Estimate','Lower','Upper']],rtol=0,atol=1e-12)
            same=None
            if a.compare:
                other=a.compare/name
                with np.load(out/'arrays.npz') as left,np.load(other/'arrays.npz') as right:
                    assert set(left.files)==set(right.files)
                    for key in left.files: np.testing.assert_array_equal(left[key],right[key],err_msg=name+'/'+key)
                left_tables=pd.read_excel(out/'report.xlsx',sheet_name=None)
                right_tables=pd.read_excel(other/'report.xlsx',sheet_name=None)
                assert set(left_tables)==set(right_tables)
                for key in left_tables: pd.testing.assert_frame_equal(left_tables[key],right_tables[key],check_exact=True)
                assert (out/'metric_intervals.png').read_bytes()==(other/'metric_intervals.png').read_bytes()
                same=True
            record=dict(dataset=dataset,family=family,resampling=mode,status='passed',version=__version__,
                        schema=m['schema_version'],evaluations=len(m['evaluations']),
                        bootstrap_repeats=1000,seconds=time.perf_counter()-started,
                        source_hashes=m['source_hashes'],all_arrays_tables_and_png_identical=same)
            records.append(record)
            print(f'{name}: {len(m["evaluations"])} evaluations verified',flush=True)
        after={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in source.rglob('*') if f.is_file()}
        assert before==after,'Source was modified'
(a.output/'results.json').write_text(json.dumps(dict(status='passed',version=__version__,
    outputs=len(records),source_runs_unchanged=True,records=records),indent=2)+'\n')
