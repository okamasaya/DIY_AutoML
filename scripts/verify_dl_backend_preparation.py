"""Check TensorFlow refactoring against stored predictions, weights and histories."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from diy_automl import __version__,repeat_deep_learning_csv,render_excel_report,evaluate_uncertainty
from diy_automl.validation import validate
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.reporting import source_hashes

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path.cwd())
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
p.add_argument('--reuse-existing',action='store_true',help='Reuse only already validated completed outputs from the same source hashes')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.reuse_existing);records=[]
for dataset in ['iris','wine','breast_cancer']:
    start=time.perf_counter();source=a.root/'validation/dl-repeat-0.4.0a2'/dataset
    prior_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir() if f.is_file()}
    out=a.output/dataset
    if not (a.reuse_existing and out.exists()):
        repeat_deep_learning_csv(a.root/'data/reference'/f'{dataset}.csv','target',out,
                                 training_seeds=[42,43],epochs=100)
    m=validate(out,5)
    assert m['source_hashes']==source_hashes()
    assert m['settings']['training_seeds']==[42,43] and m['settings']['epochs']==100
    assert m['dl_backend']=='tensorflow' and m['dl_result_contract']=='dense-32-16-8-v1'
    with np.load(out/'arrays.npz') as current,np.load(source/'arrays.npz') as old:
        for key in current.files: np.testing.assert_array_equal(current[key],old[key],err_msg=dataset+'/'+key)
    frames=pd.read_excel(out/'report.xlsx',sheet_name=None)
    old_frames=pd.read_excel(source/'report.xlsx',sheet_name=None)
    for name in ['DL_By_Seed','DL_History','DL_Settings']:
        original=old_frames[name];original=original[original['Training Seed'].isin([42,43])].reset_index(drop=True)
        pd.testing.assert_frame_equal(frames[name].drop(columns='Train Time (s)',errors='ignore'),
                                      original.drop(columns='Train Time (s)',errors='ignore'),check_exact=True,check_dtype=False)
    pd.testing.assert_frame_equal(frames['DL_Splits'],old_frames['DL_Splits'],check_exact=True)
    illustrated=a.output/f'{dataset}-figures'
    if not (a.reuse_existing and illustrated.exists()): render_excel_report(out,illustrated)
    validate_excel_report(illustrated)
    uncertainty=a.output/f'{dataset}-uncertainty'
    if not (a.reuse_existing and uncertainty.exists()): evaluate_uncertainty(out,uncertainty)
    validate(uncertainty,9)
    if a.compare:
        for name in [dataset,f'{dataset}-figures',f'{dataset}-uncertainty']:
            left=a.output/name;right=a.compare/name
            for array_name in ['arrays.npz',*(['figure_data.npz'] if name.endswith('-figures') else [])]:
                with np.load(left/array_name) as x,np.load(right/array_name) as y:
                    assert set(x.files)==set(y.files)
                    for key in x.files: np.testing.assert_array_equal(x[key],y[key],err_msg=name+'/'+key)
            for book in ['report.xlsx',*(['report_with_figures.xlsx'] if name.endswith('-figures') else [])]:
                x=pd.read_excel(left/book,sheet_name=None);y=pd.read_excel(right/book,sheet_name=None)
                for sheet in x:
                    pd.testing.assert_frame_equal(x[sheet].drop(columns='Train Time (s)',errors='ignore'),
                                                  y[sheet].drop(columns='Train Time (s)',errors='ignore'),check_exact=True)
            if name.endswith('-uncertainty'):
                assert (left/'metric_intervals.png').read_bytes()==(right/'metric_intervals.png').read_bytes()
            if name.endswith('-figures'):
                for image in (left/'figures').glob('*.png'):
                    assert image.read_bytes()==(right/'figures'/image.name).read_bytes()
    assert prior_hashes=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir() if f.is_file()}
    records.append(dict(dataset=dataset,status='passed',training_seeds=[42,43],epochs=100,
                        stored_predictions_weights_histories_unchanged=True,source_runs_unchanged=True,
                        cross_environment_equal=bool(a.compare),seconds=time.perf_counter()-start,
                        source_hashes=m['source_hashes']))
    print(dataset+': TensorFlow arrays/history unchanged; report and uncertainty passed',flush=True)
(a.output/'results.json').write_text(json.dumps(dict(status='passed',version=__version__,records=records),indent=2)+'\n')
