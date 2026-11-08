"""Verify all six result families, raw-run compatibility and illustrated workbooks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from diy_automl.data import load_csv
from diy_automl.excel_report_validation import validate_excel_report

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path.cwd())
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
for dataset in ['iris','wine','breast_cancer']:
    csv=a.root/'data/reference'/f'{dataset}.csv'
    numerical=a.output/f'{dataset}-source-run'
    result=subprocess.run([sys.executable,'-m','diy_automl','run','--csv',str(csv),'--target','target',
                           '--output',str(numerical)],capture_output=True,text=True)
    (a.output/f'{dataset}-training.log').write_text((result.stdout+'\n'+result.stderr).rstrip('\n')+'\n')
    if result.returncode: raise RuntimeError('New run failed: '+dataset)
    data=load_csv(csv,'target')
    with np.load(numerical/'arrays.npz') as new,np.load(a.root/'validation/current'/f'{dataset}-run/arrays.npz') as old:
        assert set(new.files)==set(old.files)|{'features','target'}
        for key in old.files: np.testing.assert_array_equal(new[key],old[key],err_msg=dataset+'/'+key)
        np.testing.assert_array_equal(new['features'],data.X);np.testing.assert_array_equal(new['target'],data.y)
    sources={'run':numerical,'select':a.root/'validation/current'/f'{dataset}-select',
             'repeat-ul':a.root/'validation/current'/f'{dataset}-repeat-ul',
             'repeat-dl':a.root/'validation/dl-repeat-0.4.0a2'/dataset,
             'subsample-ul':a.root/'validation/ul-subsample-0.4.0a3'/dataset,
             'tune':a.root/'validation/sl-tuning-0.4.0a4'/dataset}
    for command,source in sources.items():
        out=a.output/f'{dataset}-{command}';start=time.perf_counter()
        result=subprocess.run([sys.executable,'-m','diy_automl','report','--run',str(source),'--output',str(out)],capture_output=True,text=True)
        (a.output/f'{dataset}-{command}.log').write_text((result.stdout+'\n'+result.stderr).rstrip('\n')+'\n')
        if result.returncode: raise RuntimeError(f'{dataset}/{command}: see log')
        manifest=validate_excel_report(out)
        record={'dataset':dataset,'command':command,'status':'passed','seconds':time.perf_counter()-start,
                'source_schema':manifest['source_schema'],'figure_count':manifest['figure_count'],
                'workbook_bytes':manifest['workbook_bytes'],'renderer_source_hashes':manifest['source_hashes'],
                'source_run_arrays_match_initial_reimplementation':True}
        if a.compare:
            other=a.compare/f'{dataset}-{command}'
            with np.load(out/'figure_data.npz') as left,np.load(other/'figure_data.npz') as right:
                if set(left.files)!=set(right.files): raise ValueError('Figure input keys differ')
                for key in left.files: np.testing.assert_array_equal(left[key],right[key],err_msg=f'{dataset}/{command}/{key}')
            prior=json.loads((other/'render_manifest.json').read_text())
            excluded={'font'}
            for left,right in zip(manifest['figures'],prior['figures']):
                assert {k:v for k,v in left.items() if k not in excluded}=={k:v for k,v in right.items() if k not in excluded}
            record['all_figure_inputs_and_pngs_identical']=True
        records.append(record)
        print(f'{dataset}/{command}: {manifest["figure_count"]} images, {manifest["workbook_bytes"]} bytes',flush=True)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
