"""Run integration checks on bundled development data; never select using test scores."""
import argparse
from pathlib import Path
import hashlib,json,subprocess,sys,time
import numpy as np
from diy_automl.validation import validate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path.cwd())
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
p.add_argument('--ul-only',action='store_true')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
records=[]
for dataset in ['iris','wine','breast_cancer']:
    csv=a.root/'data/reference'/f'{dataset}.csv'
    commands=['repeat-ul'] if a.ul_only else ['run','select','repeat-ul']
    for command in commands:
        dest=a.output/f'{dataset}-{command}'
        cmd=[sys.executable,'-m','diy_automl',command,'--csv',str(csv),'--target','target','--output',str(dest)]
        start=time.perf_counter()
        proc=subprocess.run(cmd,text=True,capture_output=True)
        (a.output/f'{dataset}-{command}.log').write_text(proc.stdout+'\n'+proc.stderr)
        if proc.returncode: raise RuntimeError(f'{dataset}/{command}: see log')
        m=validate(dest)
        record={'dataset':dataset,'command':command,'status':'passed','seconds':time.perf_counter()-start,'sheets':m['sheets']}
        if a.compare:
            other=a.compare/f'{dataset}-{command}'
            with np.load(dest/'arrays.npz') as x,np.load(other/'arrays.npz') as y:
                if set(x.files)!=set(y.files): raise ValueError('Array names differ')
                for key in x.files: np.testing.assert_array_equal(x[key],y[key],err_msg=f'{dataset}/{key}')
            record['all_arrays_equal']=True
        records.append(record)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
