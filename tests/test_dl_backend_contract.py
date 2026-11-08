"""Check adapter outputs without installing or pretending to implement another backend."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from diy_automl import deep_learning as dl
from diy_automl import tensorflow_backend
from diy_automl.dl_contract import HIDDEN_LAYERS, validate_result
from diy_automl.dl_repeat_validation import validate_dl_repeat
from diy_automl.supervised import metrics


def prepared():
    x=np.arange(160,dtype=float).reshape(40,4)
    return dl.prepare_data(x,np.arange(40)%2,split_seed=17)


def valid_result(data,seed=42,epochs=2,batch_size=8):
    truth=data.target[data.test]
    row={'Model':'MLP','Layers':'32-16-8',**metrics(truth,truth,np.unique(data.target)),
         'Epoch':epochs,'Batch Size':batch_size,'Split Seed':data.split_seed,'Training Seed':seed}
    history=pd.DataFrame(dict(accuracy=[1.]*epochs,loss=[.1]*epochs,
                              val_accuracy=[.8]*epochs,val_loss=[.2]*epochs,Epoch=range(1,epochs+1)))
    arrays={**data.arrays(),'predictions':truth.copy()}
    dims=[data.scaled.shape[1],*HIDDEN_LAYERS,len(np.unique(data.target))]
    for i,(a,b) in enumerate(zip(dims,dims[1:])):
        arrays[f'weight_{2*i}']=np.zeros((a,b),dtype=np.float32)
        arrays[f'weight_{2*i+1}']=np.zeros(b,dtype=np.float32)
    return pd.DataFrame([row]),history,arrays


def test_dispatch_shares_preparation_and_checks_adapter(monkeypatch):
    data=prepared();calls=[]
    result=valid_result(data)
    def adapter(fixed,seed,epochs,batch_size):
        calls.append((fixed,seed,epochs,batch_size))
        return result
    monkeypatch.setattr(tensorflow_backend,'fit_seed',adapter)
    assert dl.fit_seed(data,42,2,8) is result
    assert calls==[(data,42,2,8)]
    assert not data.scaled.flags.writeable and not data.target.flags.writeable


@pytest.mark.parametrize('bad',[
    'train','validation','scaler','prediction_class','prediction_dtype','metric','training_seed',
    'split_seed','layer','missing_weight','native_weight_layout','nonfinite_weight',
    'history_length','history_epoch','history_nonfinite','extra_array',
])
def test_invalid_adapter_result_is_rejected(monkeypatch,bad):
    data=prepared();row,history,arrays=valid_result(data)
    arrays={k:v.copy() for k,v in arrays.items()}
    if bad in ('train','validation'): arrays[bad][0]=data.test[0]
    elif bad=='scaler': arrays['scaler_mean'][0]+=1
    elif bad=='prediction_class': arrays['predictions'][0]=999
    elif bad=='prediction_dtype': arrays['predictions']=arrays['predictions'].astype(float)
    elif bad=='metric': row.loc[0,'F1']=.1
    elif bad=='training_seed': row.loc[0,'Training Seed']=43
    elif bad=='split_seed': row.loc[0,'Split Seed']=42
    elif bad=='layer': row.loc[0,'Layers']='8-16-32'
    elif bad=='missing_weight': arrays.pop('weight_7')
    elif bad=='native_weight_layout': arrays['weight_0']=arrays['weight_0'].T.copy()
    elif bad=='nonfinite_weight': arrays['weight_0'][0,0]=np.nan
    elif bad=='history_length': history=history.iloc[:1]
    elif bad=='history_epoch': history.loc[0,'Epoch']=2
    elif bad=='history_nonfinite': history.loc[0,'val_loss']=np.inf
    else: arrays['unrecorded_output']=np.zeros(1)
    monkeypatch.setattr(tensorflow_backend,'fit_seed',lambda *args:(row,history,arrays))
    with pytest.raises(ValueError,match='DL result contract'): dl.fit_seed(data,42,2,8)


def test_shared_types_keep_existing_import_paths():
    from diy_automl.dl_contract import PreparedDL,TrainingFailure
    assert dl.PreparedDL is PreparedDL and dl.TrainingFailure is TrainingFailure
    data=prepared()
    validate_result(data,valid_result(data),42,2,8)


@pytest.mark.parametrize('field',['scaled','train'])
def test_adapter_cannot_silently_modify_shared_inputs(monkeypatch,field):
    data=prepared();result=valid_result(data)
    def adapter(*args):
        array=getattr(data,field)
        array.setflags(write=True);array.flat[0]+=1;array.setflags(write=False)
        return result
    monkeypatch.setattr(tensorflow_backend,'fit_seed',adapter)
    with pytest.raises(ValueError,match='shared prepared data'): dl.fit_seed(data,42,2,8)


def test_mutation_on_failure_preserves_error_context(monkeypatch):
    data=prepared()
    def adapter(*args):
        data.target.setflags(write=True);data.target[0]=1-data.target[0];data.target.setflags(write=False)
        raise dl.TrainingFailure(RuntimeError('adapter failed'),pd.DataFrame(),[])
    monkeypatch.setattr(tensorflow_backend,'fit_seed',adapter)
    with pytest.raises(ValueError,match='shared prepared data') as caught: dl.fit_seed(data,42,2,8)
    assert isinstance(caught.value.__context__,dl.TrainingFailure)


@pytest.mark.parametrize('field,value',[('dl_backend','pytorch'),('dl_result_contract','unknown')])
def test_validator_does_not_claim_an_unsupported_backend(tmp_path,field,value):
    source=Path(__file__).parents[1]/'validation/dl-repeat-0.4.0a2/iris'
    out=tmp_path/'out';shutil.copytree(source,out)
    p=out/'manifest.json';m=json.loads(p.read_text());m[field]=value;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Unsupported DL'): validate_dl_repeat(out)


def test_backend_preparation_imports_no_training_framework():
    code='''
import importlib.abc,sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in ('tensorflow','keras','torch','jax'):
            raise AssertionError('Framework imported: '+fullname)
sys.meta_path.insert(0,BlockFrameworks())
import diy_automl
from diy_automl import deep_learning
from diy_automl import dl_contract,tensorflow_backend
import numpy as np
x=np.arange(160,dtype=float).reshape(40,4)
fixed=deep_learning.prepare_data(x,np.arange(40)%2)
assert not fixed.scaled.flags.writeable
assert tensorflow_backend.MODEL_CONFIG['layers']==[32,16,8]
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
