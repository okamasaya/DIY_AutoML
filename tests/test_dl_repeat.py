from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from diy_automl import deep_learning as dl
from diy_automl import deep_learning_repeat_workflow as workflow
from diy_automl.data import load_csv
from diy_automl.dl_repeat_validation import validate_dl_repeat
from diy_automl.validation import validate

CSV = Path(__file__).parents[1]/'data/reference/iris.csv'


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp('repeat')/'run'
    return workflow.repeat_deep_learning_csv(CSV,'target',path,training_seeds=[43,42],epochs=2)


def test_complete_results_and_single_seed_equivalence(completed):
    manifest = validate(completed,5)
    assert manifest['status'] == 'completed'
    assert manifest['sheets'] == {'DL_By_Seed':2,'DL_Seed_Summary':8,'DL_History':4,
                                  'DL_Settings':2,'DL_Splits':150,'Summary':10}
    assert manifest['settings']['training_seeds'] == [43,42]
    assert 'tensorflow-cpu' in manifest['dependencies']
    frames = pd.read_excel(completed/'report.xlsx',sheet_name=None)
    assert frames['DL_By_Seed']['Training Seed'].tolist() == [43,42]
    d = load_csv(CSV,'target')
    _,h,single = dl.run(d.X,d.y,training_seed=42,epochs=2)
    with np.load(completed/'arrays.npz') as arrays:
        for name in ['train','validation','test','scaler_mean','scaler_scale']:
            np.testing.assert_array_equal(arrays[name],single[name])
        for name in ['predictions',*[f'weight_{i}' for i in range(8)]]:
            np.testing.assert_array_equal(arrays[f'seed_42_{name}'],single[name])
        assert not np.array_equal(arrays['seed_42_weight_0'],arrays['seed_43_weight_0'])
    stored = frames['DL_History'].query('`Training Seed` == 42').drop(columns='Training Seed').reset_index(drop=True)
    pd.testing.assert_frame_equal(stored,h,check_dtype=False,check_like=True)
    assert not any('best' in k.lower() or 'selected' in k.lower() for k in manifest)


@pytest.mark.parametrize('settings',[
    {'training_seeds':[]}, {'training_seeds':[42]}, {'training_seeds':[42,42]},
    {'training_seeds':[42,43.5]}, {'training_seeds':[42,True]},
    {'training_seeds':[42,-1]}, {'training_seeds':[42,2**32]},
    {'epochs':0}, {'batch_size':0}, {'split_seed':-1},
])
def test_invalid_configuration_rejected_before_output(tmp_path,settings):
    out = tmp_path/'invalid'
    with pytest.raises(ValueError): workflow.repeat_deep_learning_csv(CSV,'target',out,**settings)
    assert not out.exists()


def fake_fit(prepared, seed, epochs, batch_size):
    # Avoid TensorFlow in fault-injection tests, retaining real shape/metric contracts.
    from diy_automl.supervised import metrics
    truth = prepared.target[prepared.test]
    row = {'Model':'MLP','Layers':'32-16-8',**metrics(truth,truth,np.unique(prepared.target)),
           'Epoch':epochs,'Batch Size':batch_size,'Split Seed':prepared.split_seed,'Training Seed':seed}
    history = pd.DataFrame({'accuracy':[1.]*epochs,'loss':[.1]*epochs,
                            'val_accuracy':[1.]*epochs,'val_loss':[.2]*epochs,'Epoch':range(1,epochs+1)})
    dims = [prepared.scaled.shape[1],32,16,8,len(np.unique(prepared.target))]
    arrays = {'predictions':truth.copy()}
    for i,(a,b) in enumerate(zip(dims,dims[1:])):
        arrays[f'weight_{2*i}'] = np.zeros((a,b))
        arrays[f'weight_{2*i+1}'] = np.zeros(b)
    return pd.DataFrame([row]),history,arrays


@pytest.mark.parametrize('failure_index',[0,1])
def test_partial_failure_and_checkpoint(tmp_path,monkeypatch,failure_index):
    attempts=[]
    out=tmp_path/'failure'
    def fit(prepared, seed, epochs, batch):
        checkpoint=json.loads((out/'manifest.json').read_text())
        assert checkpoint['status']=='running'
        assert checkpoint['seed_states'][len(attempts)]['status']=='running'
        assert sum(s['status']=='completed' for s in checkpoint['seed_states'])==len(attempts)
        attempts.append(seed)
        if len(attempts)-1==failure_index:
            partial=pd.DataFrame({'accuracy':[.5],'loss':[1.], 'val_accuracy':[.4], 'val_loss':[1.2], 'Epoch':[1]})
            raise dl.TrainingFailure(RuntimeError('injected'),partial,[np.zeros((4,32))])
        return fake_fit(prepared,seed,epochs,batch)
    monkeypatch.setattr(workflow,'fit_seed',fit)
    with pytest.raises(dl.TrainingFailure):
        workflow.repeat_deep_learning_csv(CSV,'target',out,training_seeds=[42,43,44],epochs=3)
    with pytest.raises(ValueError,match='complete'): validate_dl_repeat(out)
    m=validate_dl_repeat(out,allow_failed=True)
    assert m['status']=='failed'
    assert [s['status'] for s in m['seed_states']]==['completed']*failure_index+['failed']+['not_run']*(2-failure_index)
    assert attempts==[42,43][:failure_index+1]
    with np.load(out/'arrays.npz') as arrays:
        assert f'seed_{attempts[-1]}_partial_weight_0' in arrays
    summary=pd.read_excel(out/'report.xlsx',sheet_name='DL_Seed_Summary')
    assert summary.Count.eq(failure_index).all()


def test_scaler_fit_once_and_only_on_train(tmp_path,monkeypatch):
    import diy_automl.data as data_module
    fitted=[];prepared_objects=[]
    original=data_module.StandardScaler.fit
    def fit(scaler,X,*args,**kwargs):
        fitted.append(X.copy())
        return original(scaler,X,*args,**kwargs)
    def train(prepared,*args):
        prepared_objects.append(prepared)
        assert not prepared.scaled.flags.writeable
        return fake_fit(prepared,*args)
    monkeypatch.setattr(data_module.StandardScaler,'fit',fit)
    monkeypatch.setattr(workflow,'fit_seed',train)
    out=workflow.repeat_deep_learning_csv(CSV,'target',tmp_path/'once',training_seeds=[42,43,44],epochs=2)
    assert len(fitted)==1
    assert prepared_objects[0] is prepared_objects[1] is prepared_objects[2]
    d=load_csv(CSV,'target')
    with np.load(out/'arrays.npz') as arrays:
        np.testing.assert_array_equal(fitted[0],d.X[arrays['train']])
    validate_dl_repeat(out)


def rewrite_artifact(root,name):
    p=root/'manifest.json';m=json.loads(p.read_text())
    m['artifacts'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    p.write_text(json.dumps(m))


@pytest.mark.parametrize('corruption',['predictions','overlap','summary','seed-order','checksum'])
def test_validator_rejects_corrupted_artifacts(completed,tmp_path,corruption):
    import shutil
    out=tmp_path/'tampered';shutil.copytree(completed,out)
    if corruption in ['predictions','overlap']:
        with np.load(out/'arrays.npz') as a: values={k:a[k] for k in a.files}
        if corruption=='predictions': values['seed_42_predictions']=np.full_like(values['seed_42_predictions'],99)
        else: values['validation'][0]=values['train'][0]
        np.savez_compressed(out/'arrays.npz',**values);rewrite_artifact(out,'arrays.npz')
    elif corruption in ['summary','seed-order']:
        sheets=pd.read_excel(out/'report.xlsx',sheet_name=None)
        if corruption=='summary': sheets['DL_Seed_Summary'].loc[0,'Mean']=-1
        else: sheets['DL_By_Seed']=sheets['DL_By_Seed'].iloc[::-1]
        with pd.ExcelWriter(out/'report.xlsx') as writer:
            for name,frame in sheets.items(): frame.to_excel(writer,sheet_name=name,index=False)
        rewrite_artifact(out,'report.xlsx')
    else:
        with (out/'arrays.npz').open('ab') as stream: stream.write(b'corrupted')
    with pytest.raises(ValueError): validate_dl_repeat(out)


def test_missing_input_failed_manifest(tmp_path):
    out=tmp_path/'failed'
    with pytest.raises(ValueError): workflow.repeat_deep_learning_csv(CSV,'missing',out)
    assert json.loads((out/'manifest.json').read_text())['status']=='failed'


def test_missing_tensorflow_can_import_package():
    import subprocess,sys
    code='''
import sys
sys.modules['tensorflow'] = None
import diy_automl
from diy_automl.validation import validate
print(diy_automl.__version__)
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

def test_real_training_failure_retains_finished_epochs(monkeypatch):
    import tensorflow as tf
    original = tf.keras.Model.fit
    class Interrupt(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            raise RuntimeError('after first epoch')
    def interrupted(model,*args,**kwargs):
        kwargs['callbacks'] = [*kwargs.get('callbacks',[]),Interrupt()]
        return original(model,*args,**kwargs)
    monkeypatch.setattr(tf.keras.Model,'fit',interrupted)
    d=load_csv(CSV,'target')
    with pytest.raises(dl.TrainingFailure) as caught:
        dl.run(d.X,d.y,epochs=3)
    assert caught.value.history.Epoch.tolist()==[1]
    assert len(caught.value.weights)==8
