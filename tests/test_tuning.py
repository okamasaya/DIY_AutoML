from pathlib import Path
import hashlib
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diy_automl import tuning
from diy_automl import tuning_workflow as workflow
from diy_automl.data import load_csv
from diy_automl.selection import bootstrap
from diy_automl.supervised import candidates,metrics
from diy_automl.tuning_validation import validate_tuning
from diy_automl.validation import validate

CSV=Path(__file__).parents[1]/'data/reference/iris.csv'


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    return workflow.tune_csv(CSV,'target',tmp_path_factory.mktemp('tune')/'run',bootstrap_repeats=20)


def test_grid_order_and_existing_defaults():
    grid=tuning.search_candidates(42)
    assert len(grid)==27
    assert [sum(g[0]==name for g in grid) for name,_ in candidates()]==[6,4,3,6,8]
    for index,(_,model) in zip([0,6,10,13,19],candidates()):
        assert grid[index][1].get_params()==model.get_params()


def test_complete_artifacts_and_single_test(completed):
    m=validate(completed,7)
    assert m['sheets']=={'SL_Search_Settings':27,'SL_Search_Folds':135,'SL_Search_Mean':27,
                         'SL_Search_SD':27,'SL_Test':1,'SL_Bootstrap':20,'SL_Test_Intervals':4,'Summary':10}
    means=pd.read_excel(completed/'report.xlsx',sheet_name='SL_Search_Mean')
    winner,ties=tuning.choose(means,'F1')
    assert m['selected_candidate']==winner and m['tied_candidates']==ties
    with np.load(completed/'arrays.npz') as a:
        assert len([key for key in a.files if key.endswith('_predictions')])==135
        np.testing.assert_allclose(a['selected_scaler_mean'],a['features'][a['train']].mean(axis=0))
        assert 'predictions' in a.files


def test_test_values_cannot_change_training_search():
    d=load_csv(CSV,'target')
    train,test=workflow.split_indices(d.y)
    first=tuning.search(d.X[train],d.y[train])
    changed=d.X.copy();changed[test]+=1e6
    second=tuning.search(changed[train],d.y[train])
    assert first[1:3]==second[1:3]
    pd.testing.assert_frame_equal(first[4],second[4],check_exact=True)


def test_fit_rows_and_prediction_boundary(tmp_path,monkeypatch):
    frame=pd.DataFrame({'row_id':np.arange(100),'noise':np.random.default_rng(4).normal(size=100),
                        'target':np.arange(100)%2})
    csv=tmp_path/'ids.csv';frame.to_csv(csv,index=False)
    train,test=workflow.split_indices(frame.target.to_numpy())
    test_ids=set(test);fitted=[];test_calls=[]
    real_fit=StandardScaler.fit;real_predict=Pipeline.predict
    def fit(scaler,X,*args,**kwargs):
        fitted.append(np.asarray(X)[:,0].copy())
        assert not test_ids.intersection(np.asarray(X)[:,0])
        return real_fit(scaler,X,*args,**kwargs)
    def predict(model,X,*args,**kwargs):
        ids=set(np.asarray(X)[:,0])
        if ids & test_ids:
            assert ids==test_ids
            test_calls.append(ids)
        return real_predict(model,X,*args,**kwargs)
    monkeypatch.setattr(StandardScaler,'fit',fit)
    monkeypatch.setattr(Pipeline,'predict',predict)
    out=workflow.tune_csv(csv,'target',tmp_path/'boundary',bootstrap_repeats=5)
    assert len(fitted)==136 and len(test_calls)==1
    # Verification recomputes preprocessing but never fits a classifier.
    monkeypatch.setattr(StandardScaler,'fit',real_fit)
    validate_tuning(out)


@pytest.mark.parametrize('failure_index',[0,7])
def test_failed_cv_saves_state(tmp_path,monkeypatch,failure_index):
    original=tuning.fit_checked;count=0
    def fit(*args):
        nonlocal count
        current=count;count+=1
        if current==failure_index: raise RuntimeError('injected CV failure')
        return original(*args)
    monkeypatch.setattr(tuning,'fit_checked',fit)
    out=tmp_path/'failure'
    with pytest.raises(RuntimeError): workflow.tune_csv(CSV,'target',out,bootstrap_repeats=5)
    with pytest.raises(ValueError): validate_tuning(out)
    m=validate_tuning(out,allow_failed=True)
    assert m['stage']=='cv' and 'selected_candidate' not in m
    states=pd.read_excel(out/'report.xlsx',sheet_name='SL_Search_Folds').Status.tolist()
    assert states==['completed']*failure_index+['failed']+['not_run']*(134-failure_index)


def test_refit_failure_never_tests(tmp_path,monkeypatch):
    def fail(*args): raise RuntimeError('injected refit failure')
    monkeypatch.setattr(workflow,'fit_checked',fail)
    out=tmp_path/'refit-failure'
    with pytest.raises(RuntimeError): workflow.tune_csv(CSV,'target',out,bootstrap_repeats=5)
    m=validate_tuning(out,allow_failed=True)
    assert m['stage']=='refit'
    assert m['sheets']['SL_Test']==0


def test_convergence_warning_stops_tuning(tmp_path,monkeypatch):
    original=StandardScaler.fit
    def warn(*args,**kwargs):
        result=original(*args,**kwargs)
        warnings.warn('injected convergence warning',ConvergenceWarning)
        return result
    monkeypatch.setattr(StandardScaler,'fit',warn)
    out=tmp_path/'warning'
    with pytest.raises(tuning.ConvergenceFailure): workflow.tune_csv(CSV,'target',out)
    monkeypatch.setattr(StandardScaler,'fit',original)
    m=validate_tuning(out,allow_failed=True)
    assert m['stage']=='cv' and 'ConvergenceWarning' in m['error']


@pytest.mark.parametrize('options',[
    {'seed':-1},{'seed':True},{'bootstrap_repeats':0},{'bootstrap_repeats':10001},
    {'bootstrap_seed':-1},{'confidence':0},{'confidence':1},{'confidence':np.nan},{'metric':'unknown'},
])
def test_bad_options_before_output(tmp_path,options):
    out=tmp_path/'invalid'
    with pytest.raises(ValueError): workflow.tune_csv(CSV,'target',out,**options)
    assert not out.exists()


def test_global_class_set_in_conditional_intervals():
    y=np.array([0,0,1,1]);pred=np.array([0,2,1,2]);classes=np.array([0,1,2])
    _,interval=bootstrap(y,pred,20,classes=classes)
    point=metrics(y,pred,classes)
    for _,row in interval.iterrows(): assert row.Estimate==point[row.Metric]


@pytest.mark.parametrize('kind',['test-split','fold-scaler','cv-pred','selection','test-score','bootstrap','checksum'])
def test_inconsistent_saved_results_rejected(completed,tmp_path,kind):
    out=tmp_path/'changed';shutil.copytree(completed,out)
    if kind in ['test-split','fold-scaler','cv-pred']:
        with np.load(out/'arrays.npz') as a: values={key:a[key] for key in a.files}
        if kind=='test-split': values['train'][0]=values['test'][0]
        elif kind=='fold-scaler': values['fold_0_scaler_mean']+=1
        else: values['candidate_0_fold_0_predictions'][0]=99
        np.savez_compressed(out/'arrays.npz',**values)
        artifact='arrays.npz'
    elif kind=='selection':
        p=out/'manifest.json';m=json.loads(p.read_text());m['selected_candidate']=(m['selected_candidate']+1)%27
        p.write_text(json.dumps(m));artifact=None
    elif kind in ['test-score','bootstrap']:
        sheets=pd.read_excel(out/'report.xlsx',sheet_name=None)
        name='SL_Test' if kind=='test-score' else 'SL_Bootstrap'
        sheets[name].loc[0,'F1']=-99
        with pd.ExcelWriter(out/'report.xlsx') as writer:
            for name,frame in sheets.items(): frame.to_excel(writer,sheet_name=name,index=False)
        artifact='report.xlsx'
    else:
        with (out/'arrays.npz').open('ab') as stream: stream.write(b'corrupted')
        artifact=None
    if artifact:
        p=out/'manifest.json';m=json.loads(p.read_text())
        m['artifacts'][artifact]=hashlib.sha256((out/artifact).read_bytes()).hexdigest()
        p.write_text(json.dumps(m))
    with pytest.raises(ValueError): validate_tuning(out)
