from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from diy_automl.data import load_csv,dl_indices,scale_train,split_indices
from diy_automl.supervised import run as sl_run
from diy_automl.clustering import run as ul_run
from diy_automl.selection import choose,bootstrap
from diy_automl.selection_workflow import select_csv
from diy_automl.clustering_repeated import run as repeat_run,RepeatFailure
from diy_automl.clustering_repeat_workflow import repeat_clustering_csv

CSV=Path(__file__).parents[1]/'data/reference/iris.csv'

def test_input():
    d=load_csv(CSV,'target')
    assert d.X.shape==(150,4)
    assert len(d.classes)==3

@pytest.mark.parametrize('kind',['missing','nonfinite','text','oneclass','nofeatures'])
def test_invalid_csv(tmp_path,kind):
    frame=pd.DataFrame({'x':[1.,2.,3.,4.],'target':[0,0,1,1]})
    if kind=='missing': frame.loc[0,'x']=np.nan
    if kind=='nonfinite': frame.loc[0,'x']=np.inf
    if kind=='text': frame['x']='abc'
    if kind=='oneclass': frame['target']=0
    if kind=='nofeatures': frame=frame[['target']]
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    with pytest.raises(ValueError): load_csv(path,'target')

def test_train_only_scaling_and_split():
    X,y=load_iris(return_X_y=True)
    train,val,test=dl_indices(y)
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    assert sorted(np.concatenate([train,val,test]))==list(range(len(y)))
    assert set(test)==set(split_indices(y)[1])
    changed=X.copy();changed[val]+=10000;changed[test]-=10000
    a,s1=scale_train(X,train);b,s2=scale_train(changed,train)
    np.testing.assert_array_equal(s1.mean_,s2.mean_)
    np.testing.assert_array_equal(a[train],b[train])

def test_sl_models():
    X,y=load_iris(return_X_y=True)
    result,arrays=sl_run(X,y)
    assert len(result)==5
    assert len(arrays['test'])==30
    assert result.F1.between(0,1).all()

def test_ul_54_unique():
    X,y=load_iris(return_X_y=True)
    result,arrays=ul_run(X,y)
    assert len(result)==54
    assert not result.duplicated(['Model','Cluster']).any()
    assert sum(k.startswith('labels_') for k in arrays)==54

@pytest.mark.parametrize('gap,expected',[(0,0),(5e-13,0),(2e-12,1)])
def test_ties(gap,expected):
    selected,ties=choose(pd.DataFrame({'F1':[.8,.8+gap]}))
    assert selected==expected

def test_bootstrap_class_counts_and_reproducibility():
    y=np.array([0]*8+[1]*2);pred=np.array([0]*7+[1,0,1])
    a,i=bootstrap(y,pred,30);b,j=bootstrap(y,pred,30)
    pd.testing.assert_frame_equal(a,b)
    assert (i.Lower<=i.Upper).all()
    _,perfect=bootstrap(y,y,20)
    assert (perfect.Lower==1).all() and (perfect.Upper==1).all()

def test_selection_artifacts(tmp_path):
    out=select_csv(CSV,'target',tmp_path/'selection',bootstrap_repeats=20)
    m=json.loads((out/'manifest.json').read_text())
    assert m['status']=='completed' and len(m['sheets'])==7
    assert m['sheets']['SL_Test']==1
    with np.load(out/'arrays.npz') as a:
        test=set(a['test']);train=set(a['train'])
        data=load_csv(CSV,'target')
        np.testing.assert_allclose(a['scaler_mean'],data.X[a['train']].mean(axis=0))
        for i in range(5):
            assert not test&set(a[f'fold_{i}_train'])
            assert not test&set(a[f'fold_{i}_validation'])
            assert set(a[f'fold_{i}_train'])|set(a[f'fold_{i}_validation'])==train

def test_repeat_counts_and_assignments():
    X,y=load_iris(return_X_y=True)
    sheets,arrays=repeat_run(X,y,seeds=[42,43,44],ks=[2,3])
    assert len(sheets['UL_By_Seed'])==12
    assert len(sheets['UL_Seed_Pairs'])==12
    assert len(sheets['UL_Reference'])==8
    assert (sheets['UL_By_Seed'].Status=='completed').all()
    _,single=ul_run(X,y,42,ks=[2,3])
    np.testing.assert_array_equal(arrays['labels_42_KMeans_2'],single['labels_0'])
    np.testing.assert_array_equal(arrays['labels_42_GMM_2'],single['labels_1'])

def test_repeat_failure_records(monkeypatch):
    import diy_automl.clustering_repeated as module
    X,y=load_iris(return_X_y=True)
    original=module.fit_one
    count=0
    def failure(*args):
        nonlocal count
        count+=1
        if count==2: raise RuntimeError('injected failure')
        return original(*args)
    monkeypatch.setattr(module,'fit_one',failure)
    with pytest.raises(RepeatFailure) as caught: module.run(X,y,[42,43],[2])
    states=caught.value.result[0]['UL_By_Seed'].Status.tolist()
    assert states==['completed','failed','not_run','not_run']

def test_failed_manifest(tmp_path):
    out=tmp_path/'failed'
    with pytest.raises(ValueError): select_csv(CSV,'absent',out)
    assert json.loads((out/'manifest.json').read_text())['status']=='failed'

def test_dl_training_seed_independence():
    from diy_automl.deep_learning import run
    X,y=load_iris(return_X_y=True)
    _,h,a=run(X,y,epochs=2,training_seed=42)
    _,_,b=run(X,y,epochs=2,training_seed=43)
    _,_,c=run(X,y,epochs=2,training_seed=42)
    for key in ['train','validation','test','scaler_mean','scaler_scale']:
        np.testing.assert_array_equal(a[key],b[key])
    assert not np.array_equal(a['weight_0'],b['weight_0'])
    for key in a: np.testing.assert_array_equal(a[key],c[key])
    np.testing.assert_allclose(a['scaler_mean'],X[a['train']].mean(axis=0))
    assert not np.array_equal(dl_indices(y,43)[0],a['train'])
    assert len(h)==2 and not h.isna().any().any()

def test_repeat_first_failure(monkeypatch):
    import diy_automl.clustering_repeated as module
    X,y=load_iris(return_X_y=True)
    def failure(*args): raise RuntimeError('first candidate failed')
    monkeypatch.setattr(module,'fit_one',failure)
    with pytest.raises(RepeatFailure) as caught: module.run(X,y,[42,43],[2])
    assert caught.value.result[0]['UL_By_Seed'].Status.tolist()==['failed','not_run','not_run','not_run']

def test_validator_detects_tampering(tmp_path):
    from diy_automl.validation import validate
    out=select_csv(CSV,'target',tmp_path/'selection',bootstrap_repeats=5)
    validate(out,3)
    with (out/'arrays.npz').open('ab') as stream: stream.write(b'tamper')
    with pytest.raises(ValueError,match='checksum'): validate(out,3)

def test_no_overwrite(tmp_path):
    out=tmp_path/'existing';out.mkdir();(out/'keep').write_text('original')
    with pytest.raises(FileExistsError): select_csv(CSV,'target',out,bootstrap_repeats=5)
    assert (out/'keep').read_text()=='original'

def test_duplicate_headers(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('x,x,target\n1,2,0\n2,3,1\n')
    with pytest.raises(ValueError,match='Duplicate'): load_csv(p,'target')

def test_repeat_failed_artifacts(tmp_path,monkeypatch):
    import diy_automl.clustering_repeated as module
    def failure(*args): raise RuntimeError('first failed')
    monkeypatch.setattr(module,'fit_one',failure)
    out=tmp_path/'failure'
    with pytest.raises(RuntimeError): repeat_clustering_csv(CSV,'target',out,seeds=[42,43])
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='failed'
    frames=pd.read_excel(out/'report.xlsx',sheet_name=None)
    assert frames['UL_By_Seed'].Status.tolist()==['failed']+['not_run']*35
    assert frames['UL_Reference'].Status.eq('not_run').all()
    assert (out/'arrays.npz').exists()
