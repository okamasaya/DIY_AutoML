from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from diy_automl import clustering_subsample_workflow as workflow
from diy_automl.clustering import evaluate,run as single_run
from diy_automl.clustering_subsample import compare_assignments,sample_plan
from diy_automl.data import load_csv
from diy_automl.ul_subsample_validation import validate_ul_subsample
from diy_automl.validation import validate

CSV=Path(__file__).parents[1]/'data/reference/iris.csv'


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    return workflow.subsample_clustering_csv(CSV,'target',tmp_path_factory.mktemp('subsample')/'run',
                                            sample_seeds=[43,42,44],clusters=[2])


def test_index_alignment_and_label_permutation():
    left_index=np.array([10,20,30,40,50,60]);left=np.array([0,0,1,1,2,2])
    right_index=np.array([60,10,40,20]);right=np.array([8,7,9,7])
    result=compare_assignments(left_index,left,right_index,right,min_overlap=2)
    assert result['Comparison Status']=='valid' and result['Shared Rows']==4
    assert result['Assignment ARI']==1
    assert adjusted_rand_score(left[:4],right)!=1


@pytest.mark.parametrize('indices,labels,minimum,status',[
    ([0,1],[0,1],3,'insufficient_overlap'),
    ([0,1,2,3],[8,8,8,8],2,'degenerate'),
    ([],[],2,'insufficient_overlap'),
])
def test_uninformative_comparisons_unscored(indices,labels,minimum,status):
    result=compare_assignments(np.arange(4),[0,0,1,1],indices,labels,minimum)
    assert result['Comparison Status']==status
    assert np.isnan(result['Assignment ARI'])


def test_artifacts_and_reference_equivalence(completed):
    m=validate(completed,6)
    assert m['sheets']=={'UL_Reference':6,'UL_By_Subsample':18,'UL_Sample_Summary':6,
                        'UL_Overlap_Pairs':18,'UL_Overlap_Summary':6,'UL_Samples':3,
                        'UL_Sample_Indices':360,'UL_Settings':6,'Summary':10}
    assert not any('best' in k.lower() or 'selected' in k.lower() for k in m)
    frame=pd.read_excel(completed/'report.xlsx',sheet_name='UL_By_Subsample')
    assert frame.loc[frame.Model!='GMM','Converged'].isna().all()
    assert frame.loc[frame.Model=='GMM','Converged'].notna().all()
    d=load_csv(CSV,'target');_,single=single_run(d.X,d.y,seed=42,ks=[2])
    with np.load(completed/'arrays.npz') as a:
        for cid in range(6): np.testing.assert_array_equal(a[f'reference_labels_{cid}'],single[f'labels_{cid}'])
        for seed in [43,42,44]:
            index=a[f'sample_{seed}_indices']
            assert len(index)==120 and len(np.unique(index))==120 and np.all(np.diff(index)>0)
            np.testing.assert_array_equal(index,np.sort(np.random.default_rng(seed).choice(150,120,replace=False)))
        index=a['sample_42_indices']
        _,subset=single_run(d.X[index],d.y[index],seed=42,ks=[2])
        np.testing.assert_array_equal(a['sample_42_scaler_mean'],subset['scaler_mean'])
        for cid in range(6): np.testing.assert_array_equal(a[f'sample_42_labels_{cid}'],subset[f'labels_{cid}'])


def fake_fit(X,y,name,k,model):
    labels=np.arange(len(y))%k
    return {'Model':name,'Cluster':k,**evaluate(X,y,labels),'Train Time (s)':0.,
            'Converged':True,'Iterations':1,'Warnings':'','Parameters':repr(model.get_params())},labels


def test_each_scaler_uses_only_its_subset(tmp_path,monkeypatch):
    captured=[];model_seeds=[]
    original=workflow.StandardScaler.fit
    def fit(scaler,X,*args,**kwargs):
        captured.append(X.copy());return original(scaler,X,*args,**kwargs)
    def train(X,y,name,k,model):
        if name in ['KMeans','GMM']: model_seeds.append(model.random_state)
        return fake_fit(X,y,name,k,model)
    monkeypatch.setattr(workflow.StandardScaler,'fit',fit)
    monkeypatch.setattr(workflow,'fit_one',train)
    out=workflow.subsample_clustering_csv(CSV,'target',tmp_path/'once',sample_seeds=[42,43],clusters=[2],model_seed=7)
    d=load_csv(CSV,'target')
    assert len(captured)==3 and set(model_seeds)=={7}
    np.testing.assert_array_equal(captured[0],d.X)
    with np.load(out/'arrays.npz') as a:
        for i,seed in enumerate([42,43]):
            np.testing.assert_array_equal(captured[i+1],d.X[a[f'sample_{seed}_indices']])
    validate_ul_subsample(out)


@pytest.mark.parametrize('options',[
    {'sample_seeds':[]},{'sample_seeds':[42]},{'sample_seeds':[42,42]},
    {'sample_seeds':[42,True]},{'sample_seeds':[42,1.2]}, {'sample_seeds':[42,-1]},
    {'fraction':0},{'fraction':1},{'fraction':np.nan},{'fraction':True},
    {'model_seed':-1},{'clusters':[]},{'clusters':[2,2]},{'clusters':[11]},
    {'min_overlap':1},{'sample_seeds':list(range(40))},
])
def test_invalid_configuration_before_output(tmp_path,options):
    out=tmp_path/'invalid'
    with pytest.raises(ValueError): workflow.subsample_clustering_csv(CSV,'target',out,**options)
    assert not out.exists()


@pytest.mark.parametrize('shape',[(2001,2),(30,101)])
def test_computation_limits(tmp_path,shape):
    X=np.zeros(shape);frame=pd.DataFrame(X);frame['target']=np.arange(shape[0])%2
    path=tmp_path/'input.csv';frame.to_csv(path,index=False)
    out=tmp_path/'limit'
    with pytest.raises(ValueError,match='at most'): workflow.subsample_clustering_csv(path,'target',out)
    assert json.loads((out/'manifest.json').read_text())['status']=='failed'


def test_subset_must_support_requested_k(tmp_path):
    with pytest.raises(ValueError,match='Subsample'):
        workflow.subsample_clustering_csv(CSV,'target',tmp_path/'small',fraction=.01)


@pytest.mark.parametrize('failure_index',[0,7,13])
def test_model_failure_retains_partial_results(tmp_path,monkeypatch,failure_index):
    count=0;out=tmp_path/'failure'
    def train(*args):
        nonlocal count
        checkpoint=json.loads((out/'manifest.json').read_text())
        assert checkpoint['status']=='running'
        current=count;count+=1
        if current==failure_index: raise RuntimeError('injected fit failure')
        return fake_fit(*args)
    monkeypatch.setattr(workflow,'fit_one',train)
    with pytest.raises(RuntimeError,match='injected'):
        workflow.subsample_clustering_csv(CSV,'target',out,sample_seeds=[42,43,44],clusters=[2])
    with pytest.raises(ValueError,match='complete'): validate_ul_subsample(out)
    m=validate_ul_subsample(out,allow_failed=True)
    assert m['status']=='failed' and count==failure_index+1
    frames=pd.read_excel(out/'report.xlsx',sheet_name=None)
    states=frames['UL_Reference'].Status.tolist()+frames['UL_By_Subsample'].Status.tolist()
    assert states==['completed']*failure_index+['failed']+['not_run']*(23-failure_index)


@pytest.mark.parametrize('failure_index',[0,1,2])
def test_scaler_failures_are_identified(tmp_path,monkeypatch,failure_index):
    count=0;original=workflow.StandardScaler.fit
    def fit(scaler,*args,**kwargs):
        nonlocal count
        current=count;count+=1
        if current==failure_index: raise RuntimeError('injected scaler failure')
        return original(scaler,*args,**kwargs)
    monkeypatch.setattr(workflow.StandardScaler,'fit',fit)
    monkeypatch.setattr(workflow,'fit_one',fake_fit)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='scaler'):
        workflow.subsample_clustering_csv(CSV,'target',out,sample_seeds=[42,43],clusters=[2])
    m=validate_ul_subsample(out,allow_failed=True)
    assert m['stage']==('reference_scaling' if failure_index==0 else 'sample_scaling')


def test_small_overlap_is_not_scored(tmp_path):
    out=workflow.subsample_clustering_csv(CSV,'target',tmp_path/'low-overlap',sample_seeds=[42,43],
                                         fraction=.1,clusters=[2],min_overlap=10)
    validate_ul_subsample(out)
    pairs=pd.read_excel(out/'report.xlsx',sheet_name='UL_Overlap_Pairs')
    assert pairs['Comparison Status'].eq('insufficient_overlap').all()
    assert pairs['Assignment ARI'].isna().all()


def update_checksum(out,name):
    p=out/'manifest.json';m=json.loads(p.read_text())
    m['artifacts'][name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    p.write_text(json.dumps(m))


@pytest.mark.parametrize('kind',['indices','scaler','labels','reference-ari','pair','summary','states','checksum'])
def test_semantic_corruption_detected(completed,tmp_path,kind):
    out=tmp_path/'changed';shutil.copytree(completed,out)
    if kind in ['indices','scaler','labels']:
        with np.load(out/'arrays.npz') as a: values={key:a[key] for key in a.files}
        if kind=='indices': values['sample_42_indices']=values['sample_42_indices'][::-1]
        elif kind=='scaler': values['sample_42_scaler_mean']+=1
        else: values['sample_42_labels_0'][0]=99
        np.savez_compressed(out/'arrays.npz',**values);update_checksum(out,'arrays.npz')
    elif kind!='checksum':
        frames=pd.read_excel(out/'report.xlsx',sheet_name=None)
        if kind=='reference-ari': frames['UL_By_Subsample'].loc[0,'Reference ARI']=99
        elif kind=='pair': frames['UL_Overlap_Pairs'].loc[0,'Assignment ARI']=99
        elif kind=='summary': frames['UL_Sample_Summary'].loc[0,'Reference ARI_Mean']=99
        else: frames['UL_By_Subsample'].loc[0,'Status']='not_run'
        with pd.ExcelWriter(out/'report.xlsx') as writer:
            for name,frame in frames.items(): frame.to_excel(writer,sheet_name=name,index=False)
        update_checksum(out,'report.xlsx')
    else:
        with (out/'arrays.npz').open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError): validate_ul_subsample(out)
